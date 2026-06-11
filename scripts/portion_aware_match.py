"""Portion-aware LLM candidate retrieval and prompt formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from amount_kind import AmountKind, classify_from_parsed_row
from ingredient_match_staged import (
    LLMRetrievalConfig,
    QueryRow,
    StagedFoodIndex,
    retrieve_llm_candidates,
)
from portion_candidate_index import (
    PortionSummaryLine,
    has_container_mass_portion,
    summarize_fdc_portions,
)
from portion_gram import PortionCapabilitySets, SENTINEL_FDC_ID
from resolution_plan import (
    ResolutionPlan,
    ingredient_has_mass_reference,
    plan_from_parsed_row,
)

SEMANTIC_BLEND = 0.45
PORTION_BLEND = 0.55
SEMANTIC_POOL_SIZE = 30
MIN_PORTION_VIABLE_IN_TOP10 = 7


@dataclass(frozen=True)
class PortionRetrievalResult:
    candidates: pd.DataFrame
    amount_kind: AmountKind
    retrieval_tier: str
    portion_filter_kind: str | None
    n_tier1_union: int
    tier1_max_score: float | None
    mass_in_text: bool = False
    query_tokens: tuple[str, ...] = ()


def portion_flag_for_fdc(
    fdc_id: int,
    *,
    volume_fdc_ids: set[int],
    count_fdc_ids: set[int],
    summary_lines: list[PortionSummaryLine] | None = None,
) -> str:
    if int(fdc_id) == SENTINEL_FDC_ID:
        return "-"
    has_v = int(fdc_id) in volume_fdc_ids
    has_c = int(fdc_id) in count_fdc_ids
    has_cm = has_container_mass_portion(summary_lines or [])
    if has_v and (has_c or has_cm):
        return "VC" if has_cm else "V"
    if has_c and has_v:
        return "VC"
    if has_cm:
        return "Cm"
    if has_v:
        return "V"
    if has_c:
        return "C"
    return "-"


def add_portion_flags(
    cand_df: pd.DataFrame,
    *,
    volume_fdc_ids: set[int],
    count_fdc_ids: set[int],
    summary_index: dict[int, list[PortionSummaryLine]] | None = None,
) -> pd.DataFrame:
    if cand_df.empty:
        return cand_df
    out = cand_df.copy()
    summary_index = summary_index or {}

    def _flag(fid: int) -> str:
        return portion_flag_for_fdc(
            int(fid),
            volume_fdc_ids=volume_fdc_ids,
            count_fdc_ids=count_fdc_ids,
            summary_lines=summary_index.get(int(fid), []),
        )

    out["portion_flag"] = out["fdc_id"].map(_flag)
    out["has_volume_portion"] = out["fdc_id"].map(lambda x: int(x) in volume_fdc_ids)
    out["has_count_portion"] = out["fdc_id"].map(lambda x: int(x) in count_fdc_ids)
    return out


def _query_tokens_from_plan(
    plan: ResolutionPlan,
    amount_kind: AmountKind,
) -> list[str]:
    if "count_portion" in plan.resolution_paths:
        return plan.count_query_tokens()
    if amount_kind == "volume" and plan.unit:
        return [str(plan.unit)]
    if amount_kind == "count":
        return plan.count_query_tokens()
    return []


def _attach_portion_scores(
    cand_df: pd.DataFrame,
    summary_index: dict[int, list[PortionSummaryLine]],
    query_tokens: list[str],
    *,
    amount_kind: AmountKind,
) -> pd.DataFrame:
    if cand_df.empty:
        return cand_df
    out = cand_df.copy()
    portion_scores: list[float] = []
    portion_lines: list[str] = []
    best_portion_ids: list[int | None] = []
    for row in out.itertuples(index=False):
        score, display, best_pid = summarize_fdc_portions(
            summary_index,
            int(row.fdc_id),
            query_tokens,
            amount_kind=amount_kind if amount_kind in ("volume", "count") else None,
            retrieval_score=float(getattr(row, "retrieval_score", 0.0) or 0.0),
        )
        portion_scores.append(score)
        portion_lines.append(display)
        best_portion_ids.append(best_pid)
    out["portion_match_score"] = portion_scores
    out["portion_summary"] = portion_lines
    out["best_portion_id"] = best_portion_ids
    out["blended_score"] = (
        SEMANTIC_BLEND * out["retrieval_score"] + PORTION_BLEND * out["portion_match_score"]
    ).round(4)
    return out


def _rank_and_trim(
    cand_df: pd.DataFrame,
    rc: LLMRetrievalConfig,
    *,
    require_portion_match: bool,
) -> pd.DataFrame:
    if cand_df.empty:
        return cand_df
    pool = cand_df.copy()
    if require_portion_match:
        with_match = pool[pool["portion_match_score"] > 0]
        if not with_match.empty:
            pool = with_match
    pool = pool.sort_values(
        ["blended_score", "retrieval_score", "staged_final_score", "fdc_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    pool = pool.head(SEMANTIC_POOL_SIZE)
    pool = pool.drop(columns=["rank"], errors="ignore")
    pool.insert(0, "rank", range(1, len(pool) + 1))
    pool["in_llm_prompt"] = pool["rank"] <= rc.top10_size
    return pool


def _ensure_portion_viable_top10(
    ranked: pd.DataFrame,
    cand_df: pd.DataFrame,
    rc: LLMRetrievalConfig,
) -> pd.DataFrame:
    """Prefer portion-viable candidates in top-10; avoid P=- backfill when viable exist."""
    if ranked.empty or cand_df.empty:
        return ranked

    viable = cand_df[cand_df["portion_match_score"] > 0].sort_values(
        "blended_score", ascending=False
    )
    if viable.empty:
        return ranked

    prompt = ranked[ranked["in_llm_prompt"]].copy()
    n_viable_in_prompt = int((prompt["portion_match_score"] > 0).sum())
    target = min(rc.top10_size, max(MIN_PORTION_VIABLE_IN_TOP10, n_viable_in_prompt))

    if n_viable_in_prompt >= target:
        return ranked

    need = target - n_viable_in_prompt
    have_ids = set(prompt["fdc_id"].tolist())
    extras = viable[~viable["fdc_id"].isin(have_ids)].head(need)
    if extras.empty:
        return ranked

    non_viable = prompt[prompt["portion_match_score"] <= 0].sort_values("rank", ascending=False)
    drop_n = min(len(extras), len(non_viable))
    if drop_n == 0:
        merged = pd.concat([prompt, extras], ignore_index=True)
    else:
        keep_prompt = prompt[~prompt["fdc_id"].isin(non_viable.head(drop_n)["fdc_id"])]
        merged = pd.concat([keep_prompt, extras], ignore_index=True)

    rest = ranked[~ranked["fdc_id"].isin(merged["fdc_id"])]
    out = pd.concat([merged, rest], ignore_index=True)
    out = out.drop_duplicates(subset=["fdc_id"], keep="first")
    out = out.sort_values("blended_score", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out["in_llm_prompt"] = out["rank"] <= rc.top10_size
    return out


def retrieve_llm_candidates_portion_aware(
    query: QueryRow,
    index: StagedFoodIndex,
    capabilities: PortionCapabilitySets,
    retr_config: LLMRetrievalConfig | None = None,
    *,
    amount_kind: AmountKind | None = None,
    staged_top1_fdc_id: int | None = None,
    precomputed_sims: Any = None,
    tier1_score_floor: float = 0.45,
    parsed_row: dict[str, Any] | None = None,
    resolution_plan: ResolutionPlan | dict[str, Any] | None = None,
    portion_summary_index: dict[int, list[PortionSummaryLine]] | None = None,
) -> PortionRetrievalResult:
    """Portion-informed retrieval: mass-in-text uses semantic top-10; else portion-ranked pool."""
    rc = retr_config or LLMRetrievalConfig()
    row_dict = parsed_row or {}
    plan = (
        resolution_plan
        if isinstance(resolution_plan, ResolutionPlan)
        else plan_from_parsed_row(row_dict) if row_dict else ResolutionPlan()
    )
    if isinstance(plan, dict):
        from resolution_plan import _normalize_plan_dict

        fields = _normalize_plan_dict(plan)
        plan = ResolutionPlan(**{k: v for k, v in fields.items() if k in ResolutionPlan.__dataclass_fields__})

    kind: AmountKind = amount_kind or plan.primary_amount_kind or (
        classify_from_parsed_row(row_dict) if row_dict else "unknown"
    )
    if kind == "unknown" and plan.primary_amount_kind != "unknown":
        kind = plan.primary_amount_kind  # type: ignore[assignment]

    ingredient_raw = str(row_dict.get("ingredient") or query.dequant_text or "")
    mass_in_text = ingredient_has_mass_reference(
        ingredient_raw,
        embedded_mass_qty=plan.embedded_mass_qty,
        embedded_mass_unit=plan.embedded_mass_unit,
    )
    query_tokens = _query_tokens_from_plan(plan, kind)
    summary_index = portion_summary_index or {}

    wide_rc = LLMRetrievalConfig(
        lexical_min_token_overlap=rc.lexical_min_token_overlap,
        lexical_score_floor=rc.lexical_score_floor,
        lexical_top_k=rc.lexical_top_k,
        semantic_top_k=max(rc.semantic_top_k, SEMANTIC_POOL_SIZE),
        semantic_score_floor=rc.semantic_score_floor,
        semantic_floor_cap=rc.semantic_floor_cap,
        max_candidates=SEMANTIC_POOL_SIZE,
        description_max_chars=rc.description_max_chars,
        top10_size=rc.top10_size,
        semantic_blend_weight=rc.semantic_blend_weight,
        lexical_blend_weight=rc.lexical_blend_weight,
    )

    cand_df = retrieve_llm_candidates(
        query,
        index,
        wide_rc,
        staged_top1_fdc_id=staged_top1_fdc_id,
        precomputed_sims=precomputed_sims,
    )
    n_union = int(cand_df.attrs.get("n_union", 0)) if not cand_df.empty else 0
    tier1_max = float(cand_df["retrieval_score"].max()) if not cand_df.empty else None

    volume_fdc_ids = set(capabilities.volume_fdc_ids)
    count_fdc_ids = set(capabilities.count_fdc_ids)

    if mass_in_text or not summary_index:
        tier = "mass_in_text" if mass_in_text else "semantic_only"
        if not cand_df.empty:
            cand_df = cand_df.head(rc.top10_size).copy()
            cand_df = cand_df.drop(columns=["rank"], errors="ignore")
            cand_df.insert(0, "rank", range(1, len(cand_df) + 1))
            cand_df["in_llm_prompt"] = True
            cand_df["portion_match_score"] = 0.0
            cand_df["portion_summary"] = "-"
            cand_df["best_portion_id"] = None
            cand_df["blended_score"] = cand_df["retrieval_score"]
        cand_df = add_portion_flags(
            cand_df,
            volume_fdc_ids=volume_fdc_ids,
            count_fdc_ids=count_fdc_ids,
            summary_index=summary_index,
        )
        return PortionRetrievalResult(
            candidates=cand_df,
            amount_kind=kind,
            retrieval_tier=tier,
            portion_filter_kind=None,
            n_tier1_union=n_union,
            tier1_max_score=tier1_max,
            mass_in_text=mass_in_text,
            query_tokens=tuple(query_tokens),
        )

    require_portion = "count_portion" in plan.resolution_paths or kind in ("volume", "count")
    cand_df = _attach_portion_scores(cand_df, summary_index, query_tokens, amount_kind=kind)
    ranked = _rank_and_trim(cand_df, rc, require_portion_match=require_portion)
    ranked = _ensure_portion_viable_top10(ranked, cand_df, rc)

    ranked = add_portion_flags(
        ranked,
        volume_fdc_ids=volume_fdc_ids,
        count_fdc_ids=count_fdc_ids,
        summary_index=summary_index,
    )
    return PortionRetrievalResult(
        candidates=ranked,
        amount_kind=kind,
        retrieval_tier="portion_ranked",
        portion_filter_kind=kind if kind in ("volume", "count") else None,
        n_tier1_union=n_union,
        tier1_max_score=tier1_max,
        mass_in_text=False,
        query_tokens=tuple(query_tokens),
    )


def format_candidate_block_portion(
    prompt_candidates: pd.DataFrame,
    max_chars: int,
) -> str:
    if prompt_candidates.empty:
        return "(none)"
    lines = []
    for row in prompt_candidates.itertuples(index=False):
        desc = str(row.description)[:max_chars]
        pflag = getattr(row, "portion_flag", "-")
        portions = getattr(row, "portion_summary", "-") or "-"
        fit = getattr(row, "portion_match_score", 0.0) or 0.0
        lines.append(
            f"{row.fdc_id} | {desc} | {row.lexical_dequant:.2f} | "
            f"{row.dequant_sem:.2f} | {pflag} | portions: {portions} | fit={fit:.2f}"
        )
    return "\n".join(lines)


def build_user_prompt_portion(
    ingredient: str,
    name: str,
    preparation: str,
    unit: str,
    amount_kind: str,
    prompt_candidates: pd.DataFrame,
    steps: list[str],
    max_chars: int,
    *,
    mass_in_text: bool = False,
    query_tokens: list[str] | None = None,
) -> str:
    parts = [
        f"INGREDIENT: {ingredient}",
        (
            f"PARSED: name={name or '-'}; prep={preparation or '-'}; "
            f"unit={unit or '-'}; amount_kind={amount_kind}"
        ),
    ]
    if query_tokens:
        parts.append(f"PORTION_QUERY_TOKENS: {', '.join(query_tokens)}")
    if mass_in_text:
        parts.append(
            "NOTE: Recipe line includes explicit mass; grams will convert from mass directly."
        )
    parts.extend(
        [
            "",
            "CANDIDATES (fdc_id | description | L | S | P | portions | fit):",
            "  P = V volume, C count, Cm container-mass (e.g. can), VC both, -=none",
            "  portions = top USDA portion lines; fit = portion label match 0-1",
            "  For count/volume: prefer P!=- with fit>0 over slightly better lexical match without portions.",
            "  Prep (minced/chopped) does NOT change count amounts; prefer raw/generic with count portions.",
            format_candidate_block_portion(prompt_candidates, max_chars),
        ]
    )
    if steps:
        parts.append("")
        parts.append("STEPS:")
        for i, step in enumerate(steps, 1):
            parts.append(f"{i}. {step}")
    parts.append("")
    parts.append(
        "Select best fdc_id. Return matched_portion_id when a portion line is a good-enough match "
        "(exact unit/size, container modifier, or acceptable size fallback). "
        "Set negligible_calories=true for nutritionally insignificant amounts (baking powder, spices) "
        "when grams cannot be resolved."
    )
    return "\n".join(parts)


def build_pipeline_path(
    *,
    amount_kind: str,
    retrieval_tier: str,
    grams_status: str | None,
    llm_fdc_id: int | None,
) -> str:
    pick = "abstain" if llm_fdc_id is None else f"fdc={llm_fdc_id}"
    gram = grams_status or "pending"
    return f"{amount_kind}→{retrieval_tier}→llm→{pick}→{gram}"
