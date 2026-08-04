"""Configuration for the recipe optimization agent."""

from __future__ import annotations

from dataclasses import dataclass, field


# Identity templates: canonical title substrings → identity roles that cannot be emptied.
IDENTITY_TEMPLATES: dict[str, list[str]] = {
    "carbonara": ["pasta", "egg", "cheese", "cured_pork"],
    "cheese pizza": ["cheese", "crust"],
    "pizza": ["crust"],
    "margherita": ["cheese", "tomato", "crust"],
}


@dataclass
class AgentConfig:
    # Model tiers (see model_policy.py). Strong/complex default is gpt-4.1-mini
    # (cheaper/faster than gpt-4o); override creative/escalate/judge to gpt-4o for A/B.
    model: str = "gpt-4o-mini"
    model_escalate: str = "gpt-4.1-mini"
    creative_model: str = "gpt-4.1-mini"
    # Silent one-shot draft→optimize candidate (not shown as authored by this model).
    shadow_draft_model: str = "gpt-5.5"
    enable_shadow_gpt_candidate: bool = True
    tags_model: str = "gpt-4.1-nano"
    judge_model: str = "gpt-4.1-mini"
    # LLM judge screens for clash / obviously-bad ingredient lists and demotes
    # those recipes to the end. Ordering among normal recipes stays deterministic
    # (proportion quality). Set False to skip the LLM call entirely.
    enable_llm_judge: bool = True
    # demote_weird: flag clashes only (default). full: old arbiter picks a winner.
    llm_judge_mode: str = "demote_weird"
    max_iterations: int = 3
    F_accept: float = 1.0
    F_max: float = 1.5
    protein_min: float = 0.19
    protein_max: float = 0.23
    carb_min: float = 0.345
    carb_max: float = 0.545
    fat_min: float = 0.245
    fat_max: float = 0.445
    # Optional absolute calorie target (Atwater). Propagated into drafts + LP.
    kcal_target: float | None = None
    # Dimensionless weight on L1 PFC box violation. None restores hard bounds.
    nutrition_slack_weight: float | None = 1.0
    neighbor_k: int = 40
    loss_field_grid_n: int = 11  # keep small for speed in v1
    w_foodon: float = 0.50
    w_semantic: float = 0.35
    w_cuisine: float = 0.15
    identity_templates: dict[str, list[str]] = field(default_factory=lambda: dict(IDENTITY_TEMPLATES))
    # Creative mode scoring weights (goodness composite)
    w_score_nutrient: float = 0.4
    w_score_ratio: float = 0.3
    w_score_intent: float = 0.2
    w_score_churn: float = 0.1
    judge_epsilon: float = 0.03
    save_on_must_retry_feasible: bool = True
    min_finalists: int = 2
    max_finalists: int = 4
    agent_mode: str = "neighborhood"  # neighborhood | creative
    # Auto-apply clear LP favorite
    auto_apply_delta_eps: float = 0.01
    auto_apply_margin: float = 0.02
    # Anti-oscillation: require this much extra improvement to re-apply same fingerprint
    oscillation_improve_eps: float = 0.02
    # Identity LLM extract (uses mini by default)
    identity_extract_model: str = "gpt-4o-mini"
    # Cap FoodOn leaf→basis hops. None → half the average FoodOn leaf depth.
    max_foodon_aggregation_levels: int | None = None
    # LLM proposes this many substitution/add ideas before numeric verification
    n_ideation_candidates: int = 8
    # Give OOD/hybrid bundles a delta_L* handicap so they compete fairly with ID
    ood_delta_handicap: float = 0.015
    ideation_model: str = "gpt-4.1-mini"
    # Stop-adding policy: add-only bundles below this |delta_L*| are marginal
    marginal_add_delta_eps: float = 0.02
    # Max new ingredients per run before add-only bundles are vetoed
    max_total_adds: int = 2

    def target_box_dict(self) -> dict[str, float]:
        return {
            "protein_min": self.protein_min,
            "protein_max": self.protein_max,
            "carb_min": self.carb_min,
            "carb_max": self.carb_max,
            "fat_min": self.fat_min,
            "fat_max": self.fat_max,
        }

    def score_weights(self) -> dict[str, float]:
        return {
            "nutrient": self.w_score_nutrient,
            "ratio": self.w_score_ratio,
            "intent": self.w_score_intent,
            "churn": self.w_score_churn,
        }


def identity_roles_for_title(title: str, templates: dict[str, list[str]] | None = None) -> list[str]:
    templates = templates or IDENTITY_TEMPLATES
    t = title.lower()
    for key, roles in templates.items():
        if key in t:
            return list(roles)
    return []
