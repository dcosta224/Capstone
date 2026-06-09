import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mvp_data import parse_nlg_ingredients
from mvp_recipe_judge import (
    CandidateReview,
    FinalPickCandidate,
    llm_select_final_candidate,
    select_best_semantic_candidate,
    select_final_candidate,
    select_from_candidate_reviews,
)


def _c(
    recipe_id: int,
    portion_score: float,
    avg_pct_change: float = 0.0,
    *,
    title: str = "",
    semantic_text: str = "",
    nlg_ingredients: str = "",
    ingredients: str = "",
    semantic_sim: float = 0.5,
    macro_feasible: bool = True,
    used_fallback: bool = False,
    already_feasible: bool = False,
    stage1_rank: int = 1,
) -> FinalPickCandidate:
    return FinalPickCandidate(
        recipe_id=recipe_id,
        title_clean=title or f"Recipe {recipe_id}",
        semantic_text=semantic_text,
        nlg_ingredients=nlg_ingredients or ingredients,
        ingredient_summary=ingredients,
        semantic_sim=semantic_sim,
        portion_score=portion_score,
        avg_pct_change=avg_pct_change,
        macro_feasible=macro_feasible,
        used_fallback=used_fallback,
        already_feasible=already_feasible,
        stage1_rank=stage1_rank,
    )


def _reviews(*items: CandidateReview) -> dict[int, CandidateReview]:
    return {r.recipe_id: r for r in items}


def test_parse_nlg_ingredients():
    text = "tex-mex chilli | beef beans tomatoes chilli powder cumin"
    assert parse_nlg_ingredients(text) == "beef beans tomatoes chilli powder cumin"


def test_strong_match_wins_despite_worse_portion_score():
    chili = _c(1, portion_score=0.01, title="Tex-Mex Chilli", semantic_sim=0.7)
    chicken = _c(2, portion_score=0.6, title="Easy Chicken Casserole", semantic_sim=0.5)
    reviews = _reviews(
        CandidateReview(1, True, "no_match", "Official ingredients list beef, not chicken."),
        CandidateReview(2, True, "strong_match", "Official ingredients include chicken."),
    )
    chosen, _ = select_from_candidate_reviews([chili, chicken], reviews)
    assert chosen.recipe_id == 2


def test_prefers_lower_portion_among_strong_matches():
    a = _c(1, portion_score=0.8, title="Chicken A", semantic_sim=0.6)
    b = _c(2, portion_score=0.3, title="Chicken B", semantic_sim=0.55)
    reviews = _reviews(
        CandidateReview(1, True, "strong_match", "Has chicken."),
        CandidateReview(2, True, "strong_match", "Has chicken."),
    )
    chosen, _ = select_from_candidate_reviews([a, b], reviews)
    assert chosen.recipe_id == 2


def test_weak_match_only_when_all_strong_severely_distorted():
    strong = _c(1, portion_score=1.5, avg_pct_change=55.0, title="Chicken Stew", semantic_sim=0.7)
    weak = _c(2, portion_score=0.1, title="Chicken-ish Side", semantic_sim=0.4)
    reviews = _reviews(
        CandidateReview(1, True, "strong_match", "Full chicken stew."),
        CandidateReview(2, True, "weak_match", "Only minor chicken flavor."),
    )
    chosen, note = select_from_candidate_reviews([strong, weak], reviews)
    assert chosen.recipe_id == 2
    assert "severe" in note.lower()


def test_prefers_macro_feasible_over_lower_infeasible_score():
    result = select_best_semantic_candidate(
        [
            _c(1, portion_score=0.01, used_fallback=True, macro_feasible=False),
            _c(2, portion_score=1.2, avg_pct_change=40.0),
        ]
    )
    assert result.chosen_recipe_id == 2


def test_no_api_key_uses_friendly_rationale():
    seasoning = _c(1, portion_score=0.01, title="Poultry Seasoning", semantic_sim=0.9)
    chicken = _c(2, portion_score=0.8, title="Roasted Chicken Thighs", semantic_sim=0.4)
    with patch.dict("os.environ", {}, clear=True):
        result = select_final_candidate("chicken dinner", [seasoning, chicken])
    assert result.chosen_recipe_id == 1
    assert "portion score" not in result.rationale.lower()
    assert "LLM judge unavailable" not in result.rationale


def _llm_payload(reviews: list[dict], **extra) -> dict:
    return {
        "candidate_reviews": reviews,
        "why_this_recipe": extra.get(
            "why_this_recipe",
            "A hearty, satisfying chicken dish that matches what you asked for.",
        ),
        "runner_up_notes": extra.get("runner_up_notes", ""),
    }


@patch("mvp_recipe_judge.get_sync_openai_client")
def test_llm_rejects_seasoning_picks_chicken_casserole(mock_client_fn):
    seasoning = _c(
        1,
        portion_score=0.01,
        title="Poultry Seasoning",
        nlg_ingredients="sage thyme rosemary marjoram",
        semantic_sim=0.9,
    )
    casserole = _c(
        2,
        portion_score=0.5,
        title="Easy Chicken Casserole",
        nlg_ingredients="chicken rice celery cream of mushroom soup",
        semantic_sim=0.6,
        stage1_rank=2,
    )
    chicken = _c(
        3,
        portion_score=0.8,
        title="Roasted Chicken Thighs",
        nlg_ingredients="chicken thighs garlic lemon",
        semantic_sim=0.55,
        stage1_rank=3,
    )

    llm_payload = _llm_payload(
        [
            {
                "recipe_id": 1,
                "edible_alone": False,
                "query_fit": "no_match",
                "fit_reason": "Spice blend only, not a meal.",
            },
            {
                "recipe_id": 2,
                "edible_alone": True,
                "query_fit": "strong_match",
                "fit_reason": "Official ingredients include chicken.",
            },
            {
                "recipe_id": 3,
                "edible_alone": True,
                "query_fit": "strong_match",
                "fit_reason": "Official ingredients include chicken thighs.",
            },
        ],
        why_this_recipe=(
            "You wanted a hearty chicken dinner, and this casserole includes chicken as a "
            "main ingredient in a complete meal."
        ),
    )
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_payload)))]
    mock_client_fn.return_value.chat.completions.create.return_value = mock_resp

    result = llm_select_final_candidate(
        "hearty chicken dinner",
        [seasoning, casserole, chicken],
    )
    assert result.chosen_recipe_id == 2
    assert "hearty chicken dinner" in result.rationale.lower()
    assert "portion score" not in result.rationale.lower()
    assert "reshaping" not in result.rationale.lower()


@patch("mvp_recipe_judge.get_sync_openai_client")
def test_llm_marks_chili_no_match_picks_chicken(mock_client_fn):
    chili = _c(
        1,
        portion_score=0.01,
        title="Tex-Mex Chilli",
        nlg_ingredients="beef beans tomatoes chilli powder cumin onion",
        semantic_sim=0.75,
    )
    chicken = _c(
        2,
        portion_score=0.6,
        title="Easy Chicken Casserole",
        nlg_ingredients="chicken rice celery",
        semantic_sim=0.5,
    )
    llm_payload = _llm_payload(
        [
            {
                "recipe_id": 1,
                "edible_alone": True,
                "query_fit": "no_match",
                "fit_reason": (
                    "Official ingredients list beef and beans; user asked for chicken."
                ),
            },
            {
                "recipe_id": 2,
                "edible_alone": True,
                "query_fit": "strong_match",
                "fit_reason": "Official ingredients include chicken.",
            },
        ],
        why_this_recipe=(
            "You asked for chicken, and this casserole delivers with real chicken, rice, "
            "and celery in a filling, home-style dish."
        ),
    )
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_payload)))]
    mock_client_fn.return_value.chat.completions.create.return_value = mock_resp

    result = llm_select_final_candidate(
        "Give me a meal containing chicken.",
        [chili, chicken],
    )
    assert result.chosen_recipe_id == 2
    assert "chicken" in result.rationale.lower()
    assert "portion" not in result.rationale.lower()


@patch("mvp_recipe_judge.get_sync_openai_client")
def test_rationale_always_matches_chosen_recipe(mock_client_fn):
    seasoning = _c(
        1,
        portion_score=0.01,
        title="Poultry Seasoning",
        nlg_ingredients="sage thyme",
    )
    chicken = _c(
        2,
        portion_score=0.2,
        title="Easy Chicken Casserole",
        nlg_ingredients="chicken rice",
    )

    llm_payload = _llm_payload(
        [
            {
                "recipe_id": 1,
                "edible_alone": False,
                "query_fit": "no_match",
                "fit_reason": "Seasoning only.",
            },
            {
                "recipe_id": 2,
                "edible_alone": True,
                "query_fit": "strong_match",
                "fit_reason": "Contains chicken.",
            },
        ],
        why_this_recipe="You asked for chicken, and this casserole lists chicken among its ingredients.",
    )
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_payload)))]
    mock_client_fn.return_value.chat.completions.create.return_value = mock_resp

    result = llm_select_final_candidate("chicken", [seasoning, chicken])
    assert result.chosen_recipe_id == 2
    assert "casserole" in result.rationale.lower()
