# Recipe Opt Agent playground

Live UI for the LangGraph recipe optimization agent: inputs, step stream with LLM rationale, and a flow graph that highlights active nodes.

Partner setup (branch, env, local store): see the **Recipe optimization agent** section in the [project README](../README.md).

## Run

```bash
# from repo root
PYTHONPATH=scripts:. uv run python -m recipe_opt_web --reload
# → http://127.0.0.1:8010
```

Or:

```bash
PYTHONPATH=scripts:. uv run uvicorn recipe_opt_web.server:app --reload --port 8010
```

## Inputs

| Field | Meaning |
|-------|---------|
| Mode | **Neighborhood** (canonical dish) or **Creative** (free-text request) |
| Canonical recipe dropdown | List from the local cap40 store by default (`RECIPE_DATA_SOURCE=local`), ordered by `n_matches`. Defines the FoodOn neighborhood; `taste_text` = dish title. |
| Creative request | Free-text dish idea; agent drafts then grounds to FDC before the diagnose loop. |
| Macro % targets | Calorie-share percents → 0–1 fractions. Starting NLG instance chosen closest to this box during load. |
| F_accept / F_max / Max iterations | Fidelity bands + loop budget |

On **Run**, the UI streams live load phases (build neighborhood → pick start recipe, or creative draft/ground) then the LangGraph loop. **propose** retrieves modification candidates live (not from a JSON fixture).

There is no “fixture” or “title override” in this playground — those were only for offline unit tests.

## Related

- Agent package: [`recipe_opt_agent/`](../recipe_opt_agent/)
- Notebook sandbox: [`notebooks/recipe_opt_agent_sandbox.ipynb`](../notebooks/recipe_opt_agent_sandbox.ipynb)
- Design notes: [`docs/recipe_opt_agent.md`](../docs/recipe_opt_agent.md)
