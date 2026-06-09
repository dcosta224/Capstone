const form = document.getElementById("query-form");
const timeline = document.getElementById("timeline");
const stageDetail = document.getElementById("stage-detail");
const activeStep = document.getElementById("active-step");
const progress = document.getElementById("progress");
const result = document.getElementById("result");
const submitBtn = document.getElementById("submit-btn");
const kcalMidDisplay = document.getElementById("kcal_mid_display");

const optimizerRows = new Map();
const progressAnchor = document.getElementById("progress-scroll-anchor");

let scrollPending = false;

/** Keep the latest pipeline output in view as content grows downward. */
function scrollToLatestOutput() {
  if (scrollPending) return;
  scrollPending = true;
  requestAnimationFrame(() => {
    scrollPending = false;
    let target = progressAnchor;
    if (!result.classList.contains("hidden")) {
      target = result;
    } else if (!activeStep.classList.contains("hidden")) {
      target = activeStep;
    }
    target?.scrollIntoView({ behavior: "smooth", block: "end" });
  });
}

function updateKcalMid() {
  const lo = Number(document.getElementById("kcal_min").value) || 0;
  const hi = Number(document.getElementById("kcal_max").value) || 0;
  kcalMidDisplay.textContent = ((lo + hi) / 2).toFixed(0);
}
["kcal_min", "kcal_max"].forEach((id) => {
  document.getElementById(id).addEventListener("input", updateKcalMid);
});
updateKcalMid();

function pctToFrac(id) {
  return Number(document.getElementById(id).value) / 100;
}

function setActiveStep(text) {
  if (!text) {
    activeStep.classList.add("hidden");
    activeStep.textContent = "";
    return;
  }
  activeStep.classList.remove("hidden");
  activeStep.innerHTML = `<span class="spinner"></span> ${text}`;
  scrollToLatestOutput();
}

function addTimeline(text, className) {
  const li = document.createElement("li");
  li.textContent = text;
  if (className) li.classList.add(className);
  timeline.appendChild(li);
  scrollToLatestOutput();
  return li;
}

function nutrientFitLabel(fit, inRange) {
  if (inRange || fit <= 1e-6) return '<span class="badge ok">PFC in range</span>';
  return fit.toFixed(3);
}

function renderRankTable(rows) {
  if (!rows || !rows.length) return "<p>No rankings yet.</p>";
  const head = `<tr><th>Rank</th><th>Recipe</th><th>Semantic sim</th><th>Nutrient Dist.</th><th>Combined Score (/100)</th></tr>`;
  const body = rows.map(r => `
    <tr>
      <td>${r.rank}</td>
      <td>${r.recipe_name || r.recipe_id}</td>
      <td>${r.semantic_sim?.toFixed(3) ?? ""}</td>
      <td>${nutrientFitLabel(r.nutrient_fit, r.pfc_in_range)}</td>
      <td>${r.combined_score?.toFixed(1) ?? ""}</td>
    </tr>`).join("");
  return `<table id="rank-table">${head}${body}</table>`;
}

function optimizerStatusLabel(c) {
  if (c.already_feasible) return "already_feasible";
  if (c.macro_feasible === false) return "infeasible";
  if (c.used_fallback) return c.optimizer_status || "fallback";
  return c.optimizer_status ?? "";
}

function renderOptimizerTable(candidates) {
  if (!candidates || !candidates.length) {
    return `<table id="opt-table"><tr><th>Recipe</th><th>Status</th><th>Portion score</th><th>Kcal after</th><th>Target kcal</th></tr><tbody id="opt-tbody"></tbody></table>`;
  }
  const head = `<tr><th>Recipe</th><th>Status</th><th>Portion score</th><th>Kcal after</th><th>Target kcal</th></tr>`;
  const body = candidates.map(c => optimizerRowHtml(c)).join("");
  return `<table id="opt-table">${head}<tbody id="opt-tbody">${body}</tbody></table>`;
}

function recipeDisplayName(c) {
  return c.recipe_name || `Recipe ${c.recipe_id}`;
}

function optimizerRowHtml(c) {
  const cls = c.macro_feasible === false ? "row-infeasible" : (c.already_feasible ? "row-ok" : "");
  return `
    <tr class="${cls}" data-recipe-id="${c.recipe_id}">
      <td>${recipeDisplayName(c)}</td>
      <td>${optimizerStatusLabel(c)}</td>
      <td>${c.portion_score?.toFixed(4) ?? ""}</td>
      <td>${c.macros_after?.energy_kcal?.toFixed(0) ?? ""}</td>
      <td>${c.kcal_target?.toFixed(0) ?? ""}</td>
    </tr>`;
}

function upsertOptimizerRow(candidate) {
  let tbody = document.getElementById("opt-tbody");
  if (!tbody) {
    stageDetail.innerHTML += `<h3>Optimizer results</h3>${renderOptimizerTable([])}`;
    tbody = document.getElementById("opt-tbody");
  }
  const existing = tbody.querySelector(`tr[data-recipe-id="${candidate.recipe_id}"]`);
  const html = optimizerRowHtml(candidate);
  if (existing) {
    existing.outerHTML = html;
  } else {
    tbody.insertAdjacentHTML("beforeend", html);
  }
}

function renderIngredients(ingredients) {
  return ingredients.map(ing => `
    <div class="ingredient-card">
      <h4>${ing.ingredient}</h4>
      <div class="meta">${ing.fdc_description || "USDA food"}</div>
      <div class="meta">Portion: ${ing.portion_label || ing.unit || "grams"}</div>
      <div>
        Quantity: ${ing.quantity_original ?? "—"}
        → ${ing.quantity_optimized != null ? ing.quantity_optimized.toFixed(2) : ing.gram_weight_optimized?.toFixed(1)}
      </div>
      <div class="meta">Scale factor: ${ing.adjustment_factor?.toFixed(3)}</div>
    </div>`).join("");
}

function handleStage(data) {
  const { stage, payload } = data;

  if (stage === "embed_query") {
    if (payload.status === "running") {
      setActiveStep(payload.message);
    } else {
      addTimeline(payload.message || "Taste embedding complete", "done");
    }
  } else if (stage === "load_corpus") {
    setActiveStep(payload.message);
    if (payload.status === "done") addTimeline(payload.message, "done");
  } else if (stage === "stage1_rank") {
    if (payload.status === "running") {
      setActiveStep(payload.message);
    } else {
      setActiveStep(null);
      addTimeline(`Ranked ${payload.n_recipes} recipes`, "done");
      stageDetail.innerHTML = `<h3>Stage 1 — Top 20</h3><p class="hint">Combined: weighted sum (0–100; 100 = perfect taste + PFC in range). Semantic sim: 1 = best match. PFC fit 0 = in range. Target kcal: ${payload.kcal_target?.toFixed(0)}</p>${renderRankTable(payload.top_20)}`;
      scrollToLatestOutput();
    }
  } else if (stage === "optimize") {
    if (payload.status === "running") {
      setActiveStep(payload.message);
      stageDetail.innerHTML += `<h3>Optimizer results</h3><p class="hint">Avg % change = mean |scale − 1| per ingredient (0% = unchanged).</p>${renderOptimizerTable([])}`;
      scrollToLatestOutput();
    } else {
      setActiveStep(null);
      const nFeas = payload.n_already_feasible ?? 0;
      addTimeline(
        nFeas > 0 ? `Optimization complete — ${nFeas} already feasible` : "Optimization complete",
        "done"
      );
      if (payload.infeasible_note) {
        addTimeline(payload.infeasible_note, "warn");
        stageDetail.innerHTML += `<p class="warn-banner">${payload.infeasible_note}</p>`;
        scrollToLatestOutput();
      }
    }
  } else if (stage === "optimize_progress") {
    const c = payload.candidate;
    upsertOptimizerRow(c);
    setActiveStep(`Optimizing ${recipeDisplayName(c)} — ${payload.index} of ${payload.total}…`);
  } else if (stage === "judge") {
    if (payload.status === "running") {
      setActiveStep(payload.message);
    } else {
      setActiveStep("Preparing your recommendation…");
      addTimeline(payload.message || "Judge complete", "done");
    }
  } else if (stage === "finalize") {
    if (payload.status === "running") {
      setActiveStep(payload.message);
    }
  } else if (stage === "format_result") {
    setActiveStep("Rendering final recipe…");
    renderResult(payload);
    setActiveStep(null);
    addTimeline("Recommendation ready", "done");
  }
}

function renderResult(payload) {
  const optNote = payload.optimization_note
    ? `<p class="warn-banner">${payload.optimization_note}</p>`
    : "";
  const feasBadge = payload.already_feasible
    ? '<span class="badge ok">Already feasible — portions unchanged</span>'
    : (payload.macro_feasible === false
      ? '<span class="badge warn">Macro target unreachable</span>'
      : '<span class="badge ok">Optimized</span>');

  document.getElementById("result-summary").innerHTML = `
    <h3>${payload.recipe_name}</h3>
    ${optNote}
    <p>${feasBadge} · Recipe ID: ${payload.chosen_recipe_id} · Target kcal: ${payload.kcal_target?.toFixed(0)}</p>
    <p class="hint">Portion change vs kcal-scaled baseline: <strong>${payload.avg_pct_change?.toFixed(1) ?? 0}%</strong> avg per ingredient
       (max ${payload.max_pct_change?.toFixed(1) ?? 0}%).
       0% = only uniform scaling to target kcal; higher = ingredient ratios shifted.
       Log-ratio score ${payload.portion_score?.toFixed(4) ?? 0} (optimizer objective vs kcal-scaled baseline; 0 = uniform scaling only).</p>
    <p>Macros after: protein ${payload.macros_after?.protein_g?.toFixed(1)}g,
       fat ${payload.macros_after?.fat_g?.toFixed(1)}g,
       carbs ${payload.macros_after?.carbs_g?.toFixed(1)}g,
       ${payload.macros_after?.energy_kcal?.toFixed(0)} kcal</p>`;
  document.getElementById("result-ingredients").innerHTML = renderIngredients(payload.ingredients || []);
  const j = payload.judge || {};
  document.getElementById("result-explanation").innerHTML = `
    <h3>Why this recipe?</h3>
    <p>${j.rationale || ""}</p>
    <h3>Portion changes</h3>
    <p>${j.portion_summary || ""}</p>
    ${j.runner_up_notes ? `<p class="meta">${j.runner_up_notes}</p>` : ""}`;
  result.classList.remove("hidden");
  scrollToLatestOutput();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  timeline.innerHTML = "";
  stageDetail.innerHTML = "";
  result.classList.add("hidden");
  progress.classList.remove("hidden");
  optimizerRows.clear();
  setActiveStep("Starting pipeline…");

  const body = {
    taste_text: document.getElementById("taste_text").value,
    kcal_min: Number(document.getElementById("kcal_min").value),
    kcal_max: Number(document.getElementById("kcal_max").value),
    fat_frac_min: pctToFrac("fat_min"),
    fat_frac_max: pctToFrac("fat_max"),
    carb_frac_min: pctToFrac("carb_min"),
    carb_frac_max: pctToFrac("carb_max"),
    protein_frac_min: pctToFrac("protein_min"),
    protein_frac_max: pctToFrac("protein_max"),
    top_k: 10,
  };

  try {
    const resp = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.trim()) continue;
        const lines = part.split("\n");
        let eventType = "message";
        let dataLine = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) eventType = line.slice(7).trim();
          if (line.startsWith("data: ")) dataLine = line.slice(6);
        }
        if (!dataLine) continue;
        const parsed = JSON.parse(dataLine);
        if (eventType === "error") {
          setActiveStep(null);
          addTimeline(`Error: ${parsed.error}`, "warn");
          break;
        }
        if (eventType === "done") {
          setActiveStep(null);
          if (result.classList.contains("hidden") && parsed.payload?.chosen_recipe_id) {
            renderResult(parsed.payload);
          }
        } else if (eventType === "stage") {
          handleStage(parsed);
        }
      }
    }
  } catch (err) {
    setActiveStep(null);
    addTimeline(`Failed: ${err.message}`, "warn");
  } finally {
    submitBtn.disabled = false;
    setActiveStep(null);
  }
});
