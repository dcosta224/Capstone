const treeRoot = document.getElementById("tree-root");
const statusText = document.getElementById("status-text");
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const searchResults = document.getElementById("search-results");
const containsFilterPanel = document.getElementById("contains-filter-panel");
const containsFilters = document.getElementById("contains-filters");
const containsFilterClear = document.getElementById("contains-filter-clear");
const detailPanel = document.getElementById("detail-panel");
const detailTitle = document.getElementById("detail-title");
const detailId = document.getElementById("detail-id");
const detailContains = document.getElementById("detail-contains");
const carbonaraLink = document.getElementById("carbonara-link");

const nodeCache = new Map();
const nodeDataById = new Map();
let selectedNodeId = null;
let activeContainsFilter = null;
let containsBrowseMode = false;
let baseStatusText = "";
let containsSlugs = [];
let containsSummary = null;

const CONTAINS_COLORS = {
  alcohol: "#7c3aed",
  dairy: "#2563eb",
  egg: "#f59e0b",
  fish: "#0891b2",
  honey: "#ca8a04",
  peanut: "#b45309",
  pork: "#e11d48",
  poultry: "#ea580c",
  red_meat: "#dc2626",
  root_vegetable: "#65a30d",
  sesame: "#a16207",
  shellfish: "#0e7490",
  soy: "#4d7c0f",
  tree_nut: "#92400e",
  wheat: "#d97706",
};

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function slugLabel(slug) {
  return slug.replaceAll("_", " ");
}

function containsColor(slug) {
  return CONTAINS_COLORS[slug] || "#64748b";
}

function containsPillsHtml(contains, { compact = false } = {}) {
  if (!contains || !contains.length) {
    return "";
  }
  const cls = compact ? "contains-pill compact" : "contains-pill";
  return contains
    .map((slug) => {
      const color = containsColor(slug);
      return `<span class="${cls}" style="--pill-color:${color}" title="contains ${slugLabel(slug)}">${escapeHtml(slugLabel(slug))}</span>`;
    })
    .join("");
}

function nodeMatchesFilter(node) {
  if (!activeContainsFilter) {
    return true;
  }
  return (node.contains || []).includes(activeContainsFilter);
}

function nodeTitle(node) {
  const badge = node.is_bfo ? '<span class="bfo-badge">BFO</span>' : "";
  const pills = containsPillsHtml(node.contains, { compact: true });
  const count = `<span class="count">(${formatCount(node.descendant_count)})</span>`;
  const dimmed =
    containsBrowseMode && activeContainsFilter && !nodeMatchesFilter(node) ? " dimmed" : "";
  return `${badge}<span class="node-text${dimmed}">${escapeHtml(node.label)}</span>${pills}${count}`;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function apiGet(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function getSiblingNodes(li) {
  const parentList = li.parentElement;
  if (!parentList) {
    return [];
  }
  return Array.from(parentList.children).filter(
    (element) => element.classList.contains("tree-node") && element !== li,
  );
}

function hideSiblings(li) {
  getSiblingNodes(li).forEach((sibling) => sibling.classList.add("hidden"));
}

function showSiblings(li) {
  getSiblingNodes(li).forEach((sibling) => sibling.classList.remove("hidden"));
}

function isExpanded(li) {
  const childList = li.querySelector(":scope > .tree-children");
  return Boolean(childList && !childList.classList.contains("hidden"));
}

function setToggleState(toggle, node, expanded) {
  toggle.textContent = expanded ? "▼" : "▶";
  toggle.setAttribute(
    "aria-label",
    `${expanded ? "Collapse" : "Expand"} ${node.label}`,
  );
}

function resetDescendants(li) {
  const childList = li.querySelector(":scope > .tree-children");
  if (!childList) {
    return;
  }

  childList.classList.add("hidden");

  const toggle = li.querySelector(":scope > .node-row > .toggle");
  if (toggle) {
    toggle.textContent = "▶";
    const label = li.querySelector(".node-text")?.textContent || "";
    toggle.setAttribute("aria-label", `Expand ${label}`);
  }

  childList.querySelectorAll(":scope > .tree-node").forEach((childLi) => {
    showSiblings(childLi);
    resetDescendants(childLi);
  });
}

function refreshNodeLabels() {
  document.querySelectorAll(".tree-node").forEach((li) => {
    const nodeId = li.dataset.nodeId;
    const cached = findCachedNode(nodeId);
    if (!cached) {
      return;
    }
    const label = li.querySelector(":scope > .node-row > .node-label");
    if (label) {
      label.innerHTML = nodeTitle(cached);
    }
  });
}

function setStatusBrowse(slug, rootCount, taggedCount) {
  const taggedPart = taggedCount != null ? ` · ${formatCount(taggedCount)} tagged` : "";
  statusText.textContent = `Browsing ${slugLabel(slug)} · ${rootCount} root branch(es)${taggedPart}`;
}

function rememberNode(node) {
  nodeDataById.set(node.id, node);
}

function findCachedNode(nodeId) {
  return nodeDataById.get(nodeId) || null;
}

function createNodeElement(node, { autoExpand = false, select = false } = {}) {
  rememberNode(node);
  const li = document.createElement("li");
  li.className = "tree-node";
  li.dataset.nodeId = node.id;

  const row = document.createElement("div");
  row.className = `node-row${node.has_children ? "" : " leaf"}`;
  if (select) {
    row.classList.add("selected");
    selectedNodeId = node.id;
  }

  let toggle = null;
  if (node.has_children) {
    toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "toggle";
    toggle.setAttribute("aria-label", `Expand ${node.label}`);
    toggle.textContent = "▶";
    toggle.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleNode(li, node, toggle);
    });
    row.appendChild(toggle);
  } else {
    const spacer = document.createElement("span");
    spacer.className = "toggle-spacer";
    row.appendChild(spacer);
  }

  const label = document.createElement("span");
  label.className = `node-label${node.has_children ? " clickable" : ""}`;
  label.innerHTML = nodeTitle(node);
  label.addEventListener("click", () => {
    if (node.has_children) {
      toggleNode(li, node, toggle);
    } else {
      selectNode(node.id);
    }
  });
  row.appendChild(label);

  li.appendChild(row);

  if (node.has_children) {
    const childList = document.createElement("ul");
    childList.className = "tree-children hidden";
    childList.dataset.loaded = "false";
    li.appendChild(childList);

    if (autoExpand) {
      expandNode(li, node, toggle);
    }
  }

  return li;
}

async function showNodeDetail(nodeId) {
  try {
    const node = await apiGet(`/api/nodes/${encodeURIComponent(nodeId)}`);
    rememberNode(node);
    detailPanel.classList.remove("hidden");
    detailTitle.textContent = node.label;
    detailId.textContent = node.id;

    if (!node.contains_flags || !containsSlugs.length) {
      detailContains.innerHTML = '<p class="empty-state">No contains tags for this class.</p>';
      return;
    }

    const rows = containsSlugs.map((slug) => {
      const active = Boolean(node.contains_flags[slug]);
      const color = containsColor(slug);
      return `
        <div class="contains-flag-row${active ? " active" : ""}">
          <span class="contains-flag-name">${escapeHtml(slugLabel(slug))}</span>
          <span class="contains-flag-value" style="--pill-color:${color}">${active ? "yes" : "no"}</span>
        </div>
      `;
    });
    detailContains.innerHTML = `
      <h3 class="detail-subtitle">Dietary contains flags</h3>
      <div class="contains-flag-grid">${rows.join("")}</div>
    `;
  } catch (error) {
    detailContains.innerHTML = '<p class="empty-state">Could not load class details.</p>';
    console.error(error);
  }
}

function selectNode(nodeId) {
  selectedNodeId = nodeId;
  document.querySelectorAll(".node-row.selected").forEach((row) => {
    row.classList.remove("selected");
  });
  const current = document.querySelector(
    `.tree-node[data-node-id="${cssEscape(nodeId)}"] > .node-row`,
  );
  if (current) {
    current.classList.add("selected");
    current.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  showNodeDetail(nodeId);
}

function cssEscape(value) {
  if (window.CSS && CSS.escape) {
    return CSS.escape(value);
  }
  return value.replaceAll('"', '\\"');
}

async function loadChildren(nodeId) {
  if (nodeCache.has(nodeId)) {
    return nodeCache.get(nodeId);
  }
  const payload = await apiGet(`/api/nodes/${encodeURIComponent(nodeId)}/children`);
  payload.children.forEach(rememberNode);
  nodeCache.set(nodeId, payload.children);
  return payload.children;
}

async function expandNode(li, node, toggle) {
  const childList = li.querySelector(":scope > .tree-children");
  if (!childList) {
    return;
  }

  hideSiblings(li);
  childList.classList.remove("hidden");
  setToggleState(toggle, node, true);

  if (childList.dataset.loaded === "true") {
    return;
  }

  childList.innerHTML = '<li class="empty-state">Loading…</li>';
  try {
    const children = await loadChildren(node.id);
    childList.innerHTML = "";
    if (!children.length) {
      childList.innerHTML = '<li class="empty-state">No child classes</li>';
    } else {
      children.forEach((child) => {
        childList.appendChild(createNodeElement(child));
      });
    }
    childList.dataset.loaded = "true";
  } catch (error) {
    childList.innerHTML = '<li class="empty-state">Failed to load children</li>';
    console.error(error);
  }
}

function collapseNode(li, node, toggle) {
  const childList = li.querySelector(":scope > .tree-children");
  if (!childList) {
    return;
  }

  showSiblings(li);
  childList.classList.add("hidden");
  setToggleState(toggle, node, false);

  childList.querySelectorAll(":scope > .tree-node").forEach((childLi) => {
    showSiblings(childLi);
    resetDescendants(childLi);
  });
}

function toggleNode(li, node, toggle) {
  const childList = li.querySelector(":scope > .tree-children");
  if (!childList) {
    return;
  }

  if (isExpanded(li)) {
    collapseNode(li, node, toggle);
  } else {
    expandNode(li, node, toggle);
  }
  selectNode(node.id);
}

async function renderRoots() {
  containsBrowseMode = false;
  treeRoot.innerHTML = '<div class="empty-state">Loading hierarchy…</div>';
  const payload = await apiGet("/api/roots");
  treeRoot.innerHTML = "";
  const list = document.createElement("ul");
  list.className = "tree-root";
  payload.nodes.forEach((node) => {
    list.appendChild(createNodeElement(node));
  });
  treeRoot.appendChild(list);
  statusText.textContent = baseStatusText;
}

async function renderContainsBrowse(slug) {
  containsBrowseMode = true;
  treeRoot.innerHTML = '<div class="empty-state">Loading subtree…</div>';
  const payload = await apiGet(`/api/contains/${encodeURIComponent(slug)}/browse-roots`);
  treeRoot.innerHTML = "";
  if (!payload.nodes.length) {
    treeRoot.innerHTML = '<div class="empty-state">No browse roots for this dimension.</div>';
    return;
  }
  const list = document.createElement("ul");
  list.className = "tree-root";
  payload.nodes.forEach((node) => {
    list.appendChild(createNodeElement(node));
  });
  treeRoot.appendChild(list);
  const taggedCount = containsSummary?.tagged_counts?.[slug];
  setStatusBrowse(slug, payload.nodes.length, taggedCount);
  refreshNodeLabels();
}

async function revealPath(pathIds) {
  if (!pathIds.length) {
    return;
  }

  treeRoot.innerHTML = "";
  const list = document.createElement("ul");
  list.className = "tree-root";
  treeRoot.appendChild(list);

  let parentList = list;

  for (let index = 0; index < pathIds.length; index += 1) {
    const nodeId = pathIds[index];
    const node = await apiGet(`/api/nodes/${encodeURIComponent(nodeId)}`);
    rememberNode(node);
    const isLast = index === pathIds.length - 1;
    const li = createNodeElement(node, { autoExpand: !isLast, select: isLast });
    parentList.appendChild(li);
    if (!isLast) {
      hideSiblings(li);
      parentList = li.querySelector(":scope > .tree-children");
    }
  }

  selectNode(pathIds[pathIds.length - 1]);
}

function renderSearchResults(results) {
  searchResults.classList.remove("hidden");
  const filtered = activeContainsFilter
    ? results.filter((result) => (result.contains || []).includes(activeContainsFilter))
    : results;

  if (!filtered.length) {
    searchResults.innerHTML = '<div class="empty-state">No matches</div>';
    return;
  }

  searchResults.innerHTML = "";
  filtered.forEach((result) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "search-hit";
    const pills = containsPillsHtml(result.contains);
    button.innerHTML = `
      <span class="search-hit-main">
        <span>${escapeHtml(result.label)}</span>
        ${pills ? `<span class="search-hit-pills">${pills}</span>` : ""}
      </span>
      <span class="meta">${formatCount(result.descendant_count)} descendants · score ${result.score}</span>
    `;
    button.addEventListener("click", async () => {
      searchResults.classList.add("hidden");
      const payload = await apiGet(`/api/nodes/${encodeURIComponent(result.id)}/ancestors`);
      const pathIds = payload.ancestors.map((node) => node.id);
      await revealPath(pathIds);
    });
    searchResults.appendChild(button);
  });
}

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) {
    return;
  }
  searchResults.classList.remove("hidden");
  searchResults.innerHTML = '<div class="empty-state">Searching…</div>';
  try {
    const payload = await apiGet(`/api/search?${new URLSearchParams({ q: query, limit: "25" })}`);
    renderSearchResults(payload.results);
  } catch (error) {
    searchResults.innerHTML = '<div class="empty-state">Search failed</div>';
    console.error(error);
  }
}

function renderContainsFilters() {
  if (!containsSlugs.length) {
    return;
  }
  containsFilterPanel.classList.remove("hidden");
  containsFilters.innerHTML = "";
  containsSlugs.forEach((slug) => {
    const count = containsSummary?.tagged_counts?.[slug];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "filter-chip";
    button.dataset.slug = slug;
    button.style.setProperty("--pill-color", containsColor(slug));
    button.textContent = count != null ? `${slugLabel(slug)} (${formatCount(count)})` : slugLabel(slug);
    button.addEventListener("click", async () => {
      if (activeContainsFilter === slug) {
        activeContainsFilter = null;
        document.querySelectorAll(".filter-chip").forEach((chip) => chip.classList.remove("active"));
        await renderRoots();
      } else {
        activeContainsFilter = slug;
        document.querySelectorAll(".filter-chip").forEach((chip) => {
          chip.classList.toggle("active", chip.dataset.slug === activeContainsFilter);
        });
        await renderContainsBrowse(slug);
      }
      if (!searchResults.classList.contains("hidden") && searchInput.value.trim()) {
        runSearch();
      }
    });
    containsFilters.appendChild(button);
  });
}

async function boot() {
  try {
    const status = await apiGet("/api/status");
    const containsPart = status.contains_loaded
      ? ` · ${formatCount(status.contains_tagged_count)} tagged for contains`
      : "";
    baseStatusText = `${formatCount(status.class_count)} classes indexed${containsPart}`;
    statusText.textContent = baseStatusText;

    if (status.carbonara_available) {
      carbonaraLink.classList.remove("hidden");
    }

    if (status.contains_loaded) {
      containsSummary = await apiGet("/api/contains/summary");
      containsSlugs = containsSummary.contains_slugs || [];
      renderContainsFilters();
    }

    await renderRoots();
  } catch (error) {
    statusText.textContent = "Failed to load ontology";
    treeRoot.innerHTML =
      '<div class="empty-state">Could not load FoodOn index. Is the server still starting?</div>';
    console.error(error);
  }
}

containsFilterClear.addEventListener("click", async () => {
  activeContainsFilter = null;
  document.querySelectorAll(".filter-chip").forEach((chip) => chip.classList.remove("active"));
  await renderRoots();
  if (!searchResults.classList.contains("hidden") && searchInput.value.trim()) {
    runSearch();
  }
});

searchBtn.addEventListener("click", runSearch);
searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});

boot();
