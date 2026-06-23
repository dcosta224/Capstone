const treeRoot = document.getElementById("tree-root");
const statusText = document.getElementById("status-text");
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const searchResults = document.getElementById("search-results");

const nodeCache = new Map();
let selectedNodeId = null;

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function nodeTitle(node) {
  const badge = node.is_bfo ? '<span class="bfo-badge">BFO</span>' : "";
  const count = `<span class="count">(${formatCount(node.descendant_count)})</span>`;
  return `${badge}<span class="node-text">${escapeHtml(node.label)}</span>${count}`;
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

function createNodeElement(node, { autoExpand = false, select = false } = {}) {
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
  treeRoot.innerHTML = '<div class="empty-state">Loading hierarchy…</div>';
  const payload = await apiGet("/api/roots");
  treeRoot.innerHTML = "";
  const list = document.createElement("ul");
  list.className = "tree-root";
  payload.nodes.forEach((node) => {
    list.appendChild(createNodeElement(node));
  });
  treeRoot.appendChild(list);
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
  if (!results.length) {
    searchResults.innerHTML = '<div class="empty-state">No matches</div>';
    return;
  }

  searchResults.innerHTML = "";
  results.forEach((result) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "search-hit";
    button.innerHTML = `
      <span>${escapeHtml(result.label)}</span>
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

async function boot() {
  try {
    const status = await apiGet("/api/status");
    statusText.textContent = `${formatCount(status.class_count)} classes indexed`;
    await renderRoots();
  } catch (error) {
    statusText.textContent = "Failed to load ontology";
    treeRoot.innerHTML =
      '<div class="empty-state">Could not load FoodOn index. Is the server still starting?</div>';
    console.error(error);
  }
}

searchBtn.addEventListener("click", runSearch);
searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});

boot();
