const $ = (selector) => document.querySelector(selector);
let cy;

function setTooltip(text) { $("#graph-tooltip").textContent = text || "Hover a node or relationship to inspect its evidence."; }

function renderGraph(payload, focused = false) {
  cy?.destroy();
  const elements = payload.elements || [];
  $("#graph-count").textContent = `${elements.filter((item) => !item.data.source).length} nodes · ${elements.filter((item) => item.data.source).length} links`;
  cy = cytoscape({
    container: $("#cy"), elements,
    style: [
      { selector: "node", style: { "background-color": "#576a95", label: "data(label)", color: "#34415d", "font-size": 9, "text-wrap": "wrap", "text-max-width": 88, "text-valign": "bottom", "text-margin-y": 6, width: 18, height: 18 } },
      { selector: "node[type = 'character']", style: { "background-color": "#28476f", width: 24, height: 24 } },
      { selector: "node[type = 'location']", style: { "background-color": "#7b6198" } },
      { selector: "node[type = 'event']", style: { "background-color": "#ad7556" } },
      { selector: "edge", style: { width: 1.1, "line-color": "#c4cad7", "target-arrow-color": "#c4cad7", "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", color: "#8792a7", "font-size": 7, "text-rotation": "autorotate" } },
      { selector: ":selected", style: { "background-color": "#e9a545", "line-color": "#e9a545", "target-arrow-color": "#e9a545" } },
    ],
    layout: { name: "cose", animate: false, padding: 28, nodeRepulsion: 8000, idealEdgeLength: 80, gravity: 0.55, numIter: 800 },
  });
  cy.on("mouseover", "node", (event) => { const data = event.target.data(); setTooltip(`${data.label} · ${data.type}\n${data.description || data.traits || "No summary available."}`); event.target.select(); });
  cy.on("mouseover", "edge", (event) => { const data = event.target.data(); setTooltip(`${data.label}: ${data.description || "Relationship extracted from the corpus."}`); event.target.select(); });
  cy.on("mouseout", "node, edge", (event) => event.target.unselect());
  renderCommunities(payload.communities || []);
}

function renderCommunities(communities) {
  const container = $("#communities"); container.replaceChildren();
  communities.slice(0, 18).forEach((community) => { const button = document.createElement("button"); button.className = "community"; button.textContent = `${community.title} (${community.entities})`; button.title = community.summary; button.onclick = () => setTooltip(`${community.title}\n${community.summary}`); container.append(button); });
}

function appendMessage(role, content) {
  const fragment = $("#message-template").content.cloneNode(true);
  const message = fragment.querySelector(".message"); message.classList.add(role); message.querySelector(".message-role").textContent = role === "user" ? "You" : "Homer"; message.querySelector(".message-content").textContent = content; $("#conversation").append(message); message.scrollIntoView({ behavior: "smooth", block: "end" }); return message;
}

async function send(prompt) {
  appendMessage("user", prompt); const sendButton = $("#send"); sendButton.disabled = true; const loading = appendMessage("assistant", "Retrieving canon and writing…"); loading.classList.add("loading");
  try {
    const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, corpus: "sherlock", max_words: 1200 }) });
    const body = await response.json(); if (!response.ok) throw new Error(body.detail || "Story generation failed.");
    loading.remove(); appendMessage("assistant", body.story);
    const citations = body.grounding.map((item) => item.section_title || item.document_title || item.kind).filter(Boolean); const grounding = document.createElement("div"); grounding.className = "grounding"; grounding.textContent = `Grounded in ${body.grounding.length} retrieved items · ${[...new Set(citations)].join(" · ")}`; $("#conversation").append(grounding);
    renderGraph(body.graph, true); $("#graph-title").textContent = "Evidence used for this story";
  } catch (error) { loading.classList.remove("loading"); loading.querySelector(".message-content").textContent = error.message; }
  finally { sendButton.disabled = false; }
}

$("#composer").addEventListener("submit", (event) => { event.preventDefault(); const input = $("#prompt"); const prompt = input.value.trim(); if (!prompt) return; input.value = ""; send(prompt); });
$(".suggestion").addEventListener("click", (event) => { const prompt = event.currentTarget.dataset.prompt; $("#prompt").value = ""; send(prompt); });

fetch("/api/graph?corpus=sherlock").then(async (response) => { const body = await response.json(); if (!response.ok) throw new Error(body.detail); $("#graph-meta").textContent = `${body.stats.documents} books · ${body.stats.chunks} passages · ${body.stats.entities} entities`; renderGraph(body); }).catch((error) => { $("#graph-meta").textContent = error.message; setTooltip("The graph could not be loaded. Run the corpus ingestion command first."); });
