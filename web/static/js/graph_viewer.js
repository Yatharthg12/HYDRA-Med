(function () {
  "use strict";
  const data = window.PAGE_DATA && window.PAGE_DATA.graph;
  if (!data) return;
  const container = document.getElementById("graph-canvas");
  const nodeFilter = document.getElementById("node-filter");
  const edgeFilter = document.getElementById("edge-filter");
  const inspector = document.getElementById("node-inspector");
  const legend = document.getElementById("graph-legend");
  const NS = "http://www.w3.org/2000/svg";
  const palette = {
    Patient: "#7c8cff", Encounter: "#35d0ba", DiagnosisCategory: "#ffbd69",
    Medication: "#ff7b88", AdmissionType: "#64a8ff", AdmissionSource: "#ba8cff",
    LabResultCategory: "#84df85"
  };
  const types = [...new Set(data.nodes.map(node => node.type))];
  const relations = [...new Set(data.links.map(link => link.relation))];
  types.forEach(type => nodeFilter.insertAdjacentHTML("beforeend", `<option value="${type}">${type}</option>`));
  relations.forEach(relation => edgeFilter.insertAdjacentHTML("beforeend", `<option value="${relation}">${relation.replaceAll("_", " ")}</option>`));
  legend.innerHTML = types.map(type => `<span><i style="background:${palette[type] || "#bbb"}"></i>${type}</span>`).join("");

  function create(name, attributes = {}) {
    const node = document.createElementNS(NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function render() {
    const width = Math.max(600, container.clientWidth);
    const height = Math.max(420, container.clientHeight);
    const svg = create("svg", { viewBox: `0 0 ${width} ${height}` });
    const positions = new Map();
    const grouped = Object.groupBy ? Object.groupBy(data.nodes, node => node.type) :
      data.nodes.reduce((result, node) => ((result[node.type] ||= []).push(node), result), {});
    types.forEach((type, typeIndex) => {
      const nodes = grouped[type] || [];
      const angle = typeIndex / types.length * Math.PI * 2 - Math.PI / 2;
      const centerX = width / 2 + Math.cos(angle) * width * .30;
      const centerY = height / 2 + Math.sin(angle) * height * .29;
      nodes.forEach((node, index) => {
        const localAngle = index / Math.max(1, nodes.length) * Math.PI * 2;
        const radius = 20 + 6 * Math.sqrt(nodes.length);
        positions.set(node.id, {
          x: centerX + Math.cos(localAngle) * radius,
          y: centerY + Math.sin(localAngle) * radius
        });
      });
    });
    const linkLayer = create("g");
    const nodeLayer = create("g");
    svg.append(linkLayer, nodeLayer);
    const linkNodes = [];
    data.links.forEach(link => {
      const source = positions.get(link.source);
      const target = positions.get(link.target);
      if (!source || !target) return;
      const line = create("line", { x1: source.x, y1: source.y, x2: target.x, y2: target.y, "data-relation": link.relation, "data-source": link.source, "data-target": link.target });
      linkLayer.append(line);
      linkNodes.push(line);
    });
    const circles = [];
    data.nodes.forEach(node => {
      const position = positions.get(node.id);
      const circle = create("circle", {
        cx: position.x, cy: position.y,
        r: node.type === "Encounter" ? 6 : node.type === "Patient" ? 5 : 4,
        fill: palette[node.type] || "#bbb", "data-id": node.id, "data-type": node.type,
        tabindex: 0, role: "button", "aria-label": `${node.type}: ${node.label}`
      });
      const title = create("title");
      title.textContent = `${node.type} · ${node.label}`;
      circle.append(title);
      const inspect = () => showInspector(node, linkNodes);
      circle.addEventListener("click", inspect);
      circle.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") inspect(); });
      nodeLayer.append(circle);
      circles.push(circle);
    });
    function updateFilters() {
      const chosenType = nodeFilter.value;
      const chosenRelation = edgeFilter.value;
      circles.forEach(circle => {
        const visible = chosenType === "all" || circle.dataset.type === chosenType;
        circle.style.opacity = visible ? "1" : ".08";
      });
      linkNodes.forEach(line => {
        const relationVisible = chosenRelation === "all" || line.dataset.relation === chosenRelation;
        const typeVisible = chosenType === "all" ||
          data.nodes.find(node => node.id === line.dataset.source)?.type === chosenType ||
          data.nodes.find(node => node.id === line.dataset.target)?.type === chosenType;
        line.style.opacity = relationVisible && typeVisible ? ".8" : ".03";
      });
    }
    nodeFilter.onchange = updateFilters;
    edgeFilter.onchange = updateFilters;
    container.replaceChildren(svg);
  }

  function showInspector(node, links) {
    container.querySelectorAll("circle").forEach(circle => circle.classList.toggle("selected", circle.dataset.id === node.id));
    const connected = links.filter(line => line.dataset.source === node.id || line.dataset.target === node.id);
    const metadata = node.type === "Encounter" ? data.metadata[String(node.encounter_id)] : null;
    inspector.innerHTML = `<p class="eyebrow">NODE INSPECTOR</p><h2>${node.label}</h2><span class="verified">${node.type}</span>
      <dl><dt>Sample degree</dt><dd>${connected.length}</dd>
      ${node.age_group ? `<dt>Age group</dt><dd>${node.age_group}</dd>` : ""}
      ${node.outcome ? `<dt>Observed outcome</dt><dd>${node.outcome}</dd>` : ""}
      ${metadata ? `<dt>Admission</dt><dd>${metadata.admission_type}</dd><dt>Source</dt><dd>${metadata.admission_source}</dd>` : ""}</dl>
      ${metadata ? `<h3>Diagnoses</h3><p class="fine-print">${metadata.diagnoses.join(", ") || "Unknown"}</p><h3>Active medications</h3><p class="fine-print">${metadata.active_medications.join(", ") || "None recorded"}</p>` : ""}
      <h3>Connected relations</h3><div class="pair-list">${connected.slice(0, 12).map(line => `<span>${line.dataset.relation.replaceAll("_", " ")}</span>`).join("")}</div>`;
  }

  render();
  window.addEventListener("resize", () => window.requestAnimationFrame(render));
})();
