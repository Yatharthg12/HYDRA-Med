(function () {
  "use strict";

  function toast(message) {
    const node = document.getElementById("toast");
    if (!node) return;
    node.textContent = message;
    node.classList.add("show");
    window.setTimeout(() => node.classList.remove("show"), 1800);
  }

  function initializeChrome() {
    const menu = document.getElementById("menu-button");
    const sidebar = document.getElementById("sidebar");
    if (menu && sidebar) menu.addEventListener("click", () => {
      const open = sidebar.classList.toggle("open");
      menu.setAttribute("aria-expanded", String(open));
    });
    document.querySelectorAll("[data-copy]").forEach(button => button.addEventListener("click", async () => {
      await navigator.clipboard.writeText(button.dataset.copy);
      toast("Command copied");
    }));
  }

  function initializeDataset(data) {
    if (!data) return;
    HealthCharts.bar("#target-chart", data.readmission, { left: 135 });
    HealthCharts.bar("#age-chart", data.age, { left: 72 });
    HealthCharts.bar("#admission-chart", data.admission_type, { left: 105 });
    HealthCharts.bar("#diagnosis-chart", data.primary_diagnosis, { left: 105 });
    HealthCharts.bar("#medication-chart", data.active_medications, { left: 105 });
    const search = document.getElementById("dictionary-search");
    if (search) search.addEventListener("input", () => {
      const query = search.value.trim().toLowerCase();
      document.querySelectorAll("#dictionary-table tbody tr").forEach(row => {
        row.hidden = Boolean(query) && !row.textContent.toLowerCase().includes(query);
      });
    });
  }

  function initializeWarshall(data) {
    if (!data) return;
    let currentStep = 0;
    let autoPlayTimer = null;
    const matrix = document.getElementById("warshall-matrix");
    const slider = document.getElementById("warshall-step-slider");
    const panel = document.querySelector(".matrix-panel");

    function renderGraph(iteration) {
      const container = document.getElementById("warshall-graph");
      container.replaceChildren();
      const width = Math.max(600, container.clientWidth || 760);
      const height = 320;
      const namespace = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(namespace, "svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const definitions = document.createElementNS(namespace, "defs");
      const marker = document.createElementNS(namespace, "marker");
      marker.setAttribute("id", "warshall-arrow");
      marker.setAttribute("viewBox", "0 0 10 10");
      marker.setAttribute("refX", "19");
      marker.setAttribute("refY", "5");
      marker.setAttribute("markerWidth", "6");
      marker.setAttribute("markerHeight", "6");
      marker.setAttribute("orient", "auto-start-reverse");
      const arrow = document.createElementNS(namespace, "path");
      arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
      arrow.setAttribute("fill", "#93a0b5");
      marker.append(arrow);
      definitions.append(marker);
      svg.append(definitions);
      const positions = data.nodes.map((node, index) => {
        const angle = -Math.PI / 2 + index * 2 * Math.PI / data.nodes.length;
        return {
          node,
          x: width / 2 + Math.cos(angle) * Math.min(width * .33, 235),
          y: height / 2 + Math.sin(angle) * 112
        };
      });
      const newlyAdded = new Set(iteration.new_pairs.map(pair => `${pair.row}|${pair.column}`));
      iteration.matrix.forEach((row, rowIndex) => row.forEach((reachable, columnIndex) => {
        if (!reachable || rowIndex === columnIndex) return;
        const line = document.createElementNS(namespace, "line");
        line.setAttribute("x1", positions[rowIndex].x);
        line.setAttribute("y1", positions[rowIndex].y);
        line.setAttribute("x2", positions[columnIndex].x);
        line.setAttribute("y2", positions[columnIndex].y);
        line.setAttribute("marker-end", "url(#warshall-arrow)");
        line.setAttribute("class", newlyAdded.has(`${rowIndex}|${columnIndex}`) ? "new-edge" : "known-edge");
        svg.append(line);
      }));
      positions.forEach(position => {
        const circle = document.createElementNS(namespace, "circle");
        circle.setAttribute("cx", position.x);
        circle.setAttribute("cy", position.y);
        circle.setAttribute("r", "18");
        circle.setAttribute("class", iteration.intermediate_node === position.node ? "current-node" : "");
        svg.append(circle);
        const label = document.createElementNS(namespace, "text");
        label.setAttribute("x", position.x);
        label.setAttribute("y", position.y + 34);
        label.setAttribute("text-anchor", "middle");
        label.textContent = position.node;
        svg.append(label);
      });
      container.append(svg);
    }

    function stopAutoPlay() {
      if (autoPlayTimer !== null) window.clearInterval(autoPlayTimer);
      autoPlayTimer = null;
      const button = document.getElementById("warshall-autoplay");
      button.textContent = "Auto-play iterations";
      button.setAttribute("aria-pressed", "false");
    }

    function renderStep(step) {
      currentStep = Math.max(0, Math.min(5, Number(step)));
      const iteration = data.iterations[currentStep];
      const added = new Set(iteration.new_cells.map(cell => `${cell.row}|${cell.column}`));
      matrix.replaceChildren();
      ["", ...data.nodes].forEach(text => {
        const cell = document.createElement("div");
        cell.className = "matrix-cell label";
        cell.textContent = text;
        matrix.append(cell);
      });
      iteration.matrix.forEach((row, rowIndex) => {
        const label = document.createElement("div");
        label.className = "matrix-cell label";
        label.textContent = data.nodes[rowIndex];
        matrix.append(label);
        row.forEach((value, columnIndex) => {
          const cell = document.createElement("div");
          const classes = ["matrix-cell"];
          const key = `${rowIndex}|${columnIndex}`;
          if (rowIndex === columnIndex) classes.push("diagonal");
          else if (added.has(key)) classes.push("new");
          else if (value) classes.push("active");
          else classes.push("unreachable");
          cell.className = classes.join(" ");
          cell.textContent = value;
          cell.title = `${data.nodes[rowIndex]} to ${data.nodes[columnIndex]}: ${value ? "reachable" : "unreachable"}`;
          matrix.append(cell);
        });
      });
      document.getElementById("warshall-title").textContent =
        `${iteration.notation} · ${currentStep ? `via ${iteration.intermediate_node}` : "Direct relationships"}`;
      document.getElementById("warshall-intermediate").textContent = currentStep
        ? `Current intermediate: ${iteration.intermediate_node}`
        : "Current intermediate: none (direct relationships)";
      document.getElementById("warshall-description").textContent = iteration.description;
      document.getElementById("warshall-counter").textContent = `Step ${currentStep} of 5`;
      slider.value = String(currentStep);
      slider.setAttribute("aria-valuenow", String(currentStep));
      const list = document.getElementById("warshall-calculations");
      list.replaceChildren();
      if (!iteration.new_pairs.length) {
        const item = document.createElement("p");
        item.className = "fine-print";
        item.textContent = currentStep === 0
          ? "No paths are inferred at T^(0); these are the supplied direct relationships."
          : "No new reachable pairs were discovered during this iteration.";
        list.append(item);
      } else iteration.new_pairs.forEach(pair => {
        const item = document.createElement("article");
        const values = pair.calculation;
        item.innerHTML = `<b>${pair.plain_english}</b><code>T[${pair.from}, ${pair.to}] = ${pair.calculation.expression}<br>= ${values.previous_value} OR (${values.source_to_intermediate} AND ${values.intermediate_to_target}) = ${values.result}</code>`;
        list.append(item);
      });
      document.getElementById("warshall-prev").disabled = currentStep === 0;
      document.getElementById("warshall-next").disabled = currentStep === 5;
      renderGraph(iteration);
    }

    const renderSlider = () => renderStep(Number(slider.value));
    slider.addEventListener("input", renderSlider);
    slider.addEventListener("change", renderSlider);
    document.getElementById("warshall-prev").addEventListener("click", () => {
      stopAutoPlay();
      renderStep(currentStep - 1);
    });
    document.getElementById("warshall-next").addEventListener("click", () => {
      stopAutoPlay();
      renderStep(currentStep + 1);
    });
    document.getElementById("warshall-reset").addEventListener("click", () => {
      stopAutoPlay();
      renderStep(0);
    });
    document.getElementById("warshall-autoplay").addEventListener("click", event => {
      if (autoPlayTimer !== null) return stopAutoPlay();
      if (currentStep === 5) renderStep(0);
      event.currentTarget.textContent = "Pause iterations";
      event.currentTarget.setAttribute("aria-pressed", "true");
      autoPlayTimer = window.setInterval(() => {
        if (currentStep === 5) return stopAutoPlay();
        renderStep(currentStep + 1);
      }, 1200);
    });
    panel.addEventListener("keydown", event => {
      if (event.target === slider) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        stopAutoPlay();
        renderStep(currentStep + (event.key === "ArrowRight" ? 1 : -1));
      }
    });
    panel.setAttribute("tabindex", "0");
    renderStep(0);
  }

  function initializeModels(data) {
    if (!data) return;
    const models = data.comparison.models.slice().sort((a, b) => a.rank_by_test_pr_auc - b.rank_by_test_pr_auc);
    HealthCharts.line("#roc-chart", models.map((model, index) => ({
      name: model.name,
      color: HealthCharts.colors[index],
      points: model.curves.roc.fpr.map((value, i) => [value, model.curves.roc.tpr[i]])
    })), { xMin: 0, xMax: 1, yMin: 0, yMax: 1, xLabelLeft: "0 false-positive rate", xLabelRight: "1" });
    HealthCharts.line("#pr-chart", models.map((model, index) => ({
      name: model.name,
      color: HealthCharts.colors[index],
      points: model.curves.precision_recall.recall.map((value, i) => [value, model.curves.precision_recall.precision[i]])
    })), { xMin: 0, xMax: 1, yMin: 0, yMax: 1, xLabelLeft: "0 recall", xLabelRight: "1" });
    const gcn = models.find(model => model.slug === "gcn");
    if (gcn) HealthCharts.line("#training-chart", [{
      name: "GCN validation PR-AUC",
      points: gcn.history.map(item => [item.epoch, item.validation_pr_auc])
    }], { yMin: 0, yMax: 1, xLabelLeft: "1", xLabelRight: "epoch" });

    const confidenceSelect = document.getElementById("confidence-metric");
    if (confidenceSelect && data.bootstrap.models) {
      const renderConfidence = () => {
        const metric = confidenceSelect.value;
        HealthCharts.interval("#confidence-chart", models.map((model, index) => {
          const value = data.bootstrap.models[model.slug].intervals[metric];
          return {
            label: model.name,
            value: value.estimate,
            lower: value.ci_95_lower,
            upper: value.ci_95_upper,
            color: HealthCharts.colors[index]
          };
        }), { yMin: 0, yMax: 1, connect: false });
      };
      confidenceSelect.addEventListener("change", renderConfidence);
      renderConfidence();
    }

    const stability = data.seedStability.runs || [];
    if (stability.length) HealthCharts.line("#seed-stability-chart", ["pr_auc", "roc_auc", "recall", "f1"].map((metric, index) => ({
      name: metric.toUpperCase().replace("_", "-"),
      color: HealthCharts.colors[index],
      points: stability.map(run => [run.seed, Number(run.test_metrics[metric])])
    })), { yMin: 0, yMax: 1, xLabelLeft: String(stability[0].seed), xLabelRight: String(stability[stability.length - 1].seed) });

    const completedPca = (data.pcaAnalysis.configurations || []).filter(item => item.status === "completed");
    if (completedPca.length) {
      const maximumRuntime = Math.max(...completedPca.map(row => row.runtime_seconds), 1);
      HealthCharts.line("#pca-analysis-chart", [
        { name: "Explained variance", points: completedPca.map(item => [item.components, item.explained_variance]) },
        { name: "Validation PR-AUC", color: "#ffbd69", points: completedPca.map(item => [item.components, item.validation_pr_auc]) },
        { name: "Runtime (relative)", color: "#7c8cff", points: completedPca.map(item => [item.components, item.runtime_seconds / maximumRuntime]) }
      ], { yMin: 0, yMax: 1, xLabelLeft: String(completedPca[0].components), xLabelRight: String(completedPca[completedPca.length - 1].components) });
    }
  }

  function initializeRobustness(data) {
    if (!data) return;
    const select = document.getElementById("robustness-metric");
    const showTrials = document.getElementById("show-robustness-trials");
    function render() {
      const key = select.value;
      const colorFor = type => type === "edge_removal" ? "#ff7b88" : type === "noise_addition" ? "#7c8cff" : "#35d0ba";
      const items = data.summary.map(row => ({
        label: row.label,
        value: Number(row.metrics[key].mean),
        lower: Number(row.metrics[key].ci_95_lower),
        upper: Number(row.metrics[key].ci_95_upper),
        color: colorFor(row.perturbation_type)
      }));
      const scenarioIndex = Object.fromEntries(data.summary.map((row, index) => [row.scenario, index]));
      const trials = showTrials.checked ? data.trials.map(row => ({
        index: scenarioIndex[row.scenario],
        value: Number(row[key]),
        color: colorFor(row.perturbation_type)
      })) : [];
      HealthCharts.interval("#robustness-chart", items, {
        yMin: 0,
        yMax: key === "probability_shift_mean_absolute" ? Math.max(.01, ...items.map(row => row.upper * 1.15)) : 1,
        trials,
        lineColor: "#93a0b5",
        decimals: key === "probability_shift_mean_absolute" ? 3 : 2
      });
    }
    select.addEventListener("change", render);
    showTrials.addEventListener("change", render);
    render();
  }

  function initializeCases() {
    const form = document.getElementById("case-form");
    if (!form) return;
    const result = document.getElementById("case-result");
    const empty = document.getElementById("case-empty");
    function probabilityCard(name, probability, prediction) {
      const value = Number(probability);
      return `<article class="card probability-card"><p class="eyebrow">${name}</p><div class="probability">${(value * 100).toFixed(1)}%</div><div class="probability-track"><i style="width:${value * 100}%"></i></div><p class="fine-print">Thresholded decision: <b>${prediction ? "Flag" : "No flag"}</b></p></article>`;
    }
    form.addEventListener("submit", async event => {
      event.preventDefault();
      const identifier = document.getElementById("case-id").value.trim();
      if (!identifier) return toast("Enter an encounter ID");
      const button = form.querySelector("button");
      button.disabled = true;
      button.textContent = "Loading…";
      try {
        const response = await fetch(`/api/cases/${encodeURIComponent(identifier)}`);
        if (!response.ok) throw new Error(response.status === 404 ? "Encounter is not in the test set" : "Case lookup failed");
        const data = await response.json();
        empty.classList.add("hidden");
        result.classList.remove("hidden");
        result.querySelector(".case-outcome").innerHTML = `<div><p class="eyebrow">ENCOUNTER ${data.encounter_id}</p><h2>Observed test outcome</h2><p>Anonymous patient reference ${data.patient_nbr}</p></div><span class="outcome-badge ${data.actual_target ? "positive" : "negative"}">${data.actual_target ? "Readmitted <30 days" : "Not within 30 days"}</span>`;
        result.querySelector(".probability-cards").innerHTML =
          probabilityCard("Logistic Regression", data.lr_probability, data.lr_prediction) +
          probabilityCard("PCA + kNN", data.pca_knn_probability, data.pca_knn_prediction) +
          probabilityCard("Graph Convolutional Network", data.gcn_probability, data.gcn_prediction);
        result.querySelector(".clinical-features").innerHTML = `<p class="eyebrow">KEY DISCHARGE-TIME FEATURES</p><h2>Encounter context</h2><div class="feature-list">${[
          ["Age group", data.age_group], ["Admission type", data.admission_type],
          ["Admission source", data.admission_source], ["Diagnosis categories", data.diagnosis_categories],
          ["Active medications", data.active_medications || "None recorded"]
        ].map(item => `<div><small>${item[0]}</small>${item[1]}</div>`).join("")}</div>`;
        const neighbours = data.nearest_graph_neighbours || [];
        result.querySelector(".graph-neighbours").innerHTML = `<p class="eyebrow">NEAREST GRAPH NEIGHBOURS</p><h2>Shared relation evidence</h2><div class="neighbour-list">${neighbours.length ? neighbours.map(item => `<div><b>Encounter ${item.encounter_id}</b> · similarity ${Number(item.similarity).toFixed(3)}<small>${item.shared_relations.join(", ") || "No displayed shared tokens"}</small></div>`).join("") : "<p class='fine-print'>No neighbours were available.</p>"}</div>`;
      } catch (error) {
        toast(error.message);
      } finally {
        button.disabled = false;
        button.textContent = "Load case";
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initializeChrome();
    const page = document.body.dataset.page;
    if (page === "dataset") initializeDataset(window.PAGE_DATA);
    if (page === "warshall") initializeWarshall(window.PAGE_DATA);
    if (page === "models") initializeModels(window.PAGE_DATA);
    if (page === "robustness") initializeRobustness(window.PAGE_DATA);
    if (page === "cases") initializeCases();
  });
})();
