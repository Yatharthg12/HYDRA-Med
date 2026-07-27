(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const colors = ["#35d0ba", "#7c8cff", "#ffbd69", "#ff7b88", "#64a8ff"];

  function element(name, attributes = {}) {
    const node = document.createElementNS(NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function empty(container) {
    container.replaceChildren();
    const width = Math.max(320, container.clientWidth || 600);
    const height = Math.max(200, container.clientHeight || 240);
    const svg = element("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
    container.append(svg);
    return { svg, width, height };
  }

  function bar(selector, data, options = {}) {
    const container = typeof selector === "string" ? document.querySelector(selector) : selector;
    if (!container || !data || !data.length) return;
    const { svg, width, height } = empty(container);
    const margin = { top: 12, right: 35, bottom: 28, left: options.left || 115 };
    const usable = width - margin.left - margin.right;
    const row = (height - margin.top - margin.bottom) / data.length;
    const maximum = Math.max(...data.map(item => Number(item.count)), 1);
    data.forEach((item, index) => {
      const y = margin.top + index * row + row * .18;
      const barWidth = Number(item.count) / maximum * usable;
      const label = element("text", { x: margin.left - 8, y: y + row * .38, "text-anchor": "end" });
      label.textContent = String(item.label).slice(0, 18);
      svg.append(label);
      svg.append(element("rect", { x: margin.left, y, width: Math.max(1, barWidth), height: row * .48, rx: 3, class: "bar-rect" }));
      const value = element("text", { x: Math.min(width - 2, margin.left + barWidth + 5), y: y + row * .38 });
      value.textContent = Number(item.count).toLocaleString();
      svg.append(value);
    });
  }

  function line(selector, series, options = {}) {
    const container = typeof selector === "string" ? document.querySelector(selector) : selector;
    if (!container || !series || !series.length) return;
    const { svg, width, height } = empty(container);
    const margin = { top: 22, right: 20, bottom: 38, left: 45 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const allPoints = series.flatMap(item => item.points || []);
    if (!allPoints.length) return;
    const xValues = allPoints.map(point => Number(point[0]));
    const yValues = allPoints.map(point => Number(point[1]));
    const xMin = options.xMin ?? Math.min(...xValues);
    const xMax = options.xMax ?? Math.max(...xValues);
    const yMin = options.yMin ?? 0;
    const yMax = options.yMax ?? Math.max(1, ...yValues);
    const sx = value => margin.left + (value - xMin) / ((xMax - xMin) || 1) * plotWidth;
    const sy = value => margin.top + plotHeight - (value - yMin) / ((yMax - yMin) || 1) * plotHeight;
    for (let tick = 0; tick <= 4; tick += 1) {
      const y = margin.top + tick / 4 * plotHeight;
      svg.append(element("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "grid" }));
      const label = element("text", { x: margin.left - 7, y: y + 3, "text-anchor": "end" });
      label.textContent = (yMax - tick / 4 * (yMax - yMin)).toFixed(2);
      svg.append(label);
    }
    svg.append(element("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "axis" }));
    svg.append(element("line", { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "axis" }));
    series.forEach((item, index) => {
      const pathData = item.points.map((point, pointIndex) => `${pointIndex ? "L" : "M"}${sx(Number(point[0])).toFixed(1)},${sy(Number(point[1])).toFixed(1)}`).join(" ");
      const color = item.color || colors[index % colors.length];
      svg.append(element("path", { d: pathData, class: "line-path", stroke: color }));
      if (item.points.length < 20) item.points.forEach(point => svg.append(element("circle", { cx: sx(Number(point[0])), cy: sy(Number(point[1])), r: 3.5, fill: color, class: "point" })));
      const legendX = margin.left + index * Math.min(145, plotWidth / series.length);
      svg.append(element("line", { x1: legendX, x2: legendX + 18, y1: 9, y2: 9, stroke: color, "stroke-width": 3 }));
      const legend = element("text", { x: legendX + 23, y: 12 });
      legend.textContent = item.name;
      svg.append(legend);
    });
    const xLeft = element("text", { x: margin.left, y: height - 10 });
    xLeft.textContent = options.xLabelLeft ?? xMin.toFixed(1);
    svg.append(xLeft);
    const xRight = element("text", { x: width - margin.right, y: height - 10, "text-anchor": "end" });
    xRight.textContent = options.xLabelRight ?? xMax.toFixed(1);
    svg.append(xRight);
  }

  function interval(selector, items, options = {}) {
    const container = typeof selector === "string" ? document.querySelector(selector) : selector;
    if (!container || !items || !items.length) return;
    const { svg, width, height } = empty(container);
    const margin = { top: 24, right: 20, bottom: 72, left: 48 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const yMin = options.yMin ?? 0;
    const maximum = Math.max(...items.map(item => Number(item.upper)), ...(options.trials || []).map(item => Number(item.value)), 0);
    const yMax = options.yMax ?? Math.max(1, maximum * 1.08);
    const sx = index => margin.left + (index + .5) / items.length * plotWidth;
    const sy = value => margin.top + plotHeight - (value - yMin) / ((yMax - yMin) || 1) * plotHeight;
    for (let tick = 0; tick <= 4; tick += 1) {
      const y = margin.top + tick / 4 * plotHeight;
      svg.append(element("line", { x1: margin.left, x2: width - margin.right, y1: y, y2: y, class: "grid" }));
      const label = element("text", { x: margin.left - 7, y: y + 3, "text-anchor": "end" });
      label.textContent = (yMax - tick / 4 * (yMax - yMin)).toFixed(options.decimals ?? 2);
      svg.append(label);
    }
    (options.trials || []).forEach((trial, trialIndex) => {
      const jitter = ((trialIndex % 7) - 3) * 2.2;
      svg.append(element("circle", {
        cx: sx(Number(trial.index)) + jitter,
        cy: sy(Number(trial.value)),
        r: 2.8,
        class: "trial-point",
        fill: trial.color || "#ffbd69"
      }));
    });
    const meanPath = [];
    items.forEach((item, index) => {
      const x = sx(index);
      const color = item.color || colors[index % colors.length];
      svg.append(element("line", { x1: x, x2: x, y1: sy(Number(item.lower)), y2: sy(Number(item.upper)), stroke: color, "stroke-width": 2 }));
      svg.append(element("line", { x1: x - 7, x2: x + 7, y1: sy(Number(item.lower)), y2: sy(Number(item.lower)), stroke: color, "stroke-width": 2 }));
      svg.append(element("line", { x1: x - 7, x2: x + 7, y1: sy(Number(item.upper)), y2: sy(Number(item.upper)), stroke: color, "stroke-width": 2 }));
      svg.append(element("circle", { cx: x, cy: sy(Number(item.value)), r: 5, fill: color, class: "interval-point" }));
      const label = element("text", { x, y: height - 48, "text-anchor": "end", transform: `rotate(-28 ${x} ${height - 48})` });
      label.textContent = item.label;
      svg.append(label);
      meanPath.push(`${index ? "L" : "M"}${x.toFixed(1)},${sy(Number(item.value)).toFixed(1)}`);
    });
    if (options.connect !== false && items.length > 1) {
      svg.insertBefore(element("path", { d: meanPath.join(" "), class: "line-path interval-mean-line", stroke: options.lineColor || "#35d0ba" }), svg.firstChild);
    }
  }

  window.HealthCharts = { bar, line, interval, colors };
})();
