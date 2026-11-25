// Tiny dependency-free SVG charts (no CDN needed - works offline and in air-gapped demos).
const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const fmtK = v => v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "k" : String(Math.round(v));

// series: [{name, color, points:[{x:Date,y:Number}], dashed?, band?:[{x,lo,hi}]}]
function lineChart(container, series, opts = {}) {
  container.innerHTML = "";
  const W = container.clientWidth || 640, H = opts.height || 240, m = {l: 48, r: 12, t: 12, b: 26};
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart"}, container);
  const all = series.flatMap(s => s.points.concat((s.band || []).map(b => ({x: b.x, y: b.hi}))));
  if (!all.length) return;
  const xs = all.map(p => +p.x), ys = all.map(p => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys) * 1.08;
  const X = v => m.l + (+v - x0) / (x1 - x0 || 1) * (W - m.l - m.r);
  const Y = v => H - m.b - v / (y1 || 1) * (H - m.t - m.b);
  for (let i = 0; i <= 4; i++) {
    const v = y1 * i / 4, y = Y(v);
    el("line", {x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid"}, svg);
    el("text", {x: m.l - 6, y: y + 4, "text-anchor": "end", class: "axis"}, svg).textContent = fmtK(v);
  }
  const months = new Set();
  all.map(p => new Date(p.x)).sort((a, b) => a - b).forEach(d => {
    const key = d.getFullYear() + "-" + d.getMonth();
    if (!months.has(key) && d.getDate() <= 7) {
      months.add(key);
      el("text", {x: X(d), y: H - 6, "text-anchor": "middle", class: "axis"}, svg).textContent =
        d.toLocaleString("en", {month: "short"});
    }
  });
  series.forEach(s => {
    if (s.band && s.band.length) {
      const top = s.band.map(b => `${X(b.x)},${Y(b.hi)}`), bot = s.band.slice().reverse().map(b => `${X(b.x)},${Y(b.lo)}`);
      el("polygon", {points: top.concat(bot).join(" "), fill: s.color, opacity: 0.15}, svg);
    }
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    el("path", {d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.8, "stroke-dasharray": s.dashed ? "5 4" : ""}, svg);
  });
  // hover
  const tip = el("text", {x: m.l + 6, y: m.t + 12, class: "tip"}, svg);
  const vline = el("line", {y1: m.t, y2: H - m.b, class: "vline", opacity: 0}, svg);
  svg.addEventListener("mousemove", e => {
    const r = svg.getBoundingClientRect(), px = e.clientX - r.left;
    const t = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
    let best = null;
    series.forEach(s => s.points.forEach(p => { if (!best || Math.abs(+p.x - t) < Math.abs(+best.p.x - t)) best = {p, s}; }));
    if (!best) return;
    vline.setAttribute("x1", X(best.p.x)); vline.setAttribute("x2", X(best.p.x)); vline.setAttribute("opacity", 1);
    tip.textContent = `${new Date(best.p.x).toDateString().slice(4)} · ${best.s.name}: ${Math.round(best.p.y).toLocaleString()} SAR`;
  });
  svg.addEventListener("mouseleave", () => { vline.setAttribute("opacity", 0); tip.textContent = ""; });
}

// rows: [{label, value, note?}]
function hbar(container, rows, opts = {}) {
  container.innerHTML = "";
  const max = Math.max(...rows.map(r => r.value), 1);
  rows.forEach(r => {
    const row = document.createElement("div");
    row.className = "hbar";
    row.innerHTML = `<span class="hbar-label" title="${r.label}">${r.label}</span>
      <span class="hbar-track"><span class="hbar-fill" style="width:${(r.value / max * 100).toFixed(1)}%;background:${opts.color || "var(--accent)"}"></span></span>
      <span class="hbar-val">${opts.fmt ? opts.fmt(r.value) : fmtK(r.value)}</span>`;
    container.appendChild(row);
  });
}

function sparkline(container, values, color = "var(--accent)") {
  container.innerHTML = "";
  const W = container.clientWidth || 120, H = 28;
  const svg = el("svg", {width: W, height: H}, container);
  if (values.length < 2) return;
  const mx = Math.max(...values), mn = Math.min(...values);
  const d = values.map((v, i) => `${i ? "L" : "M"}${(i / (values.length - 1) * W).toFixed(1)},${(H - 2 - (v - mn) / (mx - mn || 1) * (H - 4)).toFixed(1)}`).join("");
  el("path", {d, fill: "none", stroke: color, "stroke-width": 1.5}, svg);
}
