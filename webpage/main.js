// LINE project page: the explanation explorer and the activation-maximization comparison.
// The explorer reads the released CSVs in assets/ at runtime, so the page has to be served
// over HTTP (GitHub Pages, or `python -m http.server` from the repository root).
"use strict";

/* ------------------------------------------------------------------ data ------- */

const MODELS = [
  { id: "resnet50", name: "ResNet50", data: "ImageNet", layers: ["layer1", "layer2", "layer3", "layer4", "avgpool"] },
  { id: "resnet18", name: "ResNet18", data: "ImageNet", layers: ["layer1", "layer2", "layer3", "layer4", "avgpool"] },
  { id: "resnet50_places", name: "ResNet50", data: "Places365", layers: ["layer1", "layer2", "layer3", "layer4", "avgpool"] },
  { id: "resnet18_places", name: "ResNet18", data: "Places365", layers: ["layer1", "layer2", "layer3", "layer4", "avgpool"] },
  { id: "vit_b_16", name: "ViT-B/16", data: "ImageNet", layers: ["encoder_layer_6", "encoder_layer_8", "encoder_layer_10", "ln_1", "ln"] },
  { id: "dinov3-vitb16", name: "DINOv3 ViT-B/16", data: "LVD-1689M", layers: ["norm"] },
];
// Layers of the main CoSy table; LINE with FLUX.1-dev was run on these only.
const HEADLINE = new Set(["resnet50-avgpool", "resnet18-avgpool", "resnet50_places-avgpool",
  "resnet18_places-avgpool", "vit_b_16-ln", "dinov3-vitb16-norm"]);
const HAS_DND = new Set([...HEADLINE, "resnet50-layer1", "resnet50-layer2", "resnet50-layer3", "resnet50-layer4"]);
// The paper calls ViT-B/16's final encoder LayerNorm (`ln` in the CSVs) the "encoder" layer.
const layerName = (modelId, layer) => (modelId === "vit_b_16" && layer === "ln" ? "encoder" : layer);
const METHOD_NAMES = { LINE: "LINE", LINE_flux: "LINE (FLUX)", "CLIP-Dissect": "CLIP-Dissect", INVERT: "INVERT", DnD: "DnD" };
const DEFAULT = { model: "resnet50_places", layer: "avgpool", unit: 1912 };

// The paper's comparison with DEXTER and DiffExplainer (robust ResNet50, avgpool), with the
// activations computed in notebooks/dexter_comparison.ipynb. `image` is the top-activating of
// LINE's five images of the neuron's concept.
const AM_CLASSES = [
  { label: 433, name: "Bathing Cap", neurons: [
      { neuron: 121, concept: "head", image: 5, act: { DiffExplainer: 3.42, DEXTER: 2.51, LINE: 1.99 } },
      { neuron: 1340, concept: "swimsuit", image: 2, act: { DiffExplainer: 13.18, DEXTER: 1.48, LINE: 8.01 } },
      { neuron: 1591, concept: "macaque", image: 3, act: { DiffExplainer: 7.84, DEXTER: 2.67, LINE: 7.09 } },
      { neuron: 1609, concept: "shower cap", image: 5, act: { DiffExplainer: 5.38, DEXTER: 2.11, LINE: 5.78 } },
      { neuron: 755, concept: "graduation attire", image: 3, act: { DiffExplainer: 11.58, DEXTER: 2.77, LINE: 1.60 } },
  ] },
  { label: 609, name: "Jeep", neurons: [
      { neuron: 1067, concept: "golfcart", image: 2, act: { DiffExplainer: 10.52, DEXTER: 3.01, LINE: 6.02 } },
      { neuron: 1100, concept: "european fire salamander", image: 2, act: { DiffExplainer: 7.09, DEXTER: 0.66, LINE: 2.74 } },
      { neuron: 1208, concept: "tractor", image: 5, act: { DiffExplainer: 0.03, DEXTER: 2.50, LINE: 3.94 } },
      { neuron: 1515, concept: "pickup", image: 3, act: { DiffExplainer: 1.47, DEXTER: 0.57, LINE: 3.01 } },
      { neuron: 691, concept: "limousine", image: 2, act: { DiffExplainer: 5.77, DEXTER: 2.84, LINE: 4.13 } },
  ] },
  { label: 985, name: "Daisy", neurons: [
      { neuron: 1105, concept: "sulphur-crested cockatoo", image: 2, act: { DiffExplainer: 6.60, DEXTER: 0.36, LINE: 4.70 } },
      { neuron: 120, concept: "ground beetle", image: 4, act: { DiffExplainer: 7.89, DEXTER: 3.61, LINE: 2.95 } },
      { neuron: 298, concept: "spiny", image: 1, act: { DiffExplainer: 13.77, DEXTER: 3.57, LINE: 6.29 } },
      { neuron: 595, concept: "daisy", image: 5, act: { DiffExplainer: 17.01, DEXTER: 12.64, LINE: 12.03 } },
      { neuron: 859, concept: "tick spider", image: 4, act: { DiffExplainer: 6.74, DEXTER: 2.70, LINE: 4.10 } },
  ] },
  { label: 764, name: "Rifle", neurons: [
      { neuron: 1259, concept: "revolver", image: 4, act: { DiffExplainer: 9.81, DEXTER: 2.45, LINE: 2.30 } },
      { neuron: 1928, concept: "tripod", image: 2, act: { DiffExplainer: 4.21, DEXTER: 0.53, LINE: 2.80 } },
      { neuron: 400, concept: "grand piano", image: 4, act: { DiffExplainer: 4.78, DEXTER: 3.72, LINE: 2.87 } },
      { neuron: 515, concept: "gun accessory", image: 4, act: { DiffExplainer: 8.20, DEXTER: 1.86, LINE: 2.67 } },
      { neuron: 522, concept: "revolver", image: 5, act: { DiffExplainer: 10.63, DEXTER: 3.11, LINE: 4.50 } },
  ] },
  { label: 706, name: "Patio", neurons: [
      { neuron: 1016, concept: "sliding door", image: 1, act: { DiffExplainer: 5.98, DEXTER: 4.85, LINE: 2.94 } },
      { neuron: 1633, concept: "oxcart", image: 2, act: { DiffExplainer: 6.57, DEXTER: 1.83, LINE: 4.55 } },
      { neuron: 194, concept: "ipod", image: 5, act: { DiffExplainer: 12.93, DEXTER: 3.73, LINE: 2.57 } },
      { neuron: 451, concept: "golfcart", image: 5, act: { DiffExplainer: 9.84, DEXTER: 4.14, LINE: 3.24 } },
      { neuron: 654, concept: "barber chair", image: 5, act: { DiffExplainer: 5.13, DEXTER: 1.35, LINE: 3.20 } },
  ] },
  { label: 785, name: "Seat Belt", neurons: [
      { neuron: 1010, concept: "scooter mower", image: 1, act: { DiffExplainer: 5.45, DEXTER: 1.14, LINE: 1.49 } },
      { neuron: 108, concept: "eye coverings", image: 4, act: { DiffExplainer: 2.11, DEXTER: 3.00, LINE: 4.53 } },
      { neuron: 116, concept: "car mirror", image: 5, act: { DiffExplainer: 6.07, DEXTER: 2.71, LINE: 2.76 } },
      { neuron: 1493, concept: "car mirror", image: 5, act: { DiffExplainer: 3.97, DEXTER: 1.85, LINE: 2.55 } },
      { neuron: 50, concept: "car mirror", image: 5, act: { DiffExplainer: 5.77, DEXTER: 0.56, LINE: 2.28 } },
  ] },
];

/* --------------------------------------------------------------- helpers ------- */

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value == null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** RFC 4180 CSV (quoted fields, doubled quotes, CRLF) into an array of row objects. */
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body
    .filter((r) => r.length === head.length)
    .map((r) => Object.fromEntries(head.map((h, j) => [h, r[j]])));
}

async function fetchCSV(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} for ${url}`);
  return parseCSV(await response.text());
}

const fmt = (x, digits = 2) => (Number.isFinite(x) ? x.toFixed(digits) : "–");
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

/* -------------------------------------------------------------- explorer ------- */

const cache = new Map();

function methodsFor(stem) {
  const methods = ["LINE"];
  if (HEADLINE.has(stem)) methods.push("LINE_flux");
  methods.push("CLIP-Dissect", "INVERT");
  if (HAS_DND.has(stem)) methods.push("DnD");
  return methods;
}

/** Explanations of every method and LINE's history for one (model, layer) file. */
function loadLayer(stem) {
  if (!cache.has(stem)) {
    const methods = methodsFor(stem);
    const promise = Promise.all([
      ...methods.map((m) => fetchCSV(`assets/explanations/${m}/${stem}.csv`)),
      fetchCSV(`assets/histories/LINE/${stem}.csv`),
    ]).then((tables) => {
      const history = tables.pop();
      const byMethod = {};
      methods.forEach((m, i) => { byMethod[m] = new Map(tables[i].map((r) => [Number(r.unit), r])); });
      const steps = new Map();
      for (const r of history) {
        const unit = Number(r.unit);
        if (!steps.has(unit)) steps.set(unit, []);
        steps.get(unit).push({ iteration: Number(r.iteration), stage: r.stage, concept: r.concept, score: Number(r.score) });
      }
      const units = [...byMethod.LINE.keys()].sort((a, b) => a - b);
      return { stem, methods, byMethod, steps, units };
    });
    promise.catch(() => cache.delete(stem));
    cache.set(stem, promise);
  }
  return cache.get(stem);
}

const state = { model: DEFAULT.model, layer: DEFAULT.layer, unit: DEFAULT.unit, data: null, request: 0 };

function fillModels() {
  const select = $("ex-model");
  for (const m of MODELS) select.append(el("option", { value: m.id, text: `${m.name} · ${m.data}` }));
  select.value = state.model;
}

function fillLayers() {
  const select = $("ex-layer");
  const model = MODELS.find((m) => m.id === state.model);
  select.replaceChildren(...model.layers.map((layer) => {
    const main = HEADLINE.has(`${model.id}-${layer}`);
    return el("option", { value: layer, text: main ? `${layerName(model.id, layer)} (main table)` : layerName(model.id, layer) });
  }));
  if (!model.layers.includes(state.layer)) state.layer = model.layers.at(-1);
  select.value = state.layer;
}

function fillUnits(data) {
  const select = $("ex-unit");
  select.replaceChildren(...data.units.map((unit) => {
    let label = data.byMethod.LINE.get(unit).description;
    if (label.length > 48) label = `${label.slice(0, 46)}…`;
    return el("option", { value: unit, text: `${unit} · ${label}` });
  }));
  if (!data.units.includes(state.unit)) state.unit = data.units[0];
  select.value = String(state.unit);
  select.disabled = false;
}

function setStatus(message, isError = false) {
  const status = $("ex-status");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

async function showLayer() {
  const stem = `${state.model}-${state.layer}`;
  const request = ++state.request;
  $("ex-unit").disabled = true;
  setStatus("Loading explanations…");
  try {
    const data = await loadLayer(stem);
    if (request !== state.request) return;
    state.data = data;
    setStatus("");
    fillUnits(data);
    renderLayerSummary(data);
    renderNeuron();
  } catch (error) {
    if (request !== state.request) return;
    const local = location.protocol === "file:";
    setStatus(local
      ? "The explorer reads the CSV files in assets/, which browsers block on file:// pages. Serve the repository root over HTTP (python -m http.server) and open http://localhost:8000."
      : `Could not load the explanations for ${stem} (${error.message}).`, true);
    $("ex-layer-summary").textContent = "";
    $("ex-methods").replaceChildren();
    $("ex-trace").replaceChildren();
  }
}

function renderLayerSummary(data) {
  const parts = data.methods.map((m) => {
    const auc = mean([...data.byMethod[m].values()].map((r) => Number(r.AUC)));
    return [`${METHOD_NAMES[m]} `, el("b", { text: fmt(auc) })];
  });
  const line = [`Mean CoSy AUC over the ${data.units.length} neurons of this layer: `];
  parts.forEach((p, i) => line.push(...(i ? [" · "] : []), ...p));
  $("ex-layer-summary").replaceChildren(...line);
}

function renderNeuron() {
  const data = state.data;
  if (!data) return;
  const unit = state.unit;
  const reported = data.byMethod.LINE.get(unit).description;
  const steps = data.steps.get(unit) || [];
  renderMethods(data, unit, reported, steps);
  renderTrace(reported, steps);
  if (location.hash.startsWith("#neuron=") || state.touched) {
    history.replaceState(null, "", `#neuron=${state.model}/${state.layer}/${unit}`);
  }
}

function renderMethods(data, unit, reported, steps) {
  const rows = data.methods.map((m) => ({ m, r: data.byMethod[m].get(unit) })).filter((x) => x.r);
  const bestAuc = Math.max(...rows.map((x) => Number(x.r.AUC)));
  const bestMad = Math.max(...rows.map((x) => Number(x.r.MAD)));
  const body = rows.map(({ m, r }) => el("tr", { class: m === "LINE" ? "is-line" : null },
    el("th", { scope: "row", text: METHOD_NAMES[m] }),
    el("td", {}, el("span", { class: m === "LINE" ? "concept final" : "concept", text: r.description })),
    el("td", { class: Number(r.AUC) === bestAuc ? "num best" : "num", text: fmt(Number(r.AUC)) }),
    el("td", { class: Number(r.MAD) === bestMad ? "num best" : "num", text: fmt(Number(r.MAD)) }),
  ));
  const model = MODELS.find((x) => x.id === state.model);
  $("ex-methods").replaceChildren(
    el("h3", { text: `Neuron ${unit}` }),
    el("p", { class: "sub", text: `${model.name} (${model.data}) · ${layerName(model.id, state.layer)}` }),
    el("table", { class: "mtable" },
      el("thead", {}, el("tr", {},
        el("th", { scope: "col", text: "Method" }),
        el("th", { scope: "col", text: "Description" }),
        el("th", { scope: "col", class: "num", text: "AUC" }),
        el("th", { scope: "col", class: "num", text: "MAD" }))),
      el("tbody", {}, body)),
    provenance(reported, steps),
  );
}

/** Where LINE's reported concept first appeared in its search. */
function provenance(reported, steps) {
  const first = steps.find((s) => s.concept === reported);
  let origin = "comes from its search";
  if (first?.stage === "initial") origin = "is a predefined concept, from the initial ImageNet vocabulary";
  else if (first?.stage === "iteration") origin = `is a generated concept, proposed by the LLM at iteration ${first.iteration}`;
  else if (first?.stage === "summary") origin = "is the summary concept, proposed at the final summary step";
  return el("p", { class: "provenance" },
    "LINE's label ", el("strong", { text: reported }), ` ${origin}.`);
}

function renderTrace(reported, steps) {
  const container = $("ex-trace");
  if (!steps.length) { container.replaceChildren(); return; }
  const max = Math.max(...steps.map((s) => s.score), 1e-9);
  const row = (index, s) => {
    const width = Math.max(0, s.score) / max * 100;
    return el("li", { class: s.concept === reported ? "is-final" : null },
      el("span", { class: "i", text: index }),
      el("span", { class: "c", text: s.concept }),
      el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `width:${width.toFixed(1)}%` })),
      el("span", { class: "v", text: fmt(s.score) }));
  };
  const initial = steps.filter((s) => s.stage === "initial").sort((a, b) => b.score - a.score);
  const iterations = steps.filter((s) => s.stage === "iteration").sort((a, b) => a.iteration - b.iteration);
  const summary = steps.filter((s) => s.stage === "summary");
  container.replaceChildren(
    el("h3", { text: "LINE's search" }),
    el("p", { class: "sub", text: "Mean activation of the neuron on each concept's generated images." }),
    el("p", { class: "trace-group", text: `Initial scoreboard · ${initial.length} ImageNet classes` }),
    el("ol", { class: "trace" }, initial.map((s) => row("", s))),
    el("p", { class: "trace-group", text: "LLM proposals" }),
    el("ol", { class: "trace" }, iterations.map((s) => row(s.iteration, s))),
    summary.length ? el("p", { class: "trace-group", text: "Summary" }) : null,
    summary.length ? el("ol", { class: "trace" }, summary.map((s) => row("S", s))) : null,
    el("p", { class: "trace-key" }, el("span", { class: "swatch", "aria-hidden": "true" }),
      "the reported concept: the highest score on the final scoreboard."),
  );
}

/** Apply a `#neuron=<model>/<layer>/<unit>` link; returns whether the hash was one. */
function readHash() {
  const hash = /^#neuron=([^/]+)\/([^/]+)\/(\d+)$/.exec(decodeURIComponent(location.hash));
  const model = hash && MODELS.find((m) => m.id === hash[1]);
  if (!model || !model.layers.includes(hash[2])) return false;
  Object.assign(state, { model: model.id, layer: hash[2], unit: Number(hash[3]) });
  return true;
}

function initExplorer() {
  if (readHash()) $("explore").scrollIntoView();
  fillModels();
  fillLayers();
  window.addEventListener("hashchange", () => {
    if (!readHash()) return;
    $("ex-model").value = state.model;
    fillLayers();
    showLayer();
    $("explore").scrollIntoView();
  });
  $("ex-model").addEventListener("change", (e) => {
    state.model = e.target.value; state.touched = true;
    fillLayers(); showLayer();
  });
  $("ex-layer").addEventListener("change", (e) => {
    state.layer = e.target.value; state.touched = true;
    showLayer();
  });
  $("ex-unit").addEventListener("change", (e) => {
    state.unit = Number(e.target.value); state.touched = true;
    renderNeuron();
  });
  $("ex-random").addEventListener("click", () => {
    if (!state.data) return;
    const units = state.data.units;
    state.unit = units[Math.floor(Math.random() * units.length)];
    state.touched = true;
    $("ex-unit").value = String(state.unit);
    renderNeuron();
  });
  showLayer();
}

/* ------------------------------------------ activation-maximization grid ------- */

const AM_ROWS = [
  { key: "SalientImageNet", label: "Salient ImageNet", heatmap: true },
  { key: "DiffExplainer", label: "DiffExplainer" },
  { key: "DEXTER", label: "DEXTER" },
  { key: "LINE", label: "LINE" },
];

const amState = { index: 0, heatmaps: false };

function renderAmTabs() {
  const tabs = $("am-tabs");
  tabs.replaceChildren(...AM_CLASSES.map((c, i) => {
    const tab = el("button", {
      type: "button", class: "tab", role: "tab", id: `am-tab-${c.label}`,
      "aria-selected": String(i === amState.index), "aria-controls": "am-grid",
      tabindex: i === amState.index ? "0" : "-1", text: c.name,
    });
    tab.addEventListener("click", () => { amState.index = i; renderAmTabs(); renderAmGrid(); });
    tab.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      amState.index = (amState.index + step + AM_CLASSES.length) % AM_CLASSES.length;
      renderAmTabs(); renderAmGrid();
      $(`am-tab-${AM_CLASSES[amState.index].label}`).focus();
    });
    return tab;
  }));
}

function renderAmGrid() {
  const cls = AM_CLASSES[amState.index];
  const dir = `assets/dexter/${cls.label}`;
  const cells = [el("div", { class: "am-corner" })];
  cls.neurons.forEach((n) => cells.push(el("div", { class: "am-colhead", text: `Neuron ${n.neuron}` })));
  for (const row of AM_ROWS) {
    if (row.heatmap && !amState.heatmaps) continue;
    cells.push(el("div", { class: row.key === "LINE" ? "am-rowlabel is-line" : "am-rowlabel", text: row.label }));
    cls.neurons.forEach((n, col) => {
      const src = row.key === "LINE"
        ? `${dir}/LINE/neuron_${n.neuron}/image_${n.image}.jpeg`
        : `${dir}/${row.key}_${col}.jpg`;
      const alt = row.key === "LINE"
        ? `LINE image of "${n.concept}" for neuron ${n.neuron}`
        : `${row.label} image for neuron ${n.neuron}`;
      const caption = [];
      if (!row.heatmap) {
        const best = Math.max(n.act.DiffExplainer, n.act.DEXTER, n.act.LINE);
        caption.push(el("figcaption", { class: n.act[row.key] === best ? "best" : null },
          fmt(n.act[row.key]),
          row.key === "LINE" ? [el("br"), el("span", { class: "concept", text: n.concept })] : null));
      }
      cells.push(el("figure", { class: "am-cell" },
        el("img", { src, alt, width: "256", height: "256", loading: "lazy", decoding: "async" }), caption));
    });
  }
  $("am-grid").setAttribute("aria-labelledby", `am-tab-${cls.label}`);
  $("am-grid").replaceChildren(...cells);
}

function initAm() {
  renderAmTabs();
  renderAmGrid();
  $("am-heatmaps").addEventListener("change", (e) => { amState.heatmaps = e.target.checked; renderAmGrid(); });
}

/* --------------------------------------------------------- search player ------- */
// Replays a few LINE searches step by step from webpage/search/: search.json (concepts and scores from
// assets/histories/LINE, labels and AUC from assets/explanations/LINE) plus the images LINE generated.
// Step 0 is the initial scoreboard, steps 1..10 the LLM's proposals, the last step the summary.

const SVG_NS = "http://www.w3.org/2000/svg";
const SP_STEP_MS = 2400;
const SP_ROWS = 10;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const sp = { neurons: [], neuron: null, step: 0, playing: false, timer: 0, seen: false };

/** replaceChildren that skips null/false, as el() does (replaceChildren would print "null"). */
function setChildren(node, ...children) {
  node.replaceChildren(...children.flat().filter((c) => c != null && c !== false));
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value == null) continue;
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) if (child != null && child !== false) node.append(child);
  return node;
}

/** Adds each concept's step and whether it topped the scoreboard when it arrived. */
function prepareSearch(n) {
  const nInitial = n.concepts.filter((c) => c.stage === "initial").length;
  n.concepts.forEach((c, index) => {
    c.index = index;
    c.step = c.stage === "initial" ? 0 : index - nInitial + 1;
  });
  n.last = n.concepts.at(-1).step;
  n.iterations = n.concepts.filter((c) => c.stage === "iteration").length;
  n.best = [];  // n.best[s]: top of the scoreboard after step s
  let top = null;
  for (let s = 0; s <= n.last; s++) {
    for (const c of n.concepts) {
      if (c.step !== s) continue;
      c.setsBest = s > 0 && c.score > top.score;
      if (!top || c.score > top.score) top = c;
    }
    n.best[s] = top;
  }
  n.best[0].setsBest = true;  // where the best-so-far line starts
  n.max = Math.max(...n.concepts.map((c) => c.score));
  n.min = Math.min(...n.concepts.map((c) => c.score));
  return n;
}

const spSrc = (n, c, k) => `webpage/search/${n.id}/${String(c.index).padStart(2, "0")}-${k}.webp`;
const spAtStep = (n, s) => n.concepts.find((c) => c.step === s);
const spStage = (c) => (c.stage === "initial" ? "initial scoreboard" : c.stage === "summary" ? "summary step" : `iteration ${c.step}`);
const spTickLabel = (n, s) => (s === 0 ? "0" : spAtStep(n, s).stage === "summary" ? "S" : String(s));

function spImg(n, c, k) {
  return el("img", { src: spSrc(n, c, k), alt: `Generated image ${k} of "${c.name}"`, width: "256", height: "256", decoding: "async" });
}

function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((v) => v >= raw);
  const ticks = [];
  for (let v = Math.floor(lo / step) * step; v < hi + step - 1e-9; v += step) ticks.push(Number(v.toFixed(6)));
  return ticks;
}

function drawSpChart(pop = false) {
  const n = sp.neuron, box = $("sp-chart"), root = $("sp-svg");
  if (!n || !box.clientWidth) return;
  const W = box.clientWidth, H = box.clientHeight, step = sp.step;
  const pad = { l: 44, r: 12, t: 26, b: 24 };
  const cols = n.last + 1;
  const band = (W - pad.l - pad.r) / cols;
  const X = (s) => pad.l + band * (s + 0.5);
  const ticks = niceTicks(Math.min(0, n.min), n.max, H < 220 ? 3 : 4);
  const lo = ticks[0], hi = ticks.at(-1);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);

  // Dot positions; the initial concepts share column 0, so nudge overlapping ones sideways.
  const pos = new Map(), placed = [];
  const offsets = [0, 7, -7, 14, -14, 21, -21].filter((dx) => Math.abs(dx) <= band / 2 - 5);
  for (const c of n.concepts) {
    let x = X(c.step);
    const y = Y(c.score);
    if (c.step === 0) {
      x += offsets.find((dx) => placed.every((p) => Math.hypot(p.x - x - dx, p.y - y) >= 9)) ?? 0;
      placed.push({ x, y });
    }
    pos.set(c, { x, y });
  }

  const marks = [svg("rect", { class: "band", x: X(step) - band / 2 + 1, y: pad.t - 8, width: band - 2, height: H - pad.t + 8, rx: 6 })];
  for (const t of ticks) {
    marks.push(svg("line", { class: "grid", x1: pad.l, x2: W - pad.r, y1: Y(t), y2: Y(t), style: t === 0 && lo < 0 ? "stroke:var(--bar)" : null }));
    marks.push(svg("text", { class: "tick", x: pad.l - 8, y: Y(t) + 4, "text-anchor": "end", text: String(t) }));
  }
  marks.push(svg("text", { class: "tick", x: pad.l - 8, y: 10, "text-anchor": "end", text: "score" }));
  for (let s = 0; s <= n.last; s++) {
    marks.push(svg("text", { class: s === step ? "tick is-current" : "tick", x: X(s), y: H - 6, "text-anchor": "middle", text: spTickLabel(n, s) }));
  }
  let d = `M${X(0)},${Y(n.best[0].score)}`;
  for (let s = 1; s <= step; s++) d += `H${X(s)}V${Y(n.best[s].score)}`;
  marks.push(svg("path", { class: "best-line", d }));

  const current = step === 0 ? n.best[0] : spAtStep(n, step);
  const final = step === n.last ? n.best[n.last] : null;
  for (const c of n.concepts) {
    if (c.step > step) marks.push(svg("circle", { class: "dot ghost", cx: pos.get(c).x, cy: pos.get(c).y, r: 3.5 }));
  }
  const shown = n.concepts.filter((c) => c.step <= step)
    .sort((a, b) => a.setsBest - b.setsBest || (a === current) - (b === current));
  for (const c of shown) {
    const big = c === current && step > 0;
    marks.push(svg("circle", {
      class: ["dot", c.setsBest ? "is-best" : "", big && pop ? "pop" : ""].join(" ").trim(),
      cx: pos.get(c).x, cy: pos.get(c).y, r: big ? 6.5 : c.setsBest ? 5 : 4.5,
    }));
  }
  if (final) marks.push(svg("circle", { class: "final-ring", cx: pos.get(final).x, cy: pos.get(final).y, r: 10 }));

  root.setAttribute("viewBox", `0 0 ${W} ${H}`);
  root.setAttribute("aria-label",
    `Scores of the ${n.concepts.length} concepts LINE tried for neuron ${n.unit}, by iteration. ` +
    `The best score rises from ${fmt(n.best[0].score)} (${n.best[0].name}) to ${fmt(n.best[n.last].score)} (${n.best[n.last].name}).`);
  root.replaceChildren(...marks);

  // Direct labels: the step's concept, and the final label once the search is over.
  const boxes = [];
  for (const c of final && final !== current ? [final, current] : [current]) {
    const p = pos.get(c);
    const text = svg("text", { class: "label", x: p.x, y: p.y - 13, "text-anchor": "middle", text: c.name });
    root.append(text);
    let bb = text.getBBox();
    const overlaps = boxes.some((b) => bb.x < b.x + b.width && b.x < bb.x + bb.width && bb.y < b.y + b.height && b.y < bb.y + bb.height);
    if (bb.y < 0 || overlaps) text.setAttribute("y", p.y + 22);
    if (bb.x < pad.l - 6) text.setAttribute("x", p.x + (pad.l - 6 - bb.x));
    else if (bb.x + bb.width > W - 2) text.setAttribute("x", p.x - (bb.x + bb.width - W + 2));
    bb = text.getBBox();
    boxes.push(bb);
  }

  // Hit areas: a click on a column jumps there; dots show their concept and score.
  const tip = $("sp-tip");
  for (let s = 0; s <= n.last; s++) {
    const col = svg("rect", { class: "col", x: X(s) - band / 2, y: 0, width: band, height: H });
    col.addEventListener("click", () => { spPause(); spGo(s); });
    root.append(col);
  }
  for (const c of shown) {
    const p = pos.get(c);
    const hit = svg("circle", { class: "hit", cx: p.x, cy: p.y, r: 12 });
    hit.addEventListener("pointerenter", () => {
      tip.replaceChildren(el("b", { text: fmt(c.score) }), el("span", { class: "c", text: c.name }), el("span", { class: "s", text: spStage(c) }));
      tip.style.left = `${Math.min(Math.max(p.x, 80), W - 80)}px`;
      tip.style.top = `${p.y}px`;
      tip.hidden = false;
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
    hit.addEventListener("click", () => { spPause(); spGo(c.step); });
    root.append(hit);
  }
  tip.hidden = true;

  // Line the slider's stops up with the chart's columns.
  const range = $("sp-range");
  range.style.left = `${pad.l + band / 2 - 8}px`;
  range.style.width = `${(cols - 1) * band + 16}px`;
}

function renderSpStep() {
  const n = sp.neuron, s = sp.step;
  if (s === 0) {
    const initial = n.concepts.filter((c) => c.step === 0);
    $("sp-step").replaceChildren(
      el("p", { class: "sp-kicker", text: "Start · initial scoreboard" }),
      el("h3", { text: `${initial.length} ImageNet class names` }),
      el("p", { class: "sp-sub", text: "The top 5 highest-scoring classes on natural ImageNet images, alongside 5 randomly selected ones, to balance exploitation and exploration. One synthesized image of each:" }),
      el("div", { class: "sp-images" }, initial.map((c) => el("figure", {}, spImg(n, c, 1),
        el("figcaption", {}, el("span", { class: "c", text: c.name }), fmt(c.score))))));
    return;
  }
  const c = spAtStep(n, s);
  const summary = c.stage === "summary";
  const seen = n.concepts.filter((x) => x.step <= s);
  const rank = seen.filter((x) => x.score > c.score).length + 1;
  const final = n.best[n.last];
  setChildren($("sp-step"),
    el("p", { class: "sp-kicker", text: summary ? "Summary step · after the last iteration" : `Iteration ${s} of ${n.iterations} · LLM proposal` }),
    el("h3", {},
      el("span", { class: "concept", text: c.name }),
      el("span", { class: "sp-score", text: fmt(c.score) }),
      c.setsBest
        ? el("span", { class: "sp-badge is-best", text: `new best, +${fmt(c.score - n.best[s - 1].score)}` })
        : el("span", { class: "sp-badge", text: `rank ${rank} of ${seen.length}` })),
    summary ? el("p", { class: "sp-sub", text: "The LLM is provided with the top three concepts from the scoreboard to generate a generalized, higher-order concept, which is evaluated via the standard pipeline." }) : null,
    el("div", { class: "sp-images" }, Array.from({ length: c.images }, (_, i) => el("figure", {}, spImg(n, c, i + 1)))),
    c.reasoning ? el("blockquote", { class: "sp-why" }, el("span", { text: "The LLM's reasoning" }), c.reasoning) : null,
    s === n.last ? el("p", { class: "sp-final" },
      "LINE reports ", el("span", { class: "concept final", text: final.name }),
      `, the top of the final scoreboard${final === c ? "" : `, first proposed ${final.step === 0 ? "in the initial vocabulary" : `at iteration ${final.step}`}`}. CoSy AUC ${fmt(n.auc)}.`) : null,
  );
}

function renderSpBoard(animate) {
  const n = sp.neuron, s = sp.step;
  const seen = n.concepts.filter((c) => c.step <= s).sort((a, b) => b.score - a.score);
  const fresh = s > 0 ? spAtStep(n, s) : null;
  const row = (c, rank) => el("li", { class: [c === seen[0] ? "is-final" : "", c === fresh ? "is-new" : ""].join(" ").trim() || null, "data-key": c.index },
    el("span", { class: "i", text: rank }),
    el("span", { class: "c", text: c.name }),
    el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `width:${(Math.max(0, c.score) / n.max * 100).toFixed(1)}%` })),
    el("span", { class: "v", text: fmt(c.score) }));
  const rows = seen.slice(0, SP_ROWS).map((c, i) => row(c, i + 1));
  const freshRank = seen.indexOf(fresh);
  if (freshRank >= SP_ROWS) rows.push(el("li", { class: "gap", "aria-hidden": "true", "data-key": "gap", text: "⋮" }), row(fresh, freshRank + 1));

  const box = $("sp-board");
  const old = box.querySelector("ol");
  const before = new Map();
  if (old && animate) {
    const top = old.getBoundingClientRect().top;
    for (const li of old.children) before.set(li.dataset.key, li.getBoundingClientRect().top - top);
  }
  const list = el("ol", { class: "trace" }, rows);
  const hidden = seen.length - Math.max(SP_ROWS, freshRank + 1);
  const when = s === 0 ? "" : `, after ${fresh.stage === "summary" ? "the summary step" : `iteration ${s}`}`;
  setChildren(box,
    el("h3", {}, "Scoreboard", el("span", { class: "sub", text: `${seen.length} concepts${when}` })),
    list,
    el("p", { class: "trace-key" }, el("span", { class: "swatch", "aria-hidden": "true" }),
      s === n.last ? "the final label: the top concept once the search is over." : "the top concept: the label LINE would report if it stopped here.",
      hidden > 0 ? ` ${hidden} lower-scoring concept${hidden > 1 ? "s are" : " is"} not shown.` : null));
  if (!animate || reduceMotion.matches) return;
  const top = list.getBoundingClientRect().top;
  for (const li of list.children) {
    const was = before.get(li.dataset.key);
    if (was === undefined) li.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 450, easing: "ease-out" });
    else {
      const dy = was - (li.getBoundingClientRect().top - top);
      if (Math.abs(dy) > 1) li.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 500, easing: "cubic-bezier(.2,.7,.2,1)" });
    }
  }
}

function spPreload(step) {
  const n = sp.neuron;
  for (const c of n.concepts) {
    if (c.step !== step) continue;
    for (let k = 1; k <= (step === 0 ? 1 : c.images); k++) new Image().src = spSrc(n, c, k);
  }
}

function spGo(step, animate = true) {
  const n = sp.neuron;
  sp.step = step;
  const range = $("sp-range");
  range.value = String(step);
  const c = step === 0 ? null : spAtStep(n, step);
  range.setAttribute("aria-valuetext", c
    ? `${c.stage === "summary" ? "Summary step" : `Iteration ${step}`}: ${c.name}, score ${fmt(c.score)}`
    : "Initial scoreboard");
  drawSpChart(animate);
  renderSpStep();
  renderSpBoard(animate);
  spPreload(step + 1);
  spButton();
}

function spButton() {
  const state = sp.playing ? "playing" : sp.step >= sp.neuron.last ? "ended" : "paused";
  $("sp-play").dataset.state = state;
  $("sp-play").setAttribute("aria-label", { playing: "Pause", ended: "Replay", paused: "Play" }[state]);
}

function spSchedule() {
  clearTimeout(sp.timer);
  sp.timer = setTimeout(() => {
    if (!sp.playing) return;
    spGo(sp.step + 1);
    if (sp.step >= sp.neuron.last) spPause();
    else spSchedule();
  }, sp.step === 0 ? SP_STEP_MS * 1.5 : SP_STEP_MS);
}

function spPlay() {
  if (sp.step >= sp.neuron.last) spGo(0, false);
  sp.playing = true;
  spButton();
  spSchedule();
}

function spPause() {
  sp.playing = false;
  clearTimeout(sp.timer);
  spButton();
}

function spTabLabel(n) {
  return n.model.startsWith("dinov3") ? "DINOv3" : `${n.name} · ${n.data}`;
}

function renderSpTabs() {
  const index = sp.neurons.indexOf(sp.neuron);
  $("sp-tabs").replaceChildren(...sp.neurons.map((n, i) => {
    const tab = el("button", {
      type: "button", class: "tab", role: "tab", id: `sp-tab-${i}`,
      "aria-selected": String(i === index), "aria-controls": "sp-player",
      tabindex: i === index ? "0" : "-1", text: spTabLabel(n),
    });
    tab.addEventListener("click", () => selectSearch(i));
    tab.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      selectSearch((sp.neurons.indexOf(sp.neuron) + step + sp.neurons.length) % sp.neurons.length);
      $(`sp-tab-${sp.neurons.indexOf(sp.neuron)}`).focus();
    });
    return tab;
  }));
  $("sp-player").setAttribute("aria-labelledby", `sp-tab-${index}`);
}

function selectSearch(i) {
  if (sp.neurons[i] === sp.neuron) return;
  spPause();
  showSearch(sp.neurons[i]);
  if (!reduceMotion.matches) spPlay();
}

function showSearch(n) {
  sp.neuron = n;
  $("sp-range").max = String(n.last);
  renderSpTabs();
  $("sp-head").replaceChildren(
    el("span", {}, el("strong", { text: `Neuron ${n.unit}` }), `${n.name} (${n.data}) · ${layerName(n.model, n.layer)}`),
    el("a", { href: `#neuron=${n.model}/${n.layer}/${n.unit}`, text: "Compare with other methods in the explorer" }));
  spGo(0, false);
}

async function initSearch() {
  try {
    const response = await fetch("webpage/search/search.json");
    if (!response.ok) throw new Error(`${response.status} for webpage/search/search.json`);
    const data = await response.json();
    sp.neurons = data.neurons.map(prepareSearch);
    sp.neuron = sp.neurons.find((n) => n.id === data.default) || sp.neurons[0];
  } catch (error) {
    $("sp-player").hidden = true;
    $("sp-status").classList.add("error");
    $("sp-status").textContent = location.protocol === "file:"
      ? "The replay reads webpage/search/search.json, which browsers block on file:// pages. Serve the repository root over HTTP (python -m http.server) and open http://localhost:8000."
      : `Could not load the searches (${error.message}).`;
    return;
  }
  showSearch(sp.neuron);
  $("sp-play").addEventListener("click", () => (sp.playing ? spPause() : spPlay()));
  $("sp-range").addEventListener("input", (e) => { spPause(); spGo(Number(e.target.value)); });
  let width = 0;
  new ResizeObserver(() => {
    if ($("sp-chart").clientWidth !== width) { width = $("sp-chart").clientWidth; drawSpChart(); }
  }).observe($("sp-chart"));
  // Start playing the first time the chart scrolls into view; pause when it leaves.
  new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting && !sp.seen) {
      sp.seen = true;
      if (!reduceMotion.matches) spPlay();
    } else if (!entry.isIntersecting && sp.playing) spPause();
  }, { threshold: 0.6 }).observe($("sp-chart"));
}

/* -------------------------------------------------------------- citation ------- */

function initCopy() {
  const button = $("bib-copy");
  button.addEventListener("click", async () => {
    const text = $("bib-text").textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
    } catch {
      const range = document.createRange();
      range.selectNodeContents($("bib-text"));
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Press Ctrl+C";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 1800);
  });
}

initSearch();
initExplorer();
initAm();
initCopy();
