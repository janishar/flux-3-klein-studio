// flux klein studio — the page. Talks to web/server.py; helmstudio's components come through /helm/.
"use strict";

const $ = (id) => document.getElementById(id);

// flux_3_klein.generate.ASPECT_RATIOS
const RATIOS = {
  "1:1": [1024, 1024], "4:3": [1184, 880], "3:4": [880, 1184], "3:2": [1248, 832],
  "2:3": [832, 1248], "16:9": [1360, 768], "9:16": [768, 1360],
};
const STAGES = [
  ["upsample", "Upsample"], ["load", "Load"], ["denoise", "Denoise"], ["decode", "Decode"], ["save", "Save"],
];
const FIELDS = ["prompt", "width", "height", "steps", "seed", "upsample", "guidance", "boxes", "device", "model"];
// The image models by path key (web/server.py MODELS), with their default steps. Only Base samples with guidance.
const MODELS = {
  model: { name: "FLUX 3 Image Klein", glyph: "K", steps: 16 },
  base: { name: "FLUX 3 Image Klein Base", glyph: "B", steps: 50, guidance: 3.5 },
};

const ui = {
  session: "", paths: {}, ratio: null, inputs: [], takes: [], queue: [],
  selected: null, // gallery item id of the take in the preview
  live: null, // the running job's summary, for the stage bar
};

// ── helpers ─────────────────────────────────────────────────────────────
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...kids.filter((kid) => kid !== null && kid !== undefined && kid !== false));
  return node;
}

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || (data && data.error)) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}

const q = (path) => `${path}?session=${encodeURIComponent(ui.session)}`;
const floor16 = (v) => Math.floor(v / 16) * 16;

function showError(node, err) {
  node.textContent = err ? err.message || String(err) : "";
  node.hidden = !err;
}

/** A small in-page dialog: a name to type, or a yes/no. Resolves to the text, true, or null. */
function ask({ title, message = "", value = null, ok = "OK" }) {
  const d = $("askDialog");
  $("askTitle").textContent = title;
  $("askMessage").textContent = message;
  $("askInput").hidden = value === null;
  $("askInput").value = value || "";
  $("askOk").textContent = ok;
  showError($("askError"), null);
  d.showModal();
  if (value !== null) $("askInput").select();
  return new Promise((resolve) => {
    const done = (result) => { d.close(); cleanup(); resolve(result); };
    const onOk = () => done(value === null ? true : $("askInput").value.trim() || null);
    const onCancel = () => done(null);
    const onKey = (e) => { if (e.key === "Enter" && value !== null) onOk(); };
    const cleanup = () => {
      $("askOk").removeEventListener("click", onOk);
      $("askCancel").removeEventListener("click", onCancel);
      $("askInput").removeEventListener("keydown", onKey);
      d.removeEventListener("cancel", onCancel);
    };
    $("askOk").addEventListener("click", onOk);
    $("askCancel").addEventListener("click", onCancel);
    $("askInput").addEventListener("keydown", onKey);
    d.addEventListener("cancel", onCancel);
  });
}

// ── the form ────────────────────────────────────────────────────────────
function settings() {
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  const model = $("model").value;
  return {
    prompt: $("prompt").value, ratio: ui.ratio, width: num("width"), height: num("height"),
    steps: num("steps") ?? MODELS[model].steps, seed: num("seed") ?? 42,
    upsample: $("upsample").checked, guidance: num("guidance") ?? MODELS.base.guidance, boxes: $("boxes").value,
    device: $("device").value, model,
  };
}

function applySettings(s) {
  $("prompt").value = s.prompt ?? "";
  $("width").value = s.width ?? "";
  $("height").value = s.height ?? "";
  $("model").value = MODELS[s.model] ? s.model : "model";
  $("steps").value = s.steps ?? MODELS[$("model").value].steps;
  $("seed").value = s.seed ?? 42;
  $("upsample").checked = s.upsample ?? true; // on unless this session turned it off
  $("guidance").value = s.guidance ?? MODELS.base.guidance;
  $("boxes").value = s.boxes ?? "";
  $("device").value = s.device || "auto";
  ui.ratio = s.ratio || null;
  update();
}

/** What size the CLI will render, mirroring generate.py and the pipeline's own default. */
function plannedSize() {
  const s = settings();
  let base;
  let source;
  if (s.ratio) {
    base = RATIOS[s.ratio];
    source = "chosen";
  } else if (ui.inputs.length) {
    // Flux3ImageProcessor.preprocess: one image is padded to the grid and cropped back, so its fitted size stands;
    // of several, each is fitted under 1 MiB pixels and cropped to the grid. The first one sets the size.
    const w = ui.inputs[0].width || 1;
    const h = ui.inputs[0].height || 1;
    if (ui.inputs.length === 1) {
      // The canvas is what must fit, so the scale backs off 0.5% at a time, as _resize_if_canvas_exceeds_area does.
      const up = (v) => Math.min(4096, Math.max(64, Math.ceil(v / 16) * 16));
      let scale = Math.min(1, Math.sqrt(4194304 / (w * h)), 4096 / Math.max(w, h));
      let fw;
      let fh;
      for (;;) {
        fw = Math.max(1, Math.round(w * scale));
        fh = Math.max(1, Math.round(h * scale));
        if (up(fw) * up(fh) <= 4194304) break;
        scale *= 0.995;
      }
      // A typed side turns the crop-back off, and the pipeline takes the other side from the canvas.
      base = s.width || s.height ? [up(fw), up(fh)] : [fw, fh];
    } else {
      const scale = Math.min(1, Math.sqrt(1048576 / (w * h)));
      base = [floor16(w * scale), floor16(h * scale)];
    }
    source = "follows the first image";
  } else {
    base = RATIOS["1:1"];
    source = "default 1:1";
  }
  return { w: s.width || base[0], h: s.height || base[1], source };
}

function update() {
  renderModel();
  const edit = ui.inputs.length > 0;
  for (const span of $("modeSeg").children) span.classList.toggle("on", (span.dataset.m === "i2i") === edit);
  $("modeSeg").lastElementChild.textContent = edit ? `Edit · ${ui.inputs.length} image${ui.inputs.length > 1 ? "s" : ""}` : "Edit";
  $("boxes").disabled = !$("upsample").checked;
  $("guidanceRow").hidden = !MODELS[$("model").value].guidance;
  $("promptCount").textContent = `${$("prompt").value.length} chars`;
  $("refCount").textContent = `${ui.inputs.length}/10`;

  const { w, h, source } = plannedSize();
  $("ratioSrc").textContent = $("width").value && $("height").value ? "width × height" : ui.ratio ? "chosen" : `auto · ${source}`;
  $("dims").textContent = w && h ? `${w} × ${h}` : "—";
  $("mp").textContent = w && h ? ((w * h) / 1e6).toFixed(2) : "—";
  // Only a typed side can be off the grid: a lone reference's own size is cropped back on purpose.
  const ok = $("width").value % 16 === 0 && $("height").value % 16 === 0;
  $("grid16").textContent = ok ? "ok" : "not a multiple";
  $("grid16").classList.toggle("warn", !ok);
  renderRatios();
  renderRefs();
  renderCommand();
  renderPipeline();
  if (!ui.takes.some((t) => t.id === ui.selected)) $("frame").style.setProperty("--r", plannedRatio());
}

function renderRatios() {
  const options = [[null, "Auto"], ...Object.keys(RATIOS).map((k) => [k, k])];
  $("ratios").replaceChildren(...options.map(([key, label]) => {
    const [w, h] = key ? RATIOS[key] : [1, 1];
    const s = 22 / Math.max(w, h);
    return el("button", {
      type: "button", class: `ratio${key ? "" : " auto"}`, "aria-pressed": String(ui.ratio === key),
      onclick: () => { ui.ratio = key; update(); saveSoon(); },
    }, el("i", { style: `width:${Math.round(w * s)}px;height:${Math.round(h * s)}px` }), label);
  }));
}

function renderCommand() {
  const s = settings();
  const quote = (t) => (/^[\w./:=@+,-]+$/.test(t) ? t : `"${t.replace(/"/g, '\\"')}"`);
  const tagged = resolveMentions(s.prompt);
  const p = tagged.length > 40 ? `${tagged.slice(0, 40)}…` : tagged;
  const parts = [`<span class="k">flux-3-klein</span> ${escapeHtml(quote(p || "…"))}`];
  if (s.model !== "model") parts.push(`-m ${escapeHtml(quote(ui.paths[s.model]?.path || MODELS[s.model].name))}`);
  if (ui.inputs.length) parts.push(`--input ${escapeHtml(ui.inputs.map((i) => i.name).join(","))}`);
  if (s.ratio) parts.push(`--ratio ${s.ratio}`);
  if (s.width) parts.push(`--width ${s.width}`);
  if (s.height) parts.push(`--height ${s.height}`);
  if (s.steps !== MODELS[s.model].steps) parts.push(`--steps ${s.steps}`);
  if (MODELS[s.model].guidance && s.guidance !== MODELS[s.model].guidance) parts.push(`--guidance ${s.guidance}`);
  if (s.seed !== 42) parts.push(`--seed ${s.seed}`);
  if (!s.upsample) parts.push("--no-upsample");
  if (s.upsample && s.boxes.trim()) parts.push("--boxes boxes.json");
  if (s.device !== "auto") parts.push(`--device ${s.device}`);
  $("cmd").innerHTML = parts.join(" \\\n  ");
}

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

let saveTimer;
function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => api("/api/session/save", { session: ui.session, settings: settings() }).catch(() => {}), 600);
}

// ── references ──────────────────────────────────────────────────────────
// ── @ mentions: a reference image named in the prompt ───────────────────
// The page writes @name; the server turns it into "the first image" by the list's order (web/server.py
// resolve_mentions), so a mention still points at its image after another is removed.
const MENTION_RE = /@([A-Za-z0-9._-]*[A-Za-z0-9_-])/g;
const ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"];
const mention = { start: -1, items: [], at: 0 };

function resolveMentions(text) {
  const index = new Map(ui.inputs.map((input, i) => [input.name, i]));
  return text.replace(MENTION_RE, (m, name) => (index.has(name) ? `the ${ORDINALS[index.get(name)]} image` : m));
}

/** The "@query" being typed just before the caret, or null. An @ inside a word (an email) is not one. */
function mentionQuery() {
  const box = $("prompt");
  const before = box.value.slice(0, box.selectionStart);
  const m = /(^|[^A-Za-z0-9._-])@([A-Za-z0-9._-]*)$/.exec(before);
  return m ? { start: before.length - m[2].length - 1, query: m[2].toLowerCase() } : null;
}

function renderMentions() {
  const found = ui.inputs.length ? mentionQuery() : null;
  mention.items = found ? ui.inputs.filter((input) => input.name.toLowerCase().includes(found.query)) : [];
  mention.start = found ? found.start : -1;
  mention.at = Math.min(mention.at, Math.max(0, mention.items.length - 1));
  $("mentions").hidden = !mention.items.length;
  $("mentions").replaceChildren(...mention.items.map((input, i) => el("button", {
    type: "button", class: "mention", role: "option", "aria-selected": String(i === mention.at),
    onmousedown: (e) => { e.preventDefault(); pickMention(input.name); }, // before the textarea blurs
  },
  el("img", { src: input.thumb, alt: "" }),
  el("span", { class: "num", text: `the ${ORDINALS[ui.inputs.indexOf(input)]} image` }),
  el("span", { class: "name", text: input.name }))));
}

function pickMention(name) {
  const box = $("prompt");
  const end = box.selectionStart;
  box.setRangeText(`@${name} `, mention.start, end, "end");
  $("mentions").hidden = true;
  box.dispatchEvent(new Event("input"));
  box.focus();
}

function insertMention(name) {
  const box = $("prompt");
  const at = box.selectionStart ?? box.value.length;
  const pad = at > 0 && !/\s/.test(box.value[at - 1]) ? " " : "";
  box.setRangeText(`${pad}@${name} `, at, box.selectionEnd ?? at, "end");
  box.dispatchEvent(new Event("input"));
  box.focus();
}

function onMentionKey(e) {
  if ($("mentions").hidden) return;
  const n = mention.items.length;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    mention.at = (mention.at + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
    renderMentions();
  } else if (e.key === "Enter" || e.key === "Tab") {
    e.preventDefault();
    pickMention(mention.items[mention.at].name);
  } else if (e.key === "Escape") {
    e.preventDefault();
    $("mentions").hidden = true;
  }
}

function renderRefs() {
  const followFirst = !ui.ratio && !$("width").value && !$("height").value;
  const cells = ui.inputs.map((input, i) => el("div", {
    class: `ref${followFirst && i === 0 ? " follow" : ""}`,
    title: `${input.original_name} · ${input.width}×${input.height}`,
  },
  el("img", { src: input.thumb, alt: input.original_name, loading: "lazy", title: `Click to add @${input.name} to the prompt`, onclick: () => insertMention(input.name) }),
  el("span", { class: "n", text: String(i + 1) }),
  el("button", {
    class: "x", type: "button", "aria-label": `Remove ${input.original_name}`, text: "✕",
    onclick: async () => { await api("/api/inputs/delete", { session: ui.session, name: input.name }); loadInputs(); },
  })));
  for (let i = ui.inputs.length; i < Math.min(10, Math.max(5, ui.inputs.length + 1)); i++) {
    cells.push(el("div", { class: "ref empty", "aria-hidden": "true" }));
  }
  $("refs").replaceChildren(...cells);
}

async function loadInputs() {
  ui.inputs = await api(q("/api/inputs"));
  update();
}

async function uploadFiles(files) {
  for (const file of files) {
    const res = await fetch(q("/api/upload"), {
      method: "POST", headers: { "X-Filename": encodeURIComponent(file.name) }, body: file,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { termLine(`[studio] ${file.name}: ${data.error || res.status}`, "e"); break; }
  }
  loadInputs();
}

// ── takes ───────────────────────────────────────────────────────────────
async function loadTakes(selectNewest = false) {
  ui.takes = await api(q("/api/takes"));
  if (selectNewest || !ui.takes.some((t) => t.id === ui.selected)) ui.selected = ui.takes[0]?.id ?? null;
  renderTakes();
  showTake();
}

function renderTakes() {
  $("takeCount").textContent = `(${ui.takes.length})`;
  if (!ui.takes.length) {
    $("takes").replaceChildren(el("div", { class: "none", text: "No takes in this session yet." }));
    return;
  }
  $("takes").replaceChildren(...ui.takes.map((take) => {
    const p = take.params;
    const act = (label, glyph, fn) => el("span", {
      title: label, role: "button", tabindex: "0", text: glyph,
      onclick: (e) => { e.stopPropagation(); fn(take); },
    });
    return el("button", {
      type: "button", class: `take${take.id === ui.selected ? " on" : ""}`,
      onclick: () => { ui.selected = take.id; renderTakes(); showTake(); },
    },
    el("div", { class: "thumb" },
      el("img", { src: take.thumb, alt: "", loading: "lazy" }),
      el("div", { class: "flags" },
        el("b", { text: p.edit ? "EDIT" : "T2I" }),
        p.settings?.model === "base" && el("b", { text: "BASE" }),
        p.upsampled_prompt && el("b", { text: "UP" })),
      el("div", { class: "acts" },
        act("Use as reference", "↩", useTake), act("Reuse settings", "⟲", reuseTake), act("Delete", "✕", deleteTake))),
    el("div", { class: "meta" }, `seed ${p.seed} · ${p.width}×${p.height}`, el("span", { text: p.prompt || "" })));
  }));
}

function showTake() {
  const take = ui.takes.find((t) => t.id === ui.selected);
  const p = take?.params || {};
  $("frameImg").hidden = !take;
  $("frameEmpty").hidden = !!take;
  $("frameBadge").hidden = !take;
  $("useTake").disabled = $("reuseTake").disabled = !take;
  $("openTake").hidden = !take;
  $("stageRatio").hidden = !take;
  if (!take) {
    $("stageTitle").textContent = "No take yet";
    $("frame").style.setProperty("--r", plannedRatio());
    showUpsampled(null);
    return;
  }
  $("frameImg").src = take.url;
  $("openTake").href = take.url;
  $("frame").style.setProperty("--r", String(p.width / p.height));
  $("stageTitle").textContent = p.name || take.id;
  $("stageRatio").textContent = `${p.width}×${p.height}`;
  const base = p.settings?.model === "base";
  $("frameBadge").textContent = `${base ? "Base · " : ""}seed ${p.seed} · ${p.steps} steps${base ? ` · guidance ${p.guidance}` : ""} · ${p.mode}${p.elapsed ? ` · ${Math.round(p.elapsed)}s` : ""}`;
  showUpsampled(p.upsampled_prompt ? { prompt: p.prompt, upsampled: p.upsampled_prompt } : null);
}

function plannedRatio() {
  const { w, h } = plannedSize();
  return w && h ? String(w / h) : "1";
}

function showUpsampled(u) {
  $("upsampledBox").hidden = !u;
  if (!u) return;
  $("origPrompt").textContent = u.prompt || "";
  $("upsampledPrompt").textContent = u.upsampled;
}

async function useTake(take) {
  try {
    await api("/api/takes/use", { session: ui.session, id: take.id });
    loadInputs();
  } catch (err) {
    termLine(`[studio] ${err.message}`, "e");
  }
}

function reuseTake(take) {
  applySettings(take.params.settings || take.params);
  saveSoon();
}

async function deleteTake(take) {
  const yes = await ask({ title: "Delete this take?", message: `${take.params.name} leaves the gallery. Takes made from it keep their history.`, ok: "Delete" });
  if (!yes) return;
  await api("/api/takes/delete", { id: take.id });
  loadTakes();
}

// ── queue, progress, terminal ───────────────────────────────────────────
function renderQueue() {
  const running = ui.queue.find((j) => j.status === "running" || j.status === "cancelling");
  $("lamp").className = `dot ${running ? "busy" : "on"}`;
  $("lampText").textContent = running ? "rendering" : ui.queue.length ? "queued" : "idle";
  if (!ui.queue.length) {
    $("queue").replaceChildren(el("div", { class: "none", text: "Nothing queued." }));
    return;
  }
  $("queue").replaceChildren(...ui.queue.map((job, i) => el("div", { class: "qi" },
    el("span", { class: `dot ${job.status === "running" ? "busy" : ""}` }),
    el("span", { text: job.status === "running" ? "Rendering" : job.status === "cancelling" ? "Cancelling" : `Queued #${i + 1}` }),
    el("button", { class: "link", type: "button", text: "cancel", onclick: () => api("/api/cancel", { id: job.id }) }),
    el("small", { text: `${job.prompt} · ${job.ratio || "auto"} · seed ${job.seed}` }))));
}

function renderPipeline() {
  const job = ui.live;
  const upsample = job ? !job.argv_display?.includes("--no-upsample") : $("upsample").checked;
  const at = job ? STAGES.findIndex(([key]) => key === job.progress.stage) : -1;
  const steps = job?.progress.total || $("steps").value || MODELS[$("model").value].steps;
  $("pipeline").replaceChildren(...STAGES.map(([key, name], i) => {
    const skip = key === "upsample" && !upsample;
    let cls = "";
    let width = null;
    if (skip) cls = "skip";
    else if (job && i < at) cls = "done";
    else if (job && i === at) {
      cls = "now";
      width = key === "denoise" && job.progress.total ? (100 * job.progress.step) / job.progress.total : 35;
    }
    const sub = { upsample: "pup 9B", load: `bf16 · ${($("device").value === "auto" ? "auto" : $("device").value).toUpperCase()}`, denoise: `${steps} steps`, decode: "VAE", save: "PNG" }[key];
    return el("div", { class: `stepc ${cls}` },
      el("div", { class: "bar" }, el("i", { style: width === null ? null : `width:${width}%` })), name, el("small", { text: sub }));
  }));
  $("progressActions").hidden = !job;
  $("frameLive").hidden = !job;
  $("generate").textContent = job ? "Add to queue" : "Generate";
  if (!job) return;
  const p = job.progress;
  const label = p.stage === "denoise" && p.total ? `step ${p.step}/${p.total}` : (STAGES.find(([k]) => k === p.stage)?.[1] ?? "Starting") + "…";
  $("frameLive").textContent = label;
  $("progressText").textContent = label;
  if (job.result?.upsampled_prompt) showUpsampled({ prompt: job.prompt, upsampled: job.result.upsampled_prompt });
}

let lastRewrites = false;
function termLine(text, cls) {
  const term = $("term");
  const line = el("span", { class: cls || null, text: `${text}\n` });
  if (lastRewrites && term.lastChild) term.lastChild.replaceWith(line);
  else term.append(line);
  while (term.childNodes.length > 2000) term.firstChild.remove();
  term.scrollTop = term.scrollHeight;
}

function onEvent(event) {
  const { type, data } = JSON.parse(event.data);
  if (type === "queue") {
    ui.queue = data;
    const running = data.find((j) => j.status === "running" || j.status === "cancelling");
    ui.live = running && running.session === ui.session ? running : null;
    renderQueue();
    renderPipeline();
    if (running && window.showRenderLog) window.showRenderLog(running.helm_job);
  } else if (type === "progress" && ui.live && data.id === ui.live.id) {
    ui.live.progress = data.progress;
    ui.live.result = data.result;
    renderPipeline();
  } else if (type === "log" && data.session === ui.session) {
    const cls = { cmd: "m", done: "a", failed: "e", cancelled: "e" }[data.kind] || (data.line.startsWith("Saved ") ? "a" : null);
    termLine(data.line, cls);
    lastRewrites = data.rewrites;
  } else if (type === "job" && data.session === ui.session) {
    if (ui.live && ui.live.id === data.id) ui.live = null;
    renderPipeline();
    if (data.status === "failed") showError($("formError"), new Error(data.error || "the render failed"));
    loadTakes(data.status === "done");
  } else if (type === "inputs") {
    loadInputs();
  }
}

function connectEvents() {
  const source = new EventSource("/api/events");
  source.onmessage = onEvent;
  source.onerror = () => {
    $("lamp").className = "dot off";
    $("lampText").textContent = "reconnecting…";
  };
}

// ── sessions and paths ──────────────────────────────────────────────────
function renderSessions(names) {
  $("sessionSelect").replaceChildren(...names.map((n) => el("option", { value: n, text: n, selected: n === ui.session })));
}

async function openSession(name, settingsOverride) {
  const res = await api("/api/session/activate", { session: name });
  ui.session = res.name;
  applySettings(settingsOverride || res.settings || {});
  $("term").replaceChildren();
  const cfg = await api("/api/config");
  renderSessions(cfg.sessions);
  await Promise.all([loadInputs(), loadTakes()]);
}

function renderPaths(paths) {
  ui.paths = paths;
  renderModel();
}

/** The chosen model's card: its name, its directory, and whether that is there. */
function renderModel() {
  const key = $("model").value;
  const path = ui.paths[key] || {};
  $("modelName").textContent = MODELS[key].name;
  $("modelGlyph").textContent = MODELS[key].glyph;
  $("modelPath").textContent = path.path || "not set — choose it under Paths";
  $("modelBad").hidden = !!path.ok;
}

async function savePaths() {
  try {
    renderPaths(await api("/api/paths", { paths: { model: $("pModel").value, base: $("pBase").value, pup: $("pPup").value } }));
    $("pathsDialog").close();
  } catch (err) {
    showError($("pathsError"), err);
  }
}

// ── helmstudio's components: the shared gallery and the render log ──────
async function connectHelm() {
  let connect;
  try {
    ({ connect } = await import("/helm/sdk/v1/helm-runtime.js"));
    await import("/helm/sdk/v1/helm-ui.js");
  } catch {
    return; // nothing behind the proxy: no components
  }
  window.helm = connect();

  // Custom elements are written as markup into a container already in the document, so they upgrade.
  $("galleryMount").insertAdjacentHTML("beforeend", '<helm-gallery scope="self" kind="image"></helm-gallery>');
  const gallery = $("galleryMount").lastElementChild;
  let choose = null;
  gallery.addEventListener("pick", (e) => { const c = choose; choose = null; $("galleryDialog").close(); c?.(e.detail?.item); });
  $("galleryDialog").addEventListener("close", () => { gallery.removeAttribute("picker"); choose?.(null); choose = null; });
  $("galleryClose").addEventListener("click", () => $("galleryDialog").close());
  $("galleryBtn").hidden = false;
  $("galleryBtn").addEventListener("click", () => { $("galleryTitle").textContent = "Gallery"; $("galleryDialog").showModal(); });

  $("refFromGallery").hidden = false;
  $("refFromGallery").addEventListener("click", async () => {
    $("galleryTitle").textContent = "Use an image as a reference";
    gallery.setAttribute("picker", "");
    $("galleryDialog").showModal();
    const item = await new Promise((resolve) => { choose = resolve; });
    if (!item) return;
    // Any studio's image will do: fetch its bytes through the proxy and add them like an upload.
    const res = await fetch(`/helm/api/v1/assets/${encodeURIComponent(item.asset_id)}`);
    if (!res.ok) return termLine(`[studio] the gallery image could not be read (${res.status})`, "e");
    const blob = await res.blob();
    const ext = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" }[blob.type];
    if (!ext) return termLine(`[studio] ${blob.type || "that file"} is not an image FLUX 3 Image Klein reads`, "e");
    uploadFiles([new File([blob], `${(item.title || "reference").replace(/[^\w.-]+/g, "-")}.${ext}`, { type: blob.type })]);
  });

  $("renderLog").insertAdjacentHTML("beforeend", "<helm-terminal follow></helm-terminal>");
  const term = $("renderLog").lastElementChild;
  term.client = window.helm;
  document.querySelector('#termTabs [data-tab="render"]').hidden = false;
  window.showRenderLog = (id) => { if (id && term.getAttribute("job") !== id) term.setAttribute("job", id); };
}

// ── wiring ──────────────────────────────────────────────────────────────
function wire() {
  $("model").addEventListener("input", () => { $("steps").value = MODELS[$("model").value].steps; }); // before update()
  for (const id of FIELDS) $(id).addEventListener("input", () => { update(); saveSoon(); });
  $("prompt").addEventListener("input", () => { mention.at = 0; renderMentions(); });
  $("prompt").addEventListener("keydown", onMentionKey);
  $("prompt").addEventListener("click", renderMentions);
  $("prompt").addEventListener("blur", () => { $("mentions").hidden = true; });
  $("dice").addEventListener("click", () => { $("seed").value = Math.floor(Math.random() * 2 ** 31); update(); saveSoon(); });
  $("generate").addEventListener("click", async () => {
    showError($("formError"), null);
    try {
      const s = settings();
      // Boxes need the upsampler; with it off their text just waits in the box.
      await api("/api/render", { session: ui.session, ...s, boxes: s.upsample ? s.boxes.trim() : "" });
    } catch (err) {
      showError($("formError"), err);
    }
  });
  $("cancel").addEventListener("click", () => ui.live && api("/api/cancel", { id: ui.live.id }));

  $("fileInput").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("drag"); uploadFiles([...e.dataTransfer.files]); });

  const current = () => ui.takes.find((t) => t.id === ui.selected);
  $("useTake").addEventListener("click", () => current() && useTake(current()));
  $("reuseTake").addEventListener("click", () => current() && reuseTake(current()));
  $("expandUpsampled").addEventListener("click", () => {
    const open = $("upsampledBox").classList.toggle("open");
    $("expandUpsampled").textContent = open ? "collapse" : "expand";
    $("expandUpsampled").setAttribute("aria-expanded", String(open));
  });
  $("copyUpsampled").addEventListener("click", () => {
    $("prompt").value = $("upsampledPrompt").textContent;
    $("upsample").checked = false; // already upsampled
    update();
    saveSoon();
  });

  $("sessionSelect").addEventListener("change", (e) => openSession(e.target.value));
  $("sessionNew").addEventListener("click", async () => {
    const name = await ask({ title: "New session", value: `session-${$("sessionSelect").options.length + 1}`, ok: "Create" });
    if (name) openSession(name, {});
  });
  $("sessionDup").addEventListener("click", async () => {
    const name = await ask({ title: "Duplicate session", message: "Settings and references are copied; takes stay with the original.", value: `${ui.session}-copy`, ok: "Duplicate" });
    if (!name) return;
    try {
      const res = await api("/api/session/duplicate", { session: ui.session, new_name: name });
      openSession(res.name);
    } catch (err) {
      termLine(`[studio] ${err.message}`, "e");
    }
  });
  $("sessionDel").addEventListener("click", async () => {
    const yes = await ask({ title: `Delete ${ui.session}?`, message: "Its settings and references go. Its takes stay in helmstudio's gallery.", ok: "Delete" });
    if (!yes) return;
    const res = await api("/api/session/delete", { session: ui.session });
    openSession(res.name);
  });

  $("pathsBtn").addEventListener("click", () => {
    $("pModel").value = ui.paths.model?.path || "";
    $("pBase").value = ui.paths.base?.path || "";
    $("pPup").value = ui.paths.pup?.path || "";
    showError($("pathsError"), null);
    $("pathsDialog").showModal();
  });
  $("pathsCancel").addEventListener("click", () => $("pathsDialog").close());
  $("pathsSave").addEventListener("click", savePaths);

  $("termTabs").addEventListener("click", (e) => {
    const tab = e.target.closest("button[data-tab]");
    if (!tab) return;
    for (const b of $("termTabs").children) b.setAttribute("aria-selected", String(b === tab));
    $("term").hidden = tab.dataset.tab !== "output";
    $("renderLog").hidden = tab.dataset.tab !== "render";
    $("termClear").hidden = tab.dataset.tab !== "output";
  });
  $("termClear").addEventListener("click", () => $("term").replaceChildren());
}

async function start() {
  wire();
  const cfg = await api("/api/config");
  renderPaths(cfg.paths);
  ui.session = cfg.active;
  renderSessions(cfg.sessions.includes(cfg.active) ? cfg.sessions : [cfg.active, ...cfg.sessions]);
  applySettings(cfg.settings);
  await Promise.all([loadInputs(), loadTakes()]);
  connectEvents();
  connectHelm().catch((err) => console.warn("helmstudio: not connected:", err));
}

start().catch((err) => {
  $("lamp").className = "dot off";
  $("lampText").textContent = err.message;
});
