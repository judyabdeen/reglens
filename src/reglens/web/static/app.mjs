// The RegLens page: state, the question box, the scope and settings drawer, and the passage reader.
// Answers are drawn by render.mjs; the logic that reads them is in lib.mjs (tested with Node); the server is api.mjs.

import { getJson, postJson } from "./api.mjs";
import { announce, clear, externalLink, h, icon, reducedMotion, replaceChildren, withIcon, wireDialog } from "./dom.mjs";
import {
  SEARCH_METHODS,
  STATUS,
  activeFilters,
  cleanFilters,
  formatCount,
  formatSeconds,
  joinList,
  mergeSettings,
  passageMarks,
  plural,
  scopeSummary,
  statusInfo,
  stripCitations,
  toMarkdown,
} from "./lib.mjs";
import { WIDE, disposeTurn, layoutAll, layoutNotes, markup, renderTurn, stamp } from "./render.mjs";

const MODES = [
  { value: "rag", label: "Answer", needsModel: true, hint: "Answers from the documents and cites the passages it used.", title: "Answer from the documents, with citations" },
  { value: "compare", label: "Compare", needsModel: true, hint: "Answers twice, with the documents and from the model alone, and shows where the two differ.", title: "Answer with and without the documents, side by side" },
  { value: "passages", label: "Passages", needsModel: false, hint: "Shows only the matching passages. The language model is not asked.", title: "Show the matching passages without asking the model" },
  { value: "llm_only", label: "Model alone", needsModel: true, hint: "Asks the model with no documents: what a plain chatbot would say.", title: "Ask the model without any documents" },
];

const EXAMPLES = [
  "When do the high-risk obligations of the EU AI Act start to apply?",
  "How quickly must a personal data breach be reported to the regulator under the GDPR?",
  "What are the fines for breaking the AI Act's ban on prohibited practices?",
  "Does the CCPA give people a right to opt out of automated decision-making?",
  "How does the HIPAA Breach Notification Rule differ from the GDPR on breach notification?",
];

const STORAGE = { settings: "reglens.settings", filters: "reglens.filters", mode: "reglens.mode", theme: "reglens.theme" };
const POLL_MS = 4000;

const $ = (id) => document.getElementById(id);
const el = {
  thread: $("thread"),
  notices: $("notices"),
  barStatus: $("bar-status"),
  newChat: $("new-chat"),
  saveChat: $("save-chat"),
  theme: $("theme"),
  aboutOpen: $("about-open"),
  dock: $("dock"),
  question: $("question"),
  ask: $("ask"),
  hint: $("hint"),
  modes: $("modes"),
  scopeOpen: $("scope-open"),
  scopeText: $("scope-text"),
  scope: $("scope"),
  scopeBody: $("scope-body"),
  reset: $("reset-settings"),
  reader: $("reader"),
  readerTitle: $("reader-title"),
  readerBody: $("reader-body"),
  readerFoot: $("reader-foot"),
  about: $("about"),
  aboutBody: $("about-body"),
};

const state = {
  status: null,
  documents: [],
  defaults: {},
  settings: {},
  filters: { jurisdiction: [], status: [], doc_type: [], doc_id: [] },
  mode: "rag",
  exchanges: [],
  nextId: 1,
  running: null,
};

// ------------------------------------------------------------------------------------------------ storage

function load(key) {
  try {
    return JSON.parse(window.localStorage.getItem(key) ?? "null");
  } catch {
    return null;
  }
}

function save(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage is blocked (private window): the choices last until the page is closed
  }
}

// ------------------------------------------------------------------------------------------------ the page frame

const root = document.documentElement;
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
// The question box is narrow on a phone, where a long placeholder would be cut off; this matches the CSS breakpoint.
const narrow = window.matchMedia("(max-width: 40rem)");
const currentTheme = () => root.dataset.theme ?? (systemDark.matches ? "dark" : "light");

function paintThemeButton() {
  const dark = currentTheme() === "dark";
  replaceChildren(el.theme, icon(dark ? "sun" : "moon"));
  el.theme.setAttribute("aria-label", dark ? "Switch to the light theme" : "Switch to the dark theme");
}

function setLabel(button, text) {
  const label = button.querySelector(".btn-text");
  if (label) label.textContent = text;
}

function paintAskButton() {
  const running = Boolean(state.running);
  replaceChildren(el.ask, icon(running ? "stop" : "arrowUp"), h("span", { class: "btn-text", text: running ? "Stop" : "Ask" }));
  el.ask.title = running ? "Stop waiting for the answer" : "Ask (Enter)";
  el.ask.classList.toggle("is-stop", running);
}

function modelAvailable() {
  return Boolean(state.status?.llm?.available);
}

function paintBar() {
  const status = state.status;
  let text = "Connecting to the server";
  if (status && !status.ready) text = "The library has not been built yet";
  else if (status) {
    text = `${status.index.documents} documents, ${formatCount(status.index.passages)} passages`;
    if (status.llm.available) text += `, ${status.llm.model}`;
  }
  el.barStatus.textContent = text;
  const answered = state.exchanges.some((exchange) => exchange.answer || exchange.baseline);
  el.newChat.disabled = !state.exchanges.length;
  el.saveChat.disabled = !answered;
}

function scopeInfo() {
  return scopeSummary(state.filters, state.documents);
}

function paintScopeButton() {
  const ready = Boolean(state.status?.ready);
  const info = scopeInfo();
  replaceChildren(el.scopeText, icon("sliders"), h("span", { text: ready ? info.text : "Scope and settings" }));
  el.scopeOpen.setAttribute("aria-label", ready ? `Change scope and search settings. Searching: ${info.text}` : "Scope and search settings");
  el.scopeOpen.dataset.active = ready && info.active ? "true" : "false";
}

function updateHint() {
  const mode = MODES.find((entry) => entry.value === state.mode) ?? MODES[0];
  const length = el.question.value.length;
  const max = Number(el.question.maxLength);
  el.hint.textContent = length > max * 0.8 ? `${formatCount(length)} of ${formatCount(max)} characters. ${mode.hint}` : mode.hint;
}

function autosize() {
  el.question.style.height = "auto";
  el.question.style.height = `${Math.min(el.question.scrollHeight + 2, 190)}px`;
}

function updateDock() {
  const ready = Boolean(state.status?.ready);
  const running = Boolean(state.running);
  el.question.disabled = !ready;
  el.scopeOpen.disabled = !ready;
  el.ask.disabled = !ready || (!running && !el.question.value.trim());
  for (const input of el.modes.querySelectorAll("input")) {
    const mode = MODES.find((entry) => entry.value === input.value);
    input.disabled = !ready || (mode.needsModel && !modelAvailable());
  }
  if (!ready) el.question.placeholder = "Available when the library is ready";
  else el.question.placeholder = narrow.matches ? "Ask a question" : "Ask about the AI Act, the GDPR, HIPAA, the CCPA";
  paintAskButton();
  paintScopeButton();
  updateHint();
}

function buildModes() {
  replaceChildren(
    el.modes,
    MODES.map((mode) =>
      h(
        "label",
        { class: "mode", title: mode.title },
        h("input", { type: "radio", name: "mode", value: mode.value, class: "sr-only", checked: state.mode === mode.value }),
        h("span", { class: "mode-label", text: mode.label }),
      ),
    ),
  );
}

function chooseMode(value) {
  state.mode = value;
  save(STORAGE.mode, value);
  for (const input of el.modes.querySelectorAll("input")) input.checked = input.value === value;
  updateHint();
}

function renderNotices() {
  clear(el.notices);
  const status = state.status;
  if (!status?.ready) return;
  const items = [];
  if (!status.llm.available) {
    items.push(`No language model is connected (${status.llm.problem ?? "no API key"}). RegLens shows matching passages only. Add your key to the .env file and restart the server to get written answers.`);
  }
  for (const warning of status.warnings ?? []) items.push(warning);
  for (const text of items) {
    const notice = h(
      "div",
      { class: "notice" },
      icon("info", "icon icon-inline"),
      h("p", { text }),
      h("button", { type: "button", class: "icon-btn small", "aria-label": "Dismiss this notice", on: { click: () => notice.remove() } }, icon("close")),
    );
    el.notices.append(notice);
  }
}

// ------------------------------------------------------------------------------------------------ welcome and setup panels

function libraryCounts() {
  const docs = state.documents;
  const kinds = { law: 0, guidance: 0, draft: 0 };
  for (const doc of docs) {
    const kind = statusInfo(doc.status).kind;
    if (kind in kinds) kinds[kind] += 1;
  }
  const places = new Map();
  for (const doc of docs) places.set(doc.jurisdiction, (places.get(doc.jurisdiction) ?? 0) + 1);
  const ranked = [...places.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return { kinds, ranked };
}

function jurisdictionName(name) {
  return name === "International" ? "international" : name;
}

function welcome() {
  const { kinds, ranked } = libraryCounts();
  const places = joinList(ranked.map(([name]) => jurisdictionName(name)));
  const rows = [
    ["in_force", kinds.law, plural(kinds.law, "binding text", "binding texts")],
    ["final", kinds.guidance, plural(kinds.guidance, "final guideline, opinion or framework", "final guidelines, opinions and frameworks")],
    ["draft", kinds.draft, plural(kinds.draft, "draft, consultation or proposal", "drafts, consultations and proposals")],
  ];
  return h(
    "section",
    { class: "welcome", "aria-labelledby": "welcome-title" },
    h(
      "div",
      { class: "main" },
      h("h1", { class: "welcome-title", id: "welcome-title", text: "Ask about AI and data-protection law" }),
      h("p", {
        class: "welcome-lead",
        text: `RegLens searches ${state.documents.length} public documents from ${places} sources. Each statement in an answer points to the passage behind it, shown beside the answer with its legal status.`,
      }),
      h("h2", { class: "welcome-sub", text: "Try a question" }),
      h(
        "ul",
        { class: "examples" },
        EXAMPLES.map((question) =>
          h("li", {}, h("button", { type: "button", class: "example", on: { click: () => askFromExample(question) } }, h("span", { text: question }), icon("chevron", "icon icon-inline"))),
        ),
      ),
      h("p", { class: "disclaimer", text: "Research support, not legal advice." }),
    ),
    h(
      "aside",
      { class: "welcome-side", "aria-label": "What is in the library" },
      h("h2", { class: "side-title", text: "In the library" }),
      h(
        "ul",
        { class: "library" },
        rows
          .filter(([, count]) => count)
          .map(([status, count, words]) => h("li", {}, stamp(status), h("span", { class: "library-count", text: `${count} ${words}` }))),
      ),
      h("button", { type: "button", class: "btn small", on: { click: (event) => openScope(event.currentTarget) } }, icon("sliders"), h("span", { class: "btn-text", text: "Browse and choose documents" })),
    ),
  );
}

function setup({ title, message, fix = "", note = "" }) {
  return h(
    "section",
    { class: "setup", "aria-labelledby": "setup-title" },
    h("h1", { class: "welcome-title", id: "setup-title", text: title }),
    h("p", { class: "welcome-lead", text: message }),
    fix
      ? h(
          "div",
          { class: "setup-step" },
          h("p", { text: "Run this in the project folder, then come back to this page:" }),
          h("div", { class: "command" }, h("code", { text: fix }), h("button", { type: "button", class: "btn quiet small", on: { click: (event) => copyText(fix, event.currentTarget) } }, icon("copy"), h("span", { class: "btn-text", text: "Copy" }))),
        )
      : null,
    note ? h("p", { class: "setup-note", text: note }) : null,
    h("button", { type: "button", class: "btn small", on: { click: () => refresh() } }, "Check again"),
  );
}

function showThread() {
  clear(el.thread);
  if (!state.status) {
    el.thread.append(h("p", { class: "loading", text: "Connecting to the server" }));
  } else if (!state.status.ready) {
    el.thread.append(
      setup({
        title: "The library has not been built yet",
        message: state.status.problem,
        fix: state.status.fix,
        note: "This page checks again every few seconds, so it will open as soon as the library is ready.",
      }),
    );
  } else if (!state.exchanges.length) {
    el.thread.append(welcome());
  } else {
    for (const exchange of state.exchanges) mountTurn(exchange);
  }
}

// ------------------------------------------------------------------------------------------------ loading the server's state

let pollTimer = null;

function poll() {
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(refresh, POLL_MS);
}

async function refresh() {
  window.clearTimeout(pollTimer);
  let status;
  try {
    status = await getJson("/api/status");
  } catch (error) {
    if (state.status?.ready) return; // a hiccup while a ready page is open: the next question reports it
    state.status = null;
    paintBar();
    replaceChildren(el.thread, setup({ title: "RegLens could not be reached", message: error.message, note: "This page checks again every few seconds." }));
    updateDock();
    poll();
    return;
  }
  const wasReady = Boolean(state.status?.ready);
  state.status = status;
  state.defaults = status.defaults;
  state.settings = mergeSettings(status.defaults, load(STORAGE.settings), status.limits);
  if (!status.ready) {
    paintBar();
    showThread();
    updateDock();
    poll();
    return;
  }
  if (!wasReady) {
    try {
      state.documents = (await getJson("/api/documents")).documents;
    } catch (error) {
      state.status = null;
      replaceChildren(el.thread, setup({ title: "The documents could not be loaded", message: error.message }));
      updateDock();
      poll();
      return;
    }
    state.filters = cleanFilters(load(STORAGE.filters), state.documents);
    const saved = load(STORAGE.mode);
    state.mode = MODES.some((mode) => mode.value === saved) ? saved : "rag";
    if (!status.llm.available) state.mode = "passages";
    buildModes();
    renderNotices();
    showThread();
  }
  paintBar();
  updateDock();
}

// ------------------------------------------------------------------------------------------------ asking

const failureText = {
  abort: "You stopped this question before the answer arrived. The server may still finish it in the background.",
};

function describeFailure(error) {
  if (error?.name === "AbortError") return { kind: "cancelled", message: failureText.abort };
  if (error?.kind) return { kind: error.kind, message: error.message, fix: error.fix };
  return { kind: "error", message: "Something unexpected went wrong in the page. Reload it and ask again." };
}

function historyFor(exchange) {
  const index = state.exchanges.indexOf(exchange);
  return state.exchanges
    .slice(0, Math.max(index, 0))
    .filter((past) => past.answer?.mode === "rag" && !past.answer.abstained && !past.answer.error_kind && past.answer.text)
    .slice(-6)
    .map((past) => ({ question: past.question, answer: stripCitations(past.answer.text).slice(0, 1500) }));
}

function exportMeta() {
  const index = state.status?.index;
  return {
    exportedAt: new Date().toISOString().slice(0, 10),
    passages: index?.passages,
    documents: index?.documents,
    model: state.status?.llm?.available ? state.status.llm.model : "",
  };
}

function summaryOf(exchange) {
  if (exchange.error) return exchange.error.kind === "cancelled" ? "Stopped." : `The question failed: ${exchange.error.message}`;
  const answer = exchange.answer;
  if (answer?.abstained) return "The documents do not answer this question.";
  if (answer?.mode === "extractive") return `${answer.sources.length} matching ${plural(answer.sources.length, "passage", "passages")} found.`;
  if (answer) return `Answer ready, citing ${answer.cited.length} ${plural(answer.cited.length, "source", "sources")}.`;
  return "The answer from the model alone is ready.";
}

async function run(exchange) {
  if (state.running) return;
  const controller = new AbortController();
  const started = performance.now();
  exchange.pending = true;
  exchange.error = null;
  exchange.answer = null;
  exchange.baseline = null;
  state.running = { exchange, controller };
  const timer = window.setInterval(() => {
    if (exchange.elapsedEl) exchange.elapsedEl.textContent = formatSeconds((performance.now() - started) / 1000);
  }, 500);
  mountTurn(exchange);
  updateDock();
  paintBar();
  announce("Working on your question.");
  try {
    const payload = { question: exchange.question, mode: exchange.mode, history: historyFor(exchange), options: exchange.settings, filters: exchange.filters };
    const data = await postJson("/api/ask", payload, { signal: controller.signal });
    exchange.answer = data.answer;
    exchange.baseline = data.baseline;
    exchange.elapsed = data.elapsed;
  } catch (error) {
    exchange.error = describeFailure(error);
  } finally {
    window.clearInterval(timer);
    exchange.pending = false;
    exchange.elapsedEl = null;
    if (state.running?.exchange === exchange) state.running = null;
  }
  // "New conversation" while a question was running removed this exchange: its stopped turn must not come back.
  if (state.exchanges.includes(exchange)) {
    mountTurn(exchange);
    announce(summaryOf(exchange));
  }
  updateDock();
  paintBar();
}

function startAsking(question) {
  const text = question.trim();
  if (!text || state.running || !state.status?.ready) return;
  const info = scopeInfo();
  const exchange = {
    id: state.nextId++,
    question: text,
    mode: state.mode,
    scope: info.text,
    scopeText: `Searched ${info.active ? info.text : info.text.charAt(0).toLowerCase() + info.text.slice(1)}`,
    settings: { ...state.settings },
    filters: activeFilters(state.filters),
    pending: true,
    answer: null,
    baseline: null,
    error: null,
    elapsed: null,
    el: null,
  };
  if (!state.exchanges.length) clear(el.thread);
  state.exchanges.push(exchange);
  el.question.value = "";
  autosize();
  run(exchange);
  window.requestAnimationFrame(() => exchange.el?.scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" }));
}

function askFromExample(question) {
  el.question.value = question;
  autosize();
  startAsking(question);
}

function mountTurn(exchange) {
  const turn = renderTurn(exchange, ctx);
  const old = exchange.el;
  exchange.el = turn;
  if (old?.isConnected) {
    disposeTurn(old);
    old.replaceWith(turn);
  } else {
    el.thread.append(turn);
  }
  layoutNotes(turn);
}

async function copyText(text, button) {
  const label = button.querySelector(".btn-text");
  const original = label?.textContent ?? "";
  let message = "Copied to the clipboard.";
  let shown = "Copied";
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    message = "Copying was blocked by the browser. Select the text and copy it by hand.";
    shown = "Copy blocked";
  }
  if (label) {
    label.textContent = shown;
    window.setTimeout(() => {
      label.textContent = original;
    }, 1800);
  }
  announce(message);
}

function saveMarkdown() {
  const done = state.exchanges.filter((exchange) => exchange.answer || exchange.baseline);
  if (!done.length) return;
  const blob = new Blob([toMarkdown(done, exportMeta())], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = h("a", { href: url, download: `reglens-${new Date().toISOString().slice(0, 10)}.md` });
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
  announce("The conversation was saved as a Markdown file.");
}

let clearTimer = null;

function resetClearButton() {
  window.clearTimeout(clearTimer);
  clearTimer = null;
  setLabel(el.newChat, "New conversation");
  el.newChat.classList.remove("is-armed");
}

function clearConversation() {
  const running = state.running;
  state.running = null;
  running?.controller.abort();
  for (const exchange of state.exchanges) if (exchange.el) disposeTurn(exchange.el);
  state.exchanges = [];
  showThread();
  paintBar();
  updateDock();
  el.question.focus();
  announce("Started a new conversation.");
}

// ------------------------------------------------------------------------------------------------ the passage reader

const reader = wireDialog(el.reader);
let readerToken = 0;

function openReader(source, claim, opener) {
  reader.open(opener);
  loadPassage(source.chunk_id, claim, "");
}

async function loadPassage(chunkId, claim, focus) {
  const token = ++readerToken;
  replaceChildren(el.readerBody, h("p", { class: "loading", text: "Loading the passage" }));
  clear(el.readerFoot);
  let data;
  try {
    data = await getJson(`/api/passages/${encodeURIComponent(chunkId)}`);
  } catch (error) {
    if (token === readerToken) replaceChildren(el.readerBody, h("p", { class: "problem-text", text: error.message }));
    return;
  }
  if (token !== readerToken) return;
  const passage = data.passage;
  const info = statusInfo(passage.status, passage.status_text);
  const marks = claim ? passageMarks(passage.text, claim) : [];
  el.readerTitle.textContent = passage.title;
  replaceChildren(
    el.readerBody,
    h(
      "div",
      { class: "reader-meta" },
      stamp(passage.status, passage.status_text),
      h("span", { text: [passage.jurisdiction, passage.kind === "notes" ? "Footnotes" : "", passage.section, passage.pages].filter(Boolean).join(", ") }),
    ),
    info.kind === "draft" ? h("p", { class: "reader-warning" }, icon("warn", "icon icon-inline"), "This text has not been adopted. It shows where the rules may go, not what applies today.") : null,
    h("div", { class: "reader-text" }, markup(passage.text, marks)),
    marks.length ? h("p", { class: "note-caption", text: "Highlighted: the sentence closest to the statement you came from, and the figures they share." }) : null,
  );
  el.readerBody.scrollTop = 0;
  const step = (record, label, iconName, id) =>
    h(
      "button",
      {
        type: "button",
        id,
        class: "btn quiet small",
        disabled: !record,
        title: record ? `${record.section || "Neighbouring passage"}` : "There is no such passage",
        on: { click: () => loadPassage(record.chunk_id, claim, id) },
      },
      iconName === "back" ? icon("back") : null,
      h("span", { class: "btn-text", text: label }),
      iconName === "forward" ? icon("forward") : null,
    );
  replaceChildren(
    el.readerFoot,
    h("div", { class: "reader-steps" }, step(data.before, "Previous passage", "back", "reader-prev"), step(data.after, "Next passage", "forward", "reader-next")),
    externalLink(passage.url, "Open the source document"),
  );
  if (focus) {
    const target = $(focus);
    if (target && !target.disabled) target.focus();
  }
}

// ------------------------------------------------------------------------------------------------ scope and settings drawer

const scopeDialog = wireDialog(el.scope);

function openScope(opener) {
  buildScope();
  scopeDialog.open(opener ?? el.scopeOpen);
}

function saveFilters() {
  save(STORAGE.filters, state.filters);
  paintScopeButton();
}

function setFilter(key, value, on) {
  const current = state.filters[key];
  state.filters[key] = on ? [...new Set([...current, value])] : current.filter((entry) => entry !== value);
  saveFilters();
}

function setSetting(key, value) {
  state.settings[key] = value;
  save(STORAGE.settings, state.settings);
}

function option({ type = "checkbox", name, checked, label, detail, count, lead, onChange }) {
  const input = h("input", { type, name, checked, on: { change: () => onChange(input.checked) } });
  return h(
    "label",
    { class: "opt" },
    input,
    h("span", { class: "opt-body" }, h("span", { class: "opt-label" }, lead ?? null, label ? h("span", { text: label }) : null), detail ? h("span", { class: "opt-detail" }, detail) : null),
    count === undefined ? null : h("span", { class: "opt-count", text: String(count) }),
  );
}

function docTypeLabel(value) {
  const text = String(value).replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function countBy(key) {
  const counts = new Map();
  for (const doc of state.documents) counts.set(doc[key], (counts.get(doc[key]) ?? 0) + 1);
  return counts;
}

function buildScope() {
  const summary = h("p", { class: "scope-summary" });
  const warning = h("p", { class: "scope-warning", hidden: true, text: "No document matches these choices, so every question would come back empty. Clear one of the choices." });
  const repaint = () => {
    const info = scopeInfo();
    summary.textContent = info.active ? `Searching ${info.text}` : `Searching all ${info.count} documents`;
    warning.hidden = info.count > 0;
  };
  const group = (key, legend, entries) =>
    h(
      "fieldset",
      { class: "group" },
      h("legend", { text: legend }),
      entries.map((entry) =>
        option({ checked: state.filters[key].includes(entry.value), label: entry.label, count: entry.count, lead: entry.lead, onChange: (on) => { setFilter(key, entry.value, on); repaint(); } }),
      ),
    );

  const jurisdictions = [...countBy("jurisdiction").entries()].sort((a, b) => b[1] - a[1]).map(([value, count]) => ({ value, label: value === "International" ? "International bodies" : value, count }));
  const statusCounts = countBy("status");
  const statuses = Object.keys(STATUS).filter((value) => statusCounts.has(value)).map((value) => ({ value, label: null, count: statusCounts.get(value), lead: stamp(value) }));
  const typeCounts = countBy("doc_type");
  const types = [...typeCounts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([value, count]) => ({ value, label: docTypeLabel(value), count }));

  // individual documents, grouped by jurisdiction, with a search box
  const finder = h("input", { type: "search", class: "field", placeholder: "Find a document by name", "aria-label": "Find a document by name" });
  const rows = [];
  const docGroups = [];
  for (const [name] of [...countBy("jurisdiction").entries()].sort((a, b) => b[1] - a[1])) {
    const members = state.documents.filter((doc) => doc.jurisdiction === name);
    const items = members.map((doc) => {
      const row = option({
        checked: state.filters.doc_id.includes(doc.doc_id),
        label: doc.title,
        detail: stamp(doc.status),
        onChange: (on) => { setFilter("doc_id", doc.doc_id, on); repaint(); },
      });
      rows.push({ row, text: `${doc.title} ${doc.publisher ?? ""}`.toLowerCase() });
      return row;
    });
    docGroups.push({ heading: h("h4", { class: "doc-group", text: name === "International" ? "International bodies" : name }), items });
  }
  finder.addEventListener("input", () => {
    const query = finder.value.trim().toLowerCase();
    for (const entry of rows) entry.row.hidden = Boolean(query) && !entry.text.includes(query);
    for (const part of docGroups) part.heading.hidden = part.items.every((item) => item.hidden);
  });
  const chosen = state.filters.doc_id.length;

  const radio = (key, legend, entries) =>
    h(
      "fieldset",
      { class: "group" },
      h("legend", { text: legend }),
      entries.map(([value, label, detail]) => option({ type: "radio", name: key, checked: state.settings[key] === value, label, detail, onChange: (on) => on && setSetting(key, value) })),
    );
  const toggle = (key, label, detail) => option({ checked: Boolean(state.settings[key]), label, detail, onChange: (on) => setSetting(key, on) });
  const slider = (key, label, min, max, words) => {
    const output = h("output", { class: "slider-value" });
    const input = h("input", { type: "range", min, max, step: 1, value: String(state.settings[key]), "aria-label": label });
    const paint = () => { output.textContent = words(Number(input.value)); };
    input.addEventListener("input", () => { setSetting(key, Number(input.value)); paint(); });
    paint();
    return h("div", { class: "slider" }, h("span", { class: "opt-label", text: label }), h("div", { class: "slider-row" }, input, output));
  };

  replaceChildren(
    el.scopeBody,
    h(
      "section",
      { class: "pane" },
      h("h3", { text: "Which documents to search" }),
      summary,
      warning,
      group("jurisdiction", "Where it comes from", jurisdictions),
      group("status", "Legal status", statuses),
      group("doc_type", "Kind of document", types),
      h(
        "details",
        { class: "doc-picker", open: chosen > 0 },
        h("summary", {}, `Choose individual documents (${state.documents.length})`),
        finder,
        docGroups.map((part) => [part.heading, h("div", { class: "doc-rows" }, part.items)]),
      ),
    ),
    h(
      "section",
      { class: "pane" },
      h("h3", { text: "How RegLens searches" }),
      radio("search", "Search method", SEARCH_METHODS.map((method) => [method.value, method.name, method.detail])),
      slider("top_k", "Passages given to the model", 1, 12, (n) => `${n} ${plural(n, "passage", "passages")}`),
      toggle("rerank", "Re-rank the best matches with a second model", "Reads each candidate together with the question and reorders them. Slower, and the first use downloads a small model."),
      toggle("glossary", "Add related legal terms to the search", "A question about a leak also searches for “personal data breach”."),
      toggle("condense", "Rewrite follow-up questions so they stand alone", "Turns “and in the US?” into a full question before searching."),
      slider("multi_query", "Extra searches with rephrased questions", 0, 3, (n) => (n === 0 ? "None" : `${n} extra ${plural(n, "search", "searches")}`)),
      toggle("hyde", "Search with a hypothetical answer first", "The model drafts a short answer and the draft is used to find passages. Uses one more model call."),
      toggle("include_notes", "Include footnotes", "Footnotes are searched as separate passages."),
    ),
  );
  repaint();
}

function resetSettings() {
  state.settings = { ...state.defaults };
  state.filters = { jurisdiction: [], status: [], doc_type: [], doc_id: [] };
  save(STORAGE.settings, state.settings);
  saveFilters();
  buildScope();
}

// ------------------------------------------------------------------------------------------------ about

const aboutDialog = wireDialog(el.about);

function buildAbout() {
  const legend = (status, text) => h("li", {}, stamp(status), h("span", { text }));
  const index = state.status?.index;
  const facts = index
    ? [
        ["Documents", String(index.documents)],
        ["Passages", formatCount(index.passages)],
        ["Embedding model", index.embedding_model],
        ["Vector store", `${index.vector_store}, ${index.space} similarity`],
        ["Re-ranking model", index.reranker],
        ["Language model", state.status.llm.available ? `${state.status.llm.provider}, ${state.status.llm.model}` : "none connected"],
        ["Index built", index.built_at],
      ]
    : [];
  const step = (text) => h("li", { text });
  replaceChildren(
    el.aboutBody,
    h(
      "section",
      { class: "pane" },
      h("h3", { text: "What RegLens is for" }),
      h("p", { text: "RegLens helps legal, compliance and product teams find what EU and US rules on AI and data protection say, and where the text is. It answers only from the documents in its library, and every statement points to the passage behind it." }),
      h("p", { text: "It is research support, not legal advice." }),
    ),
    h(
      "section",
      { class: "pane" },
      h("h3", { text: "How an answer is made" }),
      h(
        "ol",
        { class: "steps" },
        step("A follow-up question is rewritten so that it stands on its own."),
        step("The library is searched twice, by meaning and by keywords, and the two rankings are merged."),
        step("The best passages go to the language model together with their legal status and date."),
        step("The model writes an answer and must cite a passage for each statement."),
        step("RegLens checks that each citation exists and that each number appears in the passages, then shows the cited passages beside the answer."),
      ),
    ),
    h(
      "section",
      { class: "pane" },
      h("h3", { text: "How to read the marks" }),
      h("ul", { class: "legend" }, legend("in_force", "Binding law that applies."), legend("final", "Official guidance, an opinion or a framework. Authoritative, but not binding law."), legend("draft", "A draft, consultation or proposal. It shows where the rules may go, not what applies today.")),
      h("p", { text: "A citation such as S2 takes the colour and outline of its source. Selecting it shows that source beside the answer. In the source, the sentence closest to the statement has a tinted background, and dates, amounts and numbers that the statement shares with the source are marked in yellow." }),
    ),
    h(
      "section",
      { class: "pane" },
      h("h3", { text: "What the checks do not do" }),
      h("p", { text: "The automatic checks catch a citation that does not exist and a number that appears in no passage. They cannot tell whether a passage truly supports a statement. Reading the highlighted sentence is the check that matters." }),
      h("p", { text: "Compare mode shows what the same model says without the documents, so that differences in dates and amounts are visible." }),
    ),
    facts.length
      ? h("section", { class: "pane" }, h("h3", { text: "This installation" }), h("dl", { class: "facts" }, facts.flatMap(([label, value]) => [h("dt", { text: label }), h("dd", { text: value })])))
      : null,
    h(
      "section",
      { class: "pane" },
      h("h3", { text: "Keyboard" }),
      h("ul", { class: "plain-list" }, h("li", { text: "Enter asks. Shift and Enter start a new line." }), h("li", { text: "The slash key moves to the question box." }), h("li", { text: "Escape closes a panel." })),
    ),
  );
}

// ------------------------------------------------------------------------------------------------ wiring

const ctx = { openReader, retry: (exchange) => run(exchange), copy: copyText, exportMeta };

function isTyping(target) {
  return target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
}

function wire() {
  withIcon(el.newChat, "plus", "New conversation");
  withIcon(el.saveChat, "download", "Save as Markdown");
  el.aboutOpen.append(icon("info"));
  for (const button of document.querySelectorAll(".icon-btn[data-close]")) button.append(icon("close"));
  paintThemeButton();
  paintAskButton();
  buildAbout();

  el.theme.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      window.localStorage.setItem(STORAGE.theme, next);
    } catch {
      // storage is blocked: the choice lasts until the page is closed
    }
    paintThemeButton();
  });
  systemDark.addEventListener("change", paintThemeButton);

  el.aboutOpen.addEventListener("click", (event) => {
    buildAbout();
    aboutDialog.open(event.currentTarget);
  });
  el.scopeOpen.addEventListener("click", (event) => openScope(event.currentTarget));
  el.reset.addEventListener("click", resetSettings);
  el.saveChat.addEventListener("click", saveMarkdown);
  el.newChat.addEventListener("click", () => {
    if (clearTimer === null) {
      setLabel(el.newChat, "Click again to clear");
      el.newChat.classList.add("is-armed");
      clearTimer = window.setTimeout(resetClearButton, 3000);
      return;
    }
    resetClearButton();
    clearConversation();
  });

  el.modes.addEventListener("change", (event) => {
    if (event.target instanceof HTMLInputElement) chooseMode(event.target.value);
  });
  el.question.addEventListener("input", () => {
    autosize();
    updateDock();
  });
  el.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!state.running && el.question.value.trim()) el.dock.requestSubmit();
    }
  });
  el.dock.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.running) state.running.controller.abort();
    else startAsking(el.question.value);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey && !isTyping(event.target) && !document.querySelector("dialog[open]")) {
      event.preventDefault();
      el.question.focus();
    }
  });

  window.matchMedia(WIDE).addEventListener("change", () => layoutAll(el.thread));
  narrow.addEventListener("change", updateDock);
  document.fonts?.ready.then(() => layoutAll(el.thread));
}

wire();
paintBar();
showThread();
updateDock();
refresh();
