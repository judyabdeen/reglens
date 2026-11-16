// Builds the conversation. One <article> per question: the question, then the answer with its citations, and beside it,
// in the margin, a note for every source the answer cites. Notes show the sentence of the source that is closest to the
// statement they support, so a reader can check a claim without leaving the page.
//
// Everything is built from data with createElement and textContent (see dom.mjs): nothing the server or the model
// sends is ever parsed as markup.

import { externalLink, h, icon, reducedMotion } from "./dom.mjs";
import {
  authority,
  citationContexts,
  closestSentence,
  compareFigures,
  describeOptions,
  describeTimings,
  excerpt,
  excerptAround,
  formatSeconds,
  joinList,
  parseAnswer,
  passageMarks,
  plainText,
  plural,
  segments,
  sourcesWithFigure,
  statusInfo,
  toMarkdown,
} from "./lib.mjs";

export const WIDE = "(min-width: 68rem)";

const WORKING = {
  rag: "Searching the documents and writing the answer.",
  compare: "Writing two answers: one from the documents and one from the model alone.",
  passages: "Searching the documents.",
  llm_only: "Asking the model without any documents.",
};

const PROBLEM_TITLES = {
  not_ready: "The library is not ready",
  unreachable: "RegLens could not be reached",
  busy: "Another question is still running",
  cancelled: "Stopped",
  bad_request: "That question could not be used",
  unavailable: "A search component failed",
};

// ------------------------------------------------------------------------------------------------ small parts

// A legal status as a stamp. The shape and the border style say the same as the words, so colour is never the only cue:
// solid fill = in force, outline = final guidance, dashed = not adopted.
export function stamp(status, fallback = "") {
  const info = statusInfo(status, fallback);
  return h(
    "span",
    { class: `stamp stamp-${info.kind}` },
    h("span", { class: "stamp-mark", "aria-hidden": "true" }),
    h("span", { class: "stamp-text", text: info.label }),
  );
}

export function whereText(source) {
  return [source.kind === "notes" ? "Footnotes" : "", source.section, source.pages].filter(Boolean).join(", ");
}

// Text with marks as DOM: plain strings and <mark> elements.
export function markup(text, marks) {
  return segments(text, marks).map((part) => {
    if (!part.classes.length) return part.text;
    const figure = part.classes.includes("figure");
    return h(
      "mark",
      { class: part.classes.map((name) => `mark-${name}`).join(" "), title: figure ? "A figure that also appears in the statement" : "The sentence closest to the statement" },
      part.text,
    );
  });
}

function button(label, { iconName, className = "btn quiet small", onClick, title } = {}) {
  return h("button", { type: "button", class: className, title, on: { click: onClick } }, iconName ? icon(iconName) : null, h("span", { class: "btn-text", text: label }));
}

function lowerFirst(text) {
  return text.charAt(0).toLowerCase() + text.slice(1);
}

// ------------------------------------------------------------------------------------------------ source notes

const retargets = new WeakMap();

// `captioned`: say what the highlights mean. Only the first note of an answer does, to keep the margin short; a note
// that has no highlight always explains why.
function fillQuote(quote, caption, source, claim, captioned) {
  const text = source.text ?? "";
  const index = claim ? closestSentence(text, claim) : -1;
  const view = excerptAround(text, index, 340);
  const marks = claim ? passageMarks(text, claim, view.start, view.end - view.start) : [];
  quote.replaceChildren(view.before ? "… " : "", ...markup(view.text, marks), view.after ? " …" : "");
  if (!claim) caption.textContent = "";
  else if (index < 0) caption.textContent = "No single sentence matches closely. Read the whole passage.";
  else if (marks.some((mark) => mark.cls === "figure")) caption.textContent = "Highlighted: the sentence closest to the statement, and the figures they share.";
  else caption.textContent = "Highlighted: the sentence closest to the statement.";
  caption.hidden = !caption.textContent || (!captioned && index >= 0);
}

// One source: where it comes from, its legal status, and the passage itself. `claim` is the statement the note supports.
export function renderNote(source, { claim = "", inline = false, prefix, ctx, captioned = false }) {
  const info = statusInfo(source.status, source.status_text);
  let current = claim;
  const quote = h("blockquote", { class: "note-quote" });
  const caption = h("p", { class: "note-caption" });
  const where = whereText(source);
  const note = h(
    "section",
    {
      class: `note note-${info.kind}${inline ? " note-inline" : ""}`,
      id: `${prefix}-${source.label}`,
      tabindex: "-1",
      "aria-label": `Source ${source.label}: ${source.title}`,
      dataset: { label: source.label },
    },
    h("div", { class: "note-head" }, h("span", { class: "note-label", text: source.label }), stamp(source.status, source.status_text)),
    h("h4", { class: "note-title", text: source.title, title: source.title }),
    where ? h("p", { class: "note-where", text: where }) : null,
    quote,
    caption,
    h(
      "div",
      { class: "note-actions" },
      h("button", { type: "button", class: "link-btn", on: { click: (event) => ctx.openReader(source, current, event.currentTarget) } }, "Read the full passage"),
      externalLink(source.url, "Open the source"),
    ),
  );
  retargets.set(note, (next) => {
    current = next;
    fillQuote(quote, caption, source, current, captioned);
  });
  fillQuote(quote, caption, source, current, captioned);
  return note;
}

function passageList(sources, { claim, prefix, ctx }) {
  return h("div", { class: "passage-list" }, sources.map((source, index) => renderNote(source, { claim, inline: true, prefix, ctx, captioned: index === 0 })));
}

// ------------------------------------------------------------------------------------------------ the answer text

function makeView(turn, sources, citations = true) {
  return {
    sources,
    citations,
    claim: "",
    activate(label, claim) {
      const note = turn.querySelector(`.note[data-label="${label}"]`);
      if (!note) return;
      retargets.get(note)?.(claim);
      note.focus({ preventScroll: true });
      note.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
      note.classList.remove("note-flash");
      void note.offsetWidth; // restart the animation when the same note is chosen twice
      note.classList.add("note-flash");
      window.setTimeout(() => note.classList.remove("note-flash"), 1500);
    },
    link(label, on) {
      turn.querySelector(`.note[data-label="${label}"]`)?.classList.toggle("is-linked", on);
    },
  };
}

function chip(label, view) {
  const source = view.sources.get(label);
  if (!source) {
    return h(
      "span",
      { class: "cite cite-missing", title: `${label} is not among the passages shown` },
      label,
      h("span", { class: "sr-only", text: " (not among the passages shown)" }),
    );
  }
  const info = statusInfo(source.status, source.status_text);
  const claim = view.claim;
  return h(
    "button",
    {
      type: "button",
      class: `cite cite-${info.kind}`,
      dataset: { label },
      title: `${source.title}, ${info.label}`,
      "aria-label": `Source ${label}: ${source.title}, ${info.label}`,
      on: {
        click: () => view.activate(label, claim),
        mouseenter: () => view.link(label, true),
        mouseleave: () => view.link(label, false),
        focus: () => view.link(label, true),
        blur: () => view.link(label, false),
      },
    },
    label,
  );
}

function inline(tokens, view) {
  const out = [];
  for (const token of tokens) {
    if (token.t === "text") out.push(token.v);
    else if (token.t === "strong") out.push(h("strong", {}, inline(token.children, view)));
    else if (token.t === "em") out.push(h("em", {}, inline(token.children, view)));
    else if (token.t === "code") out.push(h("code", { text: token.v }));
    else if (!view.citations) out.push(token.raw);
    else out.push(...token.labels.map((label) => chip(label, view)));
  }
  return out;
}

function prose(blocks, view) {
  return blocks.map((block) => {
    if (block.type === "p") return h("p", {}, inline(block.inline, { ...view, claim: plainText(block.inline) }));
    if (block.type === "heading") return h("h3", { class: "prose-heading" }, inline(block.inline, { ...view, claim: plainText(block.inline) }));
    const start = block.type === "ol" && block.start > 1 ? block.start : null;
    return h(block.type, { start }, block.items.map((item) => h("li", {}, inline(item, { ...view, claim: plainText(item) }))));
  });
}

// ------------------------------------------------------------------------------------------------ checks, actions, details

function authorityLine(sources, lead) {
  const found = authority(sources, lead);
  if (!found.sentence) return null;
  const warn = found.draft > 0;
  return h(
    "p",
    { class: `authority${warn ? " authority-warn" : ""}` },
    icon(warn ? "warn" : "book", "icon icon-inline"),
    h("span", { text: warn ? `${found.sentence} The parts that cite a text that is not adopted are proposals, not rules.` : found.sentence }),
  );
}

function checkItem(kind, iconName, ...content) {
  return h("li", { class: `check check-${kind}` }, icon(iconName, "icon icon-inline"), h("span", {}, content));
}

function checks(answer, citedCount) {
  const items = [];
  const invalid = answer.invalid_citations ?? [];
  const figures = answer.unverified_figures ?? [];
  const uncited = answer.uncited_sentences ?? [];
  if (invalid.length) {
    items.push(checkItem("warn", "warn", `The answer cites ${joinList(invalid)}, which ${plural(invalid.length, "is", "are")} not among the passages shown. Disregard ${plural(invalid.length, "that citation", "those citations")}.`));
  }
  if (figures.length) {
    items.push(
      checkItem("warn", "warn", `The ${plural(figures.length, "number", "numbers")} ${joinList(figures)} ${plural(figures.length, "is", "are")} not in any retrieved passage or in your question. Check ${plural(figures.length, "it", "them")} in the sources.`),
    );
  }
  if (answer.truncated) items.push(checkItem("warn", "warn", "The answer was cut off by the length limit, so it may be incomplete."));
  if (!citedCount && !answer.abstained) items.push(checkItem("warn", "warn", "The answer cites no source. Do not rely on it without reading the passages."));
  if (!items.length) items.push(checkItem("ok", "check", "Automatic checks found no problem: every citation exists and every number appears in the passages."));
  if (uncited.length) {
    items.push(
      checkItem(
        "note",
        "info",
        `${uncited.length} ${plural(uncited.length, "statement has", "statements have")} no citation:`,
        h("ul", { class: "check-list" }, uncited.slice(0, 3).map((sentence) => h("li", { text: excerpt(sentence, 140) }))),
        uncited.length > 3 ? `and ${uncited.length - 3} more.` : null,
      ),
    );
  }
  return h("ul", { class: "checks" }, items);
}

function rank(value) {
  return Number.isFinite(value) ? `#${value}` : "–";
}

function number(value, digits) {
  return Number.isFinite(value) ? value.toFixed(digits) : "–";
}

function how(answer, ctx) {
  const facts = [];
  const add = (label, value) => value && facts.push(h("dt", { text: label }), h("dd", {}, value));
  add("Your question", answer.question);
  if (answer.standalone_question && answer.standalone_question !== answer.question) add("Searched as", answer.standalone_question);
  if (answer.plan_notes?.length) add("Question handling", h("ul", { class: "plain-list" }, answer.plan_notes.map((note) => h("li", { text: note }))));
  add("Settings", h("ul", { class: "plain-list" }, describeOptions(answer.options).map((row) => h("li", { text: `${row.label}: ${row.value}` }))));
  if (answer.model) {
    const fell = answer.fell_back_from ? ` (${answer.fell_back_from} was unavailable)` : "";
    const tokens = Number.isFinite(answer.input_tokens) ? `, ${answer.input_tokens.toLocaleString("en-US")} tokens read and ${(answer.output_tokens ?? 0).toLocaleString("en-US")} written` : "";
    add("Model", `${answer.model}${fell}${tokens}`);
  }
  add("Time", describeTimings(answer.timings));
  if (Number.isFinite(answer.citation_coverage)) add("Citation coverage", `${Math.round(answer.citation_coverage * 100)}% of the factual statements carry a citation.`);

  const rows = answer.sources.map((source) =>
    h(
      "tr",
      { class: source.cited ? "is-cited" : "" },
      h(
        "th",
        { scope: "row" },
        h("span", { class: "table-label", text: source.label }),
        h("span", { class: "table-title", text: source.title }),
        whereText(source) ? h("span", { class: "table-where", text: whereText(source) }) : null,
        h("button", { type: "button", class: "link-btn", on: { click: (event) => ctx.openReader(source, answer.standalone_question, event.currentTarget) } }, "Read"),
      ),
      h("td", { text: rank(source.dense_rank), title: `Meaning-search score ${number(source.dense_score, 3)}` }),
      h("td", { text: rank(source.lexical_rank), title: `Keyword-search score ${number(source.lexical_score, 2)}` }),
      h("td", { text: number(source.fused_score, 4) }),
      h("td", { text: number(source.rerank_score, 2) }),
      h("td", { text: source.cited ? "Yes" : "No" }),
    ),
  );
  const table = answer.sources.length
    ? h(
        "div",
        { class: "table-scroll" },
        h(
          "table",
          { class: "how-table" },
          h("caption", { class: "sr-only", text: "The passages that were retrieved for this question" }),
          h(
            "thead",
            {},
            h("tr", {}, ["Passage", "Meaning search", "Keyword search", "Combined score", "Reranker score", "Cited"].map((label) => h("th", { scope: "col", text: label }))),
          ),
          h("tbody", {}, rows),
        ),
      )
    : null;
  const notes = answer.notes?.length ? h("ul", { class: "plain-list how-notes" }, answer.notes.map((note) => h("li", { text: note }))) : null;
  return h(
    "details",
    { class: "how" },
    h("summary", {}, "How this answer was made"),
    h("div", { class: "how-body" }, h("dl", { class: "facts" }, facts), table, notes),
  );
}

function actions(exchange, answer, ctx) {
  const parts = [];
  if (answer.model) parts.push(`Written by ${answer.model}`);
  if (Number.isFinite(exchange.elapsed)) parts.push(formatSeconds(exchange.elapsed));
  return h(
    "div",
    { class: "turn-actions" },
    button("Copy answer with sources", { iconName: "copy", onClick: (event) => ctx.copy(toMarkdown([exchange], ctx.exportMeta()), event.currentTarget) }),
    parts.length ? h("p", { class: "turn-meta", text: parts.join(", ") }) : null,
  );
}

// ------------------------------------------------------------------------------------------------ problems

function modelProblem(answer) {
  return h(
    "section",
    { class: "problem problem-soft" },
    h("h3", { class: "problem-title" }, icon("warn", "icon icon-inline"), "The language model was not used"),
    h("p", { text: answer.hint ?? "The model could not answer, so the matching passages are shown instead." }),
    answer.error ? h("details", { class: "technical" }, h("summary", {}, "Technical detail"), h("p", { class: "code", text: answer.error })) : null,
  );
}

function problem(error, exchange, ctx) {
  return h(
    "section",
    { class: "problem", role: "alert" },
    h("h3", { class: "problem-title" }, icon("warn", "icon icon-inline"), PROBLEM_TITLES[error.kind] ?? "Something went wrong"),
    h("p", { text: error.message }),
    error.fix
      ? h("div", { class: "command" }, h("code", { text: error.fix }), button("Copy", { iconName: "copy", onClick: (event) => ctx.copy(error.fix, event.currentTarget) }))
      : null,
    error.kind === "bad_request" ? null : button(error.kind === "cancelled" ? "Ask again" : "Try again", { className: "btn small", onClick: () => ctx.retry(exchange) }),
  );
}

function working(exchange) {
  const elapsed = h("span", { class: "elapsed", text: "0 s" });
  exchange.elapsedEl = elapsed;
  return h("div", { class: "working" }, h("span", { class: "working-bar", "aria-hidden": "true" }), h("p", {}, WORKING[exchange.mode] ?? WORKING.rag, " ", elapsed));
}

// ------------------------------------------------------------------------------------------------ answer from the documents

function renderRag(turn, exchange, answer, ctx, prefix) {
  const blocks = parseAnswer(answer.text);
  const contexts = citationContexts(blocks);
  const sourceMap = new Map(answer.sources.map((source) => [source.label, source]));
  const order = [...new Set(contexts.map((entry) => entry.label))].filter((label) => sourceMap.has(label));
  const claims = new Map();
  for (const entry of contexts) if (!claims.has(entry.label)) claims.set(entry.label, entry.claim);
  const view = makeView(turn, sourceMap, true);
  const cited = order.map((label) => sourceMap.get(label));

  const main = h("div", { class: "main" });
  if (answer.abstained) {
    main.append(
      h("p", { class: "verdict" }, icon("info", "icon icon-inline"), "The documents do not answer this question"),
      h("div", { class: "prose" }, prose(blocks, view)),
    );
    if (answer.sources.length) {
      main.append(
        h(
          "details",
          { class: "searched", open: true },
          h("summary", {}, `The ${answer.sources.length} closest ${plural(answer.sources.length, "passage", "passages")} that were searched`),
          passageList(answer.sources, { claim: answer.standalone_question, prefix, ctx }),
        ),
        h("p", { class: "advice", text: "Try other words, or widen the scope with Change scope and settings." }),
      );
    } else {
      main.append(h("p", { class: "advice", text: "Nothing in the chosen scope matches. Try other words, or widen the scope with Change scope and settings." }));
    }
    main.append(how(answer, ctx));
    turn.append(h("div", { class: "turn-body" }, main));
    return;
  }

  main.append(authorityLine(cited, "Rests on"), h("div", { class: "prose" }, prose(blocks, view)), checks(answer, cited.length));
  if (!cited.length && answer.sources.length) {
    main.append(
      h(
        "details",
        { class: "searched", open: true },
        h("summary", {}, "The passages that were searched"),
        passageList(answer.sources, { claim: answer.standalone_question, prefix, ctx }),
      ),
    );
  }
  main.append(actions(exchange, answer, ctx), how(answer, ctx));

  const body = h("div", { class: "turn-body" }, main);
  if (cited.length) {
    const margin = h(
      "aside",
      { class: "margin", "aria-label": "Sources cited in this answer" },
      h("h3", { class: "margin-title", text: "Sources cited" }),
      cited.map((source, index) => renderNote(source, { claim: claims.get(source.label), prefix, ctx, captioned: index === 0 })),
    );
    body.append(margin);
  }
  turn.append(body);

  if (cited.length && typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => window.requestAnimationFrame(() => layoutNotes(turn)));
    observer.observe(main.querySelector(".prose"));
    for (const note of turn.querySelectorAll(".margin > .note")) observer.observe(note);
    observers.set(turn, observer);
  }
}

// ------------------------------------------------------------------------------------------------ passages only

function renderPassages(turn, exchange, answer, ctx, prefix) {
  const main = h("div", { class: "main main-wide" });
  if (answer.error_kind) main.append(modelProblem(answer));
  const lead = exchange.mode === "passages" ? "These are the passages that match your question best, best match first. No language model was asked." : answer.text;
  main.append(h("p", { class: "lead", text: lead }), authorityLine(answer.sources, "The best matches are"));
  main.append(passageList(answer.sources, { claim: answer.standalone_question, prefix, ctx }), how(answer, ctx));
  turn.append(h("div", { class: "turn-body turn-body-single" }, main));
}

// ------------------------------------------------------------------------------------------------ the model alone

function renderBaseline(baseline) {
  const section = h("section", { class: "baseline", "aria-label": "The model alone" });
  section.append(h("h3", { class: "section-title", text: "The model alone, without the documents" }));
  if (baseline.error_kind) {
    section.append(modelProblem({ ...baseline, hint: baseline.hint ?? "The model could not answer." }));
    return section;
  }
  section.append(
    h("p", { class: "caution", text: "Written from the model's memory. It had no access to the documents, so nothing here is cited or checked." }),
    h("div", { class: "prose" }, prose(parseAnswer(baseline.text), makeView(section, new Map(), false))),
  );
  const parts = [];
  if (baseline.model) parts.push(`Written by ${baseline.model}`);
  if (Number.isFinite(baseline.timings?.total)) parts.push(formatSeconds(baseline.timings.total));
  if (parts.length) section.append(h("p", { class: "turn-meta", text: parts.join(", ") }));
  return section;
}

function renderFigures(answer, baseline) {
  const rows = compareFigures(answer.text, baseline.text);
  const section = h("section", { class: "figures", "aria-label": "Dates and amounts compared" }, h("h3", { class: "section-title", text: "Dates and amounts, side by side" }));
  if (!rows.length) {
    section.append(h("p", { class: "figures-lead", text: "Neither answer names a date, an amount, a percentage or a time limit." }));
    return section;
  }
  const differing = rows.filter((row) => row.first !== row.second).length;
  section.append(
    h(
      "p",
      { class: "figures-lead" },
      differing
        ? `${differing} of ${rows.length} ${plural(rows.length, "figure appears", "figures appear")} in only one of the two answers. The last column shows whether the retrieved passages contain the figure.`
        : `The two answers name the same ${rows.length} ${plural(rows.length, "figure", "figures")}.`,
    ),
  );
  const yes = (on, who) =>
    on ? h("td", { class: "yes" }, h("span", { class: "dot", "aria-hidden": "true" }), h("span", { class: "sr-only", text: `${who} names it` }), "Named") : h("td", { class: "no" }, h("span", { "aria-hidden": "true", text: "–" }), h("span", { class: "sr-only", text: `${who} does not name it` }));
  section.append(
    h(
      "div",
      { class: "table-scroll" },
      h(
        "table",
        { class: "figure-table" },
        h("caption", { class: "sr-only", text: "Each date, amount, percentage and time limit of the two answers" }),
        h("thead", {}, h("tr", {}, ["Figure", "With the documents", "Model alone", "In the retrieved passages"].map((label) => h("th", { scope: "col", text: label })))),
        h(
          "tbody",
          {},
          rows.map((row) => {
            const found = sourcesWithFigure(row.key, answer.sources);
            return h(
              "tr",
              { class: row.first !== row.second ? "differs" : "" },
              h("th", { scope: "row", text: row.label }),
              yes(row.first, "The answer from the documents"),
              yes(row.second, "The model alone"),
              found.length
                ? h("td", { class: "found" }, found.map((label) => h("span", { class: "cite cite-plain", text: label })))
                : h("td", { class: "missing" }, "Not in the retrieved passages"),
            );
          }),
        ),
      ),
    ),
  );
  return section;
}

// ------------------------------------------------------------------------------------------------ the turn

const observers = new WeakMap();

export function disposeTurn(turn) {
  observers.get(turn)?.disconnect();
  observers.delete(turn);
}

export function renderTurn(exchange, ctx) {
  const prefix = `t${exchange.id}`;
  const turn = h(
    "article",
    { class: "turn", id: prefix, "aria-labelledby": `${prefix}-q` },
    h("header", { class: "turn-head" }, h("h2", { class: "question", id: `${prefix}-q`, text: exchange.question }), h("p", { class: "scope", text: exchange.scopeText })),
  );
  if (exchange.pending) {
    turn.append(working(exchange));
    return turn;
  }
  if (exchange.error) {
    turn.append(problem(exchange.error, exchange, ctx));
    return turn;
  }
  const { answer, baseline } = exchange;
  if (answer) {
    if (answer.mode === "extractive") renderPassages(turn, exchange, answer, ctx, prefix);
    else renderRag(turn, exchange, answer, ctx, prefix);
  }
  if (exchange.mode === "compare") {
    if (answer && baseline) {
      turn.append(renderBaseline(baseline));
      if (!baseline.error_kind && answer.mode === "rag") turn.append(renderFigures(answer, baseline));
    } else {
      turn.append(h("p", { class: "advice", text: "The model alone was not asked, because the answer from the documents could not be written." }));
    }
  } else if (exchange.mode === "llm_only" && baseline) {
    turn.append(renderBaseline(baseline));
  }
  return turn;
}

// ------------------------------------------------------------------------------------------------ margin layout

// On a wide screen every note sits in the margin beside the first statement that cites it; notes that would overlap
// are pushed down. On a narrow screen the notes simply follow the answer.
export function layoutNotes(turn) {
  const margin = turn.querySelector(".margin");
  if (!margin) return;
  const notes = [...margin.querySelectorAll(":scope > .note")];
  if (!window.matchMedia(WIDE).matches) {
    for (const note of notes) note.style.top = "";
    margin.style.minHeight = "";
    return;
  }
  const origin = margin.getBoundingClientRect().top;
  let floor = 0;
  for (const note of notes) {
    const anchor = turn.querySelector(`.prose .cite[data-label="${note.dataset.label}"]`);
    let wanted = floor;
    if (anchor) {
      // beside the start of the statement; in a very long paragraph, beside the citation itself
      const block = anchor.closest("p, li");
      const blockTop = block ? block.getBoundingClientRect().top : anchor.getBoundingClientRect().top;
      const anchorTop = anchor.getBoundingClientRect().top;
      wanted = (anchorTop - blockTop > 360 ? anchorTop - 4 : blockTop - 2) - origin;
    }
    const top = Math.max(wanted, floor, 0);
    note.style.top = `${Math.round(top)}px`;
    floor = top + note.offsetHeight + 16;
  }
  margin.style.minHeight = `${Math.max(0, Math.ceil(floor - 16))}px`;
}

export function layoutAll(container) {
  for (const turn of container.querySelectorAll(".turn")) layoutNotes(turn);
}
