// Pure helpers of the RegLens page. Nothing here touches the document, so Node can test all of it
// (tests/js/lib.test.mjs). The page itself is app.mjs.

// ------------------------------------------------------------------------------------------------ legal status

// The manifest's status values and how the page names them. "kind" decides the look of stamps, citation chips and notes:
// law = binding text in force; guidance = final but not law; draft = not adopted (draft, consultation, proposal).
export const STATUS = {
  in_force: { label: "In force", kind: "law" },
  final: { label: "Final guidance", kind: "guidance" },
  draft: { label: "Draft, not adopted", kind: "draft" },
  consultation: { label: "Consultation draft, not final", kind: "draft" },
  proposed: { label: "Proposed, not adopted", kind: "draft" },
};

export function statusInfo(status, fallback = "") {
  return STATUS[status] ?? { label: fallback || status || "Status unknown", kind: "unknown" };
}

export const JURISDICTIONS = { EU: "EU", US: "US", UK: "UK", International: "International" };

export function plural(n, one, many) {
  return n === 1 ? one : many;
}

export function joinList(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

export function formatSeconds(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 10) return `${seconds.toFixed(1)} s`;
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${String(Math.round(seconds - minutes * 60)).padStart(2, "0")} s`;
}

export function formatCount(n) {
  return Number.isFinite(n) ? n.toLocaleString("en-US") : "";
}

// ------------------------------------------------------------------------------------------------ the answer text

// The model is told to cite as [S2] or [S1][S3]; the server also accepts [S1, S3].
const CITATION = String.raw`\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]`;
const INLINE = new RegExp(
  [
    String.raw`\*\*(?=\S)(.+?)(?<=\S)\*\*`, // 1 bold
    String.raw`__(?=\S)(.+?)(?<=\S)__`, // 2 bold
    String.raw`(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])`, // 3 italic
    String.raw`(?<![\w_])_(?![\s_])(.+?)(?<![\s_])_(?![\w_])`, // 4 italic
    "`([^`]+)`", // 5 code
    CITATION, // 6 citation
  ].join("|"),
  "g",
);

// The text of an answer without its [S1] markers, for sending an earlier answer back as conversation history.
export function stripCitations(text) {
  return String(text ?? "")
    .replace(new RegExp(CITATION, "g"), "")
    .replace(/[ \t]+/g, " ")
    .replace(/ +([.,;:])/g, "$1")
    .replace(/ *\n */g, "\n")
    .trim();
}

export function citationLabels(group) {
  return group.match(/S\d+/g) ?? [];
}

// Inline tokens: {t: "text"|"strong"|"em"|"code"|"cite", ...}. Anything that is not understood stays plain text.
export function parseInline(text) {
  const tokens = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    if (match.index > last) tokens.push({ t: "text", v: text.slice(last, match.index) });
    const [, bold1, bold2, italic1, italic2, code, cite] = match;
    if (bold1 !== undefined || bold2 !== undefined) tokens.push({ t: "strong", children: parseInline(bold1 ?? bold2) });
    else if (italic1 !== undefined || italic2 !== undefined) tokens.push({ t: "em", children: parseInline(italic1 ?? italic2) });
    else if (code !== undefined) tokens.push({ t: "code", v: code });
    else tokens.push({ t: "cite", labels: citationLabels(cite), raw: match[0] });
    last = match.index + match[0].length;
  }
  if (last < text.length) tokens.push({ t: "text", v: text.slice(last) });
  return tokens;
}

// The text of inline tokens without the citation markers: what a statement says.
export function plainText(tokens) {
  return tokens
    .map((token) => {
      if (token.t === "cite") return "";
      if (token.t === "strong" || token.t === "em") return plainText(token.children);
      return token.v;
    })
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

// Blocks: {type: "p"|"heading", inline} and {type: "ul"|"ol", start, items: [inline, ...]}.
export function parseAnswer(text) {
  const blocks = [];
  let paragraph = [];
  let list = null;
  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ type: "p", inline: parseInline(paragraph.join(" ")) });
    paragraph = [];
  };
  const flushList = () => {
    if (list) blocks.push({ type: list.type, start: list.start, items: list.items.map((lines) => parseInline(lines.join(" "))) });
    list = null;
  };
  for (const raw of String(text ?? "").replace(/\r\n?/g, "\n").split("\n")) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    const heading = /^\s{0,3}#{1,6}\s+(.*)$/.exec(line);
    const bullet = /^\s*[-*•]\s+(.*)$/.exec(line);
    const number = /^\s*(\d+)[.)]\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", inline: parseInline(heading[1]) });
    } else if (bullet || number) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (!list || list.type !== type) {
        flushList();
        list = { type, start: number ? Number(number[1]) : 1, items: [] };
      }
      list.items.push([(bullet ?? number)[bullet ? 1 : 2]]);
    } else if (list && /^\s+\S/.test(line)) {
      list.items[list.items.length - 1].push(line.trim()); // an indented line continues the item above
    } else {
      flushList();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  flushList();
  return blocks;
}

// Every citation of an answer with the statement it stands under (the whole paragraph or list item, without markers).
export function citationContexts(blocks) {
  const out = [];
  const visit = (tokens) => {
    const claim = plainText(tokens);
    const walk = (list) => {
      for (const token of list) {
        if (token.t === "cite") for (const label of token.labels) out.push({ label, claim });
        else if (token.children) walk(token.children);
      }
    };
    walk(tokens);
  };
  for (const block of blocks) {
    if (block.items) block.items.forEach(visit);
    else visit(block.inline);
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ figures

const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
const MONTH = "(January|February|March|April|May|June|July|August|September|October|November|December)";
const UNIT = String.raw`(?:calendar\s+|working\s+|business\s+)?(?:hours?|days?|weeks?|months?|years?|minutes?)`;
const CURRENCY = String.raw`(?:€|EUR|USD|US\$|\$|£|GBP)`;
const AMOUNT = String.raw`\d{1,3}(?:[ ,.  ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?`;
const SCALE = String.raw`(?:\s?(?:million|billion|thousand|bn|m|k)\b)?`;

const FIGURE_PATTERNS = [
  { kind: "date", re: new RegExp(String.raw`(?<![\w/.])(\d{1,2})(?:st|nd|rd|th)?\s+${MONTH}\s+(\d{4})(?![\w/])`, "gi"), key: (m) => iso(m[3], m[2], m[1]) },
  { kind: "date", re: new RegExp(String.raw`(?<![\w/.])${MONTH}\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?![\w/])`, "gi"), key: (m) => iso(m[3], m[1], m[2]) },
  { kind: "date", re: /(?<![\w/.])(\d{4})-(\d{2})-(\d{2})(?![\w/])/g, key: (m) => `${m[1]}-${m[2]}-${m[3]}` },
  { kind: "date", re: new RegExp(String.raw`(?<![\w/.])${MONTH}\s+(\d{4})(?![\w/])`, "gi"), key: (m) => iso(m[2], m[1]) },
  { kind: "money", re: new RegExp(String.raw`(${CURRENCY})\s?(${AMOUNT})${SCALE}`, "gi"), key: (m) => money(m[1], m[2], m[0]) },
  { kind: "money", re: new RegExp(String.raw`(?<![\w/.])(${AMOUNT})${SCALE}\s?(euros?|dollars?|pounds?)\b`, "gi"), key: (m) => money(m[2], m[1], m[0]) },
  { kind: "percent", re: /(?<![\w/.])(\d+(?:[.,]\d+)?)\s?(?:%|per\s?cent\b|percent\b)/gi, key: (m) => `${number(m[1])}%` },
  { kind: "period", re: new RegExp(String.raw`(?<![\w/.])(\d+(?:[.,]\d+)?)[\s-]?(${UNIT})\b`, "gi"), key: (m) => `${number(m[1])} ${m[2].toLowerCase().replace(/\s+/g, " ").replace(/s$/, "")}` },
  { kind: "year", re: /(?<![\w/.-])((?:19|20)\d{2})(?![\w/-])/g, key: (m) => m[1] },
];

function iso(year, month, day) {
  const index = MONTHS.indexOf(String(month).toLowerCase());
  const mm = String(index + 1).padStart(2, "0");
  return day === undefined ? `${year}-${mm}` : `${year}-${mm}-${String(day).padStart(2, "0")}`;
}

function number(text) {
  return String(Number(String(text).replace(",", ".")));
}

function money(currency, amount, whole) {
  const code = { "€": "EUR", "£": "GBP", $: "USD", "us$": "USD", euro: "EUR", euros: "EUR", dollar: "USD", dollars: "USD", pound: "GBP", pounds: "GBP" }[currency.toLowerCase()] ?? currency.toUpperCase();
  let digits = amount.replace(/[   ]/g, "");
  if (/^\d{1,3}(,\d{3})+(\.\d+)?$/.test(digits)) digits = digits.replace(/,/g, "");
  else if (/^\d{1,3}(\.\d{3})+(,\d+)?$/.test(digits)) digits = digits.replace(/\./g, "").replace(",", ".");
  else digits = digits.replace(",", ".");
  let value = Number(digits);
  const scale = /(million|billion|thousand|bn|m|k)\s*$/i.exec(whole.replace(/\s?(euros?|dollars?|pounds?)\s*$/i, ""))?.[1]?.toLowerCase();
  if (scale) value *= { million: 1e6, m: 1e6, billion: 1e9, bn: 1e9, thousand: 1e3, k: 1e3 }[scale];
  return `${code} ${value}`;
}

// Dates, periods, amounts, percentages and years in a text, each with a key that is the same for the same figure
// however it is written ("2 December 2027" and "December 2, 2027" share a key). The earliest, longest match wins.
export function findFigures(text) {
  const taken = [];
  const found = [];
  const overlaps = (start, end) => taken.some(([a, b]) => start < b && end > a);
  for (const { kind, re, key } of FIGURE_PATTERNS) {
    for (const match of String(text ?? "").matchAll(re)) {
      const start = match.index;
      const end = start + match[0].length;
      if (overlaps(start, end)) continue;
      taken.push([start, end]);
      found.push({ kind, key: `${kind}:${key(match)}`, text: match[0], start, end });
    }
  }
  return found.sort((a, b) => a.start - b.start);
}

// Plain numbers of two digits or more ("Article 113" gives 113): what the server's check reads, and what the page marks.
export function findNumbers(text) {
  const out = [];
  for (const match of String(text ?? "").matchAll(/(?<![\w.])(\d{1,3}(?:[,   ]\d{3})+|\d{2,})(?:[.,]\d+)?(?![\w])/g)) {
    out.push({ digits: match[1].replace(/[,   ]/g, ""), text: match[0], start: match.index, end: match.index + match[0].length });
  }
  return out;
}

const KIND_ORDER = { date: 0, period: 1, money: 2, percent: 3, year: 4 };

// For the comparison table: every figure of either answer, and in which of the two it appears.
export function compareFigures(first, second) {
  const rows = new Map();
  const add = (text, side) => {
    for (const figure of findFigures(text)) {
      const row = rows.get(figure.key) ?? { key: figure.key, kind: figure.kind, label: figure.text.replace(/\s+/g, " "), first: false, second: false };
      row[side] = true;
      rows.set(figure.key, row);
    }
  };
  add(first, "first");
  add(second, "second");
  return [...rows.values()].sort((a, b) => KIND_ORDER[a.kind] - KIND_ORDER[b.kind] || a.label.localeCompare(b.label, "en", { numeric: true }));
}

// Which of the sources contain the figure with this key (their labels), for "found in S1".
export function sourcesWithFigure(key, sources) {
  return sources.filter((source) => findFigures(source.text).some((figure) => figure.key === key)).map((source) => source.label);
}

// ------------------------------------------------------------------------------------------------ passages

const STOP = new Set(
  "about above after again against also among because been before being between both can could did does doing down during each for from further had has have having here how into its itself just more most must not now off once only other our out over own same shall should such than that the their them then there these they this those through under until upon very was were what when where which while who whom why will with would you your".split(" "),
);

function words(text) {
  const out = new Set();
  for (const word of String(text ?? "").toLowerCase().match(/[a-z][a-z-]{2,}/g) ?? []) {
    if (STOP.has(word)) continue;
    out.add(word.length > 4 ? word.replace(/(ies|es|s)$/, "") : word);
  }
  return out;
}

// Sentence boundaries of a passage as offsets, so a highlight can be placed in the original text.
export function splitSentences(text) {
  const out = [];
  for (const line of String(text ?? "").matchAll(/[^\n]+/g)) {
    const base = line.index;
    const s = line[0];
    const push = (from, to) => {
      while (from < to && /\s/.test(s[from])) from++;
      while (to > from && /\s/.test(s[to - 1])) to--;
      if (to > from) out.push({ start: base + from, end: base + to });
    };
    let from = 0;
    for (const cut of s.matchAll(/([.!?;])\s+(?=[A-Z(‘“"'0-9])/g)) {
      push(from, cut.index + 1);
      from = cut.index + cut[0].length;
    }
    push(from, s.length);
  }
  return out;
}

// The sentence of the passage that shares the most words and figures with the statement it supports, or -1 when
// nothing matches convincingly. A guide for the eye, never a proof: the page says "closest sentence".
export function closestSentence(passage, claim) {
  const claimWords = words(claim);
  const claimFigures = new Set(findFigures(claim).map((figure) => figure.key));
  const claimNumbers = new Set(findNumbers(claim).map((n) => n.digits));
  let best = -1;
  let bestScore = 0;
  splitSentences(passage).forEach((span, index) => {
    const sentence = passage.slice(span.start, span.end);
    const sentenceWords = words(sentence);
    let shared = 0;
    for (const word of claimWords) if (sentenceWords.has(word)) shared++;
    const figures = findFigures(sentence).filter((figure) => claimFigures.has(figure.key)).length;
    const numbers = findNumbers(sentence).filter((n) => claimNumbers.has(n.digits)).length;
    const hits = shared + 2 * figures + numbers;
    if (hits < 2) return;
    const score = hits / Math.sqrt(sentenceWords.size + 4);
    if (score > bestScore) {
      best = index;
      bestScore = score;
    }
  });
  return best;
}

// A window of about `limit` characters of the passage around sentence `index` (index -1: the start of the passage).
export function excerptAround(passage, index, limit = 320) {
  const spans = splitSentences(passage);
  if (!spans.length) return { start: 0, end: 0, text: "", before: false, after: false };
  let from = Math.max(index, 0);
  let to = from;
  const length = () => spans[to].end - spans[from].start;
  while (to + 1 < spans.length && length() + (spans[to + 1].end - spans[to].end) <= limit) to++;
  while (from > 0 && index > 0 && length() + (spans[from].start - spans[from - 1].start) <= limit) from--;
  let start = spans[from].start;
  let end = spans[to].end;
  if (end - start > limit) {
    // one long sentence: cut it at a word boundary
    end = start + limit;
    const space = passage.lastIndexOf(" ", end);
    if (space > start + limit * 0.6) end = space;
  }
  return { start, end, text: passage.slice(start, end), before: start > 0, after: end < passage.length };
}

// Cut a text into pieces with the classes of every mark that covers them: marks may overlap.
// marks: [{start, end, cls}] -> [{text, classes: [...]}]
export function segments(text, marks) {
  const cuts = new Set([0, text.length]);
  for (const mark of marks) {
    cuts.add(Math.max(0, Math.min(text.length, mark.start)));
    cuts.add(Math.max(0, Math.min(text.length, mark.end)));
  }
  const points = [...cuts].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i + 1 < points.length; i++) {
    const [start, end] = [points[i], points[i + 1]];
    if (end <= start) continue;
    const classes = marks.filter((mark) => mark.start <= start && mark.end >= end).map((mark) => mark.cls);
    out.push({ text: text.slice(start, end), classes: [...new Set(classes)] });
  }
  return out;
}

// What to mark in a passage for a statement: the figures the two share, and the closest sentence.
export function passageMarks(passage, claim, offset = 0, length = passage.length) {
  const marks = [];
  const claimFigures = new Set(findFigures(claim).map((figure) => figure.key));
  const claimNumbers = new Set(findNumbers(claim).map((n) => n.digits));
  for (const figure of findFigures(passage)) if (claimFigures.has(figure.key)) marks.push({ start: figure.start, end: figure.end, cls: "figure" });
  for (const n of findNumbers(passage)) {
    if (claimNumbers.has(n.digits) && !marks.some((m) => n.start < m.end && n.end > m.start)) marks.push({ start: n.start, end: n.end, cls: "figure" });
  }
  const index = closestSentence(passage, claim);
  if (index >= 0) {
    const span = splitSentences(passage)[index];
    marks.push({ start: span.start, end: span.end, cls: "closest" });
  }
  return marks
    .filter((mark) => mark.end > offset && mark.start < offset + length)
    .map((mark) => ({ ...mark, start: mark.start - offset, end: mark.end - offset }));
}

// ------------------------------------------------------------------------------------------------ summaries

// `lead` starts the sentence: "Rests on ..." for an answer, "The best matches are ..." for a list of passages.
export function authority(sources, lead = "Rests on") {
  const counts = { law: 0, guidance: 0, draft: 0, unknown: 0 };
  for (const source of sources) counts[statusInfo(source.status).kind]++;
  const parts = [];
  if (counts.law) parts.push(`${counts.law} binding ${plural(counts.law, "text", "texts")}`);
  if (counts.guidance) parts.push(`${counts.guidance} ${plural(counts.guidance, "text", "texts")} of final guidance`);
  if (counts.draft) parts.push(`${counts.draft} ${plural(counts.draft, "text that is", "texts that are")} not adopted`);
  if (counts.unknown) parts.push(`${counts.unknown} ${plural(counts.unknown, "text", "texts")} of unknown status`);
  return { ...counts, total: sources.length, sentence: parts.length ? `${lead} ${joinList(parts)}.` : "" };
}

export function matchesFilters(doc, filters) {
  return ["jurisdiction", "status", "doc_type", "doc_id"].every((key) => {
    const wanted = filters?.[key];
    return !wanted || !wanted.length || wanted.includes(doc[key]);
  });
}

export function activeFilters(filters) {
  return Object.fromEntries(Object.entries(filters ?? {}).filter(([, values]) => Array.isArray(values) && values.length));
}

// A sentence about what the search covers, for the dock and for the heading of every answer.
export function scopeSummary(filters, documents) {
  const active = activeFilters(filters);
  const count = documents.filter((doc) => matchesFilters(doc, active)).length;
  if (!Object.keys(active).length) return { active: false, count, text: `All ${count} documents` };
  // Only individual documents chosen: "3 chosen documents" already says how many, so no count is added.
  if (Object.keys(active).length === 1 && active.doc_id) return { active: true, count, text: `${count} chosen ${plural(count, "document", "documents")}` };
  const parts = [];
  if (active.jurisdiction) parts.push(joinList(active.jurisdiction));
  if (active.status) parts.push(joinList(active.status.map((s) => statusInfo(s).label.toLowerCase())));
  if (active.doc_type) parts.push(joinList(active.doc_type));
  if (active.doc_id) parts.push(`${active.doc_id.length} chosen ${plural(active.doc_id.length, "document", "documents")}`);
  return { active: true, count, text: `${parts.join(", ")}: ${count} ${plural(count, "document", "documents")}` };
}

export function excerpt(text, limit = 200) {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  if (flat.length <= limit) return flat;
  const cut = flat.lastIndexOf(" ", limit);
  return `${flat.slice(0, cut > limit * 0.6 ? cut : limit)}…`;
}

// ------------------------------------------------------------------------------------------------ how an answer was made

// The search methods the server accepts (reglens.retrieval.retriever.MODES), with the words the page uses for them.
// A test compares these values with the server's list, so the two cannot drift apart unnoticed.
export const SEARCH_METHODS = [
  {
    value: "hybrid",
    name: "Meaning and keywords combined",
    detail: "Finds passages that say the same thing and passages that use the same words, then merges both rankings. Recommended.",
  },
  { value: "dense", name: "Meaning only", detail: "Finds paraphrases, but can miss exact terms such as article numbers." },
  { value: "lexical", name: "Keywords only", detail: "Matches words as written, but misses paraphrases." },
];

const SEARCH_NAMES = Object.fromEntries(SEARCH_METHODS.map((method) => [method.value, method.name]));

function describeWhere(where) {
  const parts = [];
  for (const [key, raw] of Object.entries(where ?? {})) {
    const values = [].concat(raw);
    if (key === "status") parts.push(joinList(values.map((value) => statusInfo(value).label.toLowerCase())));
    else if (key === "doc_id") parts.push(`${values.length} chosen ${plural(values.length, "document", "documents")}`);
    else parts.push(joinList(values));
  }
  return parts.length ? parts.join(", ") : "All documents";
}

// The settings an answer was made with (Answer.options from the server) as label and value pairs, in plain words.
export function describeOptions(options) {
  const retrieval = options?.retrieval;
  const transform = options?.transform;
  if (!retrieval || !transform) return [];
  const onOff = (flag) => (flag ? "on" : "off");
  return [
    { label: "Search method", value: SEARCH_NAMES[retrieval.mode] ?? String(retrieval.mode) },
    { label: "Passages kept", value: String(retrieval.top_k) },
    { label: "Re-ranking", value: onOff(retrieval.rerank) },
    { label: "Footnotes", value: retrieval.include_notes ? "included" : "left out" },
    { label: "Scope", value: describeWhere(retrieval.where) },
    { label: "Related legal terms added", value: onOff(transform.glossary) },
    { label: "Follow-ups rewritten", value: onOff(transform.condense) },
    { label: "Rephrased searches", value: String(transform.multi_query) },
    { label: "Search with a hypothetical answer", value: onOff(transform.hyde) },
  ];
}

const TIMING_NAMES = [
  ["transform", "preparing the question"],
  ["retrieve", "searching"],
  ["generate", "writing the answer"],
];

// "Preparing the question 0.0 s, searching 0.4 s, writing the answer 3.1 s. 3.6 s in total."
export function describeTimings(timings) {
  const steps = TIMING_NAMES.filter(([key]) => Number.isFinite(timings?.[key])).map(([key, name]) => `${name} ${formatSeconds(timings[key])}`);
  const total = Number.isFinite(timings?.total) ? `${formatSeconds(timings.total)} in total.` : "";
  const list = steps.length ? `${steps.join(", ").replace(/^./, (c) => c.toUpperCase())}.` : "";
  return [list, total].filter(Boolean).join(" ");
}

// ------------------------------------------------------------------------------------------------ export

function sourceLine(source) {
  const where = [source.section, source.pages].filter(Boolean).join(", ");
  const status = statusInfo(source.status, source.status_text).label;
  const tail = [source.jurisdiction, status].filter(Boolean).join(", ");
  return `- [${source.label}] ${source.title}${where ? `, ${where}` : ""} (${tail})${source.url ? ` ${source.url}` : ""}`;
}

export function toMarkdown(exchanges, meta = {}) {
  const lines = ["# RegLens conversation", ""];
  const facts = [
    meta.exportedAt ? `Exported ${meta.exportedAt}.` : "",
    meta.passages ? `${formatCount(meta.passages)} passages from ${meta.documents} documents were searchable.` : "",
    meta.model ? `Language model: ${meta.model}.` : "",
    "Research support, not legal advice.",
  ].filter(Boolean);
  lines.push(facts.join(" "), "");
  exchanges.forEach((exchange, index) => {
    lines.push(`## ${index + 1}. ${exchange.question}`, "");
    if (exchange.scope) lines.push(`Search scope: ${exchange.scope}.`, "");
    const answer = exchange.answer;
    if (answer) {
      lines.push(answer.mode === "extractive" ? "**Matching passages** (no model answer)" : "**Answer from the documents**", "");
      lines.push(answer.text, "");
      const cited = answer.sources.filter((source) => source.cited);
      const listed = cited.length ? cited : answer.sources;
      if (listed.length) lines.push("Sources:", ...listed.map(sourceLine), "");
      const checks = [];
      if (answer.invalid_citations?.length) checks.push(`citations that do not exist: ${answer.invalid_citations.join(", ")}`);
      if (answer.unverified_figures?.length) checks.push(`figures not found in the sources: ${answer.unverified_figures.join(", ")}`);
      if (checks.length) lines.push(`Checks: ${checks.join("; ")}.`, "");
    }
    if (exchange.baseline) lines.push(`**The model alone, without the documents** (${exchange.baseline.model ?? "model"})`, "", exchange.baseline.text, "");
  });
  return `${lines.join("\n").replace(/\n{3,}/g, "\n\n").trim()}\n`;
}

// ------------------------------------------------------------------------------------------------ settings

// Take what was saved earlier, keep only what is still valid, and fill the rest from the server's defaults.
// "limits" is the server's {top_k: [min, max], multi_query: [min, max]}: a saved number outside a range is pulled inside it,
// and a saved search method the server does not know is dropped, so that old browser data can never make every question fail.
export function mergeSettings(defaults, saved, limits = {}) {
  const merged = { ...defaults };
  if (!saved || typeof saved !== "object") return merged;
  for (const key of Object.keys(defaults)) {
    let value = saved[key];
    if (typeof value !== typeof defaults[key]) continue;
    if (key === "search" && !SEARCH_METHODS.some((method) => method.value === value)) continue;
    if (typeof value === "number") {
      if (!Number.isInteger(value)) continue;
      const range = limits[key];
      if (Array.isArray(range) && range.length === 2) value = Math.min(range[1], Math.max(range[0], value));
    }
    merged[key] = value;
  }
  return merged;
}

export function cleanFilters(saved, documents) {
  const known = {
    jurisdiction: new Set(documents.map((d) => d.jurisdiction)),
    status: new Set(documents.map((d) => d.status)),
    doc_type: new Set(documents.map((d) => d.doc_type)),
    doc_id: new Set(documents.map((d) => d.doc_id)),
  };
  const out = { jurisdiction: [], status: [], doc_type: [], doc_id: [] };
  if (saved && typeof saved === "object") {
    for (const key of Object.keys(out)) {
      if (Array.isArray(saved[key])) out[key] = saved[key].filter((value) => known[key].has(value));
    }
  }
  return out;
}
