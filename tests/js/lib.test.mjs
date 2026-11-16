// Tests of the pure helpers of the page.  Run:  node --test tests/js
import assert from "node:assert/strict";
import test from "node:test";

import * as lib from "../../src/reglens/web/static/lib.mjs";

const text = (v) => ({ t: "text", v });

// ------------------------------------------------------------------------------------------------ status and formatting

test("status values are named in plain words and grouped into three kinds", () => {
  assert.deepEqual(lib.statusInfo("in_force"), { label: "In force", kind: "law" });
  assert.equal(lib.statusInfo("final").kind, "guidance");
  for (const status of ["draft", "consultation", "proposed"]) assert.equal(lib.statusInfo(status).kind, "draft");
  assert.deepEqual(lib.statusInfo("weird"), { label: "weird", kind: "unknown" });
  assert.deepEqual(lib.statusInfo("", "status unknown"), { label: "status unknown", kind: "unknown" });
  assert.equal(lib.statusInfo(undefined).label, "Status unknown");
  assert.match(lib.statusInfo("draft").label, /not adopted/);
});

test("lists, counts and durations read naturally", () => {
  assert.equal(lib.joinList([]), "");
  assert.equal(lib.joinList(["a"]), "a");
  assert.equal(lib.joinList(["a", "b"]), "a and b");
  assert.equal(lib.joinList(["a", "b", "c"]), "a, b and c");
  assert.equal(lib.plural(1, "text", "texts"), "text");
  assert.equal(lib.plural(0, "text", "texts"), "texts");
  assert.equal(lib.formatSeconds(0.04), "0.0 s");
  assert.equal(lib.formatSeconds(3.14159), "3.1 s");
  assert.equal(lib.formatSeconds(12.4), "12 s");
  assert.equal(lib.formatSeconds(89.4), "89 s");
  assert.equal(lib.formatSeconds(65 + 90), "2 min 35 s");
  assert.equal(lib.formatSeconds(125), "2 min 05 s");
  assert.equal(lib.formatSeconds(-1), "");
  assert.equal(lib.formatSeconds(NaN), "");
  assert.equal(lib.formatCount(3813), "3,813");
  assert.equal(lib.formatCount(undefined), "");
});

// ------------------------------------------------------------------------------------------------ inline text

test("citations are found in every form the server accepts", () => {
  const cites = (s) => lib.parseInline(s).filter((t) => t.t === "cite").map((t) => t.labels);
  assert.deepEqual(cites("a [S1] b"), [["S1"]]);
  assert.deepEqual(cites("a [S1][S3]"), [["S1"], ["S3"]]);
  assert.deepEqual(cites("a [S1, S3]"), [["S1", "S3"]]);
  assert.deepEqual(cites("a [ S2 ; S10 ]"), [["S2", "S10"]]);
  assert.deepEqual(cites("not [S] and [1] and [Source 1] and [s1]"), []);
  assert.deepEqual(cites("[S12]"), [["S12"]]);
});

test("bold, italic and code are recognised, and stray markers stay as text", () => {
  assert.deepEqual(lib.parseInline("a **b** c"), [text("a "), { t: "strong", children: [text("b")] }, text(" c")]);
  assert.deepEqual(lib.parseInline("__b__"), [{ t: "strong", children: [text("b")] }]);
  assert.deepEqual(lib.parseInline("an *italic* word"), [text("an "), { t: "em", children: [text("italic")] }, text(" word")]);
  assert.deepEqual(lib.parseInline("an _italic_ word"), [text("an "), { t: "em", children: [text("italic")] }, text(" word")]);
  assert.deepEqual(lib.parseInline("use `Annex III` here"), [text("use "), { t: "code", v: "Annex III" }, text(" here")]);
  assert.deepEqual(lib.parseInline("snake_case_name stays"), [text("snake_case_name stays")]);
  assert.deepEqual(lib.parseInline("2 * 3 * 4"), [text("2 * 3 * 4")]);
  assert.deepEqual(lib.parseInline("unclosed **bold"), [text("unclosed **bold")]);
  assert.deepEqual(lib.parseInline("** spaced **"), [text("** spaced **")]);
  assert.deepEqual(lib.parseInline(""), []);
});

test("a citation can sit inside bold text, and the statement ignores the markers", () => {
  const tokens = lib.parseInline("**Notify within 72 hours [S1]** and then **document it** [S2].");
  assert.equal(tokens[0].t, "strong");
  assert.deepEqual(tokens[0].children.filter((t) => t.t === "cite").map((t) => t.labels), [["S1"]]);
  assert.equal(lib.plainText(tokens), "Notify within 72 hours and then document it .");
  assert.equal(lib.plainText(lib.parseInline("a   b\n c")), "a b c");
});

// ------------------------------------------------------------------------------------------------ blocks

test("paragraphs, bullets, numbered lists and headings become blocks", () => {
  const blocks = lib.parseAnswer("Intro line one\ncontinues here [S1].\n\n- First [S2]\n- Second\n  runs on\n\n3. Third\n4. Fourth\n\n## Heading\nAfter.");
  assert.deepEqual(blocks.map((b) => b.type), ["p", "ul", "ol", "heading", "p"]);
  assert.equal(lib.plainText(blocks[0].inline), "Intro line one continues here .");
  assert.deepEqual(blocks[1].items.map(lib.plainText), ["First", "Second runs on"]);
  assert.equal(blocks[2].start, 3);
  assert.equal(lib.plainText(blocks[3].inline), "Heading");
});

test("the list styles the models use are all understood", () => {
  const blocks = lib.parseAnswer("*   **Duty:** notify [S1]\n*   **Time:** 72 hours [S2]\n\n• dot bullet\n\n1) paren number");
  assert.deepEqual(blocks.map((b) => b.type), ["ul", "ul", "ol"]);
  assert.equal(blocks[0].items.length, 2);
  assert.equal(blocks[0].items[0][0].t, "strong");
  assert.equal(lib.plainText(blocks[0].items[1]), "Time: 72 hours");
});

test("a bold line is a paragraph, not a bullet", () => {
  const blocks = lib.parseAnswer("**Summary** of the rule.\n**Second** line");
  assert.deepEqual(blocks.map((b) => b.type), ["p"]);
});

test("line endings, blank input and odd values do not break the parser", () => {
  assert.deepEqual(lib.parseAnswer(""), []);
  assert.deepEqual(lib.parseAnswer(null), []);
  assert.deepEqual(lib.parseAnswer(undefined), []);
  assert.deepEqual(lib.parseAnswer("\n\n  \n"), []);
  assert.equal(lib.parseAnswer("a\r\nb\r\n\r\nc").length, 2);
  assert.equal(lib.parseAnswer("- only a bullet").length, 1);
});

test("every citation knows the statement it stands under", () => {
  const blocks = lib.parseAnswer("Notify within 72 hours [S1].\n\n- Document the facts [S2][S3]\n- Tell the data subjects [S2]");
  const found = lib.citationContexts(blocks);
  assert.deepEqual(found.map((c) => c.label), ["S1", "S2", "S3", "S2"]);
  assert.equal(found[0].claim, "Notify within 72 hours .");
  assert.equal(found[1].claim, "Document the facts");
  assert.equal(found[3].claim, "Tell the data subjects");
  const nested = lib.citationContexts(lib.parseAnswer("**Bold [S4]** end"));
  assert.deepEqual(nested.map((c) => c.label), ["S4"]);
});

// ------------------------------------------------------------------------------------------------ figures

const keys = (s) => lib.findFigures(s).map((f) => f.key);

test("a date has one key however it is written", () => {
  assert.deepEqual(keys("from 2 December 2027"), ["date:2027-12-02"]);
  assert.deepEqual(keys("from December 2, 2027"), ["date:2027-12-02"]);
  assert.deepEqual(keys("from 2nd December 2027"), ["date:2027-12-02"]);
  assert.deepEqual(keys("from 2027-12-02"), ["date:2027-12-02"]);
  assert.deepEqual(keys("by 02 december 2027"), ["date:2027-12-02"]);
  assert.deepEqual(keys("in August 2026"), ["date:2026-08"]);
  assert.deepEqual(keys("on 2 August 2026 and 2 August 2028"), ["date:2026-08-02", "date:2028-08-02"]);
});

test("periods, percentages and amounts are normalised", () => {
  assert.deepEqual(keys("within 72 hours"), ["period:72 hour"]);
  assert.deepEqual(keys("a 72-hour window"), ["period:72 hour"]);
  assert.deepEqual(keys("30 working days"), ["period:30 working day"]);
  assert.deepEqual(keys("1 year"), ["period:1 year"]);
  assert.deepEqual(keys("7 % of turnover"), ["percent:7%"]);
  assert.deepEqual(keys("7%"), ["percent:7%"]);
  assert.deepEqual(keys("7.5 per cent"), ["percent:7.5%"]);
  assert.deepEqual(keys("EUR 35 000 000"), ["money:EUR 35000000"]);
  assert.deepEqual(keys("€35 million"), ["money:EUR 35000000"]);
  assert.deepEqual(keys("35 million euros"), ["money:EUR 35000000"]);
  assert.deepEqual(keys("$1,500"), ["money:USD 1500"]);
  assert.deepEqual(keys("USD 2.5 million"), ["money:USD 2500000"]);
  assert.deepEqual(keys("£1.234,50"), ["money:GBP 1234.5"]);
});

test("years stand alone only when they are not part of a date or an identifier", () => {
  assert.deepEqual(keys("applies in 2027"), ["year:2027"]);
  assert.deepEqual(keys("Regulation (EU) 2024/1689 and 2016/679"), []);
  assert.deepEqual(keys("from 2 December 2027"), ["date:2027-12-02"]); // not also year:2027
  assert.deepEqual(keys("case 12345 and 99"), []);
  assert.deepEqual(keys("ISO-2022 code"), []);
});

test("figures are reported in reading order with their position", () => {
  const found = lib.findFigures("In 2026, notify within 72 hours; fines reach 7 % by 2 August 2028.");
  assert.deepEqual(found.map((f) => f.text), ["2026", "72 hours", "7 %", "2 August 2028"]);
  assert.ok(found.every((f, i) => i === 0 || found[i - 1].start < f.start));
  assert.equal("In 2026, notify within 72 hours; fines reach 7 % by 2 August 2028.".slice(found[1].start, found[1].end), "72 hours");
  assert.deepEqual(lib.findFigures(""), []);
  assert.deepEqual(lib.findFigures(undefined), []);
});

test("plain numbers of two digits or more are found, with their separators removed", () => {
  const found = lib.findNumbers("Article 113 and 35 000 000 and 5 and 1,500 and 3.5");
  assert.deepEqual(found.map((n) => n.digits), ["113", "35000000", "1500"]);
});

test("the comparison lists every figure and where it appears", () => {
  const rows = lib.compareFigures("It applies from 2 December 2027 within 72 hours.", "It applies from 2 August 2026 within 72 hours.");
  assert.deepEqual(rows.map((r) => [r.label, r.first, r.second]), [
    ["2 August 2026", false, true],
    ["2 December 2027", true, false],
    ["72 hours", true, true],
  ]);
  assert.deepEqual(lib.compareFigures("none here", "nor here"), []);
  assert.deepEqual(lib.compareFigures("from December 2, 2027", "from 2 December 2027").map((r) => [r.first, r.second]), [[true, true]]);
});

test("a figure is traced to the sources that contain it", () => {
  const sources = [
    { label: "S1", text: "It shall apply from 2 August 2026." },
    { label: "S2", text: "applies from 2 December 2027 for Annex III" },
    { label: "S3", text: "applies from December 2, 2027" },
  ];
  assert.deepEqual(lib.sourcesWithFigure("date:2027-12-02", sources), ["S2", "S3"]);
  assert.deepEqual(lib.sourcesWithFigure("date:2030-01-01", sources), []);
});

// ------------------------------------------------------------------------------------------------ passages

const PASSAGE = "Article 113 - Entry into force and application\n\nThis Regulation shall enter into force on the twentieth day. It shall apply from 2 August 2026. However: (a) Chapters I and II shall apply from 2 February 2025; (b) Chapter III shall apply from 2 August 2025.";

test("sentences are located by offset, so a mark can be placed in the original text", () => {
  const spans = lib.splitSentences(PASSAGE);
  const sentences = spans.map((s) => PASSAGE.slice(s.start, s.end));
  assert.deepEqual(sentences.slice(0, 3), [
    "Article 113 - Entry into force and application",
    "This Regulation shall enter into force on the twentieth day.",
    "It shall apply from 2 August 2026.",
  ]);
  assert.ok(sentences.every((s) => s === s.trim() && s.length > 0));
  assert.deepEqual(lib.splitSentences(""), []);
  assert.deepEqual(lib.splitSentences("   \n \n"), []);
  assert.equal(lib.splitSentences("One sentence only").length, 1);
});

test("the closest sentence is the one that shares the most words and figures with the statement", () => {
  const spans = lib.splitSentences(PASSAGE);
  const closest = (claim) => {
    const i = lib.closestSentence(PASSAGE, claim);
    return i < 0 ? null : PASSAGE.slice(spans[i].start, spans[i].end);
  };
  assert.equal(closest("The Regulation applies from 2 August 2026"), "It shall apply from 2 August 2026.");
  assert.equal(closest("Chapters I and II apply from 2 February 2025"), "However: (a) Chapters I and II shall apply from 2 February 2025;");
  assert.equal(lib.closestSentence(PASSAGE, "Penalties for breaching the transparency duties"), -1);
  assert.equal(lib.closestSentence(PASSAGE, ""), -1);
  assert.equal(lib.closestSentence("", "anything at all"), -1);
});

test("an excerpt is a window around the closest sentence, with a flag for what was left out", () => {
  const long = Array.from({ length: 12 }, (_, i) => `Sentence number ${i + 1} says something quite ordinary about nothing.`).join(" ");
  const middle = lib.excerptAround(long, 6, 200);
  assert.ok(middle.text.includes("Sentence number 7 "));
  assert.ok(middle.text.length <= 200 && middle.before && middle.after);
  assert.equal(long.slice(middle.start, middle.end), middle.text);
  const first = lib.excerptAround(long, -1, 200);
  assert.ok(first.text.startsWith("Sentence number 1 ") && !first.before && first.after);
  const all = lib.excerptAround("Short text.", 0, 200);
  assert.deepEqual([all.text, all.before, all.after], ["Short text.", false, false]);
  const huge = lib.excerptAround("word ".repeat(200).trim() + ".", 0, 100);
  assert.ok(huge.text.length <= 100 && huge.after && !huge.text.endsWith(" "));
  assert.deepEqual(lib.excerptAround("", -1), { start: 0, end: 0, text: "", before: false, after: false });
});

test("marks may overlap and every piece of the text keeps its own classes", () => {
  const pieces = lib.segments("abcdefghij", [
    { start: 2, end: 8, cls: "closest" },
    { start: 4, end: 6, cls: "figure" },
  ]);
  assert.deepEqual(pieces.map((p) => p.text), ["ab", "cd", "ef", "gh", "ij"]);
  assert.deepEqual(pieces.map((p) => p.classes), [[], ["closest"], ["closest", "figure"], ["closest"], []]);
  assert.deepEqual(lib.segments("abc", []), [{ text: "abc", classes: [] }]);
  assert.deepEqual(lib.segments("", []), []);
  assert.equal(lib.segments("abc", [{ start: -5, end: 99, cls: "x" }]).map((p) => p.text).join(""), "abc");
  assert.equal(lib.segments("abcdef", [{ start: 1, end: 4, cls: "x" }, { start: 3, end: 5, cls: "y" }]).map((p) => p.text).join(""), "abcdef");
});

test("a passage is marked where it shares figures and its closest sentence with the statement", () => {
  const marks = lib.passageMarks(PASSAGE, "The Regulation applies from 2 August 2026");
  const marked = marks.map((m) => [m.cls, PASSAGE.slice(m.start, m.end)]);
  assert.ok(marked.some(([cls, s]) => cls === "figure" && s === "2 August 2026"));
  assert.ok(marked.some(([cls, s]) => cls === "closest" && s === "It shall apply from 2 August 2026."));
  assert.ok(!marked.some(([, s]) => s === "2 February 2025"));
  const window = lib.passageMarks(PASSAGE, "The Regulation applies from 2 August 2026", 10, 60);
  assert.ok(window.every((m) => m.start >= -10 && m.end <= 200));
  assert.deepEqual(lib.passageMarks(PASSAGE, "nothing in common whatsoever"), []);
  const number = lib.passageMarks("Article 113 applies.", "See Article 113 for details");
  assert.deepEqual(number.map((m) => m.cls), ["figure", "closest"]);
});

// ------------------------------------------------------------------------------------------------ summaries

test("the sentence about authority names binding, guidance and not-adopted sources", () => {
  const a = (...status) => lib.authority(status.map((s) => ({ status: s }))).sentence;
  assert.equal(a("in_force"), "Rests on 1 binding text.");
  assert.equal(a("in_force", "in_force", "final", "draft"), "Rests on 2 binding texts, 1 text of final guidance and 1 text that is not adopted.");
  assert.equal(a("draft", "proposed"), "Rests on 2 texts that are not adopted.");
  assert.equal(a("final"), "Rests on 1 text of final guidance.");
  assert.equal(a("mystery"), "Rests on 1 text of unknown status.");
  assert.equal(a(), "");
  assert.deepEqual(lib.authority([{ status: "in_force" }, { status: "draft" }]), { law: 1, guidance: 0, draft: 1, unknown: 0, total: 2, sentence: "Rests on 1 binding text and 1 text that is not adopted." });
});

test("the lead of the authority sentence can be changed", () => {
  assert.equal(lib.authority([{ status: "final" }], "The best matches are").sentence, "The best matches are 1 text of final guidance.");
  assert.equal(lib.authority([], "The best matches are").sentence, "");
});

const DOCS = [
  { doc_id: "a", jurisdiction: "EU", status: "in_force", doc_type: "regulation" },
  { doc_id: "b", jurisdiction: "EU", status: "draft", doc_type: "guidelines" },
  { doc_id: "c", jurisdiction: "US", status: "final", doc_type: "framework" },
  { doc_id: "d", jurisdiction: "US", status: "in_force", doc_type: "statute" },
];

test("filters combine with AND between kinds and OR inside a kind", () => {
  const count = (filters) => DOCS.filter((d) => lib.matchesFilters(d, filters)).length;
  assert.equal(count({}), 4);
  assert.equal(count(undefined), 4);
  assert.equal(count({ jurisdiction: [] }), 4);
  assert.equal(count({ jurisdiction: ["EU"] }), 2);
  assert.equal(count({ jurisdiction: ["EU", "US"] }), 4);
  assert.equal(count({ jurisdiction: ["EU"], status: ["in_force"] }), 1);
  assert.equal(count({ doc_id: ["a", "d"] }), 2);
  assert.deepEqual(lib.activeFilters({ jurisdiction: ["EU"], status: [], doc_id: undefined, x: "y" }), { jurisdiction: ["EU"] });
});

test("the scope is described in a sentence", () => {
  assert.deepEqual(lib.scopeSummary({}, DOCS), { active: false, count: 4, text: "All 4 documents" });
  assert.equal(lib.scopeSummary({ jurisdiction: ["EU"] }, DOCS).text, "EU: 2 documents");
  assert.equal(lib.scopeSummary({ jurisdiction: ["EU", "US"], status: ["in_force", "final"] }, DOCS).text, "EU and US, in force and final guidance: 3 documents");
  assert.equal(lib.scopeSummary({ doc_id: ["a"] }, DOCS).text, "1 chosen document");
  assert.equal(lib.scopeSummary({ doc_id: ["a", "b", "d"] }, DOCS).text, "3 chosen documents");
  assert.equal(lib.scopeSummary({ jurisdiction: ["EU"], doc_id: ["a", "c"] }, DOCS).text, "EU, 2 chosen documents: 1 document");
  assert.equal(lib.scopeSummary({ jurisdiction: ["UK"] }, DOCS).text, "UK: 0 documents");
});

test("an excerpt is cut at a word and says it was cut", () => {
  assert.equal(lib.excerpt("short"), "short");
  assert.equal(lib.excerpt("  many   spaces \n here "), "many spaces here");
  const cut = lib.excerpt("alpha beta gamma delta epsilon zeta eta theta", 20);
  assert.ok(cut.endsWith("…") && cut.length <= 21 && !cut.includes("  "));
  assert.equal(lib.excerpt(undefined), "");
});

// ------------------------------------------------------------------------------------------------ export and settings

test("the export is a readable document with the sources and the checks", () => {
  const answer = {
    mode: "rag",
    text: "Notify within 72 hours [S1].",
    invalid_citations: ["S9"],
    unverified_figures: ["14"],
    sources: [
      { label: "S1", cited: true, title: "GDPR", section: "Article 33", pages: "p. 52", jurisdiction: "EU", status: "in_force", url: "https://example.org/gdpr" },
      { label: "S2", cited: false, title: "Other", section: "", pages: "", jurisdiction: "US", status: "draft", url: "" },
    ],
  };
  const md = lib.toMarkdown(
    [{ question: "How fast?", scope: "All 34 documents", answer, baseline: { text: "Within 3 days.", model: "m-1" } }],
    { exportedAt: "2026-11-15", passages: 3813, documents: 34, model: "gemini/m-1" },
  );
  assert.match(md, /^# RegLens conversation\n\nExported 2026-11-15\. 3,813 passages from 34 documents were searchable\. Language model: gemini\/m-1\. Research support, not legal advice\.\n/);
  assert.match(md, /## 1\. How fast\?\n\nSearch scope: All 34 documents\.\n/);
  assert.match(md, /\*\*Answer from the documents\*\*\n\nNotify within 72 hours \[S1\]\.\n/);
  assert.match(md, /Sources:\n- \[S1\] GDPR, Article 33, p\. 52 \(EU, In force\) https:\/\/example\.org\/gdpr\n/);
  assert.ok(!md.includes("[S2] Other"), "only cited sources are listed when there are any");
  assert.match(md, /Checks: citations that do not exist: S9; figures not found in the sources: 14\./);
  assert.match(md, /\*\*The model alone, without the documents\*\* \(m-1\)\n\nWithin 3 days\.\n/);
  assert.ok(!md.includes("\n\n\n") && md.endsWith("\n"));
});

test("passages-only answers list all their sources", () => {
  const md = lib.toMarkdown([{ question: "q", answer: { mode: "extractive", text: "shown", sources: [{ label: "S1", cited: false, title: "T", status: "final" }] } }]);
  assert.match(md, /\*\*Matching passages\*\* \(no model answer\)/);
  assert.match(md, /- \[S1\] T \(Final guidance\)/);
});

test("saved settings are used only when they still have the right shape", () => {
  const defaults = { search: "hybrid", top_k: 8, rerank: false };
  assert.deepEqual(lib.mergeSettings(defaults, { search: "dense", top_k: 5, rerank: true, extra: 1 }), { search: "dense", top_k: 5, rerank: true });
  assert.deepEqual(lib.mergeSettings(defaults, { search: 3, top_k: "5" }), defaults);
  assert.deepEqual(lib.mergeSettings(defaults, null), defaults);
  assert.deepEqual(lib.mergeSettings(defaults, "junk"), defaults);
});

test("a saved search method the server does not know is dropped, so old browser data cannot break every question", () => {
  const defaults = { search: "hybrid", top_k: 8 };
  assert.deepEqual(lib.mergeSettings(defaults, { search: "bm25", top_k: 5 }), { search: "hybrid", top_k: 5 });
  assert.deepEqual(lib.mergeSettings(defaults, { search: "lexical" }), { search: "lexical", top_k: 8 });
});

test("saved numbers are pulled inside the server's limits, and fractions are refused", () => {
  const defaults = { top_k: 8, multi_query: 0 };
  const limits = { top_k: [1, 12], multi_query: [0, 3] };
  assert.deepEqual(lib.mergeSettings(defaults, { top_k: 99, multi_query: -4 }, limits), { top_k: 12, multi_query: 0 });
  assert.deepEqual(lib.mergeSettings(defaults, { top_k: 0 }, limits), { top_k: 1, multi_query: 0 });
  assert.deepEqual(lib.mergeSettings(defaults, { top_k: 2.5, multi_query: Number.NaN }, limits), defaults);
  assert.deepEqual(lib.mergeSettings(defaults, { top_k: 99 }), { top_k: 99, multi_query: 0 }, "without limits the value is kept as it is");
});

test("the search methods are the three the server knows, each with a name and a sentence", () => {
  assert.deepEqual(lib.SEARCH_METHODS.map((method) => method.value), ["hybrid", "dense", "lexical"]);
  for (const method of lib.SEARCH_METHODS) {
    assert.ok(method.name.length > 3 && method.detail.endsWith("."), method.value);
  }
});

test("saved filters drop documents and values that no longer exist", () => {
  const cleaned = lib.cleanFilters({ jurisdiction: ["EU", "Mars"], status: "final", doc_id: ["a", "zzz"] }, DOCS);
  assert.deepEqual(cleaned, { jurisdiction: ["EU"], status: [], doc_type: [], doc_id: ["a"] });
  assert.deepEqual(lib.cleanFilters(null, DOCS), { jurisdiction: [], status: [], doc_type: [], doc_id: [] });
});

// ------------------------------------------------------------------------------------------------ how an answer was made

const OPTIONS = {
  retrieval: { mode: "hybrid", top_k: 8, candidates: 40, rerank: false, rerank_pool: 20, where: null, include_notes: true },
  transform: { glossary: true, condense: true, multi_query: 0, hyde: false },
  max_context_words: 3000,
};

test("the settings of an answer are described in plain words", () => {
  const rows = Object.fromEntries(lib.describeOptions(OPTIONS).map((row) => [row.label, row.value]));
  assert.equal(rows["Search method"], "Meaning and keywords combined");
  assert.equal(rows["Passages kept"], "8");
  assert.equal(rows["Re-ranking"], "off");
  assert.equal(rows["Footnotes"], "included");
  assert.equal(rows["Scope"], "All documents");
  assert.equal(rows["Rephrased searches"], "0");
  assert.equal(rows["Search with a hypothetical answer"], "off");
});

test("a scope with filters names them, and unknown settings do not break the description", () => {
  const where = { jurisdiction: ["EU", "US"], status: "in_force", doc_id: ["a", "b", "c"] };
  const options = { ...OPTIONS, retrieval: { ...OPTIONS.retrieval, where, mode: "dense", rerank: true, include_notes: false } };
  const rows = Object.fromEntries(lib.describeOptions(options).map((row) => [row.label, row.value]));
  assert.equal(rows["Scope"], "EU and US, in force, 3 chosen documents");
  assert.equal(rows["Search method"], "Meaning only");
  assert.equal(rows["Re-ranking"], "on");
  assert.equal(rows["Footnotes"], "left out");
  assert.deepEqual(lib.describeOptions({}), []);
  assert.deepEqual(lib.describeOptions(null), []);
  assert.equal(lib.describeOptions({ ...OPTIONS, retrieval: { ...OPTIONS.retrieval, mode: "odd" } })[0].value, "odd");
  assert.equal(lib.describeOptions({ ...OPTIONS, retrieval: { ...OPTIONS.retrieval, mode: "lexical" } })[0].value, "Keywords only");
});

test("timings become one sentence and tolerate missing steps", () => {
  assert.equal(
    lib.describeTimings({ transform: 0.02, retrieve: 0.41, generate: 3.1, total: 3.6 }),
    "Preparing the question 0.0 s, searching 0.4 s, writing the answer 3.1 s. 3.6 s in total.",
  );
  assert.equal(lib.describeTimings({ total: 12.4 }), "12 s in total.");
  assert.equal(lib.describeTimings({}), "");
  assert.equal(lib.describeTimings(undefined), "");
});

test("citation markers are removed from an answer without leaving gaps", () => {
  assert.equal(lib.stripCitations("Notify within 72 hours [S1]. See also [S2][S3]."), "Notify within 72 hours. See also.");
  assert.equal(lib.stripCitations("Both apply [S1, S2] today.\n- Fines [S4]"), "Both apply today.\n- Fines");
  assert.equal(lib.stripCitations("No markers here."), "No markers here.");
  assert.equal(lib.stripCitations(undefined), "");
  assert.equal(lib.stripCitations("Keep [this] and [S] and [1]."), "Keep [this] and [S] and [1].");
});
