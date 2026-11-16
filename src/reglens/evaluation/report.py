"""Render the results of an evaluation run as Markdown tables (the text that goes into the report)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from reglens.evaluation.metrics import KS

LIMITATIONS = """\
- The gold set has {n} questions written by the project author from the indexed documents. It is large enough to show
  clear differences between configurations and too small to estimate small ones; read a difference of one question as noise.
- *Correctness* is checked by matching the facts a correct answer must state (dates, amounts, named concepts) with
  regular expressions. It rewards an answer that states them and cannot see a wrong statement that sits beside them;
  the judge scores (if present) and the list of failures below are the check on that.
- *Faithfulness* here is what the pipeline can verify without a language model: every claim carries a citation, no cited
  label is invented, and no figure is missing from the sources. It does not prove that a cited passage really supports
  its sentence.
- The judge is the same family of model that wrote the answers, which tends to favour its own phrasing.
- Language-model answers vary from run to run; one run per configuration was made.
"""


def pct(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def count(done: int | None, total: int | None) -> str:
    return "n/a" if done is None or not total else f"{done}/{total} ({done / total * 100:.0f}%)"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def retrieval_table(runs: Sequence[Mapping]) -> str:
    shown = [k for k in KS if k in (1, 3, 5, 8)]
    header = ["Configuration"] + [f"hit@{k}" for k in shown] + ["MRR", "evidence@8", "from gold docs@8", "s/question"]
    rows = []
    for run in runs:
        if "skipped" in run:
            rows.append([run["label"]] + ["skipped"] + [""] * (len(header) - 2))
            continue
        s = run["summary"]
        rows.append(
            [run["label"]]
            + [pct(s["hit"][str(k)]) for k in shown]
            + [num(s["mrr"], 3), pct(s["coverage"]["8"]), pct(s["doc_precision"]["8"]), num(s.get("seconds_per_question"), 2)]
        )
    return table(header, rows)


def category_table(run: Mapping) -> str:
    rows = []
    for category, s in run["summary"]["by_category"].items():
        rows.append([category, str(s["n"]), pct(s["hit"]["1"]), pct(s["hit"]["5"]), pct(s["hit"]["8"]), num(s["mrr"], 3)])
    return table(["Question type", "n", "hit@1", "hit@5", "hit@8", "MRR"], rows)


def generation_table(summaries: Mapping[str, Mapping], labels: Mapping[str, str]) -> str:
    header = ["Configuration", "correct", "abstained when it should", "grounded", "citation coverage", "invented citations", "unverified figures", "s/answer", "tokens in/out"]
    rows = []
    for name, s in summaries.items():
        tokens = "n/a" if s["input_tokens"] is None else f"{s['input_tokens']:.0f} / {num(s['output_tokens'], 0)}"
        rows.append([
            labels.get(name, name), count(s["correct_count"], s["answerable"]),
            count(s["abstain_count"], s["unanswerable"]) if s["unanswerable"] else "not asked",
            pct(s["grounded_rate"]) if s["grounded_rate"] is not None else "n/a", pct(s["citation_coverage"]),
            f"{s['invalid_citation_answers']} answers", f"{s['unverified_figure_answers']} answers", num(s["latency_s"], 1), tokens,
        ])
    return table(header, rows)


def judge_table(judged: Mapping[str, Sequence[Mapping]], labels: Mapping[str, str]) -> str:
    rows = []
    for name, items in judged.items():
        if items:
            avg = lambda key: sum(i[key] for i in items) / len(items)  # noqa: E731
            rows.append([labels.get(name, name), str(len(items)), num(avg("relevance")), num(avg("faithfulness")), num(avg("correctness"))])
    return table(["Configuration", "answers judged", "relevance (1-5)", "faithfulness (1-5)", "correctness (1-5)"], rows) if rows else ""


def _clip(text: str, limit: int = 700) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def render(results: Mapping) -> str:
    """The whole report. ``results`` is the dictionary the evaluate command saves as results.json."""
    out: list[str] = ["# RegLens evaluation", ""]
    meta = results["meta"]
    out += [
        f"Run on {meta['date']}. Index: {meta['index']}. Embedding model: {meta['embedding']}. "
        f"Language model: {meta.get('llm') or 'not used in this run'}.",
        f"Gold set: {meta['n_questions']} questions ({meta['n_answerable']} answerable from the {meta['n_documents']} indexed documents, "
        f"{meta['n_unanswerable']} that the documents cannot answer).",
        "",
    ]
    retrieval = results.get("retrieval", [])
    if retrieval:
        out += ["## 1. Retrieval: is the evidence among the retrieved passages?", ""]
        out += ["hit@k = the question has a relevant passage among the first k; evidence@8 = share of the expected evidence found in the first 8; "
                "from gold docs@8 = share of the first 8 passages that come from a document that holds the answer.", ""]
        groups = {
            "Search method": [r for r in retrieval if r.get("group") == "search" and "@" not in r["config"]],
            "Re-ranking": [r for r in retrieval if r["config"] in ("hybrid", "hybrid_rerank")],
            "Query transformation": [r for r in retrieval if r["config"] in ("hybrid", "hybrid_no_glossary", "hybrid_multiquery", "hybrid_hyde")],
            "Footnote passages": [r for r in retrieval if r["config"] in ("hybrid", "hybrid_no_notes")],
            "Embedding model and chunking": [r for r in retrieval if "@" in r["config"] or r["config"] == "hybrid"],
        }
        for title, runs in groups.items():
            if len(runs) > 1 or (title == "Search method" and runs):
                out += [f"### {title}", "", retrieval_table(runs), ""]
        skipped = [r for r in retrieval if "skipped" in r]
        if skipped:
            out += ["Not run:", ""] + [f"- {r['label']}: {r['skipped']}" for r in skipped] + [""]
        base = next((r for r in retrieval if r["config"] == "hybrid" and "summary" in r), None)
        if base:
            out += ["### The default (hybrid) by type of question", "", category_table(base), ""]
    agreement = results.get("store_agreement")
    if agreement:
        out += ["### Chroma and the exact NumPy search", "",
                f"On {agreement['questions']} questions the two stores returned on average {pct(agreement['mean_overlap'])} of the same first "
                f"{agreement['k']} passages and the same first passage for {pct(agreement['same_first_passage'])} of the questions.", ""]
    generation = results.get("generation")
    if generation and generation.get("summaries"):
        labels = generation["labels"]
        out += ["## 2. Answers: correctness, faithfulness, trade-offs", "", generation_table(generation["summaries"], labels), ""]
        out += ["*grounded* = every sentence cited, no invented citation, no figure missing from the sources. This is a strict test: a heading, a lead-in or a closing disclaimer without a citation makes an answer fail it, so it understates grounding and is best used to compare configurations, not as a score on its own. *tokens* are the average prompt and answer tokens per question.", ""]
        rescored = meta.get("rescored")
        if rescored:
            moved = rescored["changed"]
            out += [f"*Re-scored on {rescored['date']}: the saved answers were scored again with corrected checks (no model was asked again; every configuration "
                    f"with the same checks). The corrections: two fact patterns that rejected correct answers were widened, and the figure check now reads each "
                    f"source as the model saw it, heading included (title, date, section). {len(moved)} correctness verdict(s) changed"
                    + (": " + ", ".join(f"{m['config']} {m['id']} {'wrong to right' if m['after'] else 'right to wrong'}" for m in moved) if moved else "")
                    + f"; {rescored.get('figures_changed', 0)} answer(s) had their figure check changed.*", ""]
        judged = generation.get("judged")
        if judged:
            out += ["### Language-model judge", "", judge_table(judged, labels), ""]
        flagship = [r for r in generation.get("rows", []) if r["id"] in ("q04", "q05", "q06")]
        if flagship:
            out += ["## 3. The questions where an out-of-date model should fail", ""]
            for qid in ("q04", "q05", "q06"):
                for r in (x for x in flagship if x["id"] == qid):
                    verdict = "states the expected facts" if r["score"]["correct"] else ("could not be asked" if r["score"]["failed"] else "does NOT state the expected facts")
                    out += [f"**{qid}, {labels.get(r['config'], r['config'])}** - {verdict}.", "", f"> {_clip(r['answer'])}", ""]
        wrong = [r for r in generation.get("rows", []) if r["config"] == "rag" and not r["score"]["failed"] and not r["expect_abstain"] and not r["score"]["correct"]]
        wrong += [r for r in generation.get("rows", []) if r["config"] == "rag" and not r["score"]["failed"] and r["expect_abstain"] and not r["score"]["correct"]]
        out += ["## 4. Where RegLens was wrong", ""]
        if wrong:
            for r in wrong:
                why = "answered a question the documents cannot answer" if r["expect_abstain"] else f"missed expected fact group(s) {r['score']['missing_facts']}" + (" (it abstained)" if r["score"]["abstained"] else "")
                out += [f"- **{r['id']}** ({r['category']}): {why}. Answer: {_clip(r['answer'], 300)}"]
            out += [""]
        else:
            out += ["No gold question was answered incorrectly in this run.", ""]
    out += ["## Limits of this evaluation", "", LIMITATIONS.format(n=meta["n_questions"])]
    return "\n".join(out)
