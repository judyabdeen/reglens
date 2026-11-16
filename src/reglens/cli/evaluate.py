"""Command line:  python scripts/evaluate.py  -  measure retrieval and answer quality on the gold questions (assignment Step 5).

What runs by default needs no language model and costs nothing: every gold question is retrieved under several settings
and scored. ``--generate`` adds the answers (it calls the language model: about one request per question and
configuration, paced), ``--judge`` adds a model's 1-5 reading of those answers.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from reglens.config import PROJECT_ROOT, ConfigError, Settings, load_settings
from reglens.console import make_console_safe, short_duration
from reglens.embeddings import EmbeddingError, make_embedder
from reglens.evaluation.experiments import (
    GENERATION_CONFIGS,
    MAIN_CONFIGS,
    RetrievalConfig,
    memory_retriever,
    recheck_figures,
    rescore,
    run_generation,
    run_retrieval,
    store_agreement,
    summarise_generation,
    variant_config,
)
from reglens.evaluation.gold import GoldError, GoldQuestion, check_corpus, load_gold
from reglens.evaluation.judge import judge
from reglens.evaluation.report import render
from reglens.indexing import IndexNotReady
from reglens.generation.prompts import source_header
from reglens.ingest.chunking import read_chunks
from reglens.retrieval.rerank import RerankError
from reglens.runtime import Runtime, build_runtime, load_sources
from reglens.vectorstore import VectorStoreError
from reglens.vectorstore.records import make_records

DEFAULT_GOLD = PROJECT_ROOT / "eval" / "gold.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "eval" / "results" / "latest"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate", description="Evaluate retrieval and answers on the gold questions.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="the gold questions (default eval/gold.jsonl)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="folder for the results (default eval/results/latest)")
    parser.add_argument("--validate", action="store_true", help="only check that the gold questions are answerable by the corpus, then stop")
    parser.add_argument("--no-rerank", action="store_true", help="skip the re-ranking comparison (it downloads a ~90 MB model once)")
    parser.add_argument("--embedding-baseline", action="store_true", help="also compare with the hashing baseline (no neural model)")
    parser.add_argument("--fixed-chunks", type=Path, metavar="CHUNKS.jsonl", help="also evaluate this fixed-size-chunk file (made with ingest.py --strategy fixed)")
    parser.add_argument("--stores", action="store_true", help="also compare Chroma with the exact NumPy search (needs build_index.py --backend both)")
    parser.add_argument("--generate", action="store_true", help="also ask the language model (llm alone vs RegLens)")
    parser.add_argument("--extra", action="store_true", help="with --generate: also answers with re-ranking and with multi-query; with retrieval: multi-query and HyDE")
    parser.add_argument("--judge", action="store_true", help="with --generate: also a language-model judge (1-5 scale)")
    parser.add_argument("--rescore", action="store_true", help="score the answers already saved in --out again with the current gold set (no model calls, no index needed)")
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N gold questions (a quick trial)")
    parser.add_argument("--min-interval", type=float, default=4.0, help="seconds between language-model requests (default 4; raise it on a free quota)")
    parser.add_argument("--fresh", action="store_true", help="forget answers saved by an earlier run in --out")
    parser.add_argument("--provider", choices=("gemini", "openai", "anthropic"), help="language model provider (default from .env)")
    parser.add_argument("--model", help="language model name (default from .env)")
    parser.add_argument("--env-file", type=Path, help="use this .env file")
    return parser


def _read_passages(settings: Settings) -> list[tuple[str, str]]:
    return [(chunk.doc_id, chunk.text) for chunk in read_chunks(settings.chunks_path)]


def validate(questions: Sequence[GoldQuestion], settings: Settings, write: Callable[[str], None]) -> bool:
    problems = check_corpus(questions, _read_passages(settings))
    if problems:
        write("The gold questions do not match the corpus:")
        for problem in problems:
            write(f"  - {problem}")
        return False
    answerable = sum(q.answerable for q in questions)
    write(f"Gold set checked against the {settings.chunks_path.name}: {len(questions)} questions ({answerable} answerable, "
          f"{len(questions) - answerable} unanswerable); every expected passage and fact exists in the documents.")
    return True


def _progress(label: str, write: Callable[[str], None]) -> Callable[[int, int], None]:
    state = {"next": 0.25}

    def update(done: int, total: int) -> None:
        if total and done / total >= state["next"] and done < total:
            write(f"      {label}: {done}/{total}")
            state["next"] += 0.25

    return update


def _chunk_stats(path: Path) -> dict:
    chunks = read_chunks(path)
    words = [c.n_words for c in chunks]
    return {"file": str(path), "chunks": len(chunks), "mean_words": round(sum(words) / len(words), 1) if words else None}


def main(
    argv: Sequence[str] | None = None,
    *,
    build: Callable[..., Runtime] = build_runtime,
    settings_loader: Callable[..., Settings] = load_settings,
    write: Callable[[str], None] = print,
) -> int:
    make_console_safe()
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    try:
        settings = settings_loader(env_file=args.env_file)
        questions = load_gold(args.gold)
    except (ConfigError, GoldError) as exc:
        write(f"Problem: {exc}")
        return 2
    if args.rescore:
        return _rescore(args.out, questions, settings, write)
    if args.limit:
        questions = questions[: args.limit]
    try:
        sound = validate(questions, settings, write)
    except (OSError, ValueError) as exc:
        write(f"Problem: the chunks file could not be read ({exc}). Run: python scripts/ingest.py")
        return 1
    if not sound:
        return 1
    if args.validate:
        return 0
    if args.judge and not args.generate:
        write("Problem: --judge needs --generate")
        return 2

    write("\nOpening the index ...")
    try:
        runtime = build(settings, provider=args.provider, model=args.model, min_interval=args.min_interval)
    except IndexNotReady as exc:
        write(f"Problem: {exc}")
        return 1
    except (EmbeddingError, VectorStoreError, OSError) as exc:
        write(f"Problem: the index could not be opened: {exc}")
        return 1
    if (args.generate or args.extra) and not runtime.has_llm:
        write(f"Problem: this needs a language model, but none is available: {runtime.llm_problem}")
        return 1

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for name in ("generation.jsonl", "judge.jsonl"):
            (out / name).unlink(missing_ok=True)

    retrievers = {"main": runtime.retriever}
    configs: list[RetrievalConfig] = [c for c in MAIN_CONFIGS if (args.extra or not c.needs_llm) and not (args.no_rerank and c.rerank)]
    extras: list[dict] = []
    sources = load_sources(settings)
    if args.embedding_baseline:
        write("Building the hashing-baseline index in memory ...")
        retrievers["hashing"] = memory_retriever(read_chunks(settings.chunks_path), sources, make_embedder("hashing"))
        configs += [
            variant_config(MAIN_CONFIGS[1], "hashing", "Meaning only, hashing baseline (no neural model)", "embedding"),
            variant_config(MAIN_CONFIGS[2], "hashing", "Hybrid, hashing baseline (no neural model)", "embedding"),
        ]
    if args.fixed_chunks:
        write(f"Embedding the fixed-size chunks of {args.fixed_chunks} in memory (a few minutes) ...")
        try:
            retrievers["fixed"] = memory_retriever(read_chunks(args.fixed_chunks), sources, runtime.embedder, progress=_progress("embedding", write))
        except (OSError, ValueError, EmbeddingError) as exc:
            write(f"Problem: the fixed-chunk file could not be used: {exc}")
            return 1
        configs += [variant_config(MAIN_CONFIGS[2], "fixed", "Hybrid, fixed-size chunks (instead of structure-aware)", "chunking")]
        extras.append({"chunking": {"structure": _chunk_stats(settings.chunks_path), "fixed": _chunk_stats(args.fixed_chunks)}})

    write(f"\nRetrieval: {len([q for q in questions if q.answerable])} answerable questions x {len(configs)} configurations")
    retrieval: list[dict] = []
    for config in configs:
        write(f"  {config.label}")
        try:
            result = run_retrieval(retrievers[config.index], questions, config, llm=runtime.pipeline.llm, progress=_progress(config.name, write))
        except (EmbeddingError, VectorStoreError) as exc:
            write(f"Problem: {exc}")
            return 1
        retrieval.append(result)
        if "skipped" in result:
            write(f"      skipped: {result['skipped']}")

    results: dict = {
        "meta": {
            "date": date.today().isoformat(), "index": f"{runtime.index.n_records} passages from {runtime.index.n_documents} documents ({runtime.index.backend})",
            "embedding": runtime.index.embedding_model, "llm": f"{runtime.llm_provider}/{runtime.llm_model}" if (args.generate and runtime.has_llm) else None,
            "n_questions": len(questions), "n_answerable": sum(q.answerable for q in questions),
            "n_unanswerable": sum(not q.answerable for q in questions), "n_documents": runtime.index.n_documents,
            "python": platform.python_version(), "platform": platform.platform(),
        },
        "retrieval": retrieval,
        "extras": extras,
    }

    if args.stores:
        write("\nComparing Chroma with the exact NumPy search ...")
        try:
            a, b = build(settings, backend="chroma"), build(settings, backend="numpy")
            results["store_agreement"] = store_agreement(a.retriever, b.retriever, questions)
        except (IndexNotReady, EmbeddingError, VectorStoreError, OSError) as exc:
            write(f"  skipped: {exc}  (build both with: python scripts/build_index.py --backend both)")

    if args.generate:
        names = ["llm_only", "rag"] + (["rag_rerank", "rag_multiquery"] if args.extra else [])
        gen_configs = [GENERATION_CONFIGS[n] for n in names]
        write(f"\nAnswers: {len(questions)} questions x {len(gen_configs)} configurations (about {args.min_interval:.0f} s apart; saved as they arrive)")
        try:
            rows, stopped = run_generation(runtime.pipeline, questions, gen_configs, out / "generation.jsonl", progress=lambda m: write(f"  {m}"))
        except (RerankError, EmbeddingError, VectorStoreError) as exc:
            write(f"Problem: {exc}")
            return 1
        generation: dict = {"labels": {c.name: c.label for c in gen_configs}, "rows": rows, "summaries": summarise_generation(rows)}
        if stopped:
            write(f"\nStopped early: {stopped}")
            generation["stopped"] = stopped
        if args.judge and not stopped:
            generation["judged"] = _judge_all(runtime, questions, rows, out / "judge.jsonl", write)
        results["generation"] = generation

    text = render(results)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    write("\n" + text)
    write(f"\nSaved: {out / 'report.md'} and {out / 'results.json'}  ({short_duration(time.perf_counter() - started)})")
    return 3 if results.get("generation", {}).get("stopped") else 0


def _judge_all(runtime: Runtime, questions: Sequence[GoldQuestion], rows: Sequence[dict], path: Path, write: Callable[[str], None]) -> dict:
    by_id = {q.id: q for q in questions}
    saved: dict[tuple[str, str], dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                saved[(item["config"], item["id"])] = item
    write("\nJudging the answers ...")
    judged: dict[str, list[dict]] = {}
    with path.open("a", encoding="utf-8") as sink:
        for row in rows:
            if row["score"]["failed"] or row["expect_abstain"]:
                continue
            key = (row["config"], row["id"])
            item = saved.get(key)
            if item is None:
                question = by_id[row["id"]]
                verdict = judge(runtime.pipeline.llm, question.question, row["answer"], row["source_texts"], [question.expected])  # type: ignore[arg-type]
                if verdict is None:
                    write(f"  {row['config']} {row['id']}: the judge gave no usable score")
                    continue
                item = {"config": row["config"], "id": row["id"], "relevance": verdict.relevance, "faithfulness": verdict.faithfulness,
                        "correctness": verdict.correctness, "reason": verdict.reason}
                sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                sink.flush()
            judged.setdefault(row["config"], []).append(item)
    return judged


def _rescore(out: Path, questions: Sequence[GoldQuestion], settings: Settings, write: Callable[[str], None]) -> int:
    results_path, answers_path = out / "results.json", out / "generation.jsonl"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        write(f"Problem: nothing to re-score in {out} ({exc}). Run: python scripts/evaluate.py --generate")
        return 1
    generation = results.get("generation")
    if not generation:
        write("Problem: results.json has no answers. Run: python scripts/evaluate.py --generate")
        return 1
    changed = rescore(rows, questions)
    figures: list[dict] = []
    try:
        sources = load_sources(settings)
        headings = {r.id: source_header(r.metadata, r.id) for r in make_records(read_chunks(settings.chunks_path), sources)}
        figures = recheck_figures(rows, questions, headings)
    except (OSError, ValueError) as exc:
        write(f"Note: the figure check was not repeated ({exc}).")
    generation["rows"], generation["summaries"] = rows, summarise_generation(rows)
    earlier = results["meta"].get("rescored") or {}  # a second pass must not erase what the first one changed
    results["meta"]["rescored"] = {
        "date": date.today().isoformat(), "changed": list(earlier.get("changed", [])) + changed,
        "figures_changed": int(earlier.get("figures_changed", 0)) + len(figures),
    }
    answers_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    text = render(results)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    write(text)
    write(f"\nRe-scored {len(rows)} saved answers: {len(changed)} verdict(s) and {len(figures)} figure check(s) changed. Saved: {out / 'report.md'}")
    return 0
