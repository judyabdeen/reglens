"""Command line:  python scripts/ask.py ["question"]  -  ask the indexed documents; without a question, chat."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path

from reglens.config import ConfigError, Settings, load_settings
from reglens.console import make_console_safe, short_duration
from reglens.embeddings import EmbeddingError
from reglens.generation.answer import EXTRACTIVE, LLM_ONLY, RAG, Answer, Source
from reglens.indexing import IndexNotReady
from reglens.pipeline import AskOptions
from reglens.retrieval import MODES, RerankError, RetrievalError, RetrievalOptions, TransformOptions, Turn
from reglens.runtime import Runtime, build_runtime
from reglens.vectorstore import VectorStoreError

FILTER_KEYS = ("doc_id", "jurisdiction", "doc_type", "status", "publisher", "version_date", "kind")
EXCERPT_CHARS = 320
HISTORY_TURNS = 6  # question/answer pairs kept for follow-up questions

HINTS = {
    "auth": "Check the API key in the .env file (python scripts/check_setup.py tests it).",
    "unavailable": "Add your API key to the .env file (see .env.example), or use --no-llm to see the passages only.",
    "quota": "The free quota of this model is used up. Try again later, or set another GEMINI_MODEL in .env.",
    "rate_limited": "Too many requests per minute. Wait a minute and ask again.",
    "overloaded": "The provider is busy right now. Ask again in a moment.",
    "server": "The provider had an internal error. Ask again in a moment.",
    "timeout": "The provider did not answer in time. Ask again, or check the internet connection.",
    "network": "The provider could not be reached. Check the internet connection, VPN or firewall.",
    "forbidden": "This key may not use this model. Set another GEMINI_MODEL in .env.",
    "not_found": "The model name is unknown or retired. Set a current model in .env (python scripts/check_setup.py lists yours).",
    "empty": "The model returned no text. Try again, or set GEMINI_THINKING_LEVEL=minimal in .env.",
    "blocked": "The provider's safety filter refused this request.",
}

HELP = """Type a question and press Enter. Follow-up questions ("and in the US?") work.
  /docs    list the indexed documents (their ids can be used with --filter doc_id=...)
  /clear   forget the conversation so far
  /quit    leave (Ctrl+C or Ctrl+Z then Enter also work)"""


# ------------------------------------------------------------------------------------------------ arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask",
        description="Ask a question about the indexed regulations. Without a question this starts an interactive chat.",
    )
    parser.add_argument("question", nargs="?", help="the question (omit it to chat)")
    parser.add_argument("--k", type=int, default=8, help="passages given to the language model (default 8)")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="search: hybrid (default), dense (meaning) or lexical (keywords)")
    parser.add_argument("--candidates", type=int, default=40, help="candidates taken from each search before fusion (default 40)")
    parser.add_argument("--rerank", action="store_true", help="re-score the best candidates with a cross-encoder (downloads ~90 MB once)")
    parser.add_argument("--rerank-pool", type=int, default=20, help="candidates the cross-encoder re-scores (default 20)")
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE", help=f"only search matching passages; keys: {', '.join(FILTER_KEYS)}; VALUE may be a,b for 'either'. Repeatable.")
    parser.add_argument("--no-notes", action="store_true", help="leave footnote passages out of the search")
    parser.add_argument("--no-glossary", action="store_true", help="do not add legal-term synonyms to the keyword query")
    parser.add_argument("--no-condense", action="store_true", help="do not rewrite follow-up questions")
    parser.add_argument("--multi-query", type=int, default=0, metavar="N", help="also search N paraphrases of the question (uses the language model)")
    parser.add_argument("--hyde", action="store_true", help="also search with a hypothetical answer (uses the language model)")
    parser.add_argument("--max-words", type=int, default=3000, help="word budget of the passages in the prompt (default 3000)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="answer length limit in tokens (default 2048)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-llm", action="store_true", help="retrieval only: show the passages, do not call the language model")
    group.add_argument("--llm-only", action="store_true", help="the language model alone, without the documents (the baseline)")
    group.add_argument("--both", action="store_true", help="answer twice, with and without the documents, to compare")
    parser.add_argument("--provider", choices=("gemini", "openai", "anthropic"), help="language-model provider (default: LLM_PROVIDER)")
    parser.add_argument("--model", help="language model (default: the one in .env)")
    parser.add_argument("--backend", choices=("chroma", "numpy"), help="vector store to search (default: VECTORSTORE)")
    parser.add_argument("--full", action="store_true", help="print the whole text of every passage, not an excerpt")
    parser.add_argument("--json", action="store_true", help="print the result as JSON (for scripts and the evaluation)")
    parser.add_argument("--documents", action="store_true", help="list the indexed documents and exit")
    parser.add_argument("--env-file", type=Path, help="use this .env file")
    return parser


def parse_filters(items: Sequence[str]) -> dict[str, str | list[str]]:
    """``["jurisdiction=EU", "status=in_force,final"]`` -> ``{"jurisdiction": "EU", "status": ["in_force", "final"]}``."""
    where: dict[str, str | list[str]] = {}
    for item in items:
        key, separator, value = item.partition("=")
        key = key.strip()
        values = [part.strip() for part in value.split(",") if part.strip()]
        if not separator or not values:
            raise ValueError(f"--filter {item!r}: write it as KEY=VALUE, for example jurisdiction=EU")
        if key not in FILTER_KEYS:
            raise ValueError(f"--filter {key!r}: unknown key; use one of {', '.join(FILTER_KEYS)}")
        where[key] = values[0] if len(values) == 1 else values
    return where


def ask_options(args: argparse.Namespace) -> AskOptions:
    where = parse_filters(args.filter)
    return AskOptions(
        retrieval=RetrievalOptions(
            mode=args.mode, top_k=args.k, candidates=max(args.candidates, args.k), rerank=args.rerank, rerank_pool=max(args.rerank_pool, args.k),
            where=where or None, include_notes=not args.no_notes,
        ),
        transform=TransformOptions(glossary=not args.no_glossary, condense=not args.no_condense, multi_query=max(args.multi_query, 0), hyde=args.hyde),
        max_context_words=args.max_words,
        max_output_tokens=args.max_tokens,
        use_llm=not args.no_llm,
    )


# ------------------------------------------------------------------------------------------------ output


def _width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns - 2, 110))


def _wrap(text: str, indent: str = "", first: str | None = None) -> str:
    width = _width()
    paragraphs = []
    for index, paragraph in enumerate(text.splitlines() or [""]):
        lead = (first if (first is not None and index == 0) else indent)
        paragraphs.append(textwrap.fill(paragraph, width=width, initial_indent=lead, subsequent_indent=indent) if paragraph.strip() else "")
    return "\n".join(paragraphs)


def _source_block(source: Source, *, full: bool) -> list[str]:
    mark = "*" if source.cited else " "
    lines = [f" {mark} [{source.label}] {source.title}"]
    facts = [f"{source.jurisdiction} {source.doc_type}".strip(), source.status_text, source.version_date, source.location]
    lines.append(_wrap(" | ".join(part for part in facts if part), indent="       "))
    evidence = []
    if source.rerank_score is not None:
        evidence.append(f"re-rank {source.rerank_score:.2f}")
    if source.dense_score is not None:
        evidence.append(f"similarity {source.dense_score:.3f}")
    if source.lexical_score is not None:
        evidence.append(f"keyword {source.lexical_score:.2f}")
    if evidence:
        lines.append(f"       {', '.join(evidence)}")
    text = " ".join(source.text.split())
    if not full and len(text) > EXCERPT_CHARS:
        text = text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " ..."
    lines.append(_wrap(f'"{text}"', indent="       "))
    if source.url:
        lines.append(f"       {source.url}")
    return lines


def format_answer(answer: Answer, *, full: bool = False) -> str:
    out: list[str] = [f"Question   {answer.question}"]
    if answer.standalone_question != answer.question:
        out.append(f"Searched   {answer.standalone_question}")
    for note in answer.plan_notes:
        out.append(f"           ({note})")
    out.append("")
    if answer.mode == EXTRACTIVE:
        out.append("ANSWER     (no language-model answer; the most relevant passages are listed below, best first)")
    else:
        who = f"{answer.provider}/{answer.model}" if answer.model else "language model"
        tokens = f", {answer.input_tokens:,} tokens in / {answer.output_tokens:,} out" if answer.input_tokens else ""
        out.append(f"ANSWER     ({who}, {short_duration(answer.timings.get('total', 0.0))}{tokens})")
        out.append(_wrap(answer.text, indent="  ") if answer.text else "  (empty)")
        if answer.abstained:
            out.append("  (the model says the indexed documents do not answer this)")
    if answer.error:
        out.append(_wrap(f"Why: {answer.error}", indent="           "))
        hint = HINTS.get(answer.error_kind or "")
        if hint:
            out.append(_wrap(f"Fix: {hint}", indent="           "))
    if answer.mode == LLM_ONLY and not answer.error:
        out.append("  (no documents were used: nothing here has been checked against a source)")
    if answer.sources:
        out += ["", "SOURCES    (* = cited in the answer)"]
        for source in answer.sources:
            out += _source_block(source, full=full)
    checks: list[str] = []
    if answer.mode == RAG and not answer.abstained and answer.citation_coverage is not None:
        checks.append(f"{answer.citation_coverage * 100:.0f}% of the factual sentences carry a citation")
        checks += [f'no citation: "{sentence[:140]}"' for sentence in answer.uncited_sentences[:3]]
    if answer.unverified_figures:
        checks.append(f"WARNING: figures in the answer that appear in no source: {', '.join(answer.unverified_figures)}")
    checks += [note for note in answer.notes if note != answer.error]
    if checks:
        out += ["", "CHECKS"]
        out += [_wrap(item, indent="    ", first="  - ") for item in checks]
    return "\n".join(out)


def list_documents(runtime: Runtime) -> str:
    rows = runtime.retriever.documents()
    width = max((len(doc_id) for doc_id, _, _ in rows), default=10)
    lines = [f"{len(rows)} documents, {sum(count for _, _, count in rows):,} passages:", ""]
    lines += [f"  {doc_id:<{width}}  {count:>4}  {title}" for doc_id, title, count in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ running


def _answer_once(runtime: Runtime, question: str, history: Sequence[Turn], options: AskOptions, args: argparse.Namespace) -> list[Answer]:
    answers: list[Answer] = []
    if not args.llm_only:
        answers.append(runtime.pipeline.ask(question, history, options))
    if args.llm_only or args.both:
        answers.append(runtime.pipeline.ask_llm_only(question, options.max_output_tokens))
    return answers


def _print_answers(answers: Sequence[Answer], args: argparse.Namespace, write: Callable[[str], None]) -> None:
    if args.json:
        payload = answers[0].to_dict() if len(answers) == 1 else {a.mode: a.to_dict() for a in answers}
        write(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for position, answer in enumerate(answers):
        if position:
            write("\n" + "=" * 30 + " the same question, without the documents " + "=" * 30 + "\n")
        write(format_answer(answer, full=args.full))


def _banner(runtime: Runtime) -> list[str]:
    info = runtime.index
    lines = [f"RegLens: {info.n_records:,} passages from {info.n_documents} documents ({info.embedding_model}, {info.backend} store)."]
    if runtime.has_llm:
        lines.append(f"Language model: {runtime.llm_provider}/{runtime.llm_model}")
    else:
        lines.append(f"No language model ({runtime.llm_problem}); answers will show the retrieved passages only.")
    lines += [f"Note: {warning}" for warning in runtime.warnings]
    return lines


def main(
    argv: Sequence[str] | None = None,
    *,
    build: Callable[..., Runtime] = build_runtime,
    settings_loader: Callable[..., Settings] = load_settings,
    read_line: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    make_console_safe()
    args = _parser().parse_args(argv)
    try:
        options = ask_options(args)
        settings = settings_loader(env_file=args.env_file)
    except (ValueError, ConfigError) as exc:
        write(f"Problem: {exc}")
        return 2
    if (args.llm_only or args.both) and args.no_llm:  # pragma: no cover - argparse already refuses the combination
        write("Problem: --no-llm cannot be combined with --llm-only or --both")
        return 2
    try:
        runtime = build(settings, backend=args.backend, provider=args.provider, model=args.model)
    except IndexNotReady as exc:
        write(f"Problem: {exc}")
        return 1
    except (EmbeddingError, VectorStoreError, OSError) as exc:
        write(f"Problem: the index could not be opened: {exc}")
        return 1
    if (args.llm_only or args.both) and not runtime.has_llm:
        write(f"Problem: this needs a language model, but none is available: {runtime.llm_problem}")
        return 1

    if args.documents:
        write(list_documents(runtime))
        return 0

    if args.question is not None:
        return _one_question(runtime, args.question, options, args, write)
    return _chat(runtime, options, args, read_line, write)


def _one_question(runtime: Runtime, question: str, options: AskOptions, args: argparse.Namespace, write: Callable[[str], None]) -> int:
    try:
        answers = _answer_once(runtime, question, (), options, args)
    except ValueError as exc:
        write(f"Problem: {exc}")
        return 2
    except (RetrievalError, RerankError, EmbeddingError, VectorStoreError) as exc:
        write(f"Problem: {exc}")
        return 1
    if not args.json:
        for warning in runtime.warnings:
            write(f"Note: {warning}")
    _print_answers(answers, args, write)
    return 0


def _chat(runtime: Runtime, options: AskOptions, args: argparse.Namespace, read_line: Callable[[str], str], write: Callable[[str], None]) -> int:
    for line in _banner(runtime):
        write(line)
    write("Type a question, or /help.\n")
    history: list[Turn] = []
    while True:
        try:
            line = read_line("> ").strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            return 0
        if not line:
            continue
        command = line.lower()
        if command in ("/quit", "/exit", "/q"):
            return 0
        if command == "/help":
            write(HELP)
            continue
        if command == "/clear":
            history.clear()
            write("(conversation cleared)")
            continue
        if command == "/docs":
            write(list_documents(runtime))
            continue
        try:
            answers = _answer_once(runtime, line, history[-HISTORY_TURNS:], options, args)
        except ValueError as exc:
            write(f"Problem: {exc}")
            continue
        except (RetrievalError, RerankError, EmbeddingError, VectorStoreError) as exc:
            write(f"Problem: {exc}")
            continue
        _print_answers(answers, args, write)
        write("")
        first = answers[0]
        if first.mode == RAG and not first.abstained:
            history.append(Turn(line, first.text))
