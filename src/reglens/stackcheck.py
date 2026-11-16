"""Checks behind ``python scripts/check_stack.py``: does the machine-learning stack work on *this* computer?

``check_setup.py`` (Step 1) proves the basics. This one proves the parts that cannot be tested anywhere but on the real
machine: ChromaDB (a real database round trip in a temporary folder), the embedding model (downloaded, run, and sanity
checked), the re-ranking model, and one real call to the language model. It also measures how long the full index will
take, and how many chunks are longer than the embedding model's input window.

Nothing here prints or stores an API key. The output is plain ASCII and ends with a block to paste back.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from reglens.config import ConfigError, Settings, load_settings, synced_folder
from reglens.console import make_console_safe
from reglens.diagnostics import Check, check_python, check_reachability, render
from reglens.embeddings import Embedder, is_hashing, make_embedder, normalize_rows
from reglens.ingest.chunking import read_chunks
from reglens.llm.base import LLMError, LLMRequest
from reglens.llm.factory import make_llm
from reglens.retrieval.rerank import CrossEncoderReranker, Reranker
from reglens.vectorstore import ChromaStore, NumpyStore, Record
from reglens.vectorstore.base import VectorStore

STACK_PACKAGES = (
    ("numpy", True),
    ("chromadb", True),
    ("sentence-transformers", True),
    ("torch", True),
    ("transformers", False),
    ("huggingface-hub", False),
)
HUGGING_FACE = ("https://huggingface.co/",)

QUESTION = "How quickly must a data breach be reported to the authority?"
PASSAGES = (
    "In the case of a personal data breach, the controller shall without undue delay and, where feasible, not later than "
    "72 hours after having become aware of it, notify the personal data breach to the supervisory authority.",
    "Combine the flour and the eggs in a large bowl, then bake the mixture for twenty minutes until it is golden.",
    "A consumer shall have the right to request that a business delete any personal information about the consumer.",
)
SPEED_SAMPLE = 48  # passages embedded to estimate how long the full index will take


def _problem(exc: BaseException, limit: int = 300) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ------------------------------------------------------------------------------------------------ libraries


def check_libraries(version: Callable[[str], str] = metadata.version) -> list[Check]:
    checks = []
    for name, required in STACK_PACKAGES:
        try:
            checks.append(Check(name, "OK", version(name)))
        except metadata.PackageNotFoundError:
            status = "FAIL" if required else "WARN"
            checks.append(Check(name, status, "not installed - run: pip install -r requirements.txt"))
    return checks


def check_accelerator() -> Check:
    """Which device the embedding model will use. A GPU is optional; the project is designed for plain CPUs."""
    try:
        import torch
    except ImportError:
        return Check("Compute device", "SKIP", "torch is not installed")
    except Exception as exc:  # a broken torch install can fail in many ways
        return Check("Compute device", "FAIL", f"torch could not be imported ({_problem(exc)})")
    threads = torch.get_num_threads()
    if torch.cuda.is_available():
        return Check("Compute device", "OK", f"CUDA GPU ({torch.cuda.get_device_name(0)}); {threads} CPU threads")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return Check("Compute device", "OK", f"Apple GPU (mps); {threads} CPU threads")
    return Check("Compute device", "OK", f"CPU only, {threads} threads (fine: the default model is small)")


def check_sync_folder(settings: Settings) -> Check:
    service = synced_folder(settings.root)
    if service is None:
        return Check("Project folder", "OK", f"{settings.root} (not inside a cloud-synced folder)")
    return Check(
        "Project folder",
        "WARN",
        f"{settings.root} is inside {service}. RegLens keeps the index outside it ({settings.vectorstore_dir}), but "
        f"{service} can still lock files while installing packages and during git. Moving the project to a plain folder "
        "such as C:\\dev\\reglens is safer",
    )


# ------------------------------------------------------------------------------------------------ the database


def check_chroma(make_store: Callable[[Path], VectorStore] = lambda path: ChromaStore(path)) -> list[Check]:
    """A real round trip: store, search, filter, reopen. The result is compared with the exact NumPy store."""
    texts = ["breach notification", "erasure of data", "automated decisions", "health information", "consumer opt out"]
    metadata_rows = [
        {"doc_id": "a", "jurisdiction": "EU", "status": "in_force", "ordinal": 1},
        {"doc_id": "a", "jurisdiction": "EU", "status": "draft", "ordinal": 2},
        {"doc_id": "b", "jurisdiction": "US", "status": "final", "ordinal": 1},
        {"doc_id": "b", "jurisdiction": "US", "status": "in_force", "ordinal": 2, "page_start": 3},
        {"doc_id": "c", "jurisdiction": "UK", "status": "final", "ordinal": 1},
    ]
    records = [Record(f"{row['doc_id']}:{index}", text, row) for index, (text, row) in enumerate(zip(texts, metadata_rows, strict=True))]
    # Random unit vectors: with real-looking vectors no two records tie, so "the 3 nearest" has exactly one right answer.
    vectors = normalize_rows(np.random.default_rng(7).normal(size=(len(texts), 32)))
    started = time.perf_counter()
    try:
        # ignore_cleanup_errors: on Windows the database files may still be open when the folder is removed
        with tempfile.TemporaryDirectory(prefix="reglens-chroma-", ignore_cleanup_errors=True) as folder:
            path = Path(folder) / "chroma"
            store = make_store(path)
            store.reset()
            store.add(records, vectors)
            store.flush()
            if store.count() != len(records):
                return [Check("ChromaDB", "FAIL", f"stored {len(records)} records but count() says {store.count()}")]
            reference = NumpyStore()
            reference.add(records, vectors)
            problems: list[str] = []
            for position in range(len(records)):
                got = [hit.id for hit in store.query(vectors[position], 3)]
                want = [hit.id for hit in reference.query(vectors[position], 3)]
                if got[:1] != want[:1] or set(got) != set(want):
                    problems.append(f"query {position}: ChromaDB {got} vs exact {want}")
            top = store.query(vectors[0], 1)
            if not top or abs(top[0].score - 1.0) > 1e-3:
                problems.append(f"the best hit for a stored vector should score 1.0, got {top[0].score if top else 'nothing'}")
            for where in ({"jurisdiction": "US"}, {"status": ["draft", "final"]}, {"jurisdiction": "EU", "status": "in_force"}):
                got = sorted(hit.id for hit in store.query(vectors[0], 5, where))
                want = sorted(hit.id for hit in reference.query(vectors[0], 5, where))
                if got != want:
                    problems.append(f"filter {where}: ChromaDB {got} vs exact {want}")
            if [r.id for r in store.records()] != sorted(r.id for r in records):
                problems.append("records() does not return every stored record")
            if [r.id for r in store.get(["b:3", "nope", "a:0"])] != ["b:3", "a:0"]:
                problems.append("get() did not return the records asked for, in order, skipping unknown ids")
            if store.get(["b:3"])[0].metadata.get("page_start") != 3:
                problems.append("metadata did not survive the round trip")
            if problems:
                return [Check("ChromaDB", "FAIL", "; ".join(problems))]
            reopened = make_store(path)
            if reopened.count() != len(records):
                return [Check("ChromaDB", "FAIL", "the stored data was not there after reopening the database")]
            del store, reopened
    except Exception as exc:
        return [Check("ChromaDB", "FAIL", _problem(exc))]
    return [Check("ChromaDB", "OK", f"stored, searched, filtered and reopened {len(records)} records; same results as the exact store ({time.perf_counter() - started:.1f} s)")]


# ------------------------------------------------------------------------------------------------ the models


def check_embedding(embedder: Embedder, chunk_texts: Sequence[str] | None = None, *, clock: Callable[[], float] = time.perf_counter) -> list[Check]:
    checks: list[Check] = []
    try:
        mark = clock()
        passages = embedder.embed_documents(PASSAGES)  # the model is loaded here: the first run downloads it
        query = embedder.embed_query(QUESTION)
        load_seconds = clock() - mark
    except Exception as exc:
        return [Check(f"Embedding model {embedder.name}", "FAIL", _problem(exc, 400))]
    norms = np.linalg.norm(passages, axis=1)
    if passages.shape != (len(PASSAGES), embedder.dim) or not np.isfinite(passages).all() or not np.allclose(norms, 1.0, atol=1e-3):
        return [Check(f"Embedding model {embedder.name}", "FAIL", f"unexpected vectors: shape {passages.shape}, norms {norms.min():.3f}..{norms.max():.3f}")]
    similarity = passages @ query
    order = [int(i) for i in np.argsort(-similarity)]
    if order[0] != 0:
        return [Check(f"Embedding model {embedder.name}", "FAIL", f"the breach question is closest to passage {order[0] + 1}, not to the breach passage: scores {np.round(similarity, 3).tolist()}")]
    checks.append(
        Check(
            f"Embedding model {embedder.name}",
            "OK",
            f"{embedder.dim} numbers per vector; the breach question ranks the breach passage first "
            f"(similarity {similarity[0]:.2f} vs {similarity[1]:.2f} for an unrelated text); loaded in {load_seconds:.1f} s",
        )
    )
    texts = list(chunk_texts[:SPEED_SAMPLE]) if chunk_texts else list(PASSAGES) * (SPEED_SAMPLE // len(PASSAGES))
    try:
        mark = clock()
        embedder.embed_documents(texts)
        seconds = max(clock() - mark, 1e-6)
    except Exception as exc:
        checks.append(Check("Embedding speed", "WARN", _problem(exc)))
        return checks
    rate = len(texts) / seconds
    if chunk_texts:
        total = len(chunk_texts)
        checks.append(Check("Embedding speed", "OK", f"{rate:.1f} passages per second; all {total:,} passages will take about {total / rate / 60:.1f} minutes"))
    else:
        checks.append(Check("Embedding speed", "OK", f"{rate:.1f} passages per second (no chunks.jsonl yet to project the full time)"))
    if chunk_texts:
        checks.append(_check_window(embedder, chunk_texts))
    return checks


def _check_window(embedder: Embedder, chunk_texts: Sequence[str]) -> Check:
    counter = getattr(embedder, "token_counts", None)
    try:
        limit = getattr(embedder, "max_tokens", None)
        counts = counter(chunk_texts) if callable(counter) else None
    except Exception as exc:
        return Check("Chunk length", "WARN", f"could not measure token lengths ({_problem(exc)})")
    if not counts or not limit:
        return Check("Chunk length", "SKIP", "this embedding model does not report token lengths")
    over = sum(count > limit for count in counts)
    detail = f"{over:,} of {len(counts):,} chunks ({100 * over / len(counts):.1f}%) are longer than the {limit}-token window; the longest has {max(counts):,} tokens"
    if over == 0:
        return Check("Chunk length", "OK", detail)
    return Check("Chunk length", "WARN", detail + ". The end of those chunks is not part of their vector; the report discusses this limit")


def check_reranker(reranker: Reranker) -> Check:
    try:
        scores = reranker.score(QUESTION, [PASSAGES[0], PASSAGES[1]])
    except Exception as exc:
        return Check(f"Re-ranking model {reranker.name}", "FAIL", _problem(exc, 400))
    if len(scores) != 2 or not scores[0] > scores[1]:
        return Check(f"Re-ranking model {reranker.name}", "FAIL", f"it scored the unrelated passage as high as the relevant one: {scores}")
    return Check(f"Re-ranking model {reranker.name}", "OK", f"relevant passage {scores[0]:.2f}, unrelated passage {scores[1]:.2f}")


def check_language_model(settings: Settings, make: Callable[..., object] = make_llm) -> Check:
    try:
        client = make(settings)
    except (LLMError, ConfigError) as exc:
        return Check("Language model", "WARN", f"not set up: {exc}")
    try:
        response = client.generate(LLMRequest.single("You are a connectivity test.", "Reply with exactly one word: ready", max_output_tokens=256))  # type: ignore[attr-defined]
    except LLMError as exc:
        status = "FAIL" if exc.kind in ("auth", "bad_request", "not_found") else "WARN"
        return Check("Language model", status, f"{exc.kind}: {exc}")
    except Exception as exc:
        return Check("Language model", "FAIL", _problem(exc))
    fallback = f"; {response.fell_back_from} was unavailable" if response.fell_back_from else ""
    tokens = response.usage.total_tokens
    return Check("Language model", "OK", f"{response.provider}/{response.model} answered {response.text.strip()[:20]!r} in {response.latency_s:.1f} s ({tokens} tokens){fallback}")


# ------------------------------------------------------------------------------------------------ orchestration


def run_stack_checks(
    settings: Settings,
    *,
    online: bool = True,
    rerank: bool = True,
    llm: bool = True,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    chroma_factory: Callable[[Path], VectorStore] | None = None,
    llm_factory: Callable[..., object] = make_llm,
    say: Callable[[str], None] = lambda text: None,
) -> list[Check]:
    checks = [check_python()]
    say("Checking the installed libraries ...")
    checks += check_libraries()
    checks.append(check_accelerator())
    checks.append(check_sync_folder(settings))
    say("Checking ChromaDB with a real database round trip ...")
    checks += check_chroma(chroma_factory) if chroma_factory else check_chroma()
    if online:
        checks += check_reachability(HUGGING_FACE)
    chunk_texts: list[str] | None = None
    if settings.chunks_path.is_file():
        try:
            chunk_texts = [chunk.embed_text for chunk in read_chunks(settings.chunks_path)]
        except ValueError as exc:
            checks.append(Check("chunks.jsonl", "WARN", _problem(exc)))
    else:
        checks.append(Check("chunks.jsonl", "WARN", f"{settings.chunks_path} not found - run: python scripts/ingest.py"))
    if is_hashing(settings.embedding_model):
        say(f"Loading the embedding model {settings.embedding_model} (the baseline model needs no download) ...")
    else:
        say(f"Loading the embedding model {settings.embedding_model} (the first run downloads about 130 MB; please wait) ...")
    checks += check_embedding(embedder or make_embedder(settings.embedding_model), chunk_texts)
    if rerank:
        say(f"Loading the re-ranking model {settings.reranker_model} (the first run downloads about 90 MB) ...")
        checks.append(check_reranker(reranker or CrossEncoderReranker(settings.reranker_model)))
    else:
        checks.append(Check("Re-ranking model", "SKIP", "--no-rerank given"))
    if llm and online:
        say("Asking the language model one short question ...")
        checks.append(check_language_model(settings, llm_factory))
    else:
        checks.append(Check("Language model", "SKIP", "--no-llm or --offline given"))
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_stack", description="Check that ChromaDB, the embedding model, the re-ranker and the language model work on this computer.")
    parser.add_argument("--offline", action="store_true", help="skip the checks that need the internet (models must already be downloaded)")
    parser.add_argument("--no-rerank", action="store_true", help="do not load the re-ranking model (saves a 90 MB download)")
    parser.add_argument("--no-llm", action="store_true", help="do not call the language model")
    parser.add_argument("--env-file", type=Path, help="use this .env file instead of the one in the project folder")
    args = parser.parse_args(argv)
    make_console_safe()
    try:
        settings = load_settings(env_file=args.env_file)
    except ConfigError as exc:
        print(f"[FAIL] .env: {exc}")
        return 1
    print("RegLens stack check - this loads the real models and can take a few minutes the first time.\n", flush=True)
    checks = run_stack_checks(
        settings, online=not args.offline, rerank=not args.no_rerank, llm=not args.no_llm, say=lambda text: print(text, flush=True)
    )
    print()
    print(render(checks, title="RegLens stack check"))
    return 1 if any(c.status == "FAIL" for c in checks) else 0


__all__ = ["check_chroma", "check_embedding", "check_language_model", "check_libraries", "check_reranker", "main", "run_stack_checks"]
