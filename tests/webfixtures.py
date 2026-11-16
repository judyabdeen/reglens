"""Helpers shared by the web tests: a service over the invented corpus, with a scripted model."""

from __future__ import annotations

import threading
import time

import ragfixtures
from ragfixtures import FakeLLM

from reglens.config import load_settings
from reglens.embeddings import HashingEmbedder
from reglens.generation.prompts import LLM_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT
from reglens.indexing import build_index, info_path
from reglens.retrieval.query_transform import CONDENSE_SYSTEM
from reglens.runtime import build_runtime
from reglens.vectorstore import open_store
from reglens.web.service import WebService

SECRET = "sk-secret-key-that-must-never-appear"
BREACH = "How quickly must we report a data leak to the regulator?"


class Reranker:
    name = "stub-reranker"

    def score(self, query, passages):
        return [float(len(p)) for p in passages]


def settings_for(tmp_path, key=SECRET):
    environ = {"VECTORSTORE": "numpy", "VECTORSTORE_DIR": str(tmp_path / "index"), "GEMINI_API_KEY": key}
    return load_settings(root=tmp_path, environ=environ)


def build_the_index(settings, embedder=None):
    store = open_store(settings.vectorstore_dir, "numpy")
    build_index(
        ragfixtures.chunks(), ragfixtures.by_source(), embedder or HashingEmbedder(), store,
        info_file=info_path(settings.vectorstore_dir, "numpy"), backend="numpy",
    )  # fmt: skip


def make_service(tmp_path, llm=None, *, embedder=None, **kwargs):
    """A service over the invented corpus. ``llm=None`` means no API key, so the runtime itself records why there is no model."""
    settings = settings_for(tmp_path, key=SECRET if llm is not None else "")
    build_the_index(settings)

    def factory(s):
        return build_runtime(s, embedder=embedder or HashingEmbedder(), reranker=Reranker(), llm=llm)

    return WebService(settings, runtime_factory=factory, **kwargs)


def scripted(answer="Notify within 72 hours [S1].", baseline="Within 72 hours.", standalone="How quickly must a data breach be reported?"):
    """A model that answers by the kind of request it receives: condensing, grounded answer, or the model alone."""

    def reply(request):
        if request.system == SYSTEM_PROMPT:
            return answer
        if request.system == LLM_ONLY_SYSTEM_PROMPT:
            return baseline
        assert request.system == CONDENSE_SYSTEM
        return standalone

    return FakeLLM(reply)


class LiveServer:
    """The real server (uvicorn) on a free local port, run in a thread: ``with LiveServer(app) as server: server.base``."""

    def __init__(self, app):
        import uvicorn

        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off", ws="none", access_log=False)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            assert time.time() < deadline, "the test server did not start"
            time.sleep(0.01)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(10)
