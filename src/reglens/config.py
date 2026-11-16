"""Central settings for RegLens.

Precedence: real environment variables > the ``.env`` file > built-in defaults.
The ``.env`` file is git-ignored; ``.env.example`` documents every variable.
API keys are excluded from ``repr()`` so they cannot leak into logs or tracebacks.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_PROVIDERS = ("gemini", "openai", "anthropic")
DEFAULT_PROVIDER = "gemini"
# Both Gemini models are listed as "free of charge" on Google's Gemini API pricing page (checked 2026-11-15).
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"
# Gemini 3 models think before they answer; "low" keeps answers quick and leaves the output budget for the answer.
THINKING_LEVELS = ("minimal", "low", "medium", "high")
DEFAULT_GEMINI_THINKING_LEVEL = "low"
# The Anthropic model ID is the one Anthropic lists for its smallest current model; OpenAI has no default
# because model names change too often to guess one (set OPENAI_MODEL when LLM_PROVIDER=openai).
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTORSTORE_BACKENDS = ("chroma", "numpy")
DEFAULT_VECTORSTORE_BACKEND = "chroma"

# Folders that a sync client watches. A database inside one of them can be locked while it is being uploaded.
_SYNC_MARKERS = (
    ("onedrive", "OneDrive"),
    ("dropbox", "Dropbox"),
    ("icloud", "iCloud Drive"),
    ("mobile documents", "iCloud Drive"),
    ("google drive", "Google Drive"),
    ("googledrive", "Google Drive"),
    ("my drive", "Google Drive"),
)


class ConfigError(ValueError):
    """Raised when a setting has an invalid value."""


def synced_folder(path: Path | str) -> str | None:
    """Name of the cloud-sync service whose folder contains ``path`` (for example "OneDrive"), or None."""
    for part in Path(path).parts:
        lowered = part.lower()
        for marker, name in _SYNC_MARKERS:
            if marker in lowered:
                return name
    return None


def default_index_dir(root: Path, environ: Mapping[str, str]) -> Path:
    """Where the vector index lives when VECTORSTORE_DIR is not set.

    Normally ``<project>/data/index``. If the project sits in a cloud-synced folder the index goes to the local
    application-data folder instead, because sync clients lock database files while uploading them.
    """
    if synced_folder(root) is None:
        return root / "data" / "index"
    base = environ.get("LOCALAPPDATA") or environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    tag = hashlib.sha1(str(root).lower().encode("utf-8")).hexdigest()[:8]
    return Path(base) / "RegLens" / f"index-{tag}"


@dataclass(frozen=True)
class Settings:
    root: Path
    env_file: Path
    env_file_found: bool
    manifest_path: Path
    raw_dir: Path
    llm_provider: str
    gemini_model: str
    gemini_fallback_model: str  # one model or a comma-separated list, tried in order
    gemini_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    anthropic_api_key: str | None = field(default=None, repr=False)
    gemini_thinking_level: str | None = DEFAULT_GEMINI_THINKING_LEVEL  # None: send no thinking setting
    openai_model: str | None = None
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    chunks_path: Path = Path("data/processed/chunks.jsonl")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    vectorstore_backend: str = DEFAULT_VECTORSTORE_BACKEND
    vectorstore_dir: Path = Path("data/index")
    vectorstore_dir_moved_from: str | None = None  # name of the sync service that made us move the index

    @property
    def gemini_fallbacks(self) -> tuple[str, ...]:
        """The fallback models as a tuple (GEMINI_FALLBACK_MODEL may list several, separated by commas)."""
        return tuple(name for name in re.split(r"[,;\s]+", self.gemini_fallback_model) if name)

    def key_for(self, provider: str) -> str | None:
        """Return the API key configured for ``provider`` (or None)."""
        keys = {
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        if provider not in keys:
            raise ConfigError(f"unknown provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}")
        return keys[provider]

    def model_for(self, provider: str) -> str | None:
        """The model that will be used for ``provider`` (None when OpenAI is chosen but OPENAI_MODEL is empty)."""
        models = {"gemini": self.gemini_model, "openai": self.openai_model, "anthropic": self.anthropic_model}
        if provider not in models:
            raise ConfigError(f"unknown provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}")
        return models[provider]


def load_settings(
    root: Path | None = None,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Read settings. ``environ`` defaults to ``os.environ`` (tests pass a dict instead)."""
    root = Path(root) if root else PROJECT_ROOT
    env_path = Path(env_file) if env_file else root / ".env"
    found = env_path.is_file()
    # utf-8-sig tolerates the byte-order mark that Windows Notepad adds.
    file_values = (
        {k: v for k, v in dotenv_values(env_path, encoding="utf-8-sig").items() if v is not None}
        if found
        else {}
    )
    env = os.environ if environ is None else environ

    def pick(name: str, default: str | None = None) -> str | None:
        for source in (env, file_values):
            value = (source.get(name) or "").strip()
            if value:
                return value
        return default

    provider = (pick("LLM_PROVIDER", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER={provider!r} is not supported; use one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    def models(name: str, default: str | None) -> str | None:
        # Users sometimes paste the API's "models/..." form; the prefix is not part of the model ID.
        value = pick(name, default)
        if value is None:
            return None
        names = [item.removeprefix("models/") for item in re.split(r"[,;\s]+", value) if item]
        return ",".join(names) or default

    def path(name: str, default: Path) -> Path:
        value = pick(name)
        if not value:
            return default
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else root / candidate

    thinking = (pick("GEMINI_THINKING_LEVEL", DEFAULT_GEMINI_THINKING_LEVEL) or "").lower()
    if thinking in ("", "off", "none", "default"):
        thinking = None
    elif thinking not in THINKING_LEVELS:
        raise ConfigError(
            f"GEMINI_THINKING_LEVEL={thinking!r} is not supported; use one of: {', '.join(THINKING_LEVELS)} "
            "(or 'default' to send no thinking setting)"
        )

    backend = (pick("VECTORSTORE", DEFAULT_VECTORSTORE_BACKEND) or DEFAULT_VECTORSTORE_BACKEND).lower()
    if backend not in VECTORSTORE_BACKENDS:
        raise ConfigError(f"VECTORSTORE={backend!r} is not supported; use one of: {', '.join(VECTORSTORE_BACKENDS)}")

    index_dir = path("VECTORSTORE_DIR", default_index_dir(root, env))
    moved_from = synced_folder(root) if pick("VECTORSTORE_DIR") is None and synced_folder(root) else None

    return Settings(
        root=root,
        env_file=env_path,
        env_file_found=found,
        manifest_path=root / "data" / "sources.csv",
        raw_dir=root / "data" / "raw",
        llm_provider=provider,
        gemini_model=models("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).split(",")[0],
        gemini_fallback_model=models("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL),
        gemini_api_key=pick("GEMINI_API_KEY"),
        openai_api_key=pick("OPENAI_API_KEY"),
        anthropic_api_key=pick("ANTHROPIC_API_KEY"),
        gemini_thinking_level=thinking,
        openai_model=(models("OPENAI_MODEL", None) or "").split(",")[0] or None,
        anthropic_model=models("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL).split(",")[0],
        chunks_path=path("CHUNKS_PATH", root / "data" / "processed" / "chunks.jsonl"),
        embedding_model=pick("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        reranker_model=pick("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        vectorstore_backend=backend,
        vectorstore_dir=index_dir,
        vectorstore_dir_moved_from=moved_from,
    )
