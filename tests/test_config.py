import pytest

from reglens.config import ConfigError, load_settings


def test_defaults_without_env_file(tmp_path):
    settings = load_settings(root=tmp_path, environ={})
    assert settings.llm_provider == "gemini"
    assert settings.gemini_model == "gemini-3.5-flash"
    assert settings.gemini_fallback_model == "gemini-3.1-flash-lite"
    assert settings.gemini_api_key is None
    assert settings.env_file_found is False
    assert settings.manifest_path == tmp_path / "data" / "sources.csv"
    assert settings.raw_dir == tmp_path / "data" / "raw"


def test_env_file_values_are_read(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=abc123\nGEMINI_MODEL=my-model\n", encoding="utf-8")
    settings = load_settings(root=tmp_path, environ={})
    assert settings.env_file_found is True
    assert settings.gemini_api_key == "abc123"
    assert settings.gemini_model == "my-model"


def test_real_environment_overrides_env_file(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    settings = load_settings(root=tmp_path, environ={"GEMINI_API_KEY": "from-environment"})
    assert settings.gemini_api_key == "from-environment"


def test_empty_values_count_as_unset(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=\nGEMINI_MODEL=\n", encoding="utf-8")
    settings = load_settings(root=tmp_path, environ={})
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-3.5-flash"


def test_windows_notepad_byte_order_mark_is_tolerated(tmp_path):
    (tmp_path / ".env").write_bytes(b"\xef\xbb\xbfGEMINI_API_KEY=abc123\r\n")
    assert load_settings(root=tmp_path, environ={}).gemini_api_key == "abc123"


def test_models_prefix_is_removed(tmp_path):
    settings = load_settings(root=tmp_path, environ={"GEMINI_MODEL": "models/gemini-3.5-flash"})
    assert settings.gemini_model == "gemini-3.5-flash"


def test_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not supported"):
        load_settings(root=tmp_path, environ={"LLM_PROVIDER": "skynet"})


def test_provider_is_case_insensitive(tmp_path):
    assert load_settings(root=tmp_path, environ={"LLM_PROVIDER": "OpenAI"}).llm_provider == "openai"


def test_api_keys_never_appear_in_repr(tmp_path):
    settings = load_settings(
        root=tmp_path,
        environ={"GEMINI_API_KEY": "SECRET-G", "OPENAI_API_KEY": "SECRET-O", "ANTHROPIC_API_KEY": "SECRET-A"},
    )
    text = repr(settings) + str(settings)
    assert "SECRET" not in text
    assert settings.key_for("openai") == "SECRET-O"


def test_key_for_unknown_provider(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(root=tmp_path, environ={}).key_for("nope")


# ------------------------------------------------------------------ settings added for the retrieval stack


def test_new_defaults(tmp_path):
    settings = load_settings(root=tmp_path, environ={})
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.vectorstore_backend == "chroma"
    assert settings.vectorstore_dir == tmp_path / "data" / "index"
    assert settings.vectorstore_dir_moved_from is None
    assert settings.chunks_path == tmp_path / "data" / "processed" / "chunks.jsonl"
    assert settings.gemini_thinking_level == "low"
    assert settings.anthropic_model == "claude-haiku-5-5"
    assert settings.openai_model is None


def test_fallback_models_can_be_a_list(tmp_path):
    env = {"GEMINI_FALLBACK_MODEL": "models/gemini-a, gemini-b;gemini-c"}
    settings = load_settings(root=tmp_path, environ=env)
    assert settings.gemini_fallbacks == ("gemini-a", "gemini-b", "gemini-c")
    assert load_settings(root=tmp_path, environ={}).gemini_fallbacks == ("gemini-3.1-flash-lite",)


def test_primary_model_is_a_single_name_even_if_a_list_is_pasted(tmp_path):
    settings = load_settings(root=tmp_path, environ={"GEMINI_MODEL": "gemini-a, gemini-b"})
    assert settings.gemini_model == "gemini-a"


@pytest.mark.parametrize("raw, expected", [("high", "high"), ("LOW", "low"), ("off", None), ("default", None), ("none", None)])
def test_thinking_level(tmp_path, raw, expected):
    assert load_settings(root=tmp_path, environ={"GEMINI_THINKING_LEVEL": raw}).gemini_thinking_level == expected


def test_unknown_thinking_level_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="GEMINI_THINKING_LEVEL"):
        load_settings(root=tmp_path, environ={"GEMINI_THINKING_LEVEL": "extreme"})


def test_unknown_vectorstore_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="VECTORSTORE"):
        load_settings(root=tmp_path, environ={"VECTORSTORE": "pinecone"})
    assert load_settings(root=tmp_path, environ={"VECTORSTORE": "Numpy"}).vectorstore_backend == "numpy"


def test_relative_paths_are_relative_to_the_project(tmp_path):
    env = {"VECTORSTORE_DIR": "somewhere/index", "CHUNKS_PATH": "my/chunks.jsonl"}
    settings = load_settings(root=tmp_path, environ=env)
    assert settings.vectorstore_dir == tmp_path / "somewhere" / "index"
    assert settings.chunks_path == tmp_path / "my" / "chunks.jsonl"


def test_absolute_paths_are_kept(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    settings = load_settings(root=tmp_path / "project", environ={"VECTORSTORE_DIR": str(elsewhere)})
    assert settings.vectorstore_dir == elsewhere


def test_model_for_each_provider(tmp_path):
    settings = load_settings(root=tmp_path, environ={"OPENAI_MODEL": "some-model"})
    assert settings.model_for("gemini") == "gemini-3.5-flash"
    assert settings.model_for("openai") == "some-model"
    assert settings.model_for("anthropic") == "claude-haiku-5-5"
    with pytest.raises(ConfigError):
        settings.model_for("nope")


@pytest.mark.parametrize(
    "path, expected",
    [
        ("C:/Users/joudy/OneDrive/Desktop/reglens", "OneDrive"),
        ("C:/Users/joudy/OneDrive - University/reglens", "OneDrive"),
        ("/home/x/Dropbox/project", "Dropbox"),
        ("C:/Users/x/iCloudDrive/project", "iCloud Drive"),
        ("/Users/x/Library/Mobile Documents/com~apple~CloudDocs/p", "iCloud Drive"),
        ("G:/My Drive/project", "Google Drive"),
        ("C:/dev/reglens", None),
        ("/home/claude/reglens", None),
    ],
)
def test_synced_folder_detection(path, expected):
    from reglens.config import synced_folder

    assert synced_folder(path) == expected


def test_index_moves_out_of_a_synced_folder(tmp_path):
    root = tmp_path / "OneDrive" / "Desktop" / "reglens"
    local = tmp_path / "Local"
    settings = load_settings(root=root, environ={"LOCALAPPDATA": str(local)})
    assert settings.vectorstore_dir.parent == local / "RegLens"
    assert settings.vectorstore_dir.name.startswith("index-")
    assert settings.vectorstore_dir_moved_from == "OneDrive"
    # the name depends on the project, so two projects never share an index
    other = load_settings(root=tmp_path / "OneDrive" / "other", environ={"LOCALAPPDATA": str(local)})
    assert other.vectorstore_dir != settings.vectorstore_dir


def test_explicit_index_dir_is_respected_even_in_a_synced_folder(tmp_path):
    root = tmp_path / "OneDrive" / "reglens"
    explicit = tmp_path / "chosen"
    settings = load_settings(root=root, environ={"VECTORSTORE_DIR": str(explicit)})
    assert settings.vectorstore_dir == explicit
    assert settings.vectorstore_dir_moved_from is None
