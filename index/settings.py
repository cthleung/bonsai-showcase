"""Where the index, the models and llama-server live, and which ports they use.

Everything sits under BONSAI_STATE (~/bonsai-state, say):

    index/index.sqlite        the index (chunks, FTS5, trigram FTS5, sqlite-vec vectors)
    index/queries.jsonl       one line per search
    index/servers.json        the llama-server processes this package started
    models/llama.cpp/         llama-server.exe and its DLLs (the pinned release build)
    models/gguf/              the GGUF model files

Any of these can be pointed elsewhere with an environment variable or a line in the repo .env:
BONSAI_LLAMA_SERVER, BONSAI_EMBED_MODEL, BONSAI_RERANK_MODEL (a path, or `none` to switch reranking
off), BONSAI_EMBED_PORT, BONSAI_RERANK_PORT.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from common import config

EMBED_DIMS = 1024              # stored dims: the first 1024 of Qwen3-Embedding-4B's 2560 (Matryoshka cut)
DEFAULT_EMBED_PORT = 8091
DEFAULT_RERANK_PORT = 8092
DEFAULT_EMBED_MODEL = "gguf/Qwen3-Embedding-4B-Q8_0.gguf"
DEFAULT_RERANK_MODEL = "gguf/Qwen3-Reranker-4B.Q8_0.gguf"
DEFAULT_LLAMA_SERVER = "llama.cpp/llama-server.exe"

# The query instruction the embedding model expects on a search query.
# Only queries get it. Documents go in plain.
QUERY_INSTRUCT = ("Given a short, loosely worded question from a person about their own past "
                  "conversations, retrieve the chat turns and notes where they talked about it")


@dataclass(frozen=True)
class Settings:
    vault: Path
    state: Path
    llama_server: Path
    embed_model: Path
    rerank_model: Path | None          # None: no reranking
    embed_port: int = DEFAULT_EMBED_PORT
    rerank_port: int = DEFAULT_RERANK_PORT

    @property
    def index_dir(self) -> Path:
        return self.state / "index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "index.sqlite"

    @property
    def query_log(self) -> Path:
        return self.index_dir / "queries.jsonl"

    @property
    def servers_file(self) -> Path:
        return self.index_dir / "servers.json"

    @property
    def embed_url(self) -> str:
        return f"http://127.0.0.1:{self.embed_port}"

    @property
    def rerank_url(self) -> str:
        return f"http://127.0.0.1:{self.rerank_port}"

    @property
    def embed_model_id(self) -> str:
        """What the index records as its embedding model, so vectors from two models never mix."""
        return f"{self.embed_model.name}@{EMBED_DIMS}"


def _value(values: dict[str, str], key: str) -> str:
    return (os.environ.get(key) or values.get(key) or "").strip()


def load(env_file: Path | None = None) -> Settings:
    cfg = config.load(env_file)
    values = config.read_env_file(env_file or config.ENV_FILE)
    models = cfg.state / "models"

    def path(key: str, default: str) -> Path:
        raw = _value(values, key)
        return Path(raw).expanduser() if raw else models / default

    rerank_raw = _value(values, "BONSAI_RERANK_MODEL")
    rerank = None if rerank_raw.lower() == "none" else path("BONSAI_RERANK_MODEL", DEFAULT_RERANK_MODEL)
    return Settings(
        vault=cfg.vault,
        state=cfg.state,
        llama_server=path("BONSAI_LLAMA_SERVER", DEFAULT_LLAMA_SERVER),
        embed_model=path("BONSAI_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        rerank_model=rerank,
        embed_port=int(_value(values, "BONSAI_EMBED_PORT") or DEFAULT_EMBED_PORT),
        rerank_port=int(_value(values, "BONSAI_RERANK_PORT") or DEFAULT_RERANK_PORT),
    )
