from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- LLM (Ollama) ---
    # Host default. Override if the app is containerized later.
    ollama_base_url: str = "http://localhost:11434"
    llm_supervisor_model: str = "qwen2.5:3b-instruct"
    llm_specialist_model: str = "qwen2.5:3b-instruct"
    llm_writer_model: str = "qwen2.5:3b-instruct"
    llm_request_timeout: int = 120

    # --- Postgres (app data) ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "afa"
    postgres_password: str = "afa"
    postgres_db: str = "afa"
    app_database_url: str = "postgresql+asyncpg://afa:afa@localhost:5432/afa"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection: str = "afa_docs"

    # --- Embeddings / Reranker ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_rerank: bool = True

    # --- FX ---
    fx_base_url: str = "https://open.er-api.com/v6/latest"
    fx_cache_ttl_hours: int = 24

    # --- RAG ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k_dense: int = 20
    retrieval_top_k_sparse: int = 20
    rerank_input_top_k: int = 20
    rerank_output_top_k: int = 5

    # --- SQL Analyst ---
    sql_max_attempts: int = 3
    sql_statement_timeout_ms: int = 5000
    sql_default_limit: int = 1000
    sql_max_limit: int = 10000

    # --- Langfuse (client side) ---
    enable_langfuse: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
