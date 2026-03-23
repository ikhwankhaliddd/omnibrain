from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str

    #Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    #Redis
    redis_url: str = "redis://localhostL6379/0"

    # API Keys
    openrouter_api_key: str
    octen_api_key: str

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440

    # File upload
    max_upload_size_bytes: int = 20 * 1024 * 1024  # 20MB
    upload_dir: str = "uploads"

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7
    chat_history_limit: int = 10
    summarization_token_threshold: int = 3000

    # Embedding
    embedding_batch_size: int = 32
    embedding_dimensions: int = 4096

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 200

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()