from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnthropicSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr
    model: str = "claude-sonnet-4-6"


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr
    chat_model: str = "gemini-3.5-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dimensions: int = 768


class PineconeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PINECONE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr
    index_name: str = "fds-reconciler"
    namespace: str = "default"
    cloud: str = "aws"
    region: str = "us-east-1"


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHUNK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    target_tokens: int = 600
    max_tokens: int = 900
    overlap_tokens: int = 80


class AlignmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALIGN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    w_heading_num: float = 0.45
    w_heading_embed: float = 0.40
    w_levenshtein: float = 0.15
    threshold: float = 0.55


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    single_doc_top_k: int = 6
    cross_doc_top_k: int = 4
    relevance_floor: float = 0.5


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
