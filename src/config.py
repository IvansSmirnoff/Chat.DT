"""
Configuration management for NL2Cypher application.
Uses pydantic-settings for environment variable handling.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    LOCAL = "local"
    GEMINI = "gemini"
    OPENAI = "openai"


# Cloud override: --cloud-direct no longer forces a specific model.
# The model from .env / LLM_MODEL_NAME is used instead.


class ExperimentSetting(str, Enum):
    """
    The 4 experimental settings for the comparative research experiment.
    
    This enum defines the 2x2 matrix of experimental conditions:
    - Axis 1: Direct QA vs Cypher Generation
    - Axis 2: Without IDS grounding vs With IDS grounding
    """
    # Setting 1: Baseline Direct QA (Text-based, no IDS)
    # Input: Text/JSON representation of IFC data + User Question
    # Output: JSON list of GlobalIDs
    DIRECT_QA_BASELINE = "direct_qa_baseline"
    
    # Setting 2: Grounded Direct QA (Text-based + IDS)
    # Input: Text/JSON representation of IFC data + IDS specs + User Question
    # Output: JSON list of GlobalIDs
    DIRECT_QA_GROUNDED = "direct_qa_grounded"
    
    # Setting 3: Soft Cypher (Graph-based, soft constraints)
    # Input: Graph Schema (nodes/rels) + User Question
    # Output: Cypher query (executed against Neo4j)
    CYPHER_SOFT = "cypher_soft"
    
    # Setting 4: Strict Cypher (Graph-based + Grammar constraints)
    # Input: Graph Schema + IDS + User Question
    # Output: Cypher query with constrained decoding (executed against Neo4j)
    CYPHER_STRICT = "cypher_strict"
    
    @property
    def is_direct_qa(self) -> bool:
        """Check if this setting uses direct QA (text-based) approach."""
        return self in (ExperimentSetting.DIRECT_QA_BASELINE, ExperimentSetting.DIRECT_QA_GROUNDED)
    
    @property
    def is_cypher_gen(self) -> bool:
        """Check if this setting uses Cypher generation (graph-based) approach."""
        return self in (ExperimentSetting.CYPHER_SOFT, ExperimentSetting.CYPHER_STRICT)
    
    @property
    def uses_ids_grounding(self) -> bool:
        """Check if this setting uses IDS grounding/constraints."""
        return self in (ExperimentSetting.DIRECT_QA_GROUNDED, ExperimentSetting.CYPHER_STRICT)
    
    @property
    def uses_strict_constraints(self) -> bool:
        """Check if this setting uses strict (grammar) constraints."""
        return self == ExperimentSetting.CYPHER_STRICT


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables can be set directly or via a .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )
    
    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://neo4j:7687",
        description="Neo4j Bolt connection URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: SecretStr = Field(
        default=SecretStr("password"),
        description="Neo4j password",
    )
    
    # LLM Configuration
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI,
        description="LLM provider to use: 'local', 'gemini', or 'openai'",
    )
    llm_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the LLM provider (required for gemini/openai)",
    )
    llm_model_name: str = Field(
        default="gemini-3-pro-preview",
        description="Model name or local path for the LLM",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible APIs (e.g. OpenRouter)",
    )
    
    # File Paths
    ifc_file_path: Path = Field(
        default=Path("/app/data/model.ifc"),
        description="Path to the IFC file",
    )
    ids_file_path: Path = Field(
        default=Path("/app/data/requirements.ids"),
        description="Path to the IDS file",
    )
    
    model_dump_path: Optional[Path] = Field(
        default=None,
        description="Path to the model dump JSON file for Direct QA mode",
    )

    test_set_path: Optional[Path] = Field(
        default=None,
        description="Path to the test-set CSV/JSON the API should serve. "
                    "Takes precedence over the /app/data/test_set.{csv,json} "
                    "defaults, so a run can point at e.g. ch9_demo_b15.json "
                    "without renaming files.",
    )
    
    # Application Settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # API Server
    api_bearer_token: SecretStr = Field(
        default=SecretStr(""),
        description="Bearer token required for authenticated API requests. "
                    "Empty disables the API (all requests return 503).",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="Interface for the FastAPI app to bind to",
    )
    api_port: int = Field(
        default=8000,
        description="Port for the FastAPI app",
    )

    @field_validator("model_dump_path", "test_set_path", mode="before")
    @classmethod
    def _blank_optional_path_is_none(cls, v):
        """Treat an empty/whitespace env var as unset.

        ``TEST_SET_PATH=${TEST_SET_PATH:-}`` in docker-compose reaches the
        process as the empty string, and ``Path("")`` is ``Path(".")`` — an
        existing directory. Without this, "unset" silently resolves to the
        working directory.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def neo4j_password_value(self) -> str:
        """Get the Neo4j password as a plain string."""
        return self.neo4j_password.get_secret_value()

    @property
    def llm_api_key_value(self) -> Optional[str]:
        """Get the LLM API key as a plain string."""
        if self.llm_api_key:
            return self.llm_api_key.get_secret_value()
        return None

    @property
    def api_bearer_token_value(self) -> str:
        """Get the API bearer token as a plain string ('' if unset)."""
        return self.api_bearer_token.get_secret_value()


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the application settings.
    
    Returns:
        Settings: The application settings instance.
    """
    return settings
