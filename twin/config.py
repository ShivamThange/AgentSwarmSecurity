from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DANGEROUS_TOOLS = [
    "transfer_funds", "wire_transfer", "make_payment", "delete_records",
    "drop_table", "write_prod", "deploy", "exfiltrate", "send_email",
    "http_post", "execute_shell", "grant_access", "disable_guardrail",
]

DEFAULT_PROHIBITION_MARKERS = [
    "do not", "don't", "must not", "never", "without moving",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TWIN_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///twin.db"
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    auth_enabled: bool = True
    bootstrap_admin_key: Optional[str] = None
    rate_limit_per_minute: int = 600

    max_batch_size: int = 500
    max_body_bytes: int = 8 * 1024 * 1024

    embeddings_backend: Literal["sentence-transformers", "hashing"] = (
        "sentence-transformers"
    )
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_cache_size: int = 8192
    embeddings_device: Optional[str] = None

    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_model_deep: Optional[str] = None
    llm_timeout: float = 20.0
    llm_max_retries: int = 2

    # Judge backend selection. "llm" (default) uses the OpenAI-compatible
    # judges; "llamafirewall" runs Meta's LlamaFirewall scanners as the small
    # tier (LLM judge stays as the deep tier when a key is configured);
    # "stub" forces the deterministic judge only.
    judge_backend: Literal["llm", "llamafirewall", "stub"] = "llm"
    llamafirewall_scanners: list[str] = Field(
        default_factory=lambda: ["prompt_guard", "alignment_check"])

    # Inline rail backend. "native" (default) is the deterministic
    # zero-inference gate; "nemo" layers NVIDIA NeMo Guardrails advisory
    # denials on top of it (requires TWIN_NEMO_CONFIG_PATH).
    guard_backend: Literal["native", "nemo"] = "native"
    nemo_config_path: Optional[str] = None

    flag_threshold: float = 0.60
    watch_threshold: float = 0.35
    escalate_threshold: float = 0.45
    hard_flag_severity: float = 0.85
    low_privilege_sample_rate: int = 3

    dangerous_tools: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DANGEROUS_TOOLS))
    prohibition_markers: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PROHIBITION_MARKERS))

    detection_cache_size: int = 4096

    # Escalation-rate anomaly monitor (guards the paid judge tier against
    # adversarial flooding). ratio = escalated / analysed spans in the window.
    escalation_window_seconds: float = 300.0
    escalation_ratio_threshold: float = 0.5
    escalation_rate_threshold_per_min: Optional[float] = None
    escalation_min_samples: int = 20

    # Per-workflow threshold overrides, keyed by Span.workflow. JSON env, e.g.
    # TWIN_THRESHOLD_PROFILES={"finance":{"flag_threshold":0.5}}. Recognised
    # keys: flag_threshold, watch_threshold, escalate_threshold,
    # hard_flag_severity.
    threshold_profiles: dict[str, dict[str, float]] = Field(
        default_factory=dict)

    retention_days: Optional[int] = None

    metrics_enabled: bool = True
    cors_origins: list[str] = Field(default_factory=list)

    # Optional path to a JSON file mapping audit action -> list of compliance
    # clauses. Entries merge over (and override) the built-in EU AI Act / NIST
    # AI RMF / ISO 42001 defaults, so an org can add its own framework without
    # forking the code.
    compliance_map_path: Optional[str] = None

    log_level: str = "INFO"
    log_json: bool = True

    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("dangerous_tools", "prohibition_markers", "cors_origins",
                     "llamafirewall_scanners",
                     mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                import json
                return json.loads(v)
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def effective_deep_model(self) -> str:
        return self.llm_model_deep or self.llm_model

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
