# ──────────────────────────────────────────────────────────────
# Sahayak API — Configuration
# ──────────────────────────────────────────────────────────────
# This file reads our .env file and turns it into typed settings.
# "Typed" means: if you typo a variable or forget to set DATABASE_URL,
# the app fails loudly at startup instead of mysteriously later.
#
# Think of this as the control panel: every tunable knob (which env
# we're in, the DB connection, API keys) lives here, in ONE place.

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime settings, loaded from environment / .env file."""

    # Tell pydantic to read values from a .env file.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown vars instead of crashing
    )

    # ── Environment ──
    environment: str = "development"

    # ── Database ──
    database_url: str = "postgresql+psycopg://sahayak:changeme_local_only@postgres:5432/sahayak"
    postgres_host: str = "postgres"

    # ── LLM keys (empty for now; we fill these in later sprints) ──
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # ── LLM knobs ──
    # Which Gemini model to call — override in .env without a code change.
    # (Gemini 2.5-flash was retired for new users; the API's 404 message
    # pointed us at 3.6-flash. Models rotate — this knob absorbs that.)
    gemini_model: str = "gemini-3.6-flash"


# @lru_cache means: build the Settings object ONCE, then reuse the same one.
# Creating it reads the .env file — we don't want to do that on every request.
@lru_cache
def get_settings() -> Settings:
    return Settings()


# A single shared instance the rest of the app imports.
settings = get_settings()
