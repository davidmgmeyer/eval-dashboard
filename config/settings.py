"""Centralized configuration management for the eval dashboard."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Try to import python-dotenv for .env file loading
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import streamlit for secrets access
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# Track where API key was loaded from
_api_key_source: str = "not configured"


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    ANTHROPIC_API_KEY: Optional[str] = None
    AUTHORIZED_EMAILS: List[str] = field(default_factory=list)
    APP_TITLE: str = "Eval Dashboard"
    APP_ICON: str = "📊"
    MAX_UPLOAD_SIZE_MB: int = 50
    ENABLE_AUTH: bool = False
    DEBUG: bool = False

    @property
    def has_api_key(self) -> bool:
        """Check if API key is configured and looks valid."""
        return bool(self.ANTHROPIC_API_KEY and self.ANTHROPIC_API_KEY.startswith('sk-'))

    @property
    def api_key_source(self) -> str:
        """Return where the API key was loaded from."""
        return _api_key_source

    @property
    def api_key_preview(self) -> str:
        """Return a safe preview of the API key (first 12 chars)."""
        if self.ANTHROPIC_API_KEY:
            return f"{self.ANTHROPIC_API_KEY[:12]}..."
        return "not set"


# Singleton instance
_settings: Optional[Settings] = None


def _get_env_value(key: str, default: str = "") -> tuple:
    """Get a value from environment, Streamlit secrets, or default.

    Priority:
    1. Environment variable
    2. Streamlit secrets (for deployed apps)
    3. Default value

    Returns:
        Tuple of (value, source) where source indicates where value came from
    """
    # Check environment variable first
    value = os.environ.get(key)
    if value is not None:
        return value, "environment variable"

    # Check Streamlit secrets (for deployed environments)
    if STREAMLIT_AVAILABLE:
        try:
            if key in st.secrets:
                return str(st.secrets[key]), "Streamlit secrets"
        except Exception:
            # st.secrets may not be available in all contexts
            pass

    return default, "default"


def _parse_bool(value: str) -> bool:
    """Parse a string value to boolean."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_list(value: str) -> List[str]:
    """Parse a comma-separated string to a list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_settings() -> Settings:
    """Load settings from environment variables and .env file."""
    global _api_key_source

    # Load .env file if python-dotenv is available
    if DOTENV_AVAILABLE:
        # Try to find .env in project root
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            env_source = ".env file"
        else:
            # Try current working directory
            load_dotenv()
            env_source = ".env file"
    else:
        env_source = "environment"

    # Get API key and track its source
    api_key, key_source = _get_env_value("ANTHROPIC_API_KEY")
    if api_key:
        if key_source == "environment variable" and DOTENV_AVAILABLE:
            _api_key_source = ".env file"
        else:
            _api_key_source = key_source
    else:
        _api_key_source = "not configured"
        api_key = None

    return Settings(
        ANTHROPIC_API_KEY=api_key or None,
        AUTHORIZED_EMAILS=_parse_list(_get_env_value("AUTHORIZED_EMAILS")[0]),
        APP_TITLE=_get_env_value("APP_TITLE", "Eval Dashboard")[0],
        APP_ICON=_get_env_value("APP_ICON", "📊")[0],
        MAX_UPLOAD_SIZE_MB=int(_get_env_value("MAX_UPLOAD_SIZE_MB", "50")[0]),
        ENABLE_AUTH=_parse_bool(_get_env_value("ENABLE_AUTH", "false")[0]),
        DEBUG=_parse_bool(_get_env_value("DEBUG", "false")[0]),
    )


def get_settings() -> Settings:
    """Get the singleton settings instance.

    Settings are loaded once on first access and cached for subsequent calls.

    Returns:
        Settings instance with configuration values
    """
    global _settings
    if _settings is None:
        _settings = _load_settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment.

    Useful for testing or when environment changes.

    Returns:
        Fresh Settings instance
    """
    global _settings
    _settings = _load_settings()
    return _settings
