"""LangChain chat model creation from LLM_Config/models.txt."""
from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.dirname(__file__))
try:
    from LLM_Config import get_model_config, get_phase_model_config
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LLM_Config"))
    from load_models import get_model_config, get_phase_model_config  # type: ignore
from llm.openrouter_utils import get_client_max_retries

load_dotenv()

DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 4096
_MODEL_CACHE = {}


def get_chat_model(
    phase: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: Optional[float] = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """Create/cache a ChatOpenAI client for a workflow phase or explicit model."""
    config = get_model_config(model_name) if model_name else get_phase_model_config(phase)
    cache_key = (
        config["base_url"],
        config["api_key"],
        config["model"],
        temperature,
        top_p,
        max_tokens,
    )
    if cache_key not in _MODEL_CACHE:
        kwargs = dict(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=120,
            max_retries=get_client_max_retries(config["base_url"]),
        )
        if top_p is not None:
            kwargs["top_p"] = top_p
        extra_body = config.get("extra_body")
        if extra_body:
            kwargs["extra_body"] = extra_body
        _MODEL_CACHE[cache_key] = ChatOpenAI(**kwargs)
    return _MODEL_CACHE[cache_key]


READREPO_CHAT_MODEL = get_chat_model("readrepo")
PHASE2_CHAT_MODEL = get_chat_model("phase2")
PHASE3_CHAT_MODEL = get_chat_model("phase3")
CHECK_CHAT_MODEL = get_chat_model("check")
DIRECT_CHAT_MODEL = get_chat_model("direct", max_tokens=DEFAULT_MAX_TOKENS)

# Backward-compatible aliases for existing imports.
DS_CHAT_MODEL = PHASE2_CHAT_MODEL
Si_DS_CHAT_MODEL = READREPO_CHAT_MODEL
