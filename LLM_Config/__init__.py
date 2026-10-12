"""
LLM 配置管理模块
提供统一的模型配置加载和访问接口
"""

from .load_models import (
    load_model_config,
    get_model_config,
    get_model_api_keys,
    get_default_config,
    get_phase_model_config,
    get_phase_model_name,
    list_available_models,
    get_openai_config,
    get_base_url,
    get_api_key,
    get_model_name,
    resolve_model_name,
    DEFAULT_MODEL,
)

__all__ = [
    "load_model_config",
    "get_model_config",
    "get_model_api_keys",
    "get_default_config",
    "get_phase_model_config",
    "get_phase_model_name",
    "list_available_models",
    "get_openai_config",
    "get_base_url",
    "get_api_key",
    "get_model_name",
    "resolve_model_name",
    "DEFAULT_MODEL",
]

__version__ = "1.0.0"
