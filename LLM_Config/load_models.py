import os
from typing import Any, Dict, Optional
from pathlib import Path

from llm.openrouter_utils import get_openrouter_extra_body

# 配置文件路径
MODEL_CONFIG_PATH = Path(__file__).parent / "models.txt"

# 默认模型名称
DEFAULT_MODEL = "deepseek-chat"

# Batch children receive one selected key through this process-local override.
# It is intentionally separate from the per-model configuration overrides.
API_KEY_OVERRIDE_ENV = "THEAUTO_LLM_API_KEY_OVERRIDE"

# 缓存配置
_cached_config: Optional[Dict[str, Dict[str, Any]]] = None
_cached_phase_models: Optional[Dict[str, str]] = None
_cached_aliases: Optional[Dict[str, str]] = None

PHASE_ALIASES = {
    "default": "default",
    "direct": "direct",
    "direct_ask": "direct",
    "ask_ds": "direct",
    "error_extract": "direct",
    "artifact_summarizer": "artifact_summarizer",
    "summarizer": "artifact_summarizer",
    "readrepo": "readrepo",
    "read_repo": "readrepo",
    "summary": "readrepo",
    "phase2": "phase2",
    "phase2_executor": "phase2",
    "constructscript": "phase2",
    "build": "phase2",
    "phase3": "phase3",
    "phase3_fixer": "phase3",
    "errorsummary": "phase3",
    "fixer": "phase3",
    "check": "check",
}


def normalize_phase_name(phase: Optional[str]) -> str:
    """Normalize phase aliases used by agents and direct LLM calls."""
    if not phase:
        return "default"
    key = phase.strip().lower().replace("-", "_")
    return PHASE_ALIASES.get(key, key)


def _parse_phase_list(raw_phases: str) -> list[str]:
    phases = []
    for item in raw_phases.replace(";", "|").split("|"):
        phase = normalize_phase_name(item)
        if phase and phase not in phases:
            phases.append(phase)
    return phases


def _parse_alias_list(raw_aliases: str) -> list[str]:
    aliases = []
    for item in raw_aliases.replace(";", "|").split("|"):
        alias = item.strip()
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _parse_api_key_list(raw_api_keys: str, line_num: int) -> list[str]:
    api_keys = [item.strip() for item in raw_api_keys.split("|")]
    if not api_keys or any(not item for item in api_keys):
        raise ValueError(f"Invalid API key list at line {line_num}: empty key entry")
    if len(set(api_keys)) != len(api_keys):
        raise ValueError(f"Invalid API key list at line {line_num}: duplicate key entry")
    return api_keys


def _alias_key(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_").replace("/", "_").replace(".", "_")


def load_model_config(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    从 models.txt 读取模型配置，返回字典格式：
    {
        "deepseek-chat": {
            "base_url": "http://10.32.15.129:32551/...",
            "api_key": "<primary-key>",
            "api_keys": ["<primary-key>", "<backup-key>"],
            "model": "DeepSeek-R1-0528"
        },
        ...
    }
    
    Args:
        force_reload: 是否强制重新加载配置，默认使用缓存
        
    Returns:
        模型配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置格式错误
    """
    global _cached_config, _cached_phase_models, _cached_aliases
    
    # 如果已有缓存且不强制重载，直接返回
    if _cached_config is not None and not force_reload:
        return _cached_config
    
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Model config file {MODEL_CONFIG_PATH} not found. "
            f"Please create it with format: model_name,base_url,primary_key[|backup_key...]"
        )
    
    model_config = {}
    phase_models = {}
    alias_models = {}
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue
            
            # 解析配置行：model_name,base_url,key1[|key2...][,phase1|phase2][,alias1|alias2]
            parts = [p.strip() for p in line.split(",", 4)]
            if len(parts) not in (3, 4, 5):
                raise ValueError(
                    f"Invalid config at line {line_num}. "
                    "Expected format: model_name,base_url,primary_key[|backup_key...]"
                    "[,phase1|phase2][,alias1|alias2]"
                )
            
            model_name, base_url, raw_api_keys = parts[:3]
            phases = _parse_phase_list(parts[3]) if len(parts) >= 4 else []
            aliases = _parse_alias_list(parts[4]) if len(parts) == 5 else []
            
            # 支持环境变量覆盖
            base_url_env = os.getenv(f"{model_name.upper().replace('-', '_')}_BASE_URL")
            api_key_env = os.getenv(f"{model_name.upper().replace('-', '_')}_API_KEY")
            api_keys = [api_key_env] if api_key_env else _parse_api_key_list(raw_api_keys, line_num)
            
            model_config[model_name] = {
                "base_url": base_url_env or base_url,
                "api_key": api_keys[0],
                "api_keys": api_keys,
                "model": model_name,
                "phases": phases,
                "aliases": aliases,
                "extra_body": get_openrouter_extra_body(base_url_env or base_url),
            }
            for phase in phases:
                phase_models[phase] = model_name
            for alias in aliases:
                alias_models[_alias_key(alias)] = model_name
    
    _cached_config = model_config
    _cached_phase_models = phase_models
    _cached_aliases = alias_models
    return model_config


def resolve_model_name(model_name: str) -> str:
    """Resolve a configured model name or alias from models.txt."""
    configs = load_model_config()
    if model_name in configs:
        return model_name
    alias_models = _cached_aliases or {}
    resolved = alias_models.get(_alias_key(model_name))
    if resolved:
        return resolved
    available = ", ".join(configs.keys())
    aliases = ", ".join(sorted(alias_models.keys()))
    suffix = f"; aliases: {aliases}" if aliases else ""
    raise KeyError(f"Model '{model_name}' not found in config. Available models: {available}{suffix}")


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    获取指定模型的配置。
    
    Args:
        model_name: 模型名称，如果为 None 则返回默认模型配置
        
    Returns:
        包含 base_url, api_key, model 的配置字典
        
    Raises:
        KeyError: 模型不存在
    """
    configs = load_model_config()
    
    # 如果没有指定模型，使用默认模型
    if model_name is None:
        model_name = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)
    
    model_name = resolve_model_name(model_name)
    
    config = dict(configs[model_name])
    api_key_override = os.getenv(API_KEY_OVERRIDE_ENV)
    if api_key_override:
        config["api_key"] = api_key_override
    return config


def get_model_api_keys(model_name: Optional[str] = None) -> tuple[str, ...]:
    """Return the configured key pool without applying a child-process override."""
    configs = load_model_config()
    if model_name is None:
        model_name = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)
    resolved_name = resolve_model_name(model_name)
    return tuple(configs[resolved_name]["api_keys"])


def get_default_config() -> Dict[str, Any]:
    """
    获取默认模型的配置。
    
    Returns:
        默认模型的配置字典
    """
    return get_model_config(DEFAULT_MODEL)


def get_phase_model_name(phase: Optional[str] = None) -> str:
    """
    获取指定 workflow 阶段使用的模型名称。

    models.txt 可在第 4 列配置阶段映射，例如：
    model_name,base_url,api_key,default|direct|readrepo|phase2|phase3|check
    """
    configs = load_model_config()
    normalized_phase = normalize_phase_name(phase)
    env_key = f"LLM_MODEL_{normalized_phase.upper()}"
    env_model = os.getenv(env_key)
    if env_model:
        try:
            return resolve_model_name(env_model)
        except KeyError as exc:
            available = ", ".join(configs.keys())
            raise KeyError(f"{env_key}={env_model} not found in config. Available models: {available}") from exc

    phase_models = _cached_phase_models or {}
    if normalized_phase in phase_models:
        return phase_models[normalized_phase]
    if normalized_phase == "artifact_summarizer" and "readrepo" in phase_models:
        return phase_models["readrepo"]
    if "default" in phase_models:
        return phase_models["default"]
    return os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)


def get_phase_model_config(phase: Optional[str] = None) -> Dict[str, Any]:
    """获取指定 workflow 阶段的 OpenAI-compatible 模型配置。"""
    return get_model_config(get_phase_model_name(phase))


def list_available_models() -> list[str]:
    """
    列出所有可用的模型名称。
    
    Returns:
        模型名称列表
    """
    return list(load_model_config().keys())


def get_openai_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    获取 OpenAI 兼容的配置（用于 langchain 等库）。
    
    Args:
        model_name: 模型名称，如果为 None 则使用默认模型
        
    Returns:
        包含 base_url, api_key, model 的配置字典
    """
    return get_model_config(model_name)


# 便捷函数：获取配置的各个部分
def get_base_url(model_name: Optional[str] = None) -> str:
    """获取指定模型的 base_url"""
    return get_model_config(model_name)["base_url"]


def get_api_key(model_name: Optional[str] = None) -> str:
    """获取指定模型的 api_key"""
    return get_model_config(model_name)["api_key"]


def get_model_name(model_name: Optional[str] = None) -> str:
    """获取模型名称（如果传入 None 则返回默认模型名）"""
    if model_name is None:
        return resolve_model_name(os.getenv("DEFAULT_MODEL", DEFAULT_MODEL))
    return resolve_model_name(model_name)
