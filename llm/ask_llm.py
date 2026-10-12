"""
基于 LangChain 的 LLM 直接调用
替代原有的 OpenAI client 调用方式
使用统一的配置管理系统
"""
import os
import re
import json
import time
import asyncio
import threading
import sys
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken
from llm.openrouter_utils import get_retry_delay, is_retryable_provider_error

# 添加项目根目录路径
sys.path.insert(0, os.path.dirname(__file__))
try:
    from LLM_Config import get_phase_model_name
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLM_Config'))
    from load_models import get_phase_model_name  # type: ignore

import llm.models as llm_models

# 默认模型名称（统一在 LLM_Config/models.txt 的 direct 阶段配置）
DEFAULT_MODEL = get_phase_model_name("direct")
DEFAULT_TOKENIZER_MODEL = "deepseek-chat"  # tiktoken 使用的模型名


class _LLMUsageTracker:
    """Thread-safe tracker for all LLM calls made through ask_ds/ask_ds_async."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_elapsed = 0.0
        self.by_caller = {}  # caller_label -> {calls, input_tokens, output_tokens, elapsed}

    def record(self, input_tokens: int, output_tokens: int, elapsed: float, caller: str = ""):
        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_elapsed += elapsed
            if caller:
                entry = self.by_caller.setdefault(caller, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed": 0.0})
                entry["calls"] += 1
                entry["input_tokens"] += input_tokens
                entry["output_tokens"] += output_tokens
                entry["elapsed"] += elapsed

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_elapsed": round(self.total_elapsed, 2),
                "by_caller": dict(self.by_caller),
            }

    def reset(self):
        with self._lock:
            self.total_calls = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_elapsed = 0.0
            self.by_caller = {}


llm_usage_tracker = _LLMUsageTracker()

# Backward-compatible direct client; direct_ask 会按传入 model 动态选择。
si_ds_client = llm_models.get_chat_model("direct", max_tokens=llm_models.DEFAULT_MAX_TOKENS)


def _get_direct_client(model=None):
    if model is None:
        return llm_models.get_chat_model("direct", max_tokens=llm_models.DEFAULT_MAX_TOKENS)
    return llm_models.get_chat_model(model_name=model, max_tokens=llm_models.DEFAULT_MAX_TOKENS)


def get_model_config(model_name):
    """获取指定模型的配置信息"""
    model_configs = {
        "deepseek-v3-250324": {"max_ctx": 120 * 1024, "max_response": 8 * 1024},
        "deepseekV31T": {"max_ctx": 60 * 1024, "max_response": 4 * 1024},
        "DeepSeek-R1-0528": {"max_ctx": 64 * 1024, "max_response": 8 * 1024},
    }
    return model_configs.get(model_name, {"max_ctx": 60 * 1024, "max_response": 4 * 1024})


def num_tokens_from_messages(messages, model=None):
    """返回 messages 列表的 token 数目"""
    if model is None:
        model = DEFAULT_TOKENIZER_MODEL
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    
    total = 0
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content", "")
        else:
            content = str(m)
        total += len(enc.encode(content))
    return total


def split_content(content, max_tokens, model=None):
    """将内容按照token数进行分片"""
    if model is None:
        model = DEFAULT_TOKENIZER_MODEL
    chunks = []
    
    if not content:
        return chunks
    
    current_chunk_tokens = num_tokens_from_messages([{"content": content}], model)
    
    if current_chunk_tokens <= max_tokens:
        chunks.append(content)
        return chunks
    
    content_chars = content.split("\n")
    current_chunk = ""
    per_chunk = 100
    
    for i in range(0, len(content_chars), per_chunk):
        chunk_lines = content_chars[i:i + per_chunk]
        temp_chunk = current_chunk + "\n".join(chunk_lines) + "\n"
        temp_tokens = num_tokens_from_messages([{"content": temp_chunk}], model)

        if temp_tokens <= max_tokens:
            current_chunk = temp_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = "\n".join(chunk_lines) + "\n"

    if current_chunk:
        chunks.append(current_chunk)
    
    print("分片数量: ", len(chunks))
    return chunks


def direct_ask(system_prompt, user_prompt, max_retries=5, retry_wait=3, model=None, caller=""):
    """同步方式直接询问 LLM"""
    if model is None:
        model = DEFAULT_MODEL
    attempt = 0

    while attempt < max_retries:
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            start = time.perf_counter()
            response = _get_direct_client(model).invoke(messages)
            elapsed = time.perf_counter() - start
            # Track usage
            meta = getattr(response, 'usage_metadata', None) or {}
            input_t = meta.get('input_tokens', 0) or num_tokens_from_messages(messages)
            output_t = meta.get('output_tokens', 0) or num_tokens_from_messages([{"content": response.content}])
            llm_usage_tracker.record(input_t, output_t, elapsed, caller=caller)
            return response.content

        except Exception as e:
            attempt += 1
            print(f"请求失败 (尝试 {attempt}/{max_retries}): {str(e)}")
            if attempt < max_retries:
                sleep_s = get_retry_delay(attempt, base_delay=retry_wait) if is_retryable_provider_error(e) else retry_wait
                time.sleep(sleep_s)
            else:
                print(f"达到最大重试次数，返回错误信息")
                return f"Error: 请求失败 - {str(e)}"


async def async_ask(system_prompt, user_prompt, max_retries=5, retry_wait=3, model=None, caller=""):
    """异步方式直接询问 LLM"""
    if model is None:
        model = DEFAULT_MODEL
    attempt = 0

    while attempt < max_retries:
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            start = time.perf_counter()
            response = await _get_direct_client(model).ainvoke(messages)
            elapsed = time.perf_counter() - start
            # Track usage
            meta = getattr(response, 'usage_metadata', None) or {}
            input_t = meta.get('input_tokens', 0) or num_tokens_from_messages(messages)
            output_t = meta.get('output_tokens', 0) or num_tokens_from_messages([{"content": response.content}])
            llm_usage_tracker.record(input_t, output_t, elapsed, caller=caller)
            return response.content

        except Exception as e:
            attempt += 1
            print(f"异步请求失败 (尝试 {attempt}/{max_retries}): {str(e)}")
            if attempt < max_retries:
                sleep_s = get_retry_delay(attempt, base_delay=retry_wait) if is_retryable_provider_error(e) else retry_wait
                await asyncio.sleep(sleep_s)
            else:
                print(f"达到最大重试次数，返回错误信息")
                return f"Error: 异步请求失败 - {str(e)}"


def ask_ds(system_prompt, user_prompt, max_retries=5, retry_wait=3, caller="ask_ds"):
    """
    DeepSeek 模型询问（保持与原 API 兼容）

    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        max_retries: 最大重试次数
        retry_wait: 重试等待时间

    Returns:
        str: LLM 响应内容
    """
    return direct_ask(system_prompt, user_prompt, max_retries, retry_wait, caller=caller)


async def ask_ds_async(system_prompt, user_prompt, max_retries=5, retry_wait=3, caller="ask_ds_async"):
    """
    DeepSeek 模型异步询问
    
    Args:
        system_prompt: 系统提示
        user_prompt: 用户提示
        max_retries: 最大重试次数
        retry_wait: 重试等待时间
    
    Returns:
        str: LLM 响应内容
    """
    return await async_ask(system_prompt, user_prompt, max_retries, retry_wait, caller=caller)


# 为了保持向后兼容，添加旧的函数名
def summary_from_chunks(chunks, system_prompt, summary_task_prompt, 
                       max_chunk_tokens=12000, model=None):
    """
    从多个文档片段生成摘要
    
    Args:
        chunks: 文档片段列表
        system_prompt: 系统提示
        summary_task_prompt: 摘要任务提示模板
        max_chunk_tokens: 每个片段最大 token 数
        model: 模型名称
    
    Returns:
        str: 合并后的摘要
    """
    if model is None:
        model = DEFAULT_MODEL
    summaries = []
    
    for i, chunk in enumerate(chunks):
        print(f"正在处理片段 {i+1}/{len(chunks)}...")
        user_prompt = summary_task_prompt.format(content=chunk)
        summary = direct_ask(system_prompt, user_prompt, model=model)
        summaries.append(summary)
    
    # 合并所有摘要
    if len(summaries) == 1:
        return summaries[0]
    else:
        # 如果有多个摘要，再次汇总
        combined = "\n\n".join(summaries)
        final_prompt = f"请将以下多个摘要合并为一个连贯的总结：\n\n{combined}"
        return direct_ask(system_prompt, final_prompt, model=model)
