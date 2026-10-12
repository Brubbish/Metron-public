"""
基于 LangChain 的 LLM 调用包装
替代原有的 autogen 调用方式
"""
import json
import logging
import re

from core.AgentTimer import AgentTimer

logger = logging.getLogger(__name__)


_CONTROL_TOKEN_PATTERNS = (
    (re.compile(r"<\|tool_call\|>.*?<tool_call\|>", re.DOTALL), ""),
    (re.compile(r"<\|/?tool_call\|>"), ""),
    (re.compile(r"<\|\"?\|>"), ""),
    (re.compile(r"<\|[^>\n]{1,200}\|>"), ""),
)


def _sanitize_agent_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    for pattern, repl in _CONTROL_TOKEN_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text")
                if value is None:
                    value = item.get("content")
                if value is None:
                    parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(_content_to_text(value))
            else:
                value = getattr(item, "text", None)
                if value is None:
                    value = getattr(item, "content", None)
                parts.append(_content_to_text(value) if value is not None else str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content", "output"):
            if content.get(key) is not None:
                return _content_to_text(content[key])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _message_content(message):
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _agent_result_content(result) -> str:
    if isinstance(result, str):
        return _sanitize_agent_text(result)
    if isinstance(result, dict):
        for key in ("output", "content"):
            if result.get(key) is not None:
                text = _content_to_text(result[key])
                if text:
                    return _sanitize_agent_text(text)
        messages = result.get("messages")
        if messages:
            for message in reversed(messages):
                text = _content_to_text(_message_content(message))
                if text:
                    return _sanitize_agent_text(text)
        tool_calls = _extract_tool_calls(result)
        if tool_calls:
            return _sanitize_agent_text("Tool calls:\n" + json.dumps(tool_calls, ensure_ascii=False, indent=2))
        return _sanitize_agent_text(repr({k: type(v).__name__ for k, v in result.items()})[:2000])
    messages = getattr(result, "messages", None)
    if messages:
        for message in reversed(messages):
            text = _content_to_text(_message_content(message))
            if text:
                return _sanitize_agent_text(text)
    content = getattr(result, "content", None)
    if content is not None:
        return _sanitize_agent_text(_content_to_text(content))
    output = getattr(result, "output", None)
    if output is not None:
        return _sanitize_agent_text(_content_to_text(output))
    tool_calls = _extract_tool_calls(result)
    if tool_calls:
        return _sanitize_agent_text("Tool calls:\n" + json.dumps(tool_calls, ensure_ascii=False, indent=2))
    return _sanitize_agent_text(repr(result)[:2000])


def _extract_tool_calls(result) -> list:
    """Extract tool call info from AgentExecutor intermediate_steps."""
    steps = []
    intermediate = None
    if isinstance(result, dict):
        intermediate = result.get("intermediate_steps", [])
    else:
        intermediate = getattr(result, "intermediate_steps", [])
    if not intermediate:
        return steps
    for action, observation in intermediate:
        tool_name = getattr(action, "tool", None) or "unknown"
        tool_input = getattr(action, "tool_input", None) or {}
        obs_str = str(observation)[:200] if observation else ""
        steps.append({"tool": tool_name, "input": tool_input, "output_preview": obs_str})
    return steps


@AgentTimer.stats_decorator
async def call_error_agent(task: str, error_summary_agent):
    """调用错误总结 agent"""
    result = await error_summary_agent.run(task=task)
    text = _agent_result_content(result)
    print(text)
    return text


@AgentTimer.stats_decorator
async def call_build_agent(task: str, build_repo_agent):
    """调用构建脚本 agent"""
    result = await build_repo_agent.run(task=task)
    text = _agent_result_content(result)
    print(text)
    return text


@AgentTimer.stats_decorator
async def call_check_agent(task: str, checker_agent):
    """调用检查 agent"""
    result = await checker_agent.run(task=task)
    text = _agent_result_content(result)
    print(text)
    return text


@AgentTimer.stats_decorator
async def call_readrepo_agent(task, read_repo_agent):
    """调用读取仓库 agent"""
    result = await read_repo_agent.run(task=task)
    text = _agent_result_content(result)
    print(text)
    return text


@AgentTimer.stats_decorator
async def ask_llm(client, messages, **kwargs):
    """
    直接调用 LLM（不使用 agent）
    
    Args:
        client: LangChain ChatModel 实例
        messages: 消息列表
        **kwargs: 其他参数（如 temperature, max_tokens 等）
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # 转换消息格式
    langchain_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            # 可以扩展其他消息类型
        else:
            langchain_messages.append(msg)
    
    # 调用模型
    response = await client.ainvoke(langchain_messages, **kwargs)
    
    # 返回兼容格式
    class Response:
        def __init__(self, content):
            self.content = content
            self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})]
    
    return Response(response.content)
