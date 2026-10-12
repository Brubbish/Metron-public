import time
import functools
import asyncio
import json
import os

from langchain_community.callbacks import get_openai_callback


def _result_content(result):
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("output", "content"):
            if result.get(key) is not None:
                v = result[key]
                return v if isinstance(v, str) else str(v)
        return str(result)
    messages = getattr(result, "messages", None)
    if messages:
        return getattr(messages[-1], "content", str(messages[-1]))
    content = getattr(result, "content", None)
    if content is not None:
        return content
    return str(result)


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
        # Safely serialize input
        try:
            input_str = json.dumps(tool_input, ensure_ascii=False, default=str)
            if len(input_str) > 200:
                input_str = input_str[:200] + "..."
        except Exception:
            input_str = str(tool_input)[:200]
        obs_str = str(observation)[:300] if observation else ""
        steps.append({"tool": tool_name, "input": input_str, "output_preview": obs_str})
    return steps


def _result_messages(result):
    if hasattr(result, "messages"):
        return result.messages
    if hasattr(result, "usage_metadata"):
        return [result]
    return []


class AsyncFuncStats:
    def __init__(self):
        self.call_counts = {}
        self.total_times = {}
        self.token_usage = {}
        self.call_history = []

    def stats_decorator(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = func.__name__
            start = time.perf_counter()

            with get_openai_callback() as cb:
                result = await func(*args, **kwargs)

            elapsed = time.perf_counter() - start

            # Record counts and times
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
            self.total_times[name] = self.total_times.get(name, 0) + elapsed

            # Extract tool calls
            tool_calls = _extract_tool_calls(result)

            # Token usage from callback (captures all LLM calls within the agent)
            input_tokens = cb.prompt_tokens
            output_tokens = cb.completion_tokens

            # Get output text
            output_text = _result_content(result)

            # Build structured record
            call_record = {
                "function": name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start)),
                "elapsed_time": round(elapsed, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": cb.total_tokens,
                "tool_calls": tool_calls,
                "tool_count": len(tool_calls),
                "output_preview": output_text[:300] if output_text else "",
            }
            prev_in, prev_out = self.token_usage.get(name, (0, 0))
            self.token_usage[name] = (prev_in + input_tokens, prev_out + output_tokens)
            self.call_history.append(call_record)
            return output_text
        return wrapper

    def print_summary(self):
        print("\nAgents run summary:")
        total_in = total_out = 0
        for name in self.call_counts:
            count = self.call_counts[name]
            total_time = self.total_times[name]
            input_tokens, output_tokens = self.token_usage.get(name, (0, 0))
            total_in += input_tokens
            total_out += output_tokens
            tool_total = sum(r["tool_count"] for r in self.call_history if r["function"] == name)
            print(f"{name}: runs {count} times, total cost: {total_time:.4f}s, "
                  f"Tokens: Input={input_tokens}, Output={output_tokens}, Tools={tool_total}")
        print(f"Total tokens: Input={total_in}, Output={total_out}, Sum={total_in + total_out}")

    def save_to_file(self, filepath):
        """Save structured call history as JSON."""
        data = {
            "summary": {
                "call_counts": self.call_counts,
                "total_times": {k: round(v, 2) for k, v in self.total_times.items()},
                "token_usage": {k: {"input": v[0], "output": v[1]}
                               for k, v in self.token_usage.items()},
                "total_tool_calls": sum(r["tool_count"] for r in self.call_history),
            },
            "call_history": self.call_history,
        }
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, "func_stats.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Full call history saved to '{filepath}'")


AgentTimer = AsyncFuncStats()
