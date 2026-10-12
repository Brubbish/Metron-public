"""LangChain agent factories for the build workflow."""
from __future__ import annotations

import json
import inspect
import logging
import os
import asyncio
import re
from typing import Any, List, Optional

from dotenv import load_dotenv
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
    create_agent = None
    _LEGACY_AGENT_API = True
except ImportError:  # LangChain 1.x
    from langchain.agents import create_agent

    AgentExecutor = None
    create_tool_calling_agent = None
    create_react_agent = None
    _LEGACY_AGENT_API = False
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import BaseTool, StructuredTool
from langchain.agents import Tool as SimpleTool

import llm.llm_prompts
import llm.models
from tools.ConstructScript import get_phase2_tools
from tools.ErrorSummary import get_phase3_tools
from tools.ReadRepo import get_readrepo_tools, get_zero_entry_tools
from llm.Run_LLM import _agent_result_content
from llm.openrouter_utils import get_retry_delay, is_retryable_provider_error

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LANGCHAIN_AGENT_DEBUG = os.environ.get("LANGCHAIN_AGENT_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

AGENT_MODE_ENV = "THEAUTO_AGENT_MODE"
DEFAULT_AGENT_MODE = "tool-calling"
JSON_PLANNER_AGENT_MODE = "json-planner"
VALID_AGENT_MODES = {DEFAULT_AGENT_MODE, JSON_PLANNER_AGENT_MODE}
_JSON_PLANNER_OBSERVATION_CHARS = 12000


def _callable_name(func: Any) -> str:
    return getattr(func, "name", getattr(func, "__name__", func.__class__.__name__))


def _tool_descriptions(tools: list) -> str:
    def _render_doc_lines(lines: list[str]) -> list[str]:
        rendered = []
        for line in lines:
            normalized = line.strip()
            if normalized.startswith(("- ", "* ")):
                normalized = normalized[2:].strip()
            rendered.append(f"- {normalized}")
        return rendered

    def _schema_lines(tool_item: Any) -> list[str]:
        schema = getattr(tool_item, "__tool_args_schema__", None)
        if not schema or not hasattr(schema, "model_fields"):
            return []
        fields = []
        for field_name, field in schema.model_fields.items():
            required = "required" if field.is_required() else "optional"
            annotation = str(getattr(field, "annotation", "")).replace("typing.", "")
            description = getattr(field, "description", "") or ""
            field_text = f"- {field_name} ({annotation}, {required})"
            if description:
                field_text += f": {description}"
            fields.append(field_text)
        return fields

    def _doc_sections(tool_item: Any) -> tuple[str, list[str], list[str], list[str]]:
        raw_doc = inspect.getdoc(tool_item) or ""
        lines = [line.strip() for line in raw_doc.splitlines() if line.strip()]
        if not lines:
            return (_callable_name(tool_item), [], [], [])

        purpose_lines: list[str] = []
        returns_lines: list[str] = []
        do_lines: list[str] = []
        dont_lines: list[str] = []
        section: str | None = None

        for line in lines:
            lowered = line.lower()
            if lowered.startswith("args:"):
                section = "args"
                continue
            if lowered.startswith("returns:"):
                section = "returns"
                continue
            if lowered.startswith("do:"):
                section = "do"
                continue
            if lowered.startswith("don't:") or lowered.startswith("dont:"):
                section = "dont"
                continue
            if section == "returns":
                returns_lines.append(line)
                continue
            if section == "do":
                do_lines.append(line)
                continue
            if section == "dont":
                dont_lines.append(line)
                continue
            if lowered.startswith(("use ", "use this", "only use", "call this", "when ", "if ")):
                do_lines.append(line)
                continue
            if lowered.startswith(("do not", "don't", "never ")):
                dont_lines.append(line)
                continue
            if not purpose_lines:
                purpose_lines.append(line)

        purpose = " ".join(purpose_lines).strip() or lines[0]
        return (purpose, returns_lines, do_lines, dont_lines)

    sections = []
    for func in tools:
        tool_name = _callable_name(func)
        purpose, returns_lines, do_lines, dont_lines = _doc_sections(func)
        schema_lines = _schema_lines(func)
        parts = [f"Tool: {tool_name}", f"Purpose: {purpose}"]
        if schema_lines:
            parts.append("Args:")
            parts.extend(schema_lines)
        if returns_lines:
            parts.append("Returns:")
            parts.extend(_render_doc_lines(returns_lines))
        if do_lines:
            parts.append("Do:")
            parts.extend(_render_doc_lines(do_lines))
        if dont_lines:
            parts.append("Don't:")
            parts.extend(_render_doc_lines(dont_lines))
        sections.append("\n".join(parts))
    return "\n\n".join(sections)


def wrap_tool(func: Any, name: Optional[str] = None, description: Optional[str] = None) -> BaseTool:
    """Wrap sync or async Python functions as LangChain tools."""
    if isinstance(func, BaseTool):
        return func
    tool_name = name or _callable_name(func)
    tool_description = description or getattr(func, "__doc__", None) or f"Tool: {tool_name}"
    args_schema = getattr(func, "__tool_args_schema__", None)
    if inspect.iscoroutinefunction(func):
        return StructuredTool.from_function(
            coroutine=func,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            handle_tool_error=True,
        )
    return StructuredTool.from_function(
        func=func,
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
        handle_tool_error=True,
    )


def _is_retryable_provider_error(exc: Exception) -> bool:
    return is_retryable_provider_error(exc)


def get_agent_mode() -> str:
    mode = os.environ.get(AGENT_MODE_ENV, DEFAULT_AGENT_MODE).strip().lower()
    return mode if mode in VALID_AGENT_MODES else DEFAULT_AGENT_MODE


def _strip_json_fence(text: str) -> str:
    stripped = str(text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    return fence_match.group(1).strip() if fence_match else stripped


def _extract_json_object(text: str) -> Optional[dict]:
    stripped = _strip_json_fence(text)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _planner_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _compact_observation(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= _JSON_PLANNER_OBSERVATION_CHARS:
        return text
    return text[:_JSON_PLANNER_OBSERVATION_CHARS] + "\n...[truncated]"


def _looks_like_provider_tool_payload(parsed: dict) -> bool:
    if "tool_calls" in parsed or "function" in parsed:
        return True
    if "name" in parsed and "arguments" in parsed:
        return True
    if "tool" in parsed and "args" in parsed:
        return True
    if "tool_name" in parsed and "tool_args" in parsed:
        return True
    return False


def _agent_class():
    return JsonPlannerAgent if get_agent_mode() == JSON_PLANNER_AGENT_MODE else LangChainAgent


def _make_agent(
    *,
    name: str,
    model,
    tools: list,
    system_message: str,
    max_iterations: int,
    fallback_output: Any,
):
    agent_cls = _agent_class()
    try:
        return agent_cls(
            name=name,
            model=model,
            tools=tools,
            system_message=system_message,
            max_iterations=max_iterations,
            fallback_output=fallback_output,
        )
    except TypeError as exc:
        if "fallback_output" not in str(exc):
            raise
        return agent_cls(
            name=name,
            model=model,
            tools=tools,
            system_message=system_message,
            max_iterations=max_iterations,
        )


class LangChainAgent:
    """Small wrapper that preserves the original agent.run(task=...) interface."""

    def __init__(
        self,
        name: str,
        model,
        tools: list,
        system_message: str,
        max_iterations: int = 10,
        fallback_output: Any = "Error occurred",
    ):
        self.name = name
        self.model = model
        self.tools = tools
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.fallback_output = fallback_output
        self.langchain_tools: List[BaseTool] = []
        self.tool_names = [_callable_name(tool_item) for tool_item in tools]
        self._agent_api = "legacy" if _LEGACY_AGENT_API else "graph"
        self._create_agent()

    def _create_agent(self) -> None:
        self.langchain_tools = [wrap_tool(tool_item) for tool_item in self.tools]
        if self._agent_api == "graph":
            self.executor = create_agent(
                model=self.model,
                tools=self.langchain_tools,
                system_prompt=self.system_message,
                debug=LANGCHAIN_AGENT_DEBUG,
                name=self.name,
            )
            return

        # Escape stray braces in system_message so ChatPromptTemplate doesn't
        # interpret JSON examples as template variables.
        safe_system = self.system_message.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", safe_system),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        model_with_tools = self.model.bind_tools(self.langchain_tools)
        agent = create_tool_calling_agent(model_with_tools, self.langchain_tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.langchain_tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    async def run(self, task: str):
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if self._agent_api == "graph":
                    result = await self.executor.ainvoke(
                        {"messages": [{"role": "user", "content": task}]},
                        config={"recursion_limit": max(4, self.max_iterations * 2 + 2)},
                    )
                    return _agent_result_content(result)
                result = await self.executor.ainvoke({"input": task})
                return _agent_result_content(result)
            except Exception as e:
                if _is_retryable_provider_error(e) and attempt < max_attempts:
                    logging.warning(
                        "Agent %s provider error on attempt %d/%d: %s",
                        self.name,
                        attempt,
                        max_attempts,
                        e,
                    )
                    await asyncio.sleep(get_retry_delay(attempt, base_delay=2.0))
                    continue
                logging.error("Agent %s error: %s", self.name, e)
                return {"output": f"Error: {str(e)}", "intermediate_steps": []}


class JsonPlannerAgent:
    """JSON-only planner loop; the controller executes requested tools."""

    def __init__(
        self,
        name: str,
        model,
        tools: list,
        system_message: str,
        max_iterations: int = 10,
        fallback_output: Any = "Error occurred",
    ):
        self.name = name
        self.model = model
        self.tools = tools
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.fallback_output = fallback_output
        self.tool_descriptions = _tool_descriptions(tools)
        self.langchain_tools = [wrap_tool(tool_item) for tool_item in tools]
        self.tool_map = {tool.name: tool for tool in self.langchain_tools}
        self.tool_names = list(self.tool_map)

    async def _call_model(self, messages: list[Any]) -> str:
        response = await self.model.ainvoke(messages)
        return _agent_result_content(response)

    async def _run_tool(self, tool_name: str, args: Any) -> Any:
        if not isinstance(args, dict):
            return {"status": "rejected", "error": "tool args must be a JSON object"}
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {
                "status": "rejected",
                "error": f"unknown tool: {tool_name}",
                "available_tools": self.tool_names,
            }
        try:
            return await tool.ainvoke(args)
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def _protocol_message(self) -> str:
        tool_text = self.tool_descriptions or ("\n".join(f"Tool: {name}" for name in self.tool_names) or "none")
        return (
            self.system_message
            + "\n\n"
            + llm_prompts.JSON_PLANNER_TOOL_PROTOCOL
            + "\nAvailable tools:\n"
            + tool_text
        )

    async def run(self, task: str):
        messages: list[Any] = [
            SystemMessage(content=self._protocol_message()),
            HumanMessage(content=task),
        ]
        max_attempts = 3
        for iteration in range(1, self.max_iterations + 1):
            raw = ""
            for attempt in range(1, max_attempts + 1):
                try:
                    raw = await self._call_model(messages)
                    break
                except Exception as exc:
                    if _is_retryable_provider_error(exc) and attempt < max_attempts:
                        logging.warning(
                            "JSON planner %s provider error on attempt %d/%d: %s",
                            self.name,
                            attempt,
                            max_attempts,
                            exc,
                        )
                        await asyncio.sleep(get_retry_delay(attempt, base_delay=2.0))
                        continue
                    logging.error("JSON planner %s error: %s", self.name, exc)
                    return {"output": f"Error: {str(exc)}", "intermediate_steps": []}

            parsed = _extract_json_object(raw)
            if not parsed:
                messages.append(AIMessage(content=raw[:2000]))
                messages.append(HumanMessage(content="Controller rejected the response: return exactly one raw JSON object."))
                continue

            status = str(parsed.get("status") or "").strip().lower()
            if status == "tool":
                tool_name = str(parsed.get("tool") or "").strip()
                observation = await self._run_tool(tool_name, parsed.get("args") or {})
                messages.append(AIMessage(content=json.dumps(parsed, ensure_ascii=False)))
                messages.append(HumanMessage(content="Tool observation:\n" + _compact_observation(observation)))
                continue

            if status == "final":
                return _planner_output_text(parsed.get("output", ""))

            if _looks_like_provider_tool_payload(parsed):
                messages.append(AIMessage(content=raw[:2000]))
                messages.append(HumanMessage(
                    content=(
                        "Controller rejected provider-native tool-call JSON. "
                        "Use the JSON controller protocol: "
                        '{"status":"tool","tool":"<exact tool name>","args":{...},"reason":"..."}'
                    )
                ))
                continue

            return json.dumps(parsed, ensure_ascii=False)

        return _planner_output_text(self.fallback_output)


class ReActAgent:
    """ReAct-format agent using text-based Thought/Action/Observation loop."""

    def __init__(self, name: str, model, tools: list, prompt_template: str, max_iterations: int = 10):
        self.name = name
        self.model = model
        self.tools = tools
        self.prompt_template = prompt_template
        self.max_iterations = max_iterations
        self.langchain_tools: List[BaseTool] = []
        self._create_agent()

    def _wrap_as_simple_tool(self, func) -> BaseTool:
        """Wrap async function as a simple Tool (single string input) for ReAct compatibility."""
        tool_name = _callable_name(func)
        tool_description = getattr(func, "__doc__", None) or f"Tool: {tool_name}"
        if inspect.iscoroutinefunction(func):
            return SimpleTool(
                name=tool_name,
                description=tool_description,
                coroutine=func,
                func=None,
            )
        return SimpleTool(
            name=tool_name,
            description=tool_description,
            func=func,
        )

    def _create_agent(self) -> None:
        self.langchain_tools = [self._wrap_as_simple_tool(t) for t in self.tools]
        prompt = PromptTemplate.from_template(self.prompt_template)
        agent = create_react_agent(self.model, self.langchain_tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.langchain_tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    async def run(self, task: str):
        try:
            result = await self.executor.ainvoke({"input": task})
            return result
        except Exception as e:
            logging.error("ReActAgent %s error: %s", self.name, e)
            return {"output": f"Error: {str(e)}", "intermediate_steps": []}


class ReadRepoAgentFactory:
    """Repository summary agent factory."""

    @staticmethod
    async def create(sysuse=None):
        tools = get_readrepo_tools()
        system_message = llm_prompts.READREPO_SYSTEM.format(tool_descriptions=_tool_descriptions(tools))
        return _make_agent(
            name="summary_assistant",
            model=models.get_chat_model("readrepo"),
            tools=tools,
            system_message=system_message,
            max_iterations=10,
            fallback_output={
                "status": "cannot_select",
                "build_entry_id": "",
                "package_ids": [],
                "source_dep_ids": [],
                "setup_command_ids": [],
                "ambiguity_decisions": [],
                "evidence_refs": [],
                "reason": "json_planner_failed",
            },
        )


class ZeroEntryReadRepoAgentFactory:
    """Bounded agent used only when deterministic entry discovery is empty."""

    @staticmethod
    async def create(repo_root: str, inventory: dict):
        tools = get_zero_entry_tools(repo_root, inventory)
        return _make_agent(
            name="zero_entry_discovery_assistant",
            model=models.get_chat_model("readrepo", temperature=0, top_p=None, max_tokens=4096),
            tools=tools,
            system_message=llm_prompts.ZERO_ENTRY_READREPO_SYSTEM,
            max_iterations=8,
            fallback_output={
                "status": "cannot_select",
                "pipelines": [],
                "reason": "json_planner_failed",
            },
        )


class ConstructScriptAgentFactory:
    """Phase2 executor agent factory."""

    @staticmethod
    async def create(
        project_path,
        sysuse,
        project_info,
        install_path,
        shell_manager,
        attempt_state=None,
    ):
        tools = get_phase2_tools(sysuse, shell_manager, attempt_state=attempt_state)
        system_message = llm_prompts.SHELL_INTERACT_SYS.format(
            tool_descriptions=_tool_descriptions(tools),
            project_path=project_path,
            sysuse=sysuse,
            project_info=project_info,
        )
        return _make_agent(
            name="phase2_executor_assistant",
            model=models.get_chat_model("phase2"),
            tools=tools,
            system_message=system_message,
            max_iterations=30,
            fallback_output="Error occurred",
        )


class Phase2ExecutorAgentFactory(ConstructScriptAgentFactory):
    """Explicit Phase2 name kept alongside the legacy ConstructScript factory."""


class ErrorSummaryAgentFactory:
    """Phase3 fixer agent factory — uses function calling for reliable tool use."""

    @staticmethod
    async def create(
        sysuse=None,
        log_dir: Optional[str] = None,
        shell_manager=None,
        operation_kinds: Optional[list[str]] = None,
    ):
        suite = sysuse or "bookworm"
        tools = get_phase3_tools(suite, log_dir, shell_manager=shell_manager)
        if operation_kinds is not None:
            allowed_tools = {"Read_Build_Log"}
            if set(operation_kinds) & {"build_plan_patch", "build_file_patch", "tool_wrapper"}:
                allowed_tools.add("Docker_Run_Command")
            tools = [tool for tool in tools if _callable_name(tool) in allowed_tools]
        system_message = llm_prompts.ERROR_SUMMARY_SYSTEM.format(
            tool_descriptions=_tool_descriptions(tools),
            current_sysuse=suite,
        )
        return _make_agent(
            name="phase3_fixer_assistant",
            model=models.get_chat_model("phase3", temperature=0, top_p=None),
            tools=tools,
            system_message=system_message,
            max_iterations=12,
            fallback_output={
                "status": "cannot_solve",
                "operation": None,
                "evidence_refs": [],
                "expected_effect": "",
                "reason": "json_planner_failed",
            },
        )

    # --- ReAct version (kept for reference) ---
    # @staticmethod
    # async def create(sysuse=None, log_dir: Optional[str] = None):
    #     suite = sysuse or "bookworm"
    #     tools = get_phase3_tools(suite, log_dir)
    #     prompt_text = llm_prompts.ERROR_SUMMARY_REACT.replace("{current_sysuse}", suite)
    #     return ReActAgent(
    #         name="phase3_fixer_assistant",
    #         model=models.get_chat_model("phase3", temperature=0, top_p=None),
    #         tools=tools,
    #         prompt_template=prompt_text,
    #         max_iterations=10,
    #     )


class Phase3FixerAgentFactory(ErrorSummaryAgentFactory):
    """Explicit Phase3 name kept alongside the legacy ErrorSummary factory."""


class BuildPlanAlternativeAgentFactory:
    """Phase3 explorer limited to controller-supplied Build Plan alternatives."""

    @staticmethod
    async def create(sysuse=None, log_dir: Optional[str] = None, shell_manager=None):
        suite = sysuse or "bookworm"
        tools = [
            tool
            for tool in get_phase3_tools(suite, log_dir, shell_manager=shell_manager)
            if _callable_name(tool) == "Read_Build_Log"
        ]
        system_message = llm_prompts.MATRIX_EMPTY_EXPLORER_SYSTEM.format(
            tool_descriptions=_tool_descriptions(tools),
            current_sysuse=suite,
        )
        return _make_agent(
            name="build_plan_alternative_explorer",
            model=models.get_chat_model("phase3", temperature=0, top_p=None),
            tools=tools,
            system_message=system_message,
            max_iterations=12,
            fallback_output={
                "status": "cannot_solve",
                "alternative_id": "",
                "evidence": [],
                "reason": "json_planner_failed",
            },
        )


class CheckAgentFactory:
    """Compatibility factory for the old script checker."""

    @staticmethod
    async def create(sysuse=None):
        suite = sysuse or "bookworm"
        tools = get_readrepo_tools()[:2]
        system_message = llm_prompts.ERROR_SUMMARY_SYSTEM.format(
            tool_descriptions=_tool_descriptions(tools),
            current_sysuse=suite,
        )
        return _make_agent(
            name="stderr_check_assistant",
            model=models.get_chat_model("check"),
            tools=tools,
            system_message=system_message,
            max_iterations=10,
            fallback_output={
                "success": False,
                "reason": "json_planner_failed",
            },
        )


async def check_tools_available(tools, required_tools):
    """Check whether a tool collection contains the requested names."""
    if isinstance(tools, dict):
        available = set(tools.keys())
    else:
        available = {_callable_name(tool_item) for tool_item in tools}
    return all(tool in available for tool in required_tools)
