"""Generic agents used only by ablation modes."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

import llm.models
from core.repo_assistant import LangChainAgent, _tool_descriptions
from llm.Run_LLM import _agent_result_content


class DockerRunCommandInput(BaseModel):
    command: str = Field(..., min_length=1, description="Shell command to run inside the Docker build container.")


_BUILD_COMMAND_RE = re.compile(
    r"(^|[;&|()]\s*)("
    r"make|gmake|ninja|"
    r"cmake\s+--build|"
    r"meson\s+compile|"
    r"cargo\s+build|"
    r"go\s+build|"
    r"npm\s+run\s+build|"
    r"python3?\s+setup\.py\s+build|"
    r"gcc|g\+\+|clang|clang\+\+|cc|c\+\+"
    r")(\s|$)",
    re.IGNORECASE,
)
_PACKAGE_INSTALL_RE = re.compile(
    r"(?:^|[\s;&|()'\"])(?:sudo\s+)?(?:"
    r"apt(?:-get)?\s+(?:install|build-dep)|aptitude\s+install|dpkg\s+-i|"
    r"pip(?:3)?\s+install|python3?\s+-m\s+pip\s+install|"
    r"gem\s+install|cargo\s+install|npm\s+(?:install|i)|yarn\s+add|pnpm\s+(?:add|install)|"
    r"go\s+install"
    r")\b",
    re.IGNORECASE,
)


def _compact_command_result(result: dict[str, Any]) -> dict[str, Any]:
    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""
    compact = {
        "command": result.get("command", ""),
        "exit_code": result.get("exit_code"),
        "status": result.get("status", ""),
    }
    if stderr:
        compact["stderr_tail"] = stderr[-3000:] if len(stderr) > 3000 else stderr
    elif stdout:
        compact["stdout_tail"] = stdout[-2000:] if len(stdout) > 2000 else stdout
    if result.get("log_path"):
        compact["log_path"] = result["log_path"]
    return compact


def _is_build_command(command: str) -> bool:
    return bool(_BUILD_COMMAND_RE.search(command or ""))


def is_package_install_command(command: str) -> bool:
    return bool(_PACKAGE_INSTALL_RE.search(command or ""))


def get_generic_docker_command_tools(
    shell_manager,
    *,
    block_build_commands: bool = False,
    block_package_installs: bool = False,
) -> list:
    async def Docker_Run_Command(command: str):
        """Run one shell command inside the Docker build container."""
        if block_build_commands and _is_build_command(command):
            return {
                "command": command,
                "status": "blocked",
                "reason": "build_or_compile_command_blocked_by_ablation",
            }
        if block_package_installs and is_package_install_command(command):
            return {
                "command": command,
                "status": "blocked",
                "reason": "package_install_blocked_in_generic_emergency",
            }
        result = await shell_manager.execute_command(command)
        return _compact_command_result(result)

    Docker_Run_Command.__name__ = "Docker_Run_Command"
    Docker_Run_Command.__tool_args_schema__ = DockerRunCommandInput
    return [Docker_Run_Command]


GENERIC_REPO_EXPLORER_SYSTEM = """
You are GenericRepoExplorer. Explore the project inside Docker and produce the repository build summary JSON.

## Context
- Path: {project_path}
- Platform: Debian {sysuse} X64

Inspect the project enough to identify its build system and likely build commands. Do not compile the project.
Return JSON only. Do not wrap it in markdown.
Required fields: build_system, build_file_path, build_commands, apt_packages, source_deps.
Use empty arrays when apt packages or source dependencies are unknown.
"""


GENERIC_REPO_EXPLORER_TASK = """
Explore `{project_path}` and return the same repository summary shape consumed by Phase 2.

Commit date: {commit_date}
Submodule preflight:
{submodule_info}
"""


GENERIC_PHASE3_SYSTEM = """
You are GenericPhase3Planner. Diagnose the current build failure and return a repair plan JSON.

## Context
- Path: {project_path}
- Platform: Debian {sysuse} X64

Use the command tool only when it helps diagnose or validate the failure. Do not execute build or compile commands.
This is an emergency exploration path: do not install packages with any system or language package manager.
Use only the current failure, selected plan, untried alternatives, and prior
repair fingerprints supplied in the request. Do not reconstruct controller
state or repeat a prior repair.
Return JSON only. Do not wrap it in markdown.
Use the existing FixPlan shape exactly:
{{"status":"fix|source_dependency|cannot_solve","reason":"hypothesis: <tag>; detail: <evidence>","commands":[],"sysuse":null,"source_dependency":null}}

- `reason` must start with `hypothesis: <tag>` so attempts can be tracked consistently.
- `commands` must contain executable repair commands only. Put diagnostic commands in tool calls.
- `sysuse` must be a Debian suite string or null.
- `source_dependency` must include non-empty `build_order` when status is `source_dependency`. Prefer `build_order` over a nested `commands` field.
"""


GENERIC_PHASE3_TASK = """
Analyze this compact current-run context and return one FixPlan JSON object:
```json
{repair_context}
```
"""


class GenericRepoExplorerAgentFactory:
    @staticmethod
    async def create(project_path: str, sysuse: str, shell_manager):
        tools = get_generic_docker_command_tools(shell_manager)
        system_message = GENERIC_REPO_EXPLORER_SYSTEM.format(
            tool_descriptions=_tool_descriptions(tools),
            project_path=project_path,
            sysuse=sysuse,
        )
        return LangChainAgent(
            name="generic_repo_explorer_assistant",
            model=models.get_chat_model("readrepo"),
            tools=tools,
            system_message=system_message,
            max_iterations=10,
        )


class GenericPhase3AgentFactory:
    @staticmethod
    async def create(project_path: str, sysuse: str, shell_manager):
        tools = get_generic_docker_command_tools(
            shell_manager,
            block_build_commands=True,
            block_package_installs=True,
        )
        system_message = GENERIC_PHASE3_SYSTEM.format(
            tool_descriptions=_tool_descriptions(tools),
            project_path=project_path,
            sysuse=sysuse,
        )
        return LangChainAgent(
            name="generic_phase3_planner_assistant",
            model=models.get_chat_model("phase3", temperature=0, top_p=None),
            tools=tools,
            system_message=system_message,
            max_iterations=12,
        )


async def call_generic_agent(task: str, agent) -> str:
    result = await agent.run(task=task)
    text = _agent_result_content(result)
    print(text)
    return text
