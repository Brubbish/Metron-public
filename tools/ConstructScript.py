import sys
import asyncio
import uuid
import json
import os
import re
import shlex
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Literal

try:
    from tools import MCPBench
except ImportError:  # pragma: no cover - script execution fallback
    import tools.MCPBench as MCPBench

from tools.error_pattern_matcher import lookup_category, AVAILABLE_CATEGORIES

mcp = FastMCP("ConstructScript", log_level="ERROR")

tool_names = [
    "Verify_Package",
    "Docker_Run_Command",
]

_READ_ONLY_PROBE_COMMANDS = {
    "pwd", "ls", "find", "test", "cat", "sed", "head", "tail",
    "grep", "rg", "stat", "file", "command", "which", "realpath",
    "readlink", "pkg-config", "apt-cache", "dpkg-query",
}


class DockerRunCommandInput(BaseModel):
    command: str = Field(
        ...,
        min_length=1,
        description='Exactly one shell command string to execute, for example {"command": "apt-get install -y cmake", "workdir": "/work"}.',
    )
    workdir: str = Field(
        default="/work",
        description="Absolute initial directory under /work or /source_deps. The command text is left unchanged.",
    )


class VerifyPackageInput(BaseModel):
    package_names: list[str] = Field(
        ...,
        min_length=1,
        description='Debian package names to verify, for example {"package_names": ["cmake", "zlib1g-dev"]}.',
    )


class DockerProbeCommandInput(BaseModel):
    command: str = Field(
        ...,
        min_length=1,
        description="One non-mutating inspection command. Its result is not added to build records.",
    )


class LookupErrorInput(BaseModel):
    category: Literal[
        "missing_header",
        "undefined_symbol",
        "missing_build_tool",
        "pkgconfig_failure",
        "version_mismatch",
        "compiler_error",
        "permission_path",
        "git_submodule",
        "cannot_solve",
    ] = Field(
        ...,
        description='Error category such as "missing_header", "missing_build_tool", or "version_mismatch".',
    )


class SearchPackageFileInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description='Missing header or library filename, for example {"query": "pcap.h"}.',
    )


def get_phase2_tools(sysuse, shell_manager, attempt_state=None):
    """Return tools available to the Phase2 executor agent.

    ``attempt_state`` is retained for callers using the previous factory
    signature. A failed command is an observation for the agent, not a latch:
    Phase 2 must be able to inspect, repair, and retry in the same container.
    """

    async def Docker_Run_Command(command: str, workdir: str = "/work"):
        """Run one command inside the persistent Docker build container.

        Args:
            command:
                Exactly one shell command string to execute inside the build container.
            workdir:
                Absolute initial working directory under ``/work`` or
                ``/source_deps``.
                The command text is not rewritten; an embedded ``cd`` keeps
                its normal shell semantics.

        Returns:
            A compact execution summary including:
              - command
              - exit_code
              - status
              - stderr_tail or stdout_tail
              - log_path when available

        Do:
            - Use the current failure output to choose a materially different
              evidence-backed inspection, repair, or retry command.

        Don't:
            - Do not pass final-answer text such as "Compilation succeeded" or "Error occurred".
            - Do not pass tool-call markup or placeholder text as a command.
        """
        raw = str(command or "")
        if not re.search(r"[\n\r;&|><`]", raw) and "$" not in raw:
            try:
                tokens = shlex.split(raw)
            except ValueError:
                tokens = []
            if tokens and os.path.basename(tokens[0]) in _READ_ONLY_PROBE_COMMANDS:
                return {
                    "command": raw,
                    "status": "rejected",
                    "reason": "read-only inspection must use Docker_Probe_Command",
                    "workdir": workdir,
                }
        try:
            result = await shell_manager.execute_command(command, workdir=workdir)
        except TypeError as exc:
            # Minimal compatibility for unit-test doubles written before the
            # explicit workdir contract. Real Docker execution always receives
            # the structured workdir.
            if "workdir" not in str(exc):
                raise
            result = await shell_manager.execute_command(command)
        # Truncate for agent scratchpad; full output is in log_path and record
        STDOUT_TAIL = 2000
        STDERR_TAIL = 3000
        obs = {
            "command": result.get("command", ""),
            "exit_code": result.get("exit_code"),
            "status": result.get("status", ""),
        }
        stdout = result.get("stdout", "") or ""
        stderr = result.get("stderr", "") or ""
        if stderr:
            obs["stderr_tail"] = stderr[-STDERR_TAIL:] if len(stderr) > STDERR_TAIL else stderr
        elif stdout:
            obs["stdout_tail"] = stdout[-STDOUT_TAIL:] if len(stdout) > STDOUT_TAIL else stdout
        if result.get("log_path"):
            obs["log_path"] = result["log_path"]
        obs["workdir"] = workdir
        return obs

    async def Docker_Probe_Command(command: str):
        """Run one bounded read-only inspection without creating a build record.

        State-changing commands, shell composition, builds, and repairs must
        use Docker_Run_Command so they remain in the causal attempt history.
        """
        raw = str(command or "")
        if re.search(r"[\n\r;&|><`]", raw) or "$" in raw:
            return {"command": raw, "status": "rejected", "reason": "probe must be one simple read-only command"}
        try:
            tokens = shlex.split(raw)
        except ValueError as exc:
            return {"command": raw, "status": "rejected", "reason": str(exc)}
        executable = os.path.basename(tokens[0]) if tokens else ""
        if executable not in _READ_ONLY_PROBE_COMMANDS:
            return {"command": raw, "status": "rejected", "reason": "command is not an approved read-only probe"}
        result = await shell_manager.probe_command(raw)
        return {
            "command": result.get("command", raw),
            "status": result.get("status", "failed"),
            "exit_code": result.get("exit_code"),
            "stdout_tail": str(result.get("stdout") or "")[-3000:],
            "stderr_tail": str(result.get("stderr") or "")[-2000:],
        }

    async def Verify_Package(package_names: list[str]):
        """Verify Debian APT package names for this build suite.

        Args:
            package_names:
                Debian package names to verify before installation.

        Returns:
            A verification result with:
              - status
              - invalid package names
              - suggestions for replacements
              - error_message when verification fails

        Do:
            - Use this only for Debian package names you actually plan to install.

        Don't:
            - Do not pass source project names.
            - Do not pass free-form dependency descriptions or contaminated token text.
        """
        return await MCPBench.Verify_Package(package_names, sysuse)

    async def Lookup_Error(category: str) -> str:
        """Look up diagnosis and fix steps for a common build error type.

        Available categories:
          missing_header    - "fatal error: foo.h: No such file or directory"
          undefined_symbol  - "undefined reference to", "cannot find -l"
          missing_build_tool - "command not found"
          pkgconfig_failure - "Package 'x' not found"
          version_mismatch  - "requires version", "version too old"
          compiler_error    - "undeclared identifier", "implicit declaration"
          permission_path   - "Permission denied"
          git_submodule     - "fatal: repository not found", submodule failures
          cannot_solve      - when to give up and report cannot_solve

        Args:
            category:
                One of the category names above.

        Returns:
            Symptoms, diagnosis steps, and fix patterns for that error type.

        Do:
            - Use this when you encounter a build error and want structured guidance.
            - Choose the category that best matches the current error.

        Don't:
            - Do not use this as a substitute for reading the actual compiler or build error.
        """
        return lookup_category(category)

    async def Search_Package_File(query: str):
        """Search which Debian package provides a given file (header, library, or binary).

        Args:
            query:
                The filename to search for, for example "pcap.h" or "libfoo.so".

        Returns:
            Matched package-file mappings, or "No matching package found."

        Do:
            - Use this when a build fails with a missing header or library file.

        Don't:
            - Do not use it as a generic package recommendation tool.
            - Do not pass project names instead of file names.
        """
        return await MCPBench.Search_Package_File(query, sysuse)

    Docker_Run_Command.__name__ = "Docker_Run_Command"
    Docker_Run_Command.__tool_args_schema__ = DockerRunCommandInput
    Docker_Probe_Command.__name__ = "Docker_Probe_Command"
    Docker_Probe_Command.__tool_args_schema__ = DockerProbeCommandInput
    Verify_Package.__name__ = "Verify_Package"
    Verify_Package.__tool_args_schema__ = VerifyPackageInput
    Lookup_Error.__name__ = "Lookup_Error"
    Lookup_Error.__tool_args_schema__ = LookupErrorInput
    Search_Package_File.__name__ = "Search_Package_File"
    Search_Package_File.__tool_args_schema__ = SearchPackageFileInput
    return [Docker_Run_Command, Docker_Probe_Command, Verify_Package, Lookup_Error, Search_Package_File]


for name in tool_names:
    func = getattr(MCPBench, name)
    mcp.tool()(func)

if __name__ == "__main__":
    mcp.run(transport="stdio")
