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

from tools.error_pattern_matcher import query_error_pattern
from tools.policy_kb.query_policy_kb import PolicyKB

mcp = FastMCP("ErrorSummary", log_level="ERROR")

tool_names = [
    "Read_Build_Log",
    "Verify_Package",
    "Search_Package_File",
    "Query_Package_Versions",
]

_policy_kb = PolicyKB()


class ReadBuildLogInput(BaseModel):
    log_name: str = Field(
        "list_recent",
        description='Use "list_recent", "list_all", or a specific file such as "command_003.log".',
    )
    max_chars: int = Field(
        20000,
        ge=1000,
        le=100000,
        description='Maximum number of characters to read when opening a specific log file.',
    )


class VerifyPackageInput(BaseModel):
    package_names: list[str] = Field(
        ...,
        min_length=1,
        description='Debian package names to verify, for example {"package_names": ["cmake", "zlib1g-dev"]}.',
    )


class SearchPackageFileInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description='Missing header or library filename, for example {"query": "openssl/ssl.h"}.',
    )


class QueryPackageVersionsInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z0-9.+:_-]+$",
        description='Debian package name whose versions across distro suites should be checked.',
    )


class QueryPolicyKbInput(BaseModel):
    error_text: str = Field(
        ...,
        min_length=1,
        description='Key stderr lines from the failed build.',
    )
    failed_phase: Literal["compile", "link", "configure"] = Field(
        "compile",
        description='One of "compile", "link", or "configure".',
    )


class DockerRunCommandInput(BaseModel):
    command: str = Field(
        ...,
        min_length=1,
        description='One read-only diagnostic shell command, for example {"command": "grep -R \"find_package\" -n CMakeLists.txt cmake 2>/dev/null | head -50"}.',
    )


def _format_policy_matches(matches: list) -> str:
    """Format Policy KB matches into readable text for the agent."""
    if not matches:
        return "No matching policy card found."
    parts = ["Found matching toolchain policy changes:"]
    for i, m in enumerate(matches, 1):
        parts.append(f"\n--- Policy {i} ({m.get('confidence','?')}) ---")
        parts.append(f"ID: {m['policy_id']}")
        parts.append(f"Tool: {m['tool']}")
        parts.append(f"Summary: {m['summary']}")
        parts.append(f"Mechanism: {m['mechanism']}")
        actions_text = "; ".join(f"[{a['type']}] {a['hint']}" for a in m.get('actions', []))
        parts.append(f"Actions: {actions_text}")
        if m.get('non_solutions'):
            parts.append(f"Avoid: {'; '.join(m['non_solutions'])}")
        if m.get('matched_patterns'):
            parts.append(f"Matched: {m['matched_patterns'][:3]}")
    return "\n".join(parts)


def get_phase3_tools(sysuse=None, log_dir=None, shell_manager=None):
    """Return tools available to the Phase3 fixer agent."""
    suite = sysuse or "bookworm"
    base_log_dir = log_dir or "."

    async def Docker_Run_Command(command: str):
        """Run one read-only diagnostic command inside the Docker build container.

        Args:
            command:
                Exactly one shell command string to execute inside the build container.

        Returns:
            A compact execution summary with command, exit_code, status, stdout_tail,
            stderr_tail, and log_path when available.

        Do:
            - Inspect current repository files, tool metadata, or environment state
              without changing them.

        Don't:
            - Do not change package, filesystem, build, process, or version-control
              state, and do not run the target build.
        """
        if shell_manager is None:
            return {"status": "error", "error": "Docker shell manager is not available."}
        blocked = _blocked_phase3_probe(command)
        if blocked:
            return {
                "status": "blocked",
                "command": command,
                "error": f"Command rejected for repair-planner diagnostics: {blocked}",
            }
        probe = getattr(shell_manager, "probe_command", None)
        if not callable(probe):
            return {
                "status": "error",
                "command": command,
                "error": "Diagnostic probe is unavailable; refusing to record it as a build command.",
            }
        result = await probe(command)
        result["origin"] = "repair_diagnostic"
        stdout = result.get("stdout", "") or ""
        stderr = result.get("stderr", "") or ""
        obs = {
            "command": result.get("command", ""),
            "exit_code": result.get("exit_code"),
            "status": result.get("status", ""),
        }
        if stderr:
            obs["stderr_tail"] = stderr[-3000:] if len(stderr) > 3000 else stderr
        if stdout:
            obs["stdout_tail"] = stdout[-3000:] if len(stdout) > 3000 else stdout
        return obs

    async def Read_Build_Log(log_name: str = "list_recent", max_chars: int = 20000):
        """Read Docker command logs to inspect build output.

        Args:
            log_name:
              - "list_recent": show the 5 most recent command logs (filename + command + exit_code)
              - "list_all": show all command logs (filename + command + exit_code)
              - "command_NNN.log": read full content of a specific log file

        Returns:
            For list modes: table of filename -> [exit_code] command_summary
            For read mode: full log with timestamp, command, exit_code, stdout, stderr

        Do:
            - Use this after at least one Docker build command has failed or timed out.

        Don't:
            - Do not use it before any Docker command has run.
        """
        return await MCPBench.Read_Build_Log(base_log_dir, log_name, max_chars)

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
            - Use this only for Debian package names you intend to install.

        Don't:
            - Do not pass source project names.
            - Do not pass free-form dependency descriptions or contaminated token text.
        """
        return await MCPBench.Verify_Package(package_names, suite)

    async def Search_Package_File(query: str):
        """Find Debian packages that ship a requested header or file path.

        Args:
            query:
                Missing header, library, or binary filename to search for.

        Returns:
            Matched package-file mappings, or "No matching package found."

        Do:
            - Use this for missing header or library file errors.

        Don't:
            - Do not use it as a generic package recommendation tool.
            - Do not pass project names instead of file names.
        """
        return await MCPBench.Search_Package_File(query, suite)

    async def Query_Package_Versions(query: str):
        """Check what version of a package is available in each Debian release.

        Args:
            query:
                Package name, for example "extra-cmake-modules" or "libssl-dev".

        Returns:
            Version information grouped by Debian distro suite.

        Do:
            - Use this when a build fails because a dependency version is too old.
            - Use the result to compare Debian suites before recommending a sysuse switch.

        Don't:
            - Do not pass header names, library filenames, or project names.
        """
        return await MCPBench.Query_Package_Versions(query, suite)

    async def Query_Policy_KB(error_text: str, failed_phase: str = "compile") -> str:
        """Find known toolchain breaking changes that match a build error.

        Args:
            error_text:
                Key error lines from stderr.
            failed_phase:
                One of "compile", "link", or "configure".

        Returns:
            Matching policy cards with root cause, mechanism, and fix actions.
            Returns "No matching policy card found." if no match.

        Do:
            - Use this when a build fails with compiler, linker, or configure errors that may be caused by toolchain changes.

        Don't:
            - Do not use it as a substitute for package lookup or file-provider lookup.
        """
        error_lines = [line.strip() for line in error_text.splitlines() if line.strip()]
        if not error_lines:
            return "No error text provided."
        result = _policy_kb.query(
            error_lines=error_lines,
            sysuse=suite,
            failed_phase=failed_phase,
        )
        return _format_policy_matches(result.get("matches", []))

    async def Query_Error_Pattern(error_text: str) -> str:
        """Match error text against common error categories (missing header, undefined symbol, etc.).

        Args:
            error_text:
                Compilation error text from stderr.

        Returns:
            Matched error category with symptom-diagnosis-fix guidance.

        Do:
            - Use this as fallback when Query_Policy_KB returns no match.

        Don't:
            - Do not use it as the primary source of truth when the exact build log is available.
        """
        result = query_error_pattern(error_text)
        if result["status"] == "no_match":
            return "No matching error pattern found in local reference files."
        parts = [f"Error category: {result['category']}"]
        if result.get("fix_pattern"):
            parts.append(f"\nReference:\n{result['fix_pattern']}")
        if result.get("cannot_solve_hint"):
            parts.append(f"\nExit conditions:\n{result['cannot_solve_hint']}")
        return "\n".join(parts)

    Docker_Run_Command.__name__ = "Docker_Run_Command"
    Docker_Run_Command.__tool_args_schema__ = DockerRunCommandInput
    Read_Build_Log.__name__ = "Read_Build_Log"
    Read_Build_Log.__tool_args_schema__ = ReadBuildLogInput
    Verify_Package.__name__ = "Verify_Package"
    Verify_Package.__tool_args_schema__ = VerifyPackageInput
    Search_Package_File.__name__ = "Search_Package_File"
    Search_Package_File.__tool_args_schema__ = SearchPackageFileInput
    Query_Package_Versions.__name__ = "Query_Package_Versions"
    Query_Package_Versions.__tool_args_schema__ = QueryPackageVersionsInput
    Query_Policy_KB.__name__ = "Query_Policy_KB"
    Query_Policy_KB.__tool_args_schema__ = QueryPolicyKbInput
    Query_Error_Pattern.__name__ = "Query_Error_Pattern"
    return [
        Docker_Run_Command,
        Read_Build_Log, Verify_Package, Search_Package_File, Query_Package_Versions,
        # Query_Policy_KB removed — candidate cards are now injected via
        # ERROR_SUMMARY_TASK's {candidate_cards} field, so the agent evaluates
        # them directly without a separate tool call.
        # Query_Error_Pattern,  # merged into system prompt
    ]


def _blocked_phase3_probe(command: str) -> str:
    """Allow only read-only diagnostics in every shell segment."""
    text = (command or "").strip()
    if not text:
        return "empty diagnostic command"
    lowered = text.lower()
    if "$(" in text or "`" in text:
        return "shell command substitution is not allowed in Phase 3 diagnostics"

    # Strip harmless stderr/stdout suppression before rejecting write
    # redirections. This keeps common probes such as `find ... 2>/dev/null`.
    safe_redirect_re = r"(?<!\S)\d*>>?\s*/dev/null\b|(?<!\S)\d*>\s*&\s*\d+\b"
    redirection_view = re.sub(safe_redirect_re, "", lowered)
    if re.search(r"(?:^|\s)\d*>>?\s*[^&]", redirection_view):
        return "write redirection is not allowed in Phase 3 diagnostics"

    try:
        parse_text = re.sub(safe_redirect_re, "", text, flags=re.I)
        lexer = shlex.shlex(parse_text, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return "diagnostic command has invalid shell syntax"

    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and all(char in ";&|" for char in token):
            if not segments[-1]:
                return "empty shell segment is not allowed in Phase 3 diagnostics"
            segments.append([])
        else:
            segments[-1].append(token)
    if not segments[-1]:
        return "empty shell segment is not allowed in Phase 3 diagnostics"

    simple_readers = {
        "pwd", "ls", "cat", "grep", "egrep", "fgrep", "rg", "head", "tail",
        "wc", "stat", "file", "strings", "readelf", "objdump", "ldd", "printenv",
        "getent", "id", "uname", "nm", "dpkg-query", "dpkg-architecture", "pkg-config",
        "apt-cache", "which", "type",
    }
    for segment in segments:
        while segment and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[0]):
            segment = segment[1:]
        if not segment:
            return "environment-only shell segments are not allowed"
        executable = os.path.basename(segment[0]).lower()
        args = [item.lower() for item in segment[1:]]
        if executable in simple_readers or executable == "cd":
            continue
        if executable == "find" and not any(
            item in {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls", "-fprint", "-fprint0", "-fprintf"}
            for item in args
        ):
            continue
        if executable in {"test", "["}:
            continue
        if executable == "command" and args and args[0] == "-v":
            continue
        if executable == "ldconfig" and "-p" in args:
            continue
        if executable == "dpkg" and args and args[0] in {"-s", "-l", "--status", "--list"}:
            continue
        if executable == "aptitude" and args and args[0] in {"search", "show"}:
            continue
        if executable == "meson" and args and args[0] == "configure":
            continue
        if executable in {"configure", "autogen.sh"} and args == ["--help"]:
            continue
        if (
            executable == "git"
            and args
            and args[0] in {"status", "show", "log", "diff", "grep", "ls-files", "rev-parse"}
            and not any(item.startswith("--output") for item in args)
        ):
            continue
        return f"{executable} is not an approved read-only Phase 3 diagnostic"

    return ""


for name in tool_names:
    func = getattr(MCPBench, name)
    mcp.tool()(func)

if __name__ == "__main__":
    mcp.run(transport='stdio')
