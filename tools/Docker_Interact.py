"""Docker-backed command execution for the linear build workflow.

The build agents run on the host, but every install/build/fix command is
executed in a short-lived Docker container rooted at a fixed repository path.
Each command is executed independently and its full output is persisted to disk;
the agent receives stdout/stderr fields plus a compact excerpt for summaries.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.debian_sysuse import VALID_DEBIAN_CODENAMES, normalize_sysuse
from core.repository_environment import repository_profile_hash

DEFAULT_WORK_DIR = "/work"
DEFAULT_CMD_TIMEOUT = 3600
DEFAULT_EXCERPT_CHARS = 6000
CONTAINER_HOSTNAME = "theauto-builder"
PROMPT_SUFFIX = f"@{CONTAINER_HOSTNAME}:"
SUPPORTED_DEBIAN_RELEASES = VALID_DEBIAN_CODENAMES

CONTROL_COMMANDS = {
    "error",
    "Error occurred",
    "Compilation succeeded",
    "done",
    "cannot_solve",
}
_CONTROL_COMMANDS_CASEFOLD = {token.casefold() for token in CONTROL_COMMANDS}
_REPO_SENTINELS = (
    ".git",
    ".hg",
    ".svn",
    "configure",
    "configure.ac",
    "configure.in",
    "CMakeLists.txt",
    "Makefile.am",
    "Makefile",
    "makefile",
    "GNUmakefile",
    "meson.build",
    "SConstruct",
    "setup.py",
    "pyproject.toml",
    "package.json",
    "README",
    "README.md",
    "readme.md",
)
_REPO_SOURCE_EXTS = (
    "*.c",
    "*.cc",
    "*.cpp",
    "*.cxx",
    "*.h",
    "*.hpp",
    "*.hh",
    "*.hxx",
    "*.rs",
    "*.go",
    "*.java",
    "*.py",
    "*.pro",
)


def image_for_sysuse(sysuse: str) -> str:
    """Map a Debian suite name to the project Docker image convention."""
    suffix = os.environ.get("THEAUTO_IMAGE_TAG_SUFFIX", "").strip(" -")
    tag = normalize_sysuse(sysuse)
    return f"theauto/debian:{tag}{('-' + suffix) if suffix else ''}"


def _repository_profile_hash() -> str:
    """Return the canonical repository-profile hash used by runtime metadata."""
    return repository_profile_hash()


def _repository_profile_validation_enabled() -> bool:
    """Enable the optional image/profile diagnostic check explicitly."""
    return os.environ.get("THEAUTO_VERIFY_REPOSITORY_PROFILE") == "1"


def _safe_container_name(compile_path: str) -> str:
    return f"{container_name_prefix(compile_path)}{uuid.uuid4().hex[:10]}"


def container_name_prefix(compile_path: str) -> str:
    """Return the stable container-name prefix for one checked-out worktree."""
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", os.path.basename(compile_path.rstrip(os.sep)) or "repo")
    return f"theauto-build-{base[:32]}-"


def _container_name_prefix(compile_path: str) -> str:
    """Backward-compatible private alias for existing DockerManager callers."""
    return container_name_prefix(compile_path)


def _normalize_command(command: str, sysuse: Optional[str] = None) -> str:
    """Compatibility helper that preserves the caller's command verbatim."""
    return str(command or "")


def _sysuse_from_image(image_name: Optional[str]) -> Optional[str]:
    if not image_name:
        return None
    match = re.search(r"theauto/debian:([A-Za-z0-9_.-]+)", image_name)
    return match.group(1) if match else None


def _tail_excerpt(text: str, limit: int = DEFAULT_EXCERPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]


def _strip_echoed_command(raw_output: str, command: str) -> str:
    """Remove the shell's echoed command prefix when docker exec surfaces it."""
    lines = raw_output.split("\n")
    output_lines = []
    cmd_seen = False
    for line in lines:
        if not cmd_seen and command[:40] in line:
            cmd_seen = True
            continue
        if cmd_seen:
            output_lines.append(line)
    return "\n".join(output_lines).strip() if output_lines else raw_output.strip()


def _split_wrapped_output(raw_output: str, command: str, stderr_start: str, stderr_end: str) -> Tuple[str, str]:
    """Split stdout and stderr markers emitted by the per-command shell wrapper."""
    if stderr_start not in raw_output or stderr_end not in raw_output:
        return _strip_echoed_command(raw_output, command), ""

    stdout_raw, rest = raw_output.split(stderr_start, 1)
    stderr_raw, _ = rest.split(stderr_end, 1)
    stdout = _strip_echoed_command(stdout_raw, command)
    stderr = stderr_raw.strip()
    return stdout, stderr


def _is_control_command(command: str) -> bool:
    return command.strip().casefold() in _CONTROL_COMMANDS_CASEFOLD


def _exit_fingerprint(exit_code: Optional[int]) -> str:
    """Compatibility field; no longer guesses error lines from command output."""
    return f"exit={exit_code}"


class DockerContainerManager:
    """Manage a Docker container with a persistent interactive shell."""

    def __init__(
        self,
        work_dir: str = DEFAULT_WORK_DIR,
        log_dir: Optional[str] = None,
        cmd_timeout: int = DEFAULT_CMD_TIMEOUT,
        excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    ):
        self.work_dir = work_dir
        self.container_id: Optional[str] = None
        self.container_name: Optional[str] = None
        self.image_name: Optional[str] = None
        self.compile_path: Optional[str] = None
        self.cmd_timeout = cmd_timeout
        self.excerpt_chars = excerpt_chars
        self.record: List[Dict[str, object]] = []
        self.temp_images: List[str] = []
        self._last_extra_volumes: List[tuple] = []
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._shell_proc: Optional[asyncio.subprocess.Process] = None
        self._shell_timed_out = False
        self.repository_profile_status: str = "unknown"

    async def _verify_repository_profile(self) -> str:
        """Check the image's repository profile without rejecting legacy tags by default."""
        if not self.container_id:
            return "unknown"
        rc, stdout, _stderr = await self._run_subprocess(
            ["docker", "exec", self.container_id, "sh", "-lc", "cat /etc/theauto-repository-profile.json 2>/dev/null"],
            timeout=15,
        )
        if rc != 0 or not stdout.strip():
            status = "legacy_unverified"
        else:
            try:
                payload = json.loads(stdout)
            except ValueError:
                status = "invalid"
            else:
                expected = _repository_profile_hash()
                status = "matched" if expected and payload.get("profile_hash") == expected else "mismatch"
        self.repository_profile_status = status
        if status == "legacy_unverified" and os.environ.get("THEAUTO_REQUIRE_REPOSITORY_PROFILE") == "1":
            raise RuntimeError("Docker image has no repository profile; candidate image is required")
        if status != "matched":
            print(f"[DockerManager] repository profile status: {status}")
        return status

    def select_image(self, sysuse: str) -> str:
        selected = image_for_sysuse(sysuse)
        print(f"[DockerManager] Selected image: {selected} for {sysuse}")
        return selected

    def set_log_dir(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def _run_subprocess(self, args: List[str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.terminate()
            try:
                stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                stdout_b, stderr_b = await process.communicate()
            stdout = stdout_b.decode("utf-8", errors="ignore") if stdout_b else ""
            stderr = stderr_b.decode("utf-8", errors="ignore") if stderr_b else ""
            return 124, stdout, stderr + "\n[DockerManager] command timed out"
        stdout = stdout_b.decode("utf-8", errors="ignore") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="ignore") if stderr_b else ""
        return int(process.returncode or 0), stdout, stderr

    async def _ensure_image_present(self, image_name: str) -> bool:
        """Pre-pull the image if missing locally. Idempotent."""
        # Check if image exists locally
        check = await self._run_subprocess(
            ["docker", "image", "inspect", image_name],
            timeout=30,
        )
        if check[0] == 0:
            return True
        # Image missing — pull it (long timeout, image can be large)
        print(f"[DockerManager] Image {image_name} not present locally, pulling...")
        pull_rc, _, pull_err = await self._run_subprocess(
            ["docker", "pull", image_name],
            timeout=900,
        )
        if pull_rc != 0:
            print(f"[DockerManager] docker pull failed: {pull_err.strip()[:200]}")
            return False
        return True

    def _host_compile_path_ready(self, compile_path: str) -> bool:
        path = Path(compile_path)
        if not path.is_dir():
            print(f"[DockerManager] compile_path is not a directory: {compile_path}")
            return False
        try:
            next(path.iterdir())
        except StopIteration:
            print(f"[DockerManager] compile_path is empty: {compile_path}")
            return False
        except Exception as exc:
            print(f"[DockerManager] cannot inspect compile_path {compile_path}: {exc}")
            return False
        return True

    async def _remove_stale_worktree_containers(self, compile_path: str) -> None:
        prefix = _container_name_prefix(compile_path)
        rc, stdout, stderr = await self._run_subprocess(
            ["docker", "ps", "-aq", "--filter", f"name=^{prefix}"],
            timeout=30,
        )
        if rc != 0:
            print(f"[DockerManager] stale container lookup failed: {(stderr or stdout).strip()[:200]}")
            return
        ids = [line.strip() for line in stdout.splitlines() if line.strip()]
        if ids:
            await self._run_subprocess(["docker", "rm", "-f", *ids], timeout=60)

    async def _verify_work_mount(self) -> bool:
        if not self.container_id:
            return False
        sentinel_expr = " || ".join(
            f'test -e {shlex.quote(self.work_dir + "/" + name)}'
            for name in _REPO_SENTINELS
        )
        source_expr = " || ".join(
            f'find {shlex.quote(self.work_dir)} -maxdepth 2 -name {shlex.quote(pattern)} -print -quit | grep -q .'
            for pattern in _REPO_SOURCE_EXTS
        )
        command = (
            f'test -d {shlex.quote(self.work_dir)} '
            f'&& test "$(find {shlex.quote(self.work_dir)} -mindepth 1 -maxdepth 1 -print -quit)" '
            f'&& ( {sentinel_expr} || {source_expr} )'
        )
        rc, stdout, stderr = await self._run_subprocess(
            ["docker", "exec", "-w", "/", self.container_id, "sh", "-lc", command],
            timeout=30,
        )
        if rc == 0:
            return True
        print(f"[DockerManager] /work mount sanity check failed: {(stderr or stdout).strip()[:200]}")
        return False

    async def start_container(
        self,
        image_name: str,
        compile_path: str,
        extra_volumes: Optional[List[tuple]] = None,
    ) -> bool:
        """Start a detached container with the repository mounted at ``work_dir``.

        Resilient to transient docker daemon congestion under concurrent batch
        load: pre-pulls the image if missing, then retries `docker run` with
        exponential backoff. Each `docker run` attempt first removes any stale
        container with the same name (left over from a previous crashed run).

        Args:
            image_name: Docker image to run.
            compile_path: Host path to mount at ``work_dir`` (the project repo).
            extra_volumes: Optional list of ``(host_path, container_path)`` tuples
                for additional mounts (e.g. source-dep clones). Each is mounted
                read-write with the given container path verbatim.
        """
        if not self._host_compile_path_ready(compile_path):
            return False
        self.image_name = image_name
        self.compile_path = compile_path
        self._last_extra_volumes = list(extra_volumes) if extra_volumes else []
        self.container_name = _safe_container_name(compile_path)
        env_args: List[str] = ["-e", "DEBIAN_FRONTEND=noninteractive"]
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            value = os.environ.get(key)
            if value:
                env_args.extend(["-e", f"{key}={value}"])

        volume_args: List[str] = ["-v", f"{os.path.abspath(compile_path)}:{self.work_dir}:rw"]
        for host_path, container_path in (extra_volumes or []):
            host_abs = os.path.abspath(host_path)
            cont_norm = container_path if container_path.startswith("/") else f"/{container_path}"
            volume_args.extend(["-v", f"{host_abs}:{cont_norm}:rw"])

        args = [
            "docker",
            "run",
            "-d",
            "--init",
            "--hostname",
            CONTAINER_HOSTNAME,
            "--name",
            self.container_name,
            *volume_args,
            "-w",
            self.work_dir,
            *env_args,
            image_name,
            "sleep",
            "infinity",
        ]

        # Pre-pull to keep the `docker run` window short
        await self._ensure_image_present(image_name)

        # Retry `docker run` with exponential backoff. 60s per attempt is fine
        # once the image is local; failures under batch load are typically
        # daemon-side congestion that resolves within tens of seconds.
        run_timeout = int(os.environ.get("DOCKER_RUN_TIMEOUT", "120"))
        max_attempts = int(os.environ.get("DOCKER_RUN_MAX_ATTEMPTS", "3"))
        backoff_seconds = [5, 15, 30]
        last_stderr = ""
        for attempt in range(1, max_attempts + 1):
            # Clean up any stale container with the same name first
            await self._remove_stale_worktree_containers(compile_path)
            await self._run_subprocess(
                ["docker", "rm", "-f", self.container_name],
                timeout=30,
            )
            return_code, stdout, stderr = await self._run_subprocess(args, timeout=run_timeout)
            if return_code == 0 and stdout.strip():
                self.container_id = stdout.strip()
                extra_msg = f" (+{len(extra_volumes)} extra mounts)" if extra_volumes else ""
                print(f"[DockerManager] Container started: {self.container_id[:12]} (attempt {attempt}/{max_attempts}){extra_msg}")
                if not await self._verify_work_mount():
                    await self._run_subprocess(["docker", "rm", "-f", self.container_id], timeout=30)
                    self.container_id = None
                    last_stderr = "work mount sanity check failed"
                    if attempt < max_attempts:
                        delay = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                        await asyncio.sleep(delay)
                    continue
                if _repository_profile_validation_enabled():
                    try:
                        await self._verify_repository_profile()
                    except RuntimeError as exc:
                        print(f"[DockerManager] {exc}")
                        await self._run_subprocess(["docker", "rm", "-f", self.container_id], timeout=30)
                        self.container_id = None
                        last_stderr = str(exc)
                        if attempt < max_attempts:
                            delay = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                            await asyncio.sleep(delay)
                        continue
                await self._start_persistent_shell()
                return True
            last_stderr = (stderr or stdout or "").strip()
            print(
                f"[DockerManager] start_container attempt {attempt}/{max_attempts} failed "
                f"(rc={return_code}): {last_stderr[:200]}"
            )
            if attempt < max_attempts:
                delay = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                await asyncio.sleep(delay)
        print(f"[DockerManager] Failed to start container after {max_attempts} attempts: {last_stderr[:200]}")
        return False

    async def _start_persistent_shell(self) -> None:
        """Attach a persistent bash session to the container via docker exec -i."""
        if self._shell_proc is not None:
            try:
                self._shell_proc.kill()
            except ProcessLookupError:
                pass
        self._shell_proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", self.container_id, "/bin/bash", "--norc", "--noprofile",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._shell_timed_out = False
        # Set a reliable prompt and wait for it
        init_cmds = [
            f'export PS1="\\u@{CONTAINER_HOSTNAME}:\\w\\$ "',
            f'cd {self.work_dir}',
            'export DEBIAN_FRONTEND=noninteractive',
        ]
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            value = os.environ.get(key)
            if value:
                init_cmds.append(f'export {key}="{value}"')

        for cmd in init_cmds:
            self._shell_proc.stdin.write((cmd + "\n").encode())
        # Send a marker to know when init is done
        self._shell_proc.stdin.write(b'echo "SHELL_READY"\n')
        await self._shell_proc.stdin.drain()
        # Read until we see SHELL_READY
        await self._read_until("SHELL_READY", timeout=10)

    async def _read_until(self, marker: str, timeout: int = None) -> str:
        """Read stdout until marker appears or timeout."""
        timeout = timeout or self.cmd_timeout
        output = ""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(
                    self._shell_proc.stdout.read(4096),
                    timeout=min(remaining, 2.0),
                )
                if not chunk:
                    break
                output += chunk.decode("utf-8", errors="ignore")
                if marker in output:
                    return output
            except asyncio.TimeoutError:
                # No data for 2s, check if marker already appeared
                if marker in output:
                    return output
                continue
        return output

    async def restart(self, sysuse: str, compile_path: Optional[str] = None) -> None:
        """Recreate the container, optionally switching Debian suite."""
        target_path = compile_path or self.compile_path
        if not target_path:
            raise RuntimeError("Cannot restart Docker container before compile_path is set")
        extra_volumes = list(self._last_extra_volumes)
        await self.cleanup()
        image_name = self.select_image(sysuse)
        if not await self.start_container(image_name, target_path, extra_volumes=extra_volumes or None):
            raise RuntimeError(f"Failed to start Docker container with image {image_name}")

    async def execute_command(self, command: str, workdir: Optional[str] = None) -> Dict[str, object]:
        """Execute one shell command inside the persistent Docker shell.

        Environment variables, working directory, and PATH persist between calls.
        ``workdir`` optionally sets the initial cwd for this command without
        modifying its text. It must stay within ``/work`` or ``/source_deps``.
        """
        normalized_command = _normalize_command(command, sysuse=_sysuse_from_image(self.image_name))
        if not normalized_command.strip():
            return {
                "input": "", "command": "", "output": "", "stdout": "", "stderr": "",
                "exit_code": None, "status": "failed", "fingerprint": "exit=None", "log_path": None,
            }
        if _is_control_command(normalized_command):
            return {
                "input": normalized_command, "command": normalized_command,
                "output": f"Control token '{normalized_command}' was not executed as a shell command.",
                "stdout": "", "stderr": "", "exit_code": None, "status": "control",
                "fingerprint": "exit=None", "log_path": None,
            }
        if normalized_command in {"^C", "\x03"}:
            return {
                "input": normalized_command, "command": normalized_command,
                "output": "Ctrl+C sent", "stdout": "", "stderr": "",
                "exit_code": None, "status": "interrupted", "fingerprint": "exit=None", "log_path": None,
            }
        if self._shell_timed_out:
            return {
                "input": normalized_command,
                "command": normalized_command,
                "output": "The persistent shell timed out and cannot execute more commands before cleanup.",
                "stdout": "",
                "stderr": "",
                "exit_code": "124",
                "status": "timeout_latched",
                "fingerprint": "exit=124",
                "log_path": None,
            }

        selected_workdir = ""
        if workdir:
            selected_workdir = os.path.normpath(str(workdir))
            allowed_roots = (os.path.normpath(self.work_dir), "/source_deps")
            if not os.path.isabs(selected_workdir) or not any(
                selected_workdir == root or selected_workdir.startswith(root + os.sep)
                for root in allowed_roots
            ):
                return {
                    "input": normalized_command,
                    "command": normalized_command,
                    "output": "workdir must be an absolute path under /work or /source_deps",
                    "stdout": "",
                    "stderr": "",
                    "exit_code": "125",
                    "status": "failed",
                    "fingerprint": "exit=125",
                    "log_path": None,
                    "workdir": str(workdir),
                }

        print(f"[DockerManager] Executing: {normalized_command}")

        # If shell died, restart it
        if self._shell_proc is None or self._shell_proc.returncode is not None:
            await self._start_persistent_shell()

        # Use unique markers to reliably detect command completion and split stderr.
        end_marker = f"__CMD_DONE_{uuid.uuid4().hex[:8]}__"
        stderr_start = f"__CMD_STDERR_START_{uuid.uuid4().hex[:8]}__"
        stderr_end = f"__CMD_STDERR_END_{uuid.uuid4().hex[:8]}__"
        cwd_marker = f"__CMD_CWD_{uuid.uuid4().hex[:8]}__"
        # Send command, capture its stderr separately, then echo exit code,
        # current cwd, and end marker. cwd is captured AFTER the command runs,
        # so the agent can see whether a `cd` inside the command drifted cwd.
        full_cmd = "\n".join(
            [
                "__THEAUTO_STDERR_FILE=$(mktemp)",
                (
                    f"if cd -- {shlex.quote(selected_workdir)}; then"
                    if selected_workdir
                    else "if true; then"
                ),
                "{",
                "set -o pipefail",
                normalized_command,
                '} 2>"$__THEAUTO_STDERR_FILE"',
                "__EXIT_CODE__=$?",
                "else",
                "__EXIT_CODE__=125",
                "fi",
                f"printf '\\n{stderr_start}\\n'",
                'cat "$__THEAUTO_STDERR_FILE"',
                f"printf '\\n{stderr_end}\\n'",
                'rm -f "$__THEAUTO_STDERR_FILE"',
                f"printf '{cwd_marker}%s\\n' \"$(pwd)\"",
                f"printf '\\n{end_marker} %s\\n' \"$__EXIT_CODE__\"",
                "",
            ]
        )
        self._shell_proc.stdin.write(full_cmd.encode())
        await self._shell_proc.stdin.drain()

        # Read until end marker
        raw_output = await self._read_until(end_marker, timeout=self.cmd_timeout)

        # Parse exit code from marker line
        exit_code = 1
        import re as _re
        marker_match = _re.search(rf"{end_marker}\s+(\d+)", raw_output)
        if marker_match:
            exit_code = int(marker_match.group(1))
            # Remove marker line and everything after from output
            raw_output = raw_output[:marker_match.start()]

        # Parse cwd marker (captured after the command ran)
        cwd_after: Optional[str] = None
        cwd_match = _re.search(rf"{cwd_marker}([^\n\r]*)", raw_output)
        if cwd_match:
            cwd_after = cwd_match.group(1).strip()
            raw_output = raw_output.replace(cwd_match.group(0), "", 1)

        stdout, stderr = _split_wrapped_output(raw_output, normalized_command, stderr_start, stderr_end)

        # Determine status
        if end_marker not in (raw_output + (marker_match.group(0) if marker_match else "") + (cwd_match.group(0) if cwd_match else "")):
            status = "timeout"
            exit_code = 124
            self._shell_timed_out = True
            if self._shell_proc is not None:
                try:
                    self._shell_proc.kill()
                except ProcessLookupError:
                    pass
        else:
            status = "success" if exit_code == 0 else "failed"

        # Surface cwd drift to the agent by appending a short footer to the
        # visible excerpt. Keeps raw stdout intact for log files / grep tools.
        excerpt = _tail_excerpt(stdout, self.excerpt_chars)
        if cwd_after and cwd_after != self.work_dir:
            excerpt = f"{excerpt}\n[cwd is now: {cwd_after}]"

        log_path = self._write_command_log(normalized_command, stdout, stderr, exit_code, cwd_after)

        result: Dict[str, object] = {
            "input": normalized_command,
            "command": normalized_command,
            "output": excerpt,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": str(exit_code),
            "status": status,
            "cwd": cwd_after,
            "fingerprint": _exit_fingerprint(exit_code),
            "log_path": log_path,
        }
        if selected_workdir:
            result["workdir"] = selected_workdir
        self.record.append(result)
        return result

    async def probe_command(self, command: str, timeout: int = 30) -> Dict[str, object]:
        """Run a read-only controller probe without adding a build record."""
        normalized_command = _normalize_command(command, sysuse=_sysuse_from_image(self.image_name))
        if not normalized_command.strip() or not self.container_id or self._shell_timed_out:
            return {"command": normalized_command, "exit_code": None, "status": "failed"}
        return_code, stdout, stderr = await self._run_subprocess(
            ["docker", "exec", "-w", self.work_dir, self.container_id, "sh", "-lc", normalized_command],
            timeout=timeout,
        )
        return {
            "command": normalized_command,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": return_code,
            "status": "success" if return_code == 0 else "failed",
        }

    def _write_command_log(self, command: str, stdout: str, stderr: str, exit_code: Optional[int], cwd_after: Optional[str] = None) -> Optional[str]:
        if not self.log_dir:
            return None
        index = len(self.record) + 1
        filename = f"command_{index:03d}.log"
        path = self.log_dir / filename
        content = [
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"image: {self.image_name or ''}",
            f"container: {self.container_id or ''}",
            f"work_dir: {self.work_dir}",
            f"exit_code: {exit_code}",
        ]
        if cwd_after is not None:
            content.append(f"cwd_after: {cwd_after}")
        content.extend([
            "",
            f"$ {command}",
            "",
            "--- stdout ---",
            stdout,
            "--- stderr ---",
            stderr,
        ])
        path.write_text("\n".join(content), encoding="utf-8", errors="replace")
        return str(path)

    def get_record(self) -> List[Dict[str, object]]:
        return self.record

    def latest_record(self) -> Optional[Dict[str, object]]:
        return self.record[-1] if self.record else None

    async def create_snapshot(self, label: str = "manual") -> str:
        """Create a temporary image snapshot. Kept for compatibility; not used by default."""
        if not self.container_id:
            return ""
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)[:40] or "snapshot"
        image_name = f"theauto-temp-{safe_label}-{uuid.uuid4().hex[:10]}"
        return_code, stdout, stderr = await self._run_subprocess(
            ["docker", "commit", self.container_id, image_name], timeout=300
        )
        if return_code == 0:
            self.temp_images.append(image_name)
            return stdout.strip() or image_name
        print(f"[DockerManager] Failed to create snapshot: {stderr or stdout}")
        return ""

    async def restore_snapshot(self, snapshot_image: str) -> bool:
        """Recreate the container from a snapshot image, preserving mounts.

        Used between multi-card trials so each card starts from a clean
        environment (the previous card's apt installs / flag edits don't
        pollute the next attempt). The snapshot_image must have been created
        by `create_snapshot()`.

        Args:
            snapshot_image: Image name returned by `create_snapshot()`.

        Returns:
            True on success, False on failure (caller should keep going with
            the current container state in that case).
        """
        if not snapshot_image or not self.compile_path:
            return False
        if snapshot_image not in self.temp_images:
            print(f"[DockerManager] restore_snapshot: unknown snapshot image {snapshot_image}")
            return False

        # Kill the persistent shell, remove the current container, then
        # start a fresh one from the snapshot image with the same mounts.
        if self._shell_proc is not None:
            try:
                self._shell_proc.kill()
            except ProcessLookupError:
                pass
            self._shell_proc = None
        self._shell_timed_out = False

        old_container = self.container_id or self.container_name
        if old_container:
            try:
                await self._run_subprocess(
                    ["docker", "rm", "-f", old_container], timeout=30,
                )
            except Exception as exc:
                print(f"[DockerManager] restore_snapshot: failed to remove old container: {exc}")
        self.container_id = None

        # Start a new container from the snapshot image, preserving the
        # original compile_path and extra_volumes.
        ok = await self.start_container(
            snapshot_image,
            self.compile_path,
            extra_volumes=self._last_extra_volumes or None,
        )
        if ok:
            print(f"[DockerManager] Restored from snapshot {snapshot_image[:24]}")
        else:
            print(f"[DockerManager] Failed to restore snapshot {snapshot_image[:24]}")
        return ok

    async def cleanup(self) -> None:
        """Stop/remove the container and remove any temporary images we created."""
        if self._shell_proc is not None:
            try:
                self._shell_proc.kill()
            except ProcessLookupError:
                pass
            self._shell_proc = None
        self._shell_timed_out = False

        container = self.container_id or self.container_name
        if container:
            try:
                subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                print(f"[DockerManager] Container cleaned up: {str(container)[:12]}")
            except Exception as exc:
                print(f"[DockerManager] Error cleaning up container: {exc}")
        self.container_id = None
        self.container_name = None

        for image_name in list(self.temp_images):
            try:
                subprocess.run(["docker", "rmi", "-f", image_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except Exception as exc:
                print(f"[DockerManager] Error removing temp image {image_name}: {exc}")
        self.temp_images.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False


async def init_docker_agent(
    sysuse: str,
    compile_path: str,
    log_dir: Optional[str] = None,
    extra_volumes: Optional[List[tuple]] = None,
) -> DockerContainerManager:
    """Initialize the Docker execution backend for a repository build.

    Args:
        sysuse: Debian codename (e.g. "bookworm") to select the base image.
        compile_path: Host path of the main project repo; mounted at ``/work``.
        log_dir: Where to write per-command docker logs.
        extra_volumes: Optional list of ``(host_path, container_path)`` mounts
            in addition to the main project. Used for source-dep clones.
    """
    manager = DockerContainerManager(work_dir=DEFAULT_WORK_DIR, log_dir=log_dir)
    image_name = manager.select_image(sysuse)
    if not await manager.start_container(image_name, compile_path, extra_volumes=extra_volumes):
        raise RuntimeError(f"Failed to start Docker container with image {image_name}")
    return manager


async def init_global_docker_agent(
    sysuse: str,
    compile_path: str,
    log_dir: Optional[str] = None,
    extra_volumes: Optional[List[tuple]] = None,
) -> DockerContainerManager:
    global docker_agent
    docker_agent = await init_docker_agent(sysuse, compile_path, log_dir=log_dir, extra_volumes=extra_volumes)
    return docker_agent


# Backwards-compatible async MCP-style function. The main workflow injects the
# manager directly, so this global is only for legacy MCP server usage.
docker_agent: Optional[DockerContainerManager] = None


async def Docker_Shell_Interact(command: str) -> Dict[str, object]:
    normalized_command = _normalize_command(command)
    if _is_control_command(normalized_command):
        return {
            "input": normalized_command,
            "command": normalized_command,
            "output": f"Control token '{normalized_command}' was not executed as a shell command.",
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "status": "control",
            "fingerprint": "exit=None",
            "log_path": None,
        }
    if docker_agent is None:
        return {
            "input": normalized_command,
            "command": normalized_command,
            "output": "Docker agent is not initialized",
            "stdout": "",
            "stderr": "Docker agent is not initialized",
            "exit_code": None,
            "status": "failed",
            "fingerprint": "exit=None",
            "log_path": None,
        }
    return await docker_agent.execute_command(command)
