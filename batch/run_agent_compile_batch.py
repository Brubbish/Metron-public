#!/usr/bin/env python3
"""Batch runner for the Docker-backed agent compiler.

The input JSON can be either the records_parse list format:
[
  {"name": "owner", "real_name": "repo", "local_path": "/path/to/repo", "commit_id": "..."}
]
or the older nested owner -> repo -> commit mapping.

For each item this runner resets the source repo to the commit, copies that clean
checkout into output-root/worktrees, then compiles the copy with agent_build_main.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import re
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from LLM_Config import get_model_api_keys, resolve_model_name
from tools.Docker_Interact import container_name_prefix
from core.debian_sysuse import debian_codename_for_date, git_commit_date, sysuse_for_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_ABLATION_MODES = {"none", "no_phase1", "no_phase3"}
_API_KEY_OVERRIDE_ENV = "THEAUTO_LLM_API_KEY_OVERRIDE"
_AGENT_MODE_ENV = "THEAUTO_AGENT_MODE"
_DEFAULT_AGENT_MODE = "tool-calling"
_VALID_AGENT_MODES = ("tool-calling", "json-planner")


@dataclass(frozen=True)
class CompileTarget:
    index: int
    owner: str
    repo: str
    local_path: str
    commit: str

    @property
    def key(self) -> str:
        return f"{self.owner}##{self.repo}##{self.commit}"


class ModelApiKeyRotator:
    """Assign configured API keys to child launches in round-robin order."""

    def __init__(self, api_keys: Iterable[str]):
        self._api_keys = tuple(api_keys)
        if not self._api_keys:
            raise ValueError("model API key pool is empty")
        self._next_index = 0
        self._lock = threading.Lock()

    @property
    def key_count(self) -> int:
        return len(self._api_keys)

    def next_key(self) -> tuple[int, str]:
        with self._lock:
            index = self._next_index % len(self._api_keys)
            self._next_index += 1
        return index, self._api_keys[index]


def get_system_by_date(commit_date: date | str) -> str:
    """Skill-compatible Debian codename selection by exact commit date."""
    return debian_codename_for_date(commit_date)


def get_system_by_year(commit_year: int) -> str:
    """Legacy helper for callers that only have a year; batch uses full dates."""
    return get_system_by_date(date(int(commit_year), 12, 31))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120]


def resolve_ablation_mode(ablation: str = "") -> str:
    mode = (ablation or os.environ.get("THEAUTO_ABLATION") or "none").strip().lower()
    if mode not in _ABLATION_MODES:
        raise ValueError(f"unsupported ablation mode: {mode}")
    return mode


def load_targets(
    input_json: str,
    offset: int = 0,
    limit: Optional[int] = None,
    only_c_cpp: bool = False,
) -> List[CompileTarget]:
    with open(input_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    targets: List[CompileTarget] = []
    if isinstance(payload, list):
        for raw_index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            if item.get("delete") is True:
                continue
            if only_c_cpp and item.get("is_c_cpp") is not True:
                continue
            owner = item.get("name") or item.get("owner")
            repo = item.get("real_name") or item.get("repo") or item.get("repository")
            local_path = item.get("local_path") or item.get("path")
            commit = item.get("commit_id") or item.get("commit") or item.get("commit_hash")
            if not (owner and repo and local_path and commit):
                continue
            targets.append(CompileTarget(raw_index, str(owner), str(repo), str(local_path), str(commit)))
    elif isinstance(payload, dict):
        raw_index = 0
        for owner, repos in payload.items():
            if not isinstance(repos, dict):
                continue
            for repo, commits in repos.items():
                if not isinstance(commits, dict):
                    continue
                for commit in commits:
                    local_path = os.environ.get("METRON_ENV_MAPPING", os.path.join(os.path.dirname(__file__), "data", "env_mapping")) + f"/{owner}/{repo}"
                    targets.append(CompileTarget(raw_index, str(owner), str(repo), local_path, str(commit)))
                    raw_index += 1
    else:
        raise ValueError("input JSON must be a list or nested owner/repo/commit dict")

    selected = targets[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_checked(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def get_commit_year(repo_path: str, commit: str) -> Optional[int]:
    commit_date = get_commit_date(repo_path, commit)
    return commit_date.year if commit_date else None


def get_commit_date(repo_path: str, commit: str) -> Optional[date]:
    return git_commit_date(repo_path, commit)


def source_commit_state(repo_path: str, commit: str) -> Dict[str, Any]:
    """Ensure the source repo can resolve the requested commit before copying."""
    def commit_exists() -> tuple[bool, str]:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            timeout=30,
        )
        return result.returncode == 0, (result.stderr or result.stdout or "").strip()

    exists, error = commit_exists()
    state: Dict[str, Any] = {
        "commit_exists": exists,
        "checkout_fetch_attempted": False,
        "checkout_fetch_error": "",
        "checkout_preflight_error": error[:500],
    }
    if exists:
        return state

    state["checkout_fetch_attempted"] = True
    try:
        fetch = subprocess.run(
            ["git", "fetch", "--all", "--tags", "--prune"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        state["checkout_fetch_error"] = repr(exc)[:500]
        return state

    if fetch.returncode != 0:
        state["checkout_fetch_error"] = (fetch.stderr or fetch.stdout or "").strip()[:500]
    exists, error = commit_exists()
    state["commit_exists"] = exists
    state["checkout_preflight_error"] = error[:500]
    return state


def checkout_worktree(repo_path: str, commit: str) -> None:
    run_checked(["git", "checkout", "-f", commit], cwd=repo_path)
    run_checked(["git", "reset", "--hard", commit], cwd=repo_path)
    run_checked(["git", "clean", "-ffdx"], cwd=repo_path)
    run_checked(["git", "submodule", "deinit", "--all", "-f"], cwd=repo_path)
    run_checked(["git", "clean", "-ffdx"], cwd=repo_path)


# Agent writes its per-commit artifacts (enhanced_dep_result.json,
# docker_commands/, BuildLog_*.txt, etc.) into a subdir named after the
# commit SHA. When the source repo was used by a prior TheAuto run, that
# subdir persists in the working tree and gets copied into the worktree,
# which then makes load_enhanced_dependency_result() read the stale result
# instead of recomputing — masking the live sysuse_select decision.
# Match full (40-hex) and short (>=7-hex) commit-SHA directory names so the
# copy never carries over historical agent output.
_COMMIT_SHA_DIR_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _ignore_agent_artifacts(_directory: str, names: List[str]) -> Set[str]:
    ignored: Set[str] = set()
    for name in names:
        # Drop agent_log_dir subdirs named after commit SHAs.
        if _COMMIT_SHA_DIR_RE.match(name):
            ignored.add(name)
    return ignored


def copy_repo(src: str, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True, ignore=_ignore_agent_artifacts)


def remove_existing_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def clear_previous_run_artifacts(worktree: Path, log_path: Path) -> None:
    remove_existing_path(worktree)
    remove_existing_path(log_path)


def elf_dbg_check(path: Path) -> bool:
    # Disabled: save_bins 已停用,此处保留签名以避免下游 NameError。原始实现见 git 历史。
    # try:
    #     if path.read_bytes()[:4] != b"\x7fELF":
    #         return False
    #     output = subprocess.getoutput(f"file {path}")
    #     return "not stripped" in output
    # except Exception:
    #     return False
    return False


def save_bins(install_path: Path, save_path: Path) -> int:
    # Disabled: ELF 收集与拷贝逻辑已停用。保留函数签名以维持向后兼容,
    # 调用方仍会拿到 bins_saved=0。原始实现见 git 历史。
    # if not install_path.exists():
    #     return 0
    # bins = [path for path in install_path.rglob("*") if path.is_file() and elf_dbg_check(path)]
    # if not bins:
    #     return 0
    # save_path.mkdir(parents=True, exist_ok=True)
    # for binary in bins:
    #     shutil.copy2(binary, save_path / binary.name)
    # return len(bins)
    return 0


def status_from_returncode(returncode: int) -> str:
    return {
        0: "done",
        1: "error",
        2: "timeout",
        5: "provider_error",
        6: "no_command",
        7: "network_failure",
    }.get(returncode, f"retcode_{returncode}")


WORKFLOW_TERMINAL_STATUSES = {
    "done",
    "cannot_solve",
    "no_plan",
    "no_command",
    "provider_error",
    "timeout",
    "network_failure",
    "error",
}


def load_done_keys(record_path: Path) -> set[str]:
    if not record_path.exists():
        return set()
    done_keys: set[str] = set()
    with record_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "done" and record.get("key"):
                done_keys.add(str(record["key"]))
    return done_keys


def load_existing_keys(record_path: Path) -> set[str]:
    """Return every key that already appears in records.jsonl, regardless of status."""
    if not record_path.exists():
        return set()
    existing_keys: set[str] = set()
    with record_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("key"):
                existing_keys.add(str(record["key"]))
    return existing_keys


def load_provider_error_keys(record_path: Path) -> set[str]:
    """Return keys whose canonical records.jsonl status is provider_error."""
    if not record_path.exists():
        return set()
    status_by_key: dict[str, str] = {}
    done_keys: set[str] = set()
    with record_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(record.get("key") or "")
            if not key:
                continue
            status = str(record.get("status") or "")
            if status == "done":
                done_keys.add(key)
                status_by_key[key] = status
                continue
            if key in done_keys:
                continue
            status_by_key[key] = status
    return {key for key, status in status_by_key.items() if status == "provider_error"}


def _log_proves_success(log_path: str) -> Optional[str]:
    if not log_path:
        return None
    path = Path(log_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = (
        "Compilation succeeded",
        "Rule-based success marker",
        "Rule-based artifact evidence",
        "Successfully built",
        "Built target",
    )
    for marker in markers:
        if marker in text:
            return marker
    return None


def reconcile_records(
    record_path: Path,
    summary_path: Path,
    ablation: str = "",
    allowed_keys: Optional[set[str]] = None,
) -> dict[str, int]:
    if not record_path.exists():
        return {"total": 0, "done": 0, "reconciled": 0}
    records: list[dict[str, Any]] = []
    for line in record_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)

    by_key: dict[str, dict[str, Any]] = {}
    reconciled = 0
    for rec in records:
        key = str(rec.get("key") or "")
        if not key:
            continue
        marker = _log_proves_success(str(rec.get("log_path") or ""))
        if marker and rec.get("status") != "done":
            rec = dict(rec)
            rec["status"] = "done"
            rec["returncode"] = 0
            rec["reconciled_from_log"] = True
            rec["reconcile_reason"] = marker
            reconciled += 1
        current = by_key.get(key)
        # A successful run wins over every non-successful retry. Otherwise the
        # later line is the current terminal observation for this target.
        if current is None or current.get("status") != "done" or rec.get("status") == "done":
            by_key[key] = rec

    canonical = [by_key[key] for key in sorted(by_key)]
    tmp = record_path.with_suffix(record_path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n" for rec in canonical),
        encoding="utf-8",
    )
    tmp.replace(record_path)
    done = sum(1 for rec in canonical if rec.get("status") == "done")
    summary = summarize_record_file(record_path, ablation, allowed_keys=allowed_keys)
    summary["reconciled"] = reconciled
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"total": len(canonical), "done": done, "reconciled": reconciled}


def filter_skip_done(targets: List[CompileTarget], done_keys: set[str]) -> List[CompileTarget]:
    if not done_keys:
        return targets
    return [target for target in targets if target.key not in done_keys]


def append_jsonl(path: Path, payload: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_run_result(worktree: Path, commit: str) -> dict[str, Any]:
    path = worktree / commit / "run_result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def summarize_record_file(
    record_path: Path,
    ablation: str,
    *,
    allowed_keys: Optional[set[str]] = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if record_path.exists():
        for line in record_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    if allowed_keys is not None:
        records = [item for item in records if str(item.get("key") or "") in allowed_keys]
    # A retry is a new attempt, but one target contributes once to the outcome.
    def attempt_rank(item: dict[str, Any]) -> int:
        try:
            return int(item.get("attempt_no", 1))
        except (TypeError, ValueError):
            return 1

    by_key: dict[str, dict[str, Any]] = {}
    for item in records:
        key = str(item.get("key") or "")
        if not key:
            continue
        current = by_key.get(key)
        if current is None:
            by_key[key] = item
            continue
        current_rank = (current.get("status") == "done", attempt_rank(current))
        item_rank = (item.get("status") == "done", attempt_rank(item))
        if item_rank >= current_rank:
            by_key[key] = item
    canonical = list(by_key.values())
    by_status: dict[str, int] = {}
    for item in canonical:
        status = str(item.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "schema_version": 2,
        "total": len(canonical),
        "success": by_status.get("done", 0),
        "failed": len(canonical) - by_status.get("done", 0),
        "by_status": dict(sorted(by_status.items())),
        "attempt_records": len(records),
        "ablation": ablation,
    }


def write_invocation_header(
    logf,
    *,
    target: CompileTarget,
    batch_argv: List[str],
    batch_args: Dict[str, Any],
    child_argv: List[str],
) -> None:
    payload = {
        "target": asdict(target),
        "batch_argv": batch_argv,
        "batch_args": batch_args,
        "child_argv": child_argv,
    }
    logf.write("[batch invocation]\n")
    logf.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    logf.write("\n\n")
    logf.flush()


def build_child_env(
    model: Optional[str],
    api_key: Optional[str] = None,
    agent_mode: str = _DEFAULT_AGENT_MODE,
) -> Dict[str, str]:
    env = os.environ.copy()
    if model:
        for phase in ("DEFAULT", "DIRECT", "READREPO", "ARTIFACT_SUMMARIZER", "PHASE2", "PHASE3", "CHECK"):
            env[f"LLM_MODEL_{phase}"] = model
    env[_AGENT_MODE_ENV] = agent_mode if agent_mode in _VALID_AGENT_MODES else _DEFAULT_AGENT_MODE
    if api_key is None:
        env.pop(_API_KEY_OVERRIDE_ENV, None)
    else:
        env[_API_KEY_OVERRIDE_ENV] = api_key
    return env


def _terminate_child_process_group(proc: subprocess.Popen[Any]) -> int:
    """End the timed-out child process group without relying on child cleanup."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return 2


def _cleanup_timeout_containers(worktree: Path) -> str:
    """Remove only containers created for the timed-out worktree."""
    prefix = container_name_prefix(str(worktree))
    container_ids: list[str] = []
    for lookup_attempt in range(2):
        try:
            lookup = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name=^{prefix}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("timeout container lookup failed for %s: %r", worktree.name, exc)
            return "lookup_error"
        if lookup.returncode != 0:
            logger.warning("timeout container lookup failed for %s: %s", worktree.name, lookup.stderr.strip()[:200])
            return "lookup_failed"
        container_ids = [line.strip() for line in lookup.stdout.splitlines() if line.strip()]
        if container_ids or lookup_attempt == 1:
            break
        time.sleep(1)
    if not container_ids:
        return "not_found"
    try:
        removal = subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("timeout container cleanup failed for %s: %r", worktree.name, exc)
        return "remove_error"
    if removal.returncode != 0:
        logger.warning("timeout container cleanup failed for %s: %s", worktree.name, removal.stderr.strip()[:200])
        return "remove_failed"
    return f"removed:{len(container_ids)}"


def run_one(
    target: CompileTarget,
    *,
    output_root: Path,
    source_locks: Dict[str, threading.Lock],
    record_path: Path,
    record_lock: threading.Lock,
    timeout_sec: int,
    max_turns: int,
    model: Optional[str],
    sysuse_override: Optional[str],
    ablation: str,
    uv_bin: str,
    batch_argv: List[str],
    batch_args: Dict[str, Any],
    attempt_no: int = 1,
    api_key_rotator: Optional[ModelApiKeyRotator] = None,
    agent_mode: str = _DEFAULT_AGENT_MODE,
) -> Dict[str, Any]:
    started = time.time()
    worktree = output_root / "worktrees" / f"{target.index:05d}_{safe_name(target.owner)}__{safe_name(target.repo)}__{target.commit[:12]}"
    log_path = output_root / "logs" / f"{target.index:05d}_{safe_name(target.owner)}__{safe_name(target.repo)}__{target.commit[:12]}.log"
    result: Dict[str, Any] = {
        **asdict(target),
        "key": target.key,
        "worktree": str(worktree),
        "log_path": str(log_path),
        "status": "error",
        "returncode": None,
        "sysuse": None,
        "elapsed_sec": None,
        "bins_saved": 0,
        "ablation": ablation or "none",
        "attempt_no": attempt_no,
        "initial_sysuse": None,
        "requested_sysuse": None,
        "phase1_selected_sysuse": None,
        "first_executed_sysuse": None,
        "final_sysuse": None,
        "outcome": {},
        "resolved_model": model,
        "agent_mode": agent_mode,
        "api_key_index": None,
        "api_key_count": api_key_rotator.key_count if api_key_rotator else 0,
        "timeout_origin": "",
        "container_cleanup_status": "",
    }

    try:
        clear_previous_run_artifacts(worktree, log_path)
        if not Path(target.local_path).is_dir():
            result["status"] = "missing_repo"
            result["error"] = f"repo not found: {target.local_path}"
            return result

        source_lock = source_locks.setdefault(str(Path(target.local_path).resolve()), threading.Lock())
        with source_lock:
            commit_state = source_commit_state(target.local_path, target.commit)
            result.update(commit_state)
            if not commit_state.get("commit_exists"):
                result["status"] = "exception"
                result["exception_stage"] = "checkout"
                result["error"] = (
                    f"commit not found in source repository after fetch: {target.commit}; "
                    f"{commit_state.get('checkout_preflight_error') or commit_state.get('checkout_fetch_error')}"
                )[:800]
                return result
            commit_date = get_commit_date(target.local_path, target.commit)
            if commit_date is None:
                result["status"] = "missing_commit_date"
                result["error"] = "failed to resolve commit date"
                return result
            sysuse = sysuse_override or sysuse_for_date(commit_date, prefix=True)
            result["sysuse"] = sysuse
            result["initial_sysuse"] = sysuse
            result["requested_sysuse"] = sysuse
            logger.info("copy+checkout %s -> %s", target.key, worktree)
            checkout_error: Optional[Exception] = None
            for checkout_attempt in range(1, 3):
                try:
                    copy_repo(target.local_path, worktree)
                    checkout_worktree(str(worktree), target.commit)
                    result["checkout_attempts"] = checkout_attempt
                    checkout_error = None
                    break
                except (OSError, subprocess.SubprocessError) as exc:
                    checkout_error = exc
                    result["checkout_attempts"] = checkout_attempt
                    if checkout_attempt < 2:
                        logger.warning("checkout retry %s attempt %d: %r", target.key, checkout_attempt, exc)
                        clear_previous_run_artifacts(worktree, log_path)
            if checkout_error is not None:
                result["exception_stage"] = "checkout"
                raise checkout_error

        child_argv = [
            uv_bin,
            "run",
            "agent_build_main.py",
            f"{worktree}##{target.commit}",
            result["sysuse"],
            "--max-turns",
            str(max_turns),
            "--ablation",
            ablation or "none",
            "--agent-mode",
            agent_mode if agent_mode in _VALID_AGENT_MODES else _DEFAULT_AGENT_MODE,
        ]
        result["batch_argv"] = batch_argv
        result["batch_args"] = batch_args
        result["child_argv"] = child_argv
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("compile %s (%s)", target.key, result["sysuse"])
        with log_path.open("w", encoding="utf-8", errors="replace") as logf:
            write_invocation_header(
                logf,
                target=target,
                batch_argv=batch_argv,
                batch_args=batch_args,
                child_argv=child_argv,
            )
            api_key = None
            if api_key_rotator is not None:
                result["api_key_index"], api_key = api_key_rotator.next_key()
            proc = subprocess.Popen(
                child_argv,
                stdout=logf,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=build_child_env(model, api_key, agent_mode),
            )
            try:
                returncode = proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                returncode = _terminate_child_process_group(proc)
                result["status"] = "timeout"
                result["timeout_origin"] = "outer_batch"
                result["container_cleanup_status"] = _cleanup_timeout_containers(worktree)
            result["returncode"] = returncode
            if result["status"] != "timeout":
                result["status"] = status_from_returncode(returncode)
            logf.write(f"\n[batch elapsed] {time.time() - started:.2f}\n")

        run_result = load_run_result(worktree, target.commit)
        if run_result:
            result["outcome"] = run_result
            child_status = str(run_result.get("status") or "").strip()
            if result["status"] != "timeout" and child_status in WORKFLOW_TERMINAL_STATUSES:
                result["status"] = child_status
            result["final_sysuse"] = run_result.get("final_sysuse") or result.get("sysuse")
            result["initial_sysuse"] = run_result.get("initial_sysuse") or result.get("initial_sysuse")
            result["phase1_selected_sysuse"] = run_result.get("phase1_selected_sysuse") or result.get("phase1_selected_sysuse")
            result["first_executed_sysuse"] = run_result.get("first_executed_sysuse") or result.get("first_executed_sysuse")
            result["metrics"] = run_result.get("metrics") or {}
            result["artifacts"] = run_result.get("artifacts") or {}
            result["failure_frame"] = run_result.get("failure_frame") or {}

        if result["status"] == "done":
            install_path = worktree / target.commit / "install"
            bin_path = output_root / "bin" / f"{target.owner}##{target.repo}" / "x64" / target.commit
            try:
                result["bins_saved"] = save_bins(install_path, bin_path)
            except Exception as save_exc:
                logger.warning("save_bins failed for %s: %r", target.key, save_exc)
                result["bins_save_error"] = repr(save_exc)
    except Exception as exc:
        if result.get("status") == "done":
            logger.error(
                "post-build exception for %s but build already succeeded; keeping status=done. error=%r",
                target.key, exc,
            )
            result.setdefault("post_build_error", repr(exc))
        else:
            result["status"] = "exception"
            result["error"] = repr(exc)
    finally:
        result["elapsed_sec"] = round(time.time() - started, 2)
        append_jsonl(record_path, result, record_lock)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compile records_parse JSON with the Docker-backed agent workflow")
    parser.add_argument("--input-json", required=True, help="records_parse list JSON or nested owner/repo/commit JSON")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument(
        "--model",
        help="Model name or alias from LLM_Config/models.txt; overrides all chat phases and rotates configured keys per child",
    )
    parser.add_argument(
        "--agent-mode",
        choices=_VALID_AGENT_MODES,
        default=_DEFAULT_AGENT_MODE,
        help="Agent execution mode. Use json-planner for models that do not support provider-native tool calling.",
    )
    parser.add_argument("--sysuse", help="Override Debian suite instead of deriving it from commit year")
    parser.add_argument("--ablation", choices=sorted(_ABLATION_MODES), default="")
    parser.add_argument("--only-c-cpp", action="store_true", help="Only include list records where is_c_cpp is true before applying offset/limit")
    parser.add_argument("--skip-done", action="store_true", help="Skip targets whose key already has status=done in output-root/records.jsonl")
    parser.add_argument("--skip-existing", action="store_true", help="Skip targets whose key already appears in output-root/records.jsonl (any status)")
    parser.add_argument("--only-provider-error", action="store_true", help="Only run targets whose canonical status is provider_error in output-root/records.jsonl")
    parser.add_argument("--reconcile-records", action="store_true", help="Rewrite records.jsonl by key with done-wins and log-proven successes")
    parser.add_argument("--uv-bin", default="/usr/local/bin/uv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if getattr(args, "only_provider_error", False) and getattr(args, "skip_existing", False):
        logger.error("--only-provider-error conflicts with --skip-existing because provider_error targets already have records")
        return 2
    batch_argv = list(sys.argv)
    batch_args = vars(args).copy()
    effective_ablation = resolve_ablation_mode(args.ablation)
    batch_args["ablation"] = effective_ablation
    resolved_model = None
    api_key_rotator = None
    if args.model:
        try:
            resolved_model = resolve_model_name(args.model)
            api_key_rotator = ModelApiKeyRotator(get_model_api_keys(resolved_model))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.error("invalid --model configuration: %s", exc)
            return 2
        batch_args["resolved_model"] = resolved_model
        batch_args["api_key_count"] = api_key_rotator.key_count
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.input_json, offset=args.offset, limit=args.limit, only_c_cpp=args.only_c_cpp)
    scope_targets = list(targets)
    scope_keys = {target.key for target in scope_targets}
    record_path = output_root / "records.jsonl"
    if args.reconcile_records:
        stats = reconcile_records(
            record_path,
            output_root / "summary.json",
            effective_ablation,
            allowed_keys=scope_keys,
        )
        logger.info("reconciled records: %s", stats)
        return 0
    loaded_count = len(targets)
    skip_existing = bool(getattr(args, "skip_existing", False))
    only_provider_error = bool(getattr(args, "only_provider_error", False))
    if only_provider_error:
        provider_error_keys = load_provider_error_keys(record_path)
        targets = [target for target in targets if target.key in provider_error_keys]
        logger.info(
            "only-provider-error selected %d/%d targets with canonical provider_error records",
            len(targets),
            loaded_count,
        )
    if skip_existing:
        existing_keys = load_existing_keys(record_path)
        targets = [t for t in targets if t.key not in existing_keys]
        logger.info(
            "skip-existing removed %d/%d targets with any existing record",
            loaded_count - len(targets),
            loaded_count,
        )
    elif args.skip_done:
        done_keys = load_done_keys(record_path)
        targets = filter_skip_done(targets, done_keys)
        logger.info(
            "skip-done removed %d/%d targets with existing done records",
            loaded_count - len(targets),
            loaded_count,
        )
    logger.info("loaded %d targets from %s", len(targets), args.input_json)

    if not targets:
        summary = summarize_record_file(record_path, effective_ablation, allowed_keys=scope_keys)
        (output_root / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0 if (args.skip_done or skip_existing or only_provider_error) and loaded_count else 1

    source_locks: Dict[str, threading.Lock] = {}
    record_lock = threading.Lock()
    success_count = 0
    # Shared kwargs for run_one — reused by the network_failure retry pass below
    # so we don't duplicate the 9-argument call signature.
    run_one_kwargs = {
        "output_root": output_root,
        "source_locks": source_locks,
        "record_path": record_path,
        "record_lock": record_lock,
        "timeout_sec": args.timeout_sec,
        "max_turns": args.max_turns,
        "model": resolved_model,
        "api_key_rotator": api_key_rotator,
        "sysuse_override": args.sysuse,
        "ablation": effective_ablation,
        "uv_bin": args.uv_bin,
        "batch_argv": batch_argv,
        "batch_args": batch_args,
        "agent_mode": args.agent_mode,
    }
    # target.key → target lookup, used to reschedule network_failure retries.
    target_by_key = {t.key: t for t in targets}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(run_one, target, **run_one_kwargs) for target in targets]

        # Track which keys ended in network_failure — they get a second chance
        # at the end of this pool (transient TLS/DNS failures are common and
        # usually resolve within minutes).
        network_failed_keys: list[str] = []
        provider_failed_keys: list[str] = []

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            logger.info("%s %s", result["status"], result["key"])
            if result["status"] == "done":
                success_count += 1
            elif result["status"] == "network_failure":
                network_failed_keys.append(result["key"])
            elif result["status"] == "provider_error":
                provider_failed_keys.append(result["key"])

        # === B3: retry pass for network_failure ===
        # These are almost always transient (TLS handshake interrupted, DNS
        # hiccup, proxy blip). Re-running them once at batch end lets the
        # network recover without manual intervention. Observed data: 21/22
        # such cases succeed on the second attempt.
        if network_failed_keys:
            logger.info(
                "[B3] retrying %d network_failure target(s): %s",
                len(network_failed_keys),
                network_failed_keys,
            )
            retry_targets = [target_by_key[k] for k in network_failed_keys if k in target_by_key]
            retry_kwargs = dict(run_one_kwargs)
            retry_kwargs["attempt_no"] = 2
            retry_futures = [pool.submit(run_one, t, **retry_kwargs) for t in retry_targets]
            for future in concurrent.futures.as_completed(retry_futures):
                result = future.result()
                logger.info("[B3 retry] %s %s", result["status"], result["key"])
                # Only count newly-succeeded retries (the original failure was
                # already counted as a miss by not incrementing success_count
                # above).
                if result["status"] == "done":
                    success_count += 1

        if provider_failed_keys:
            logger.info("retrying %d provider_error target(s) once at batch end", len(provider_failed_keys))
            retry_targets = [target_by_key[key] for key in provider_failed_keys if key in target_by_key]
            retry_kwargs = dict(run_one_kwargs)
            retry_kwargs["attempt_no"] = 2
            retry_futures = [pool.submit(run_one, target, **retry_kwargs) for target in retry_targets]
            for future in concurrent.futures.as_completed(retry_futures):
                result = future.result()
                logger.info("[provider retry] %s %s", result["status"], result["key"])
                if result["status"] == "done":
                    success_count += 1

    summary = summarize_record_file(record_path, effective_ablation, allowed_keys=scope_keys)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("summary: %s", summary)
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
