import argparse
import asyncio
import configparser
import hashlib
import json
import logging
import os
import re
import shlex
import time
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.repo_assistant import (
    BuildPlanAlternativeAgentFactory,
    ConstructScriptAgentFactory,
    ErrorSummaryAgentFactory,
    ZeroEntryReadRepoAgentFactory,
)
from core.generic_ablation_agents import (
    GENERIC_PHASE3_TASK,
    GENERIC_REPO_EXPLORER_TASK,
    GenericPhase3AgentFactory,
    GenericRepoExplorerAgentFactory,
    call_generic_agent,
    is_package_install_command,
)
from core.AgentTimer import AgentTimer
from llm.llm_prompts import (
    DIAGNOSIS_ARBITRATION_SYSTEM,
    DIAGNOSIS_ARBITRATION_TASK,
    READ_REPO_TASK,
    READREPO_SYSTEM,
    ZERO_ENTRY_READ_REPO_TASK,
    ERROR_SUMMARY_TASK,
    EXTRACT_ERROR_LINE_SYS,
    EXTRACT_ERROR_LINE_USR,
    MATRIX_EMPTY_EXPLORATION_TASK,
    SHELL_INTERACT_USR,
    VALIDATION_SYSTEM,
)
from repo_parse.doc_scanner_parse import (
    _enhanced_result_usable,
    _project_extractor_status,
    get_enhanced_dependencies,
    get_relateds_from_docs_and_scanner,
)
from llm.Run_LLM import (
    call_build_agent,
    call_error_agent,
)
from llm.ask_llm import ask_ds, async_ask
from core.debian_sysuse import git_commit_date, normalize_sysuse, debian_codename_for_date
from tools.Docker_Interact import DEFAULT_WORK_DIR, init_docker_agent
from tools.policy_kb.query_policy_kb import PolicyKB
from tools.policy_kb.translate import translate_card_to_fix_plan
from tools.MCPBench import Get_False_Pkgname, Search_PkgFile_Matches
# Source-dep resolver: import as aliased names to avoid confusion with
# BuildAutomation.{source_dep,cloned_source_dep_plan} fields.
from core.source_dep_resolver import (
    compute_host_clone_root as source_dep_resolver_compute_host_clone_root,
    collect_source_dep_candidates as source_dep_resolver_collect_candidates,
    recursive_clone_and_phase1 as source_dep_resolver_recursive,
)
from core.sysuse_probe import build_sysuse_candidate_plan
from core.build_entry_selector import materialize_build_entry, rank_build_entry_candidates
from core.repair_plan import (
    InvalidRepairDecision,
    RepairDecision,
    make_provider_evidence,
    operation_schemas_for_prompt,
    parse_repair_decision,
)
from core.build_configuration_solver import select_build_configuration
from core.build_plan_state import (
    add_source_dependency_candidate,
    build_initial_plan,
    apply_repair_operation_to_plan,
    compact_build_plan_for_prompt,
    compact_failure_frame_for_prompt,
    extract_failure_frame,
    extract_failure_frame_from_records,
    make_attempt_history_record,
    record_build_attempt,
    record_repair_decision,
    record_repair_outcome,
    record_candidate_outcome,
    rebind_environment,
    register_provider_evidence,
    render_failure_frame_text,
    save_build_plan,
    select_build_entry_in_plan,
    set_pending_fix_plan,
    update_build_plan_from_failure,
)
from diagnosis.diagnosis_router import DiagnosisRoute, fuse_diagnosis, route_failure
from diagnosis.diagnosis_probes import apply_probe_refinement, run_provenance_probes
from core.repository_environment import (
    bind_required_packages,
    best_satisfying_package_version,
    metadata_matches_profile,
    repository_profile_hash,
    resolved_repository_profile_hash,
    version_satisfies,
)

# -- sysuse auto-selection from commit date --
from datetime import date as _date
import subprocess as _subprocess

_DEBIAN_VERSIONS = [
    (_date(2007, 4, 8), _date(2009, 2, 13), "etch"),
    (_date(2009, 2, 14), _date(2011, 2, 5), "lenny"),
    (_date(2011, 2, 6), _date(2013, 5, 3), "squeeze"),
    (_date(2013, 5, 4), _date(2015, 4, 24), "wheezy"),
    (_date(2015, 4, 25), _date(2017, 6, 16), "jessie"),
    (_date(2017, 6, 17), _date(2019, 7, 5), "stretch"),
    (_date(2019, 7, 6), _date(2021, 8, 13), "buster"),
    (_date(2021, 8, 14), _date(2023, 6, 9), "bullseye"),
    (_date(2023, 6, 10), _date(2025, 8, 8), "bookworm"),
    (_date(2025, 8, 9), None, "trixie"),
]

_ABLATION_MODES = {"none", "no_phase1", "no_phase3"}
_AGENT_MODE_ENV = "THEAUTO_AGENT_MODE"
_DEFAULT_AGENT_MODE = "tool-calling"
_VALID_AGENT_MODES = ("tool-calling", "json-planner")


def _commit_date_to_sysuse(repo_path: str, commit: str) -> str:
    """Select Debian codename based on commit date, with legacy floor.

    Legacy releases (jessie and older) are bumped up to `stretch` because
    their toolchains cannot build modern C/C++ sources. Override via
    env var THEAUTO_ALLOW_LEGACY_SYSUSE=1 if you genuinely need to test
    a historical-Debian build.
    """
    commit_d = git_commit_date(repo_path, commit)
    allow_legacy = os.environ.get("THEAUTO_ALLOW_LEGACY_SYSUSE", "") == "1"
    if commit_d is None:
        return "bookworm"
    return debian_codename_for_date(commit_d, floor=not allow_legacy)


def _get_commit_date(repo_path: str, commit: str) -> str:
    """Return commit date as YYYY-MM-DD string."""
    commit_d = git_commit_date(repo_path, commit)
    if commit_d is None:
        return "2023-01-01"
    return commit_d.isoformat()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

COMMAND_SUMMARY_STREAM_CHARS = 2000

BUILD_PLAN_FILENAMES = {
    "CMakeLists.txt",
    "meson.build",
    "meson_options.txt",
    "configure.ac",
    "configure.in",
    "Makefile",
    "makefile",
    "GNUmakefile",
    "bootstrap",
    "autogen.sh",
    "configure",
    "config.m4",
    "Android.mk",
    "README",
    "README.md",
    "INSTALL",
}

OPTIONAL_COMPONENT_PATTERNS = {
    "tests": ("test", "tests", "testing"),
    "examples": ("example", "examples"),
    "docs": ("doc", "docs", "documentation", "gtk-doc", "gtk_doc"),
    "introspection": ("introspection", "gir"),
    "bash-completion": ("bash-completion", "bash_completion"),
    "plugins": ("plugin", "plugins"),
    "benchmarks": ("benchmark", "benchmarks"),
}

FLOATING_SUBMODULE_REFS = {"", "head", "main", "master", "origin/main", "origin/master"}
SOURCE_DEP_ROOT = "/source_deps"


@dataclass
class FixPlan:
    status: str
    commands: List[str] = field(default_factory=list)
    reason: str = ""
    sysuse: Optional[str] = None
    source_dependency: Optional[dict] = None
    hypothesis: str = ""
    fix_direction: str = ""
    repair_action: str = ""
    target: str = ""
    evidence: List[str] = field(default_factory=list)
    expected_effect: str = ""
    matrix_rank: int = 99
    matrix_action_id: str = ""
    intervention_target: str = ""
    evidence_gate: str = ""
    selection_origin: str = ""


def Start_LLM_Logging(task_name):
    """Initialize the per-task agent log file."""
    logger = logging.getLogger("agent_build")
    loggerfile = f"AgentLogger/{task_name}.txt"
    print(loggerfile)
    os.makedirs(os.path.dirname(loggerfile), exist_ok=True)
    open(loggerfile, "w", encoding="utf-8").close()

    file_handler = logging.FileHandler(loggerfile, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


# Module-level PolicyKB singleton — load once per process (cards.jsonl).
_POLICY_KB: Optional[PolicyKB] = None


def _policy_kb() -> PolicyKB:
    global _POLICY_KB
    if _POLICY_KB is None:
        _POLICY_KB = PolicyKB()
    return _POLICY_KB


# === Phase-0 network pre-flight ===
# Single per-process probe (cached) that checks whether the container/host
# can reach github.com over HTTPS. Many build failures are caused by
# upstream TLS/DNS issues that no amount of agent retrying can fix
# (gnutls_handshake errors, "Could not resolve host", etc.). Detecting
# this early avoids burning Phase 1/2 LLM and Docker budget on
# fore-doomed builds.
_NETWORK_CHECK_RESULT: Optional[tuple] = None
_NETWORK_CHECK_LOCK = threading.Lock()
_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
_DEFAULT_PROXY_FALLBACKS = (
    # Internal proxy removed - use env var HTTP_PROXY if needed,
)


def _network_proxy_candidates() -> List[str]:
    """Return ordered proxy URLs to try after a direct network probe fails."""
    candidates: List[str] = []

    def add(value: str) -> None:
        value = str(value or "").strip()
        if value and value not in candidates:
            candidates.append(value)

    for key in _PROXY_ENV_KEYS:
        add(os.environ.get(key, ""))
    for raw in re.split(r"[\s,]+", os.environ.get("THEAUTO_PROXY_FALLBACKS", "")):
        add(raw)
    for value in _DEFAULT_PROXY_FALLBACKS:
        add(value)
    return candidates


def _proxy_env(base_env: Dict[str, str], proxy_url: str) -> Dict[str, str]:
    env = dict(base_env)
    for key in _PROXY_ENV_KEYS:
        env[key] = proxy_url
    return env


def _persist_proxy_env(proxy_url: str) -> None:
    for key in _PROXY_ENV_KEYS:
        os.environ[key] = proxy_url


def _check_network_reachable(force: bool = False) -> tuple:
    """Probe whether github.com is reachable over HTTPS.

    Returns (reachable: bool, reason: str). Cached per-process; subsequent
    calls return the cached result unless force=True. Reason is empty on
    success or a short diagnostic on failure.

    Uses a two-stage probe to be resilient under flaky networks and to
    avoid the historical failure mode where the legacy git ls-remote
    probe against github.com/git/git.git was overwhelmed by TLS handshake
    timeouts on hosts with old GnuTLS stacks:

    1. Lightweight primary probe: ``curl -I https://github.com`` with up
       to 3 retries (5s backoff). A HEAD request is far cheaper than a
       git fetch and tolerates transient packet loss.
    2. Fallback probe: the original ``git ls-remote`` against
       github.com/git/git.git (kept for environments where curl is
       unavailable or blocked).
    """
    global _NETWORK_CHECK_RESULT
    if _NETWORK_CHECK_RESULT is not None and not force:
        return _NETWORK_CHECK_RESULT
    with _NETWORK_CHECK_LOCK:
        if _NETWORK_CHECK_RESULT is not None and not force:
            return _NETWORK_CHECK_RESULT

        probe_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        reachable = False
        reason = ""

        # Stage 1: curl HEAD probe with retries.
        max_attempts = 3
        backoff_seconds = 5
        last_curl_err = ""
        for attempt in range(1, max_attempts + 1):
            try:
                result = _subprocess.run(
                    ["timeout", "10", "curl", "-sS", "-I", "--max-time", "8",
                     "-o", "/dev/null", "-w", "%{http_code}",
                     "https://github.com"],
                    capture_output=True,
                    timeout=15,
                    env=probe_env,
                )
                if result.returncode == 0:
                    stdout = result.stdout.decode(errors="ignore").strip() if result.stdout else ""
                    # Any 2xx/3xx/4xx HTTP response means the endpoint is
                    # reachable; only network-level failures (curl exit!=0)
                    # should count as unreachable.
                    if stdout and stdout[0].isdigit():
                        reachable = True
                        reason = ""
                        break
                stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
                last_curl_err = (stderr.strip()[:160]
                                 or f"curl exit={result.returncode}")
            except _subprocess.TimeoutExpired:
                last_curl_err = "curl probe timed out (15s)"
            except Exception as exc:
                last_curl_err = f"curl probe error: {exc}"
            if attempt < max_attempts:
                time.sleep(backoff_seconds)

        # Stage 2: fall back to git ls-remote when curl could not confirm.
        if not reachable:
            try:
                result = _subprocess.run(
                    ["timeout", "10", "git", "ls-remote", "--heads",
                     "https://github.com/git/git.git", "master"],
                    capture_output=True,
                    timeout=15,
                    env=probe_env,
                )
                if result.returncode == 0:
                    reachable = True
                    reason = ""
                else:
                    stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
                    git_reason = (stderr.strip()[:160] or f"git ls-remote exit={result.returncode}")
                    reason = (f"curl probe failed after {max_attempts} attempts "
                              f"({last_curl_err}); git fallback also failed ({git_reason})")
            except _subprocess.TimeoutExpired:
                reason = (f"curl probe failed ({last_curl_err}); "
                          "git ls-remote timed out (15s)")
            except Exception as exc:
                reason = (f"curl probe failed ({last_curl_err}); "
                          f"git fallback error: {exc}")

        # Stage 3: opencode-style proxy fallback. When direct host networking
        # is flaky, keep the task alive by discovering a usable proxy and
        # exporting it for later git/Docker/cargo/npm commands.
        if not reachable:
            direct_reason = reason or f"curl probe failed ({last_curl_err})"
            proxy_errors: List[str] = []
            for proxy_url in _network_proxy_candidates():
                proxy_env = _proxy_env(probe_env, proxy_url)
                proxy_reachable = False
                proxy_err = ""
                try:
                    result = _subprocess.run(
                        ["timeout", "10", "curl", "-sS", "-I", "--max-time", "8",
                         "-o", "/dev/null", "-w", "%{http_code}",
                         "https://github.com"],
                        capture_output=True,
                        timeout=15,
                        env=proxy_env,
                    )
                    if result.returncode == 0:
                        stdout = result.stdout.decode(errors="ignore").strip() if result.stdout else ""
                        proxy_reachable = bool(stdout and stdout[0].isdigit())
                    if not proxy_reachable:
                        stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
                        proxy_err = stderr.strip()[:120] or f"curl exit={result.returncode}"
                except _subprocess.TimeoutExpired:
                    proxy_err = "curl proxy probe timed out (15s)"
                except Exception as exc:
                    proxy_err = f"curl proxy probe error: {exc}"

                if not proxy_reachable:
                    try:
                        result = _subprocess.run(
                            ["timeout", "10", "git", "ls-remote", "--heads",
                             "https://github.com/git/git.git", "master"],
                            capture_output=True,
                            timeout=15,
                            env=proxy_env,
                        )
                        proxy_reachable = result.returncode == 0
                        if not proxy_reachable and not proxy_err:
                            stderr = result.stderr.decode(errors="ignore") if result.stderr else ""
                            proxy_err = stderr.strip()[:120] or f"git exit={result.returncode}"
                    except _subprocess.TimeoutExpired:
                        proxy_err = proxy_err or "git proxy probe timed out (15s)"
                    except Exception as exc:
                        proxy_err = proxy_err or f"git proxy probe error: {exc}"

                if proxy_reachable:
                    _persist_proxy_env(proxy_url)
                    reachable = True
                    reason = f"github.com reachable via proxy {proxy_url}"
                    break
                proxy_errors.append(f"{proxy_url}: {proxy_err or 'unreachable'}")

            if not reachable and proxy_errors:
                reason = f"{direct_reason}; proxy fallbacks failed ({'; '.join(proxy_errors[:3])})"

        _NETWORK_CHECK_RESULT = (reachable, reason)
        return _NETWORK_CHECK_RESULT


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fence_match = re.search(r"```(?:json|bash|sh)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> Optional[dict]:
    stripped = _strip_code_fence(text)
    candidates = [stripped]
    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(stripped[brace_start : brace_end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _diagnosis_evidence_bundle(failure_frame: dict[str, Any]) -> List[dict[str, str]]:
    bundle = [
        {
            "id": str(item.get("id") or ""),
            "source": str(item.get("source") or "unknown"),
            "text": str(item.get("text") or "")[:500],
        }
        for item in failure_frame.get("evidence_bundle") or []
        if isinstance(item, dict) and item.get("id") and item.get("text")
    ]
    if bundle:
        return bundle[:8]
    values = [
        str(failure_frame.get("primary_failure_text") or "").strip(),
        *[str(item).strip() for item in failure_frame.get("supporting_evidence") or []],
        *[str(item).strip() for item in failure_frame.get("evidence") or []],
    ]
    compact = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        compact.append({"id": f"E{len(compact) + 1}", "source": "legacy_frame", "text": value[:500]})
    return compact[:8]


def _compact_selected_action(action: Optional[dict[str, Any]]) -> dict[str, Any]:
    selected = action or {}
    operation_kinds = [str(item) for item in selected.get("operation_kinds") or [] if item]
    return {
        key: deepcopy(selected[key])
        for key in (
            "action_id",
            "repair_action",
            "fix_direction",
            "intervention_target",
            "evidence_gate",
            "selection_tier",
        )
        if selected.get(key) not in (None, "")
    } | {
        "operation_schemas": operation_schemas_for_prompt(operation_kinds),
    }


def _compact_classification_candidates(
    rule_candidates: List[dict[str, Any]],
    card_candidates: List[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rules": [
            {
                key: deepcopy(item[key])
                for key in ("canonical_error_type", "confidence", "reason", "evidence_refs")
                if item.get(key) not in (None, "", [])
            }
            for item in rule_candidates[:6]
            if isinstance(item, dict)
        ],
        "cards": [
            {
                key: deepcopy(item[key])
                for key in (
                    "card_id",
                    "canonical_error_type_candidates",
                    "matched_patterns",
                    "confidence",
                    "summary",
                )
                if item.get(key) not in (None, "", [])
            }
            for item in card_candidates[:6]
            if isinstance(item, dict)
        ],
    }


def _parse_diagnosis_arbitration(
    raw_output: str,
    evidence_bundle: List[dict[str, str]],
    allowed_error_types: List[str],
) -> Optional[dict[str, Any]]:
    parsed = _extract_json_object(str(raw_output or ""))
    if not isinstance(parsed, dict):
        return None
    classification_status = str(parsed.get("classification_status") or "").lower()
    failure_domain = str(parsed.get("failure_domain") or "unknown").lower()
    error_type = str(parsed.get("canonical_error_type") or "")
    confidence = str(parsed.get("confidence") or "").lower()
    refs = parsed.get("evidence_refs") or []
    valid_refs = {str(item.get("id") or "") for item in evidence_bundle}
    if classification_status not in {"classified", "out_of_matrix", "insufficient_evidence"}:
        return None
    if classification_status == "classified":
        if error_type not in allowed_error_types or failure_domain != "matrix":
            return None
    else:
        if error_type not in {"", "unknown"}:
            return None
        error_type = "unknown"
        if failure_domain not in {"source", "infrastructure", "unknown"}:
            return None
    if confidence not in {"high", "medium", "low"}:
        return None
    if not isinstance(refs, list) or not refs or any(str(item) not in valid_refs for item in refs):
        return None
    return {
        "classification_status": classification_status,
        "failure_domain": failure_domain,
        "canonical_error_type": error_type,
        "confidence": confidence,
        "evidence_refs": [str(item) for item in refs],
        "reason": str(parsed.get("reason") or "")[:500],
    }


async def _request_diagnosis_arbitration(
    failure_frame: dict[str, Any],
    diagnosis_route: DiagnosisRoute,
) -> tuple[str, Optional[dict[str, Any]]]:
    evidence_bundle = _diagnosis_evidence_bundle(failure_frame)
    allowed_error_types = list(diagnosis_route.llm_allowed_error_types or [])
    task = DIAGNOSIS_ARBITRATION_TASK.format(
        classification_context=json.dumps(
            {
                "observed_phase": diagnosis_route.failed_phase,
                "allowed_error_types": allowed_error_types,
                "evidence": evidence_bundle,
                "candidate_support": _compact_classification_candidates(
                    diagnosis_route.rule_candidates,
                    diagnosis_route.card_candidates,
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    raw = await async_ask(
        DIAGNOSIS_ARBITRATION_SYSTEM.format(),
        task,
        max_retries=2,
        retry_wait=2,
        caller="diagnosis_arbitration",
    )
    return raw, _parse_diagnosis_arbitration(raw, evidence_bundle, allowed_error_types)


def _validate_readrepo_plan(raw_output: Any) -> tuple[Optional[dict], str]:
    """Validate the strict ReadRepo contract before Phase 2 sees it."""
    text = str(raw_output or "").strip()
    if not text:
        return None, "ReadRepo returned an empty response."
    candidate = _strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return None, "ReadRepo did not return a strict JSON object."
    if not isinstance(parsed, dict):
        return None, "ReadRepo returned JSON with a non-object top level."

    list_fields = (
        "build_commands",
        "apt_packages",
        "source_deps",
        "unbound_source_deps",
        "setup_commands",
        "build_entry_candidates",
        "command_candidates",
    )
    for field in list_fields:
        if field in parsed and not isinstance(parsed[field], list):
            return None, f"ReadRepo field {field!r} must be a list."
    for field in ("build_entry_candidates", "command_candidates"):
        for item in parsed.get(field) or []:
            if not isinstance(item, dict):
                return None, f"ReadRepo field {field!r} contains a non-object candidate."
            if "command_sequence" in item and not isinstance(item.get("command_sequence"), list):
                return None, f"ReadRepo candidate in {field!r} has a non-list command_sequence."

    has_build_commands = any(
        isinstance(item, str) and item.strip()
        for item in parsed.get("build_commands") or []
    )
    has_build_candidates = any(
        str(item.get("command") or "").strip()
        or any(str(command).strip() for command in item.get("command_sequence") or [])
        for item in (parsed.get("build_entry_candidates") or []) + (parsed.get("command_candidates") or [])
    )
    if not (has_build_commands or has_build_candidates):
        return None, "ReadRepo returned no executable build plan."
    return parsed, ""


def _readrepo_candidate_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part or "").strip().lower() for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _normalize_phase1_setup_command(raw: Any, source: str = "") -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    commands = _command_list(raw.get("commands") or raw.get("command"))
    if not name or not commands:
        return None
    text = "\n".join(commands)
    if re.search(r"\bgit\s+clone\b|<[^>]+>|\{\{?[^{}]+\}?\}", text, re.I):
        return None
    # Setup is controller-owned environment preparation, not a generic shell
    # escape hatch. The repository is a writable host mount, so reject shell
    # composition and permit only non-destructive package/tool subcommands.
    if any(not _is_safe_phase1_setup_command(command) for command in commands):
        return None
    return {
        "id": _readrepo_candidate_id("setup", name, *commands),
        "name": name,
        "commands": commands,
        "reason": str(raw.get("reason") or "")[:300],
        "source": source or str(raw.get("source") or ""),
        "producer": str(raw.get("producer") or "artifact_summary"),
    }


def _is_safe_phase1_setup_command(command: str) -> bool:
    value = str(command or "").strip()
    if not value or re.search(r"(?:[;&|<>`]|\$\(|\n|\r)", value):
        return False
    try:
        tokens = shlex.split(value)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = tokens[0].lower()
    args = [token.lower() for token in tokens[1:]]
    if executable in {"apt", "apt-get"}:
        subcommand = next((token for token in args if not token.startswith("-")), "")
        return subcommand in {"update", "install"}
    if executable == "dpkg":
        return args == ["--configure", "-a"]
    if executable in {"pip", "pip3", "gem", "npm", "rustup"}:
        return bool(args) and args[0] in {"install", "update"}
    if executable in {"python", "python3"}:
        return len(args) >= 3 and args[:2] == ["-m", "pip"] and args[2] == "install"
    if executable == "cargo":
        return bool(args) and args[0] == "install"
    if executable == "update-alternatives":
        return bool(args) and args[0] in {"--install", "--set"}
    return False


def _build_readrepo_catalog(
    repo_context: dict[str, Any],
    artifact_summaries: List[dict[str, Any]],
    *,
    repo_path: str,
    target_commit: str,
) -> dict[str, Any]:
    """Create a bounded selector catalog from controller-owned candidates."""
    raw_entries = [
        deepcopy(item)
        for item in repo_context.get("build_entry_candidates") or []
        if isinstance(item, dict)
    ]
    known_sequences = {
        tuple(str(command).strip() for command in item.get("command_sequence") or [] if str(command).strip())
        for item in raw_entries
    }
    for item in repo_context.get("command_candidates") or []:
        if not isinstance(item, dict):
            continue
        sequence = [
            str(command).strip()
            for command in item.get("command_sequence") or [item.get("command")]
            if str(command or "").strip()
        ]
        if not sequence or tuple(sequence) in known_sequences:
            continue
        known_sequences.add(tuple(sequence))
        raw_entries.append({
            "entry_path": item.get("path") or "",
            "evidence_paths": [item.get("path")] if item.get("path") else [],
            "source": item.get("source") or "inferred",
            "confidence": item.get("confidence") or "unknown",
            "family": item.get("family") or "",
            "command_sequence": sequence,
        })
    if not raw_entries:
        commands = [str(item).strip() for item in repo_context.get("build_commands") or [] if str(item).strip()]
        if commands:
            raw_entries.append({
                "entry_path": repo_context.get("build_file_path") or "",
                "source": "phase1",
                "confidence": "main",
                "family": repo_context.get("build_system") or "",
                "command_sequence": commands,
            })
    ranked_entries = rank_build_entry_candidates(
        raw_entries,
        repo_path=repo_path,
        target_commit=target_commit,
    )
    for item in ranked_entries:
        valid, reason = _preflight_build_entry(item, repo_path)
        if valid:
            continue
        item["eligible"] = False
        item["status"] = "ineligible"
        reasons = list(item.get("eligibility_reasons") or [])
        if reason and reason not in reasons:
            reasons.append(reason)
        item["eligibility_reasons"] = reasons
    entries = [item for item in ranked_entries if item.get("eligible", True)]
    ineligible_entries = [item for item in ranked_entries if not item.get("eligible", True)]

    packages_by_name: dict[str, dict[str, Any]] = {}

    def add_package(
        value: Any,
        source: str,
        *,
        authoritative: bool = False,
        entry_ids: Optional[List[str]] = None,
    ) -> None:
        name = str(value or "").strip()
        if not name:
            return
        key = name.lower()
        row = packages_by_name.setdefault(key, {
            "id": _readrepo_candidate_id("pkg", key),
            "name": name,
            "sources": [],
            "authoritative": False,
            "entry_ids": [],
        })
        if source and source not in row["sources"]:
            row["sources"].append(source)
        row["authoritative"] = bool(row["authoritative"] or authoritative)
        row["entry_ids"] = list(dict.fromkeys(row["entry_ids"] + list(entry_ids or [])))

    inventory = repo_context.get("dependency_inventory") or {}
    for dependency in inventory.get("dependencies") or []:
        if not isinstance(dependency, dict):
            continue
        requirement = dependency.get("requirement") or {}
        if str(requirement.get("requiredness") or "") != "required":
            continue
        evidence = dependency.get("evidence") or []
        sources = [
            str(item.get("id") or item.get("path") or "")
            for item in evidence
            if isinstance(item, dict)
        ]
        for provider in dependency.get("providers") or []:
            if not isinstance(provider, dict) or provider.get("kind") != "apt":
                continue
            add_package(
                provider.get("name"),
                sources[0] if sources else str(dependency.get("dependency_id") or "dependency_inventory"),
                authoritative=True,
                entry_ids=list((dependency.get("scope") or {}).get("entry_ids") or []),
            )
    # Compatibility with older Phase-1 results: only explicitly retained
    # apt_packages are considered, and they are not promoted over inventory.
    if not packages_by_name:
        for package in repo_context.get("apt_packages") or []:
            add_package(package, "legacy_enhanced_dep_extractor", authoritative=True)
    for summary in artifact_summaries:
        if not isinstance(summary, dict):
            continue
        source = f"{summary.get('kind') or 'artifact'}:{summary.get('path') or ''}"
        packages = summary.get("apt_packages") or []
        if summary.get("kind") == "debian_control" and summary.get("resolved_apt_package_alternatives"):
            # The raw Build-Depends alternatives are discovery input only.
            # Once the live suite probe ran, retain the provider that was
            # actually present in that suite instead of reintroducing the
            # first textual alternative later as an authoritative package.
            packages = [
                item.get("selected")
                for item in summary.get("resolved_apt_package_alternatives") or []
                if isinstance(item, dict) and item.get("selected")
            ]
        for package in packages:
            add_package(
                package,
                source,
                authoritative=(summary.get("kind") == "debian_control"),
            )
    packages = list(packages_by_name.values())

    source_deps_by_key: dict[str, dict[str, Any]] = {}
    for summary_source, values in [
        ("repo_context", repo_context.get("source_deps") or []),
        *[
            (
                f"{summary.get('kind') or 'artifact'}:{summary.get('path') or ''}",
                summary.get("source_deps") or [],
            )
            for summary in artifact_summaries
            if isinstance(summary, dict)
        ],
    ]:
        for raw in values:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            url = str(raw.get("url") or raw.get("repository") or "").strip()
            path = str(raw.get("source_path") or raw.get("submodule_path") or "").strip()
            if not (name or url or path):
                continue
            key = "\x1f".join((name.lower(), url.lower(), path.lower()))
            row = source_deps_by_key.setdefault(key, {
                "id": _readrepo_candidate_id("dep", key),
                "raw": deepcopy(raw),
                "sources": [],
            })
            if summary_source not in row["sources"]:
                row["sources"].append(summary_source)
    source_deps = list(source_deps_by_key.values())
    for row in source_deps:
        raw = row.get("raw") or {}
        raw_evidence_paths = raw.get("evidence_paths") or []
        if not isinstance(raw_evidence_paths, list):
            raw_evidence_paths = []
        source_paths = {
            str(source).split(":", 1)[-1]
            for source in row.get("sources") or []
            if ":" in str(source)
        }
        source_paths.update(
            str(path).strip().lstrip("./")
            for path in [
                raw.get("source_path"),
                raw.get("submodule_path"),
                raw.get("metadata_path"),
                *raw_evidence_paths,
            ]
            if str(path or "").strip()
        )
        row["entry_ids"] = list(dict.fromkeys(
            str(entry.get("entry_id") or "")
            for entry in entries
            if entry.get("entry_id")
            and source_paths.intersection(
                str(path).strip().lstrip("./")
                for path in [
                    entry.get("entry_path"),
                    *(entry.get("evidence_paths") or []),
                ]
                if str(path or "").strip()
            )
        ))

    setup_commands = []
    for raw in repo_context.get("setup_commands") or []:
        normalized = _normalize_phase1_setup_command(raw, str(raw.get("source") or "") if isinstance(raw, dict) else "")
        if normalized is None:
            continue
        source_path = str(normalized.get("source") or "").strip()
        normalized["entry_ids"] = [
            str(entry.get("entry_id") or "")
            for entry in entries
            if entry.get("entry_id")
            and source_path
            and source_path in {
                str(entry.get("entry_path") or ""),
                *(str(path or "") for path in entry.get("evidence_paths") or []),
            }
        ]
        producer = str(normalized.get("producer") or "").lower()
        profile_scoped = (
            any(kind in producer for kind in ("cicd", "dockerfile"))
            or source_path.startswith((".github/", ".circleci/", "Dockerfile", ".gitlab-ci", ".travis"))
        )
        normalized["global"] = not profile_scoped
        setup_commands.append(normalized)

    prompt_entries = entries[:24]
    prompt_packages = packages[:32]
    prompt_source_deps = source_deps[:16]
    prompt_setup_commands = setup_commands[:16]
    prompt_catalog = {
        "build_entries": [
            {
                "id": item.get("entry_id") or "",
                "source": item.get("source") or "unknown",
                "confidence": item.get("confidence") or "unknown",
                "build_system": item.get("build_system") or "unknown",
                "entry_path": item.get("entry_path") or "",
                "scope": item.get("scope") or "unknown",
                "root_evidence": list(item.get("root_evidence") or [])[:4],
                "entry_preflight": deepcopy(item.get("entry_preflight") or {}),
                "evidence_paths": list(item.get("evidence_paths") or [])[:4],
                "command_sequence": list(item.get("command_sequence") or [])[:8],
            }
            for item in prompt_entries
        ],
        "packages": [
            {
                "id": item.get("id") or "",
                "name": item.get("name") or "",
                "sources": list(item.get("sources") or [])[:3],
                "authoritative": bool(item.get("authoritative")),
                "entry_ids": list(item.get("entry_ids") or [])[:4],
            }
            for item in prompt_packages
        ],
        "source_dependencies": [
            {
                "id": item.get("id") or "",
                "name": (item.get("raw") or {}).get("name") or "",
                "sources": list(item.get("sources") or [])[:3],
                "entry_ids": list(item.get("entry_ids") or [])[:4],
                "executable_contract": bool(
                    _source_dependency_has_executable_contract(
                        _normalize_source_dependency_plan(item.get("raw")) or {}
                    )
                ),
            }
            for item in prompt_source_deps
        ],
        "setup_commands": [
            {
                "id": item.get("id") or "",
                "name": item.get("name") or "",
                "entry_ids": list(item.get("entry_ids") or [])[:4],
                "global": bool(item.get("global")),
            }
            for item in prompt_setup_commands
        ],
    }
    return {
        "entries": entries,
        "ineligible_entries": ineligible_entries,
        "packages": packages,
        "source_deps": source_deps,
        "setup_commands": setup_commands,
        "prompt": prompt_catalog,
        "selectable_entry_ids": {
            str(item.get("entry_id") or "")
            for item in prompt_entries
            if item.get("eligible", True)
        },
        "selectable_package_ids": {
            str(item.get("id") or "") for item in prompt_packages if item.get("id")
        },
        "selectable_source_dep_ids": {
            str(item.get("id") or "") for item in prompt_source_deps if item.get("id")
        },
        "selectable_setup_command_ids": {
            str(item.get("id") or "") for item in prompt_setup_commands if item.get("id")
        },
    }


def _preflight_build_entry(entry: dict[str, Any], repo_path: str) -> tuple[bool, str]:
    """Reject only entries whose referenced source/input is definitely absent."""
    if not isinstance(entry, dict) or not repo_path:
        return True, ""
    raw_path = str(entry.get("entry_path") or entry.get("path") or "").strip()
    path = raw_path.removeprefix("/work/").lstrip("/").replace("\\", "/")
    if not path or path in {".", "-"}:
        return True, ""
    full_path = Path(repo_path) / path
    commands = "\n".join(str(item) for item in entry.get("command_sequence") or [])
    basename = Path(path).name.lower()
    generator_present = bool(re.search(
        r"\b(?:autoreconf|autogen(?:\.sh)?|bootstrap(?:\.sh)?|buildconf|aclocal|autoheader)\b",
        commands,
        re.I,
    ))
    if full_path.is_file():
        return True, ""
    if basename == "configure" and generator_present:
        configure_ac = full_path.with_name("configure.ac")
        configure_in = full_path.with_name("configure.in")
        if configure_ac.is_file() or configure_in.is_file():
            return True, ""
    if basename in {"makefile", "gnuMAKEfile".lower()} and re.search(
        r"\b(?:configure|cmake|meson(?:\s+setup)?)\b", commands, re.I
    ):
        return True, ""
    if basename == "cmakelists.txt" and full_path.is_dir():
        return True, ""
    return False, f"missing_entry_file:{path}"


async def _resolve_debian_package_alternatives(BTarget, artifact_summaries: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Resolve debian/control alternatives against the active container suite."""
    resolutions: List[dict[str, Any]] = []
    unresolved_count = 0
    for summary in artifact_summaries:
        if not isinstance(summary, dict) or summary.get("kind") != "debian_control":
            continue
        selected_packages: List[str] = []
        for raw_group in summary.get("apt_package_alternatives") or []:
            alternatives = list(dict.fromkeys(
                str(item).strip()
                for item in raw_group
                if re.fullmatch(r"[a-z0-9][a-z0-9.+-]+", str(item).strip())
            )) if isinstance(raw_group, list) else []
            if not alternatives:
                continue
            selected = ""
            probes = []
            for package in alternatives:
                command = f"apt-cache show --no-all-versions {shlex.quote(package)} >/dev/null 2>&1"
                try:
                    probe = getattr(BTarget.shell_agent, "probe_command", None)
                    if not callable(probe):
                        probes.append({"package": package, "status": "unavailable", "error": "probe_command unavailable"})
                        continue
                    result = await probe(command)
                except Exception as exc:
                    probes.append({"package": package, "status": "error", "error": str(exc)[:300]})
                    continue
                available = result.get("status") == "success" and result.get("exit_code") in {0, "0"}
                probes.append({
                    "package": package,
                    "status": "available" if available else "unavailable",
                    "exit_code": result.get("exit_code"),
                })
                if available:
                    selected = package
                    selected_packages.append(package)
                    break
            if not selected:
                unresolved_count += 1
            resolutions.append({
                "source": str(summary.get("path") or ""),
                "alternatives": alternatives,
                "selected": selected,
                "probes": probes,
            })
        if selected_packages:
            summary["apt_packages"] = list(dict.fromkeys(
                list(summary.get("apt_packages") or []) + selected_packages
            ))
        summary["resolved_apt_package_alternatives"] = [
            item for item in resolutions
            if item["source"] == str(summary.get("path") or "")
        ]

    if resolutions:
        status = "degraded" if unresolved_count else "ok"
        error = f"{unresolved_count} debian/control alternative groups have no provider in the active suite" if unresolved_count else ""
        BTarget.phase1_producers["debian_package_alternatives"] = {
            "status": status,
            "error": error,
            "groups": len(resolutions),
            "resolved": len(resolutions) - unresolved_count,
        }
        if unresolved_count:
            BTarget.phase1_status = "degraded"
            BTarget.phase1_errors.append({
                "producer": "debian_package_alternatives",
                "type": "degraded",
                "message": error,
            })
        BTarget.save_result(
            "debian_package_alternatives.json",
            json.dumps(resolutions, indent=2, ensure_ascii=False),
        )
    return resolutions


def _parse_readrepo_selection(
    raw_output: Any,
    catalog: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], str]:
    text = str(raw_output or "").strip()
    if not text:
        return None, "empty"
    candidate = _strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        parsed = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed = value
                break
        if parsed is None:
            kind = "truncated_json" if exc.pos >= max(0, len(candidate) - 32) else "invalid_json"
            return None, f"{kind}: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "invalid_schema: top level must be an object"
    status = str(parsed.get("status") or "").strip().lower()
    if status not in {"selected", "cannot_select"}:
        return None, "invalid_schema: unsupported status"
    if status == "cannot_select":
        return {
            "status": status,
            "build_entry_id": "",
            "package_ids": [],
            "source_dep_ids": [],
            "setup_command_ids": [],
            "ambiguity_decisions": [],
            "evidence_refs": [],
            "reason": str(parsed.get("reason") or "")[:500],
        }, ""

    entry_id = str(parsed.get("build_entry_id") or "").strip()
    package_ids = parsed.get("package_ids") or []
    source_dep_ids = parsed.get("source_dep_ids") or []
    setup_command_ids = parsed.get("setup_command_ids") or []
    ambiguity_decisions = parsed.get("ambiguity_decisions") or []
    evidence_refs = parsed.get("evidence_refs") or []
    if not all(
        isinstance(values, list)
        for values in (
            package_ids,
            source_dep_ids,
            setup_command_ids,
            ambiguity_decisions,
            evidence_refs,
        )
    ):
        return None, "invalid_schema: selection arrays must be lists"
    if entry_id not in catalog.get("selectable_entry_ids", set()):
        return None, "invalid_selection: unknown build_entry_id"
    selectable_groups = (
        (package_ids, catalog.get("selectable_package_ids", set()), "package_ids"),
        (source_dep_ids, catalog.get("selectable_source_dep_ids", set()), "source_dep_ids"),
        (setup_command_ids, catalog.get("selectable_setup_command_ids", set()), "setup_command_ids"),
    )
    for values, selectable_ids, field_name in selectable_groups:
        if any(str(item) not in selectable_ids for item in values):
            return None, f"invalid_selection: unknown {field_name}"
    evidence_refs = list(dict.fromkeys(str(item) for item in evidence_refs))
    valid_refs = set().union(
        catalog.get("selectable_entry_ids", set()),
        catalog.get("selectable_package_ids", set()),
        catalog.get("selectable_source_dep_ids", set()),
        catalog.get("selectable_setup_command_ids", set()),
    )
    if not evidence_refs or any(item not in valid_refs for item in evidence_refs):
        return None, "invalid_selection: evidence_refs must cite supplied candidate IDs"
    return {
        "status": status,
        "build_entry_id": entry_id,
        "package_ids": list(dict.fromkeys(str(item) for item in package_ids))[:32],
        "source_dep_ids": list(dict.fromkeys(str(item) for item in source_dep_ids))[:16],
        "setup_command_ids": list(dict.fromkeys(str(item) for item in setup_command_ids))[:16],
        "ambiguity_decisions": ambiguity_decisions[:8],
        "evidence_refs": evidence_refs[:8],
        "reason": str(parsed.get("reason") or "")[:500],
    }, ""


_ZERO_ENTRY_BUILD_COMMAND_RE = re.compile(
    r"\b(?:make|gmake|ninja)\b|\bcmake\s+--build\b|\bmeson\s+compile\b|"
    r"\brake\s+(?:compile|build|native|extension)\b|\bgem\s+build\b|"
    r"\b(?:python|python3)\s+setup\.py\s+build\b|"
    r"\bscons\b|(?:^|\s)(?:\./)?waf\s+build\b|\bxmake(?:\s|$)|"
    r"\b(?:gcc|g\+\+|clang|clang\+\+|cc|c\+\+)\b|"
    r"(?:^|\s)(?:(?:bash|sh)\s+)?\./[^\s]*(?:build|compile)[^\s]*",
    re.I,
)

_PHASE2_PROGRESS_COMMAND_RE = re.compile(
    r"\b(?:apt-get|apt)\s+install\b|\bdpkg\s+-i\b|"
    r"(?:^|\s)(?:\./)?(?:configure|autogen\.sh|bootstrap)(?:\s|$)|"
    r"\b(?:autoreconf|autoconf|automake|aclocal|libtoolize|gtkdocize|intltoolize)\b|"
    r"\b(?:make|gmake|ninja)\b|\bcmake\s+(?:--build|-S|-B)\b|"
    r"\bmeson\s+(?:setup|compile)\b|\brake\s+(?:compile|build|native|extension)\b|"
    r"\bgem\s+build\b|\b(?:python|python3)\s+setup\.py\s+build\b|"
    r"\b(?:scons|xmake)\b|(?:^|\s)(?:\./)?waf\s+build\b|"
    r"\b(?:gcc|g\+\+|clang|clang\+\+|cc|c\+\+)\b|"
    r"\b(?:cp|install|patch|sed\s+-i|touch|mkdir)\b",
    re.I,
)

_PHASE2_LOW_PROGRESS_COMMAND_RE = re.compile(
    r"^\s*(?:cd\s+\S+\s*&&\s*)?"
    r"(?:ls|find|cat|sed\s+-n|head|tail|grep|rg|stat|file|command|which|"
    r"realpath|readlink|pkg-config|apt-cache|dpkg-query|ps|echo|pwd|true|test)\b",
    re.I,
)

_PHASE2_NONTERMINAL_TIMEOUT_RE = re.compile(
    r"\b(?:apt-get|apt|dpkg|cpan|pip|gem|cargo|git|curl|wget|"
    r"autoreconf|autoconf|automake|aclocal|libtoolize|gtkdocize|intltoolize|"
    r"configure|autogen\.sh|bootstrap|cmake\s+-S|cmake\s+-B|meson\s+setup)\b",
    re.I,
)


def _parse_zero_entry_candidates(
    raw_output: Any,
    inventory: dict[str, Any],
    repo_path: str,
) -> tuple[list[dict[str, Any]], str]:
    """Validate bounded model-discovered pipelines without rewriting commands."""
    text = str(raw_output or "").strip()
    if not text:
        return [], "empty"
    parsed = _extract_json_object(text)
    if parsed is None:
        return [], "invalid_json: no JSON object found"
    if not isinstance(parsed, dict):
        return [], "invalid_schema: top level must be an object"
    status = str(parsed.get("status") or "").strip().lower()
    if status == "cannot_select":
        return [], "cannot_select"
    pipelines = parsed.get("pipelines")
    if status != "selected" or not isinstance(pipelines, list) or not pipelines:
        return [], "invalid_schema: selected requires pipelines"

    root = Path(repo_path).resolve()
    known_files = {
        str(item.get("path") or "")
        for item in inventory.get("files") or []
        if isinstance(item, dict) and str(item.get("path") or "")
    }
    known_dirs = {str(item or "") for item in inventory.get("directories") or []}
    known_dirs.add(".")
    candidates: list[dict[str, Any]] = []
    for pipeline in pipelines[:3]:
        if not isinstance(pipeline, dict):
            continue
        entry_path = str(pipeline.get("entry_path") or "").strip().replace("\\", "/").removeprefix("./")
        evidence_paths = list(dict.fromkeys(
            str(item or "").strip().replace("\\", "/").removeprefix("./")
            for item in pipeline.get("evidence_paths") or []
            if str(item or "").strip()
        ))
        if not entry_path or entry_path not in known_files or entry_path not in evidence_paths:
            continue
        if any(path not in known_files for path in evidence_paths):
            continue
        try:
            (root / entry_path).resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            continue
        steps: list[dict[str, str]] = []
        has_build = False
        for raw_step in pipeline.get("command_steps") or []:
            if not isinstance(raw_step, dict):
                continue
            command = str(raw_step.get("command") or "")
            stage = str(raw_step.get("stage") or "").strip().lower()
            workdir = str(raw_step.get("workdir") or "").strip().replace("\\", "/")
            relative_workdir = "." if workdir == "/work" else workdir.removeprefix("/work/")
            workdir_exists = relative_workdir in known_dirs or (root / relative_workdir).is_dir()
            if (
                not command.strip()
                or len(command) > 1200
                or any(char in command for char in ("\x00", "\n", "\r"))
                or stage not in {"prepare", "bootstrap", "configure", "build", "compile", "link", "validate"}
                or not workdir.startswith("/work")
                or not workdir_exists
            ):
                steps = []
                break
            command_is_build = _ZERO_ENTRY_BUILD_COMMAND_RE.search(command) is not None
            normalized_stage = "bootstrap" if stage == "prepare" else "compile" if stage == "build" else stage
            if normalized_stage == "configure" and command_is_build:
                normalized_stage = "compile"
            has_build = has_build or normalized_stage in {"compile", "link"} or command_is_build
            steps.append({"stage": normalized_stage, "command": command, "workdir": workdir})
        if not steps or not has_build:
            continue
        candidates.append({
            "entry_path": entry_path,
            "evidence_paths": evidence_paths[:12],
            "source": "readrepo_explicit",
            "confidence": "medium",
            "family": str(pipeline.get("build_system") or "unknown")[:80],
            "command_steps": steps,
            "scope": "root" if "/" not in entry_path else "subproject",
        })
    if not candidates:
        return [], "invalid_selection: no pipeline passed evidence and stage validation"
    return candidates, ""


def _materialize_readrepo_plan(
    repo_context: dict[str, Any],
    catalog: dict[str, Any],
    selection: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    entries = list(catalog.get("entries") or [])
    selected_id = str((selection or {}).get("build_entry_id") or "")
    selected_entry = (
        next(
            (
                item
                for item in entries
                if str(item.get("entry_id") or "") == selected_id
                and item.get("eligible", True)
            ),
            None,
        )
        if selected_id
        else entries[0] if entries else None
    )
    if selected_id and selected_entry is None:
        entries = []
    selected_entry_id = str((selected_entry or {}).get("entry_id") or "")
    selected_package_ids = set((selection or {}).get("package_ids") or [])
    selected_source_dep_ids = set((selection or {}).get("source_dep_ids") or [])
    selected_setup_command_ids = set((selection or {}).get("setup_command_ids") or [])
    # Required dependencies remain controller-owned. ReadRepo may add only
    # package candidates already backed by Phase-1 evidence.
    apt_packages = [
        str(item.get("name") or "")
        for item in catalog.get("packages") or []
        if (item.get("authoritative") or str(item.get("id") or "") in selected_package_ids)
        and (
            not item.get("entry_ids")
            or selected_entry_id in item.get("entry_ids")
        )
    ]
    source_rows = []
    for item in catalog.get("source_deps") or []:
        normalized = _normalize_source_dependency_plan(item.get("raw"))
        if normalized is None:
            continue
        has_source = bool(
            normalized.get("repo_url")
            or normalized.get("source_path")
            or normalized.get("submodule_path")
        )
        has_build_contract = _source_dependency_has_executable_contract(normalized)
        if (
            has_source
            and has_build_contract
            and item.get("sources")
            and (
                not selected_source_dep_ids
                or str(item.get("id") or "") in selected_source_dep_ids
            )
            and (
                not item.get("entry_ids")
                or selected_entry_id in (item.get("entry_ids") or [])
            )
        ):
            source_rows.append(item)
    unbound_source_deps = [
        {
            **deepcopy(item.get("raw")),
            "catalog_id": str(item.get("id") or ""),
            "evidence_sources": list(item.get("sources") or []),
            "status": "unbound",
        }
        for item in catalog.get("source_deps") or []
        if isinstance(item.get("raw"), dict) and item not in source_rows
    ]
    source_deps = []
    for item in source_rows:
        normalized = _normalize_source_dependency_plan(item.get("raw"))
        if normalized:
            source_deps.append(normalized)
    build_commands = list((selected_entry or {}).get("command_sequence") or [])
    setup_commands = [
        deepcopy(item)
        for item in catalog.get("setup_commands") or []
        if (
            item.get("global")
            or selected_entry_id in (item.get("entry_ids") or [])
        )
        and (
            not selected_setup_command_ids
            or str(item.get("id") or "") in selected_setup_command_ids
        )
    ]
    return {
        "build_system": (selected_entry or {}).get("build_system") or repo_context.get("build_system") or "unknown",
        "build_file_path": (selected_entry or {}).get("entry_path") or repo_context.get("build_file_path") or "",
        "apt_packages": list(dict.fromkeys(item for item in apt_packages if item)),
        "source_deps": source_deps,
        "unbound_source_deps": unbound_source_deps,
        "setup_commands": setup_commands,
        "build_commands": build_commands,
        "selected_build_entry_id": str((selected_entry or {}).get("entry_id") or ""),
        "build_entry_candidates": entries,
        "command_candidates": [
            {
                "entry_id": item.get("entry_id") or "",
                "command": (item.get("command_sequence") or [""])[0],
                "command_sequence": list(item.get("command_sequence") or []),
                "source": item.get("source") or "unknown",
                "confidence": item.get("confidence") or "unknown",
                "path": item.get("entry_path") or "",
                "family": item.get("build_system") or "unknown",
            }
            for item in entries
            if item.get("command_sequence") and item.get("eligible", True)
        ],
    }


def _looks_like_command(line: str) -> bool:
    line = line.strip().lstrip("$ ").strip()
    if not line or line.startswith("#"):
        return False
    command_prefixes = (
        "apt ",
        "apt-get ",
        "export ",
        "cmake ",
        "make",
        "ninja",
        "meson ",
        "./configure",
        "./autogen.sh",
        "autoreconf",
        "autoconf",
        "autoheader",
        "automake",
        "aclocal",
        "libtoolize",
        "intltoolize",
        "glib-gettextize",
        "git ",
        "mkdir ",
        "cd ",
        "python ",
        "python3 ",
        "pip ",
        "pkg-config ",
        "sed ",
        "ln ",
        "cp ",
        "rm ",
    )
    return line.startswith(command_prefixes) or " && " in line


def _summary_excerpt(value: object, limit: int = COMMAND_SUMMARY_STREAM_CHARS) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]


def _looks_like_phase2_retryable_error(output: str) -> bool:
    text = str(output or "")
    lowered = text.lower()
    retryable_markers = (
        "is not a valid tool",
        "validation error for",
        "field required",
    )
    return any(marker in lowered for marker in retryable_markers)


def _looks_like_provider_error(output: str) -> bool:
    text = str(output or "")
    lowered = text.lower()
    provider_markers = (
        "provider returned error",
        "the operation was aborted",
        "authorization failed",
        "authentication failed",
        "invalid_request_error",
        "error code: 429",
        "http 429",
        "rate limit",
        "invalid api key",
        "请求失败",
    )
    return any(marker in lowered for marker in provider_markers)


def _read_small_text(path: str, limit: int = 120_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(limit)
    except OSError:
        return ""


def _iter_build_plan_files(repo_path: str, max_files: int = 120, exclude_paths: Optional[List[str]] = None) -> List[str]:
    """Return likely build/config files without walking generated dependency trees forever."""
    results: List[str] = []
    skip_dirs = {
        ".git",
        "__pycache__",
        "build",
        "builddir",
        "cmake-build-debug",
        "node_modules",
        ".cache",
        "install",
    }
    repo_path = os.path.abspath(repo_path)
    excluded = [
        os.path.abspath(os.path.join(repo_path, path))
        for path in (exclude_paths or [])
        if str(path).strip()
    ]
    for root, dirs, files in os.walk(repo_path):
        abs_root = os.path.abspath(root)
        if any(abs_root == path or abs_root.startswith(path + os.sep) for path in excluded):
            dirs[:] = []
            continue
        rel_root = os.path.relpath(root, repo_path)
        depth = 0 if rel_root == "." else len(rel_root.split(os.sep))
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".tox")]
        if depth > 4:
            dirs[:] = []
            continue
        for filename in files:
            if filename in BUILD_PLAN_FILENAMES or filename.endswith((".cmake", ".mk", ".am")):
                results.append(os.path.join(root, filename))
                if len(results) >= max_files:
                    return results
    return results


def _collect_build_plan_text(repo_path: str, include_submodules: bool = False) -> str:
    exclude_paths = [] if include_submodules else [item["path"] for item in _parse_gitmodules(repo_path)]
    chunks = []
    for path in _iter_build_plan_files(repo_path, exclude_paths=exclude_paths):
        rel_path = os.path.relpath(path, repo_path)
        text = _read_small_text(path)
        if text:
            chunks.append(f"\n# FILE: {rel_path}\n{text}")
    return "\n".join(chunks)


def _parse_gitmodules(repo_path: str) -> List[Dict[str, Any]]:
    gitmodules_path = os.path.join(repo_path, ".gitmodules")
    if not os.path.exists(gitmodules_path):
        return []
    parser = configparser.ConfigParser()
    try:
        parser.read(gitmodules_path, encoding="utf-8")
    except configparser.Error:
        return []

    submodules = []
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        name = section[len("submodule ") :].strip().strip('"')
        path = parser.get(section, "path", fallback="").strip()
        url = parser.get(section, "url", fallback="").strip()
        branch = parser.get(section, "branch", fallback="").strip()
        if not path:
            continue
        submodules.append({"name": name or os.path.basename(path), "path": path, "url": url, "branch": branch})
    return submodules


def _run_git(args: List[str], cwd: str, timeout: int = 60) -> Optional[_subprocess.CompletedProcess]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return _subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None


def _git_stdout(args: List[str], cwd: str, timeout: int = 60) -> str:
    result = _run_git(args, cwd, timeout=timeout)
    if result is None or result.returncode != 0:
        return ""
    return result.stdout.strip()


def _superproject_submodule_commit(repo_path: str, submodule_path: str, commit: str = "HEAD") -> str:
    output = _git_stdout(["ls-tree", commit, "--", submodule_path], repo_path)
    match = re.search(r"\bcommit\s+([0-9a-f]{40})\b", output)
    return match.group(1) if match else ""


def _submodule_current_commit(full_path: str) -> str:
    return _git_stdout(["rev-parse", "HEAD"], full_path)


def _submodule_current_ref(full_path: str) -> str:
    branch = _git_stdout(["symbolic-ref", "--quiet", "--short", "HEAD"], full_path)
    if branch:
        return branch
    commit = _submodule_current_commit(full_path)
    return commit or ""


def _submodule_revision_type(current_ref: str, configured_branch: str) -> str:
    normalized_ref = current_ref.strip().lower()
    normalized_branch = configured_branch.strip().lower()
    if normalized_ref in FLOATING_SUBMODULE_REFS or normalized_branch in FLOATING_SUBMODULE_REFS:
        return "floating"
    if re.fullmatch(r"[0-9a-f]{40}", current_ref.strip(), re.IGNORECASE):
        return "detached_commit"
    if current_ref:
        return "branch"
    return "unknown"


def _latest_submodule_commit_before(full_path: str, ref: str, project_commit_date: str) -> str:
    candidates = []
    clean_ref = ref.strip()
    if clean_ref:
        candidates.extend([clean_ref, f"origin/{clean_ref}"] if "/" not in clean_ref else [clean_ref])
    candidates.extend(["origin/main", "origin/master", "main", "master", "HEAD"])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        selected = _git_stdout(["rev-list", "-n1", f"--before={project_commit_date}", candidate], full_path)
        if selected:
            return selected
    return ""


def _apply_submodule_revision_policy(
    repo_path: str,
    submodules: List[dict],
    project_commit: str = "HEAD",
    project_commit_date: Optional[str] = None,
) -> List[dict]:
    """Record exact submodule checkout state and pin floating refs when possible."""
    updated = []
    for raw_item in submodules:
        item = dict(raw_item)
        path = str(item.get("path") or "")
        full_path = os.path.join(repo_path, path)
        if item.get("status") == "unreachable" or not os.path.isdir(full_path):
            item.update(
                {
                    "availability": "unavailable",
                    "current_revision": "",
                    "current_revision_type": "unknown",
                    "selected_revision": "",
                    "checkout_action": "not_available",
                    "clone_allowed": False,
                    "url_recovery_allowed": item.get("status") == "unreachable",
                }
            )
            updated.append(item)
            continue

        configured_branch = str(item.get("branch") or "")
        current_ref = _submodule_current_ref(full_path)
        current_commit = _submodule_current_commit(full_path)
        revision_type = _submodule_revision_type(current_ref, configured_branch)
        pinned_commit = _superproject_submodule_commit(repo_path, path, project_commit)
        selected_commit = pinned_commit
        selection_reason = "superproject_gitlink" if pinned_commit else "current_checkout"
        checkout_action = "none"

        if not selected_commit and revision_type == "floating" and project_commit_date:
            selected_commit = _latest_submodule_commit_before(full_path, configured_branch or current_ref, project_commit_date)
            if selected_commit:
                selection_reason = "latest_commit_before_project_date"

        if selected_commit and selected_commit != current_commit:
            result = _run_git(["checkout", "--detach", selected_commit], full_path)
            if result is not None and result.returncode == 0:
                current_commit = selected_commit
                current_ref = selected_commit
                revision_type = "detached_commit"
                checkout_action = "checked_out_selected_revision"
            else:
                checkout_action = "checkout_failed"
                item["checkout_error"] = (
                    result.stderr.strip() or result.stdout.strip() or "git checkout failed"
                    if result is not None
                    else "git checkout failed"
                )

        item.update(
            {
                "availability": "available",
                "current_revision": current_commit,
                "current_revision_type": revision_type,
                "selected_revision": selected_commit or current_commit,
                "selection_reason": selection_reason,
                "checkout_action": checkout_action,
                "clone_allowed": False,
                "url_recovery_allowed": False,
            }
        )
        updated.append(item)
    return updated


def _submodule_checkout_status(repo_path: str, submodule_path: str, preflight_by_path: Dict[str, dict]) -> str:
    if submodule_path in preflight_by_path:
        return str(preflight_by_path[submodule_path].get("status") or "missing")
    full_path = os.path.join(repo_path, submodule_path)
    try:
        if os.path.isdir(full_path):
            with os.scandir(full_path) as entries:
                if any(entries):
                    return "cloned"
    except OSError:
        pass
    return "missing"


def _path_re(pattern: str) -> str:
    return re.escape(pattern).replace("/", r"[/\\]")


def _infer_submodule_role(submodule_path: str, submodule_name: str, build_text: str, status: str) -> str:
    if status == "unreachable":
        return "unreachable"

    lowered_path_parts = {part.lower() for part in re.split(r"[/\\]+", submodule_path) if part}
    optional_markers = {
        "test",
        "tests",
        "example",
        "examples",
        "doc",
        "docs",
        "plugin",
        "plugins",
        "benchmark",
        "benchmarks",
    }
    if lowered_path_parts & optional_markers:
        return "optional_only"

    path_expr = _path_re(submodule_path.strip("/"))
    basename = os.path.basename(submodule_path.rstrip("/")) or submodule_name
    basename_expr = re.escape(basename)
    integrated_patterns = (
        rf"add_subdirectory\s*\(\s*[\"']?{path_expr}",
        rf"AC_CONFIG_SUBDIRS\s*\(\s*\[?{path_expr}",
        rf"\bmake\s+-C\s+{path_expr}\b",
        rf"\binclude\s+{path_expr}[/\\]",
        rf"\bsubproject\s*\(\s*[\"']{basename_expr}[\"']",
    )
    if any(re.search(pattern, build_text, re.IGNORECASE) for pattern in integrated_patterns):
        return "main_integrated"

    source_required_patterns = (
        rf"find_package\s*\(\s*{basename_expr}\b",
        rf"pkg_check_modules\s*\([^)]*\b{basename_expr}\b",
        rf"PKG_CHECK_MODULES\s*\([^)]*\b{basename_expr}\b",
    )
    if any(re.search(pattern, build_text, re.IGNORECASE) for pattern in source_required_patterns):
        return "prebuild_required"
    if submodule_path in build_text:
        return "main_integrated"
    return "asset_or_tooling"


def _build_submodule_plan(repo_path: str, preflight: Optional[List[dict]] = None) -> List[Dict[str, Any]]:
    submodules = _parse_gitmodules(repo_path)
    if not submodules:
        return []
    build_text = _collect_build_plan_text(repo_path)
    preflight_by_path = {
        str(item.get("path")): item
        for item in (preflight or [])
        if isinstance(item, dict) and item.get("path")
    }
    plan = []
    for item in submodules:
        path = str(item["path"])
        preflight_item = preflight_by_path.get(path, {})
        status = _submodule_checkout_status(repo_path, path, preflight_by_path)
        role = _infer_submodule_role(path, str(item.get("name") or ""), build_text, status)
        evidence = []
        if path in build_text:
            evidence.append(f"build files mention {path}")
        if role == "main_integrated":
            evidence.append("main build appears to reference the submodule directly")
        elif role == "prebuild_required":
            evidence.append("main build appears to find the dependency as an installed package")
        elif role == "optional_only":
            evidence.append("submodule path indicates optional test/example/doc/plugin content")
        plan_item = {
            "name": item.get("name") or os.path.basename(path),
            "path": path,
            "url": item.get("url") or "",
            "branch": item.get("branch") or "",
            "status": status,
            "role": role,
            "evidence": evidence,
            "build_order": [],
        }
        for key in (
            "availability",
            "current_revision",
            "current_revision_type",
            "selected_revision",
            "selection_reason",
            "checkout_action",
            "checkout_error",
            "clone_allowed",
        ):
            if key in preflight_item:
                plan_item[key] = preflight_item[key]
        if "availability" not in plan_item:
            plan_item["availability"] = "available" if status == "cloned" else "unavailable"
        plan_item.setdefault("current_revision", "")
        plan_item.setdefault("current_revision_type", "unknown")
        plan_item.setdefault("selected_revision", plan_item.get("current_revision") or "")
        plan_item.setdefault("selection_reason", "unknown")
        plan_item.setdefault("checkout_action", "none" if status == "cloned" else "not_available")
        plan_item.setdefault("clone_allowed", False)
        plan_item.setdefault("url_recovery_allowed", status == "unreachable")
        plan.append(plan_item)
    return plan


def _infer_toolchain_profile(repo_path: str, project_info: dict) -> dict:
    build_text = _collect_build_plan_text(repo_path)
    lowered = build_text.lower()
    build_system = str(project_info.get("build_system") or "").lower()
    if "phpize" in lowered or "php_arg_enable" in lowered or "zend_module" in lowered:
        return {"kind": "php-extension", "evidence": ["phpize/PHP extension markers found"]}
    if "mingw" in lowered or "cmake_system_name windows" in lowered or "windres" in lowered:
        return {"kind": "mingw", "evidence": ["MinGW/Windows cross-compile markers found"]}
    if "android_ndk" in lowered or "android.mk" in lowered or "aosp" in lowered or "platform_bionic" in repo_path.lower():
        return {"kind": "android", "evidence": ["Android/AOSP build markers found"]}
    if "gnustep" in lowered or build_system == "gnustep":
        return {"kind": "objc-gnustep", "evidence": ["GNUstep/Objective-C markers found"]}
    if re.search(r"(^|\s)-m32(\s|$)", build_text) or "multilib" in lowered:
        return {"kind": "multilib", "evidence": ["32-bit/multilib markers found"]}
    return {"kind": "native", "evidence": []}


def _infer_target_scope(project_info: dict, toolchain_profile: dict, repo_path: str) -> dict:
    kind = str(toolchain_profile.get("kind") or "native")
    if kind == "php-extension":
        return {"kind": "extension", "reason": "PHP extension toolchain detected"}
    if kind == "android":
        return {"kind": "minimal-module", "reason": "Android/AOSP projects should avoid full tree builds by default"}
    if not project_info.get("build_commands"):
        return {"kind": "unknown", "reason": "no primary build commands found"}
    if any(os.path.exists(os.path.join(repo_path, name)) for name in ("WORKSPACE", "MODULE.bazel")):
        return {"kind": "minimal-module", "reason": "Bazel-style repository may require explicit target selection"}
    return {"kind": "default", "reason": "use primary build plan"}


def _infer_optional_feature_plan(repo_path: str) -> List[dict]:
    build_text = _collect_build_plan_text(repo_path)
    lowered = build_text.lower()
    plan = []
    seen_flags = set()
    for component, markers in OPTIONAL_COMPONENT_PATTERNS.items():
        if not any(marker in lowered for marker in markers):
            continue
        disable_flags = []
        for flag in re.findall(r"--disable-[A-Za-z0-9_.-]+", build_text):
            if any(marker.replace("_", "-") in flag.lower() for marker in markers):
                disable_flags.append(flag)
        cmake_candidates = [
            f"-DBUILD_{component.upper().replace('-', '_')}=OFF",
            f"-DENABLE_{component.upper().replace('-', '_')}=OFF",
        ]
        if component == "tests":
            cmake_candidates.append("-DBUILD_TESTING=OFF")
        for flag in cmake_candidates:
            token = flag[2:].split("=")[0]
            if token in build_text:
                disable_flags.append(flag)
        for flag in disable_flags:
            if flag in seen_flags:
                continue
            seen_flags.add(flag)
            plan.append({"component": component, "disable_flag": flag, "evidence": "local build files mention optional feature"})
    return plan


def _source_dep_from_submodule(item: dict, install_prefix: str) -> Optional[dict]:
    if item.get("role") != "prebuild_required":
        return None
    name = str(item.get("name") or os.path.basename(str(item.get("path") or ""))).strip()
    path = str(item.get("path") or "").strip()
    if not name or not path:
        return None
    plan = _normalize_source_dependency_plan({
        "name": name,
        "reason": "submodule appears to be required as a prebuilt dependency",
        "apt_alternative": None,
        "source_path": path,
        "submodule_path": path,
        "build_order": item.get("build_order") or [],
        "install_prefix": install_prefix,
        "metadata_path": None,
        "instructions": "",
        "apt_compatibility": "unverified",
        "try_apt_first": False,
    })
    if plan:
        plan["apt_compatibility"] = "unverified"
    return plan


def _format_cloned_dep_reason(cloned: dict) -> str:
    """Build a human-readable reason string for a cloned source dep."""
    url = cloned.get("clone_url", "")
    origin = cloned.get("origin", "")
    vc = cloned.get("version_constraint", "")
    parts = [f"phase1-cloned source dep from {url}", f"origin={origin}"]
    if vc:
        parts.append(f"version_constraint={vc!r}")
    return " (".join([parts[0], ", ".join(parts[1:]) + ")"]) if len(parts) > 1 else parts[0]


def _build_submodule_execution_contract(submodule_plan: List[dict]) -> dict:
    required_roles = {"main_integrated", "prebuild_required"}
    required = [
        item for item in submodule_plan
        if isinstance(item, dict) and item.get("role") in required_roles
    ]
    missing = [
        item for item in required
        if item.get("availability") == "unavailable" or item.get("status") in {"missing", "unreachable"}
    ]
    checkout_required = [
        item for item in required
        if item.get("checkout_action") in {"checked_out_selected_revision", "checkout_failed"}
        or item.get("current_revision_type") == "floating"
    ]
    return {
        "has_submodules": bool(submodule_plan),
        "required_submodules_present": not missing,
        "clone_or_init_allowed": False,
        "url_recovery_allowed": bool(missing),
        "checkout_verification_required": bool(checkout_required),
        "missing_required_paths": [str(item.get("path") or "") for item in missing],
    }


def _compact_submodule_prompt_state(submodules: List[dict]) -> List[dict[str, Any]]:
    compact = []
    for item in submodules:
        if not isinstance(item, dict):
            continue
        record = {
            key: deepcopy(item[key])
            for key in (
                "name", "path", "status", "role", "availability", "selected_revision",
                "checkout_action", "evidence",
            )
            if item.get(key) not in (None, "", [], {})
        }
        for key in ("error", "checkout_error"):
            if item.get(key):
                record[key] = str(item[key])[-500:]
        compact.append(record)
    return compact


def _rewrite_git_protocol_submodule_urls(repo_path: str, submodules: List[dict]) -> bool:
    changed = False
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    for item in submodules:
        url = str(item.get("url") or "")
        name = str(item.get("name") or "")
        if not name or not url.startswith("git://"):
            continue
        https_url = "https://" + url[len("git://") :]
        result = _subprocess.run(
            ["git", "config", f"submodule.{name}.url", https_url],
            cwd=repo_path,
            env=env,
            capture_output=True,
            text=True,
        )
        changed = changed or result.returncode == 0
    return changed


def preflight_git_submodules(
    repo_path: str,
    log_dir: Optional[str] = None,
    timeout: int = 300,
    project_commit: str = "HEAD",
    project_commit_date: Optional[str] = None,
) -> List[dict]:
    """Best-effort submodule initialization before repository scanning."""
    submodules = _parse_gitmodules(repo_path)
    if not submodules:
        return []

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    attempts = [
        ["git", "submodule", "sync", "--recursive"],
        ["git", "submodule", "update", "--init", "--recursive"],
    ]
    errors = []
    for command in attempts:
        try:
            result = _subprocess.run(
                command,
                cwd=repo_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as exc:
            errors.append(f"{' '.join(command)}: {exc}")
            continue
        if result.returncode != 0:
            errors.append(result.stderr.strip() or result.stdout.strip() or f"{command} failed")

    if errors and _rewrite_git_protocol_submodule_urls(repo_path, submodules):
        try:
            result = _subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=repo_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                errors = []
            else:
                errors.append(result.stderr.strip() or result.stdout.strip() or "https submodule retry failed")
        except Exception as exc:
            errors.append(f"https submodule retry: {exc}")

    plan = _build_submodule_plan(repo_path)
    if errors:
        for item in plan:
            if item["status"] == "missing":
                item["status"] = "unreachable"
                item["error"] = errors[-1]
    plan = _apply_submodule_revision_policy(repo_path, plan, project_commit, project_commit_date)

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "submodule_preflight.json"), "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
    return plan


def _normalize_source_dependency_plan(raw: Any) -> Optional[dict]:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raw = {"name": str(raw)}

    def _string_value(*keys: str) -> str:
        for key in keys:
            value = raw.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    name = _string_value("name", "dependency", "dependency_name")
    build_order = _command_list(
        raw.get("build_order") or raw.get("build_commands") or raw.get("commands") or []
    )
    setup_commands = _command_list(raw.get("setup_commands"))
    fetch_commands = _command_list(raw.get("fetch_commands"))
    source_path = _string_value("source_path", "source_location") or None
    submodule_path = _string_value("submodule_path") or None
    container_source_path = _source_dep_container_path(raw, name, source_path, submodule_path)
    install_prefix = _string_value("install_prefix") or "/usr/local"
    env_exports = _normalize_env_exports(raw.get("env_exports") or raw.get("exports"), install_prefix)

    plan = {
        "name": name,
        "reason": _string_value("reason", "evidence"),
        "apt_alternative": _string_value("apt_alternative", "apt_package") or None,
        "source_path": source_path,
        "submodule_path": submodule_path,
        "container_source_path": container_source_path,
        "apt_packages": [
            str(item).strip()
            for item in (raw.get("apt_packages") or raw.get("apt_install_plan") or [])
            if str(item).strip()
        ],
        "build_order": build_order,
        "setup_commands": setup_commands,
        "fetch_commands": fetch_commands,
        "install_prefix": install_prefix,
        "env_exports": env_exports,
        "metadata_path": _string_value("metadata_path", "metadata") or None,
        "instructions": _string_value("instructions"),
        "repo_url": _string_value("repo_url", "url", "git_url") or None,
        "revision": _string_value("revision", "ref", "commit") or None,
        "try_apt_first": bool(_string_value("apt_alternative", "apt_package")),
        "target_commit_preserved": True,
    }
    return plan if plan["name"] else None


def _command_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _source_dependency_has_executable_contract(value: Any) -> bool:
    """A source dependency is executable only with explicit build commands."""
    if isinstance(value, dict):
        commands = value.get("build_order") or value.get("build_commands") or []
    else:
        commands = getattr(value, "build_order", [])
    return bool(_command_list(commands))


def _source_dep_container_path(raw: dict, name: str, source_path: Optional[str], submodule_path: Optional[str]) -> str:
    for key in ("container_source_path", "container_path"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    for value in (source_path, submodule_path):
        if value and str(value).startswith("/"):
            return str(value)
    if submodule_path:
        return f"{DEFAULT_WORK_DIR}/{str(submodule_path).lstrip('./')}"
    if source_path and not str(source_path).startswith("/"):
        source = str(source_path).lstrip("./")
        parts = {part.lower() for part in re.split(r"[\\/]+", source)}
        if parts & {"third_party", "3rdparty", "3dparty", "deps", "vendor", "external", "submodule", "submodules"}:
            return f"{DEFAULT_WORK_DIR}/{source}"
    safe_name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name or "dependency").strip("_") or "dependency"
    return f"{SOURCE_DEP_ROOT}/{safe_name}"


def _normalize_env_exports(value: Any, install_prefix: str) -> list[str]:
    exports = _command_list(value)
    for item in _default_source_dep_exports(install_prefix):
        if item not in exports:
            exports.append(item)
    return exports


def _default_source_dep_exports(install_prefix: str) -> list[str]:
    prefix = (install_prefix or "/usr/local").rstrip("/")
    return [
        f"export PKG_CONFIG_PATH={prefix}/lib/pkgconfig:{prefix}/share/pkgconfig:$PKG_CONFIG_PATH",
        f"export LD_LIBRARY_PATH={prefix}/lib:{prefix}/lib64:$LD_LIBRARY_PATH",
        f"export CMAKE_PREFIX_PATH={prefix}:$CMAKE_PREFIX_PATH",
        f"export C_INCLUDE_PATH={prefix}/include:$C_INCLUDE_PATH",
        f"export CPLUS_INCLUDE_PATH={prefix}/include:$CPLUS_INCLUDE_PATH",
        f"export PATH={prefix}/bin:$PATH",
    ]


def _stable_signature(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def _parse_build_plan_alternative_selection(
    raw_output: Any,
    valid_alternative_ids: set[str],
) -> Optional[dict[str, Any]]:
    parsed = _extract_json_object(str(raw_output or ""))
    if not isinstance(parsed, dict):
        return None
    status = str(parsed.get("status") or "").strip().lower()
    evidence = parsed.get("evidence") or []
    if not isinstance(evidence, list):
        return None
    if status == "cannot_solve":
        return {
            "status": status,
            "alternative_id": "",
            "evidence": [str(item)[:500] for item in evidence[:4]],
            "reason": str(parsed.get("reason") or "")[:500],
        }
    alternative_id = str(parsed.get("alternative_id") or "").strip()
    if (
        status != "select"
        or alternative_id not in valid_alternative_ids
    ):
        return None
    return {
        "status": status,
        "alternative_id": alternative_id,
        "evidence": [str(item)[:500] for item in evidence[:4]],
        "reason": str(parsed.get("reason") or "")[:500],
    }


def _extract_source_dependency_plan(parsed: dict) -> Optional[dict]:
    source_dependency = parsed.get("source_dependency") or parsed.get("source_dep")
    if not source_dependency and parsed.get("source_dependencies"):
        source_dependencies = parsed.get("source_dependencies")
        if isinstance(source_dependencies, list) and source_dependencies:
            source_dependency = source_dependencies[0]
    if not source_dependency and parsed.get("needs_source_dependency"):
        source_dependency = parsed
    return _normalize_source_dependency_plan(source_dependency)


def _extract_lenient_commands_array(text: str) -> List[str]:
    match = re.search(r'"commands"\s*:\s*\[', text or "", re.IGNORECASE)
    if not match:
        return []
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    end = -1
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return []
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    commands = []
    for item in parsed:
        if not isinstance(item, str):
            continue
        command = item.strip()
        if _looks_like_command(command):
            commands.append(command)
    return commands


def parse_fix_plan(raw_output: str) -> FixPlan:
    """Parse Phase3 fixer output into executable commands or cannot_solve."""
    text = (raw_output or "").strip()
    lowered = text.lower()
    parsed = _extract_json_object(text)

    def _extract_hypothesis(reason_text: str) -> str:
        """Extract 'hypothesis: <tag>' from reason if present."""
        m = re.match(r'hypothesis:\s*([\w-]+)', reason_text or "", re.IGNORECASE)
        return m.group(1).rstrip(";") if m else ""

    if parsed is not None:
        reason = str(parsed.get("reason") or text)
        hypothesis = _extract_hypothesis(reason)
        fix_direction = str(parsed.get("fix_direction") or "").strip()
        repair_action = str(parsed.get("repair_action") or parsed.get("matrix_action") or "").strip()
        target = str(parsed.get("target") or "").strip()
        intervention_target = str(parsed.get("intervention_target") or "").strip()
        evidence_gate = str(parsed.get("evidence_gate") or "").strip()
        selection_origin = str(parsed.get("selection_origin") or parsed.get("selection_tier") or "").strip()
        evidence = parsed.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
        expected_effect = str(parsed.get("expected_effect") or "").strip()
        matrix_rank = parsed.get("matrix_rank", 99)
        try:
            matrix_rank = int(matrix_rank)
        except (TypeError, ValueError):
            matrix_rank = 99
        status = str(parsed.get("status") or "fix").lower().replace("-", "_")
        commands = parsed.get("commands") or []
        if isinstance(commands, str):
            commands = [commands]
        commands = [str(cmd).strip() for cmd in commands if str(cmd).strip()]
        sysuse = parsed.get("sysuse") or None
        if isinstance(sysuse, str):
            sysuse = sysuse.strip() or None
        source_dependency = _extract_source_dependency_plan(parsed)
        if status in {"cannot_solve", "cannot_resolve", "cannot resolve"}:
            if sysuse:
                return FixPlan(status="fix", commands=commands, reason=reason,
                               sysuse=sysuse, source_dependency=source_dependency, hypothesis=hypothesis,
                               fix_direction=fix_direction, repair_action=repair_action, target=target,
                               evidence=evidence, expected_effect=expected_effect, matrix_rank=matrix_rank,
                               intervention_target=intervention_target, evidence_gate=evidence_gate,
                               selection_origin=selection_origin)
            if source_dependency:
                return FixPlan(status="source_dependency", reason=reason,
                               source_dependency=source_dependency, hypothesis=hypothesis,
                               fix_direction=fix_direction, repair_action=repair_action, target=target,
                               evidence=evidence, expected_effect=expected_effect, matrix_rank=matrix_rank,
                               intervention_target=intervention_target, evidence_gate=evidence_gate,
                               selection_origin=selection_origin)
            return FixPlan(status="cannot_solve", reason=reason, sysuse=sysuse, hypothesis=hypothesis,
                           fix_direction=fix_direction, repair_action=repair_action, target=target,
                           evidence=evidence, expected_effect=expected_effect, matrix_rank=matrix_rank,
                           intervention_target=intervention_target, evidence_gate=evidence_gate,
                           selection_origin=selection_origin)
        if commands or sysuse:
            return FixPlan(status="fix", commands=commands, reason=reason,
                           sysuse=sysuse, source_dependency=source_dependency, hypothesis=hypothesis,
                           fix_direction=fix_direction, repair_action=repair_action, target=target,
                           evidence=evidence, expected_effect=expected_effect, matrix_rank=matrix_rank,
                           intervention_target=intervention_target, evidence_gate=evidence_gate,
                           selection_origin=selection_origin)
        if source_dependency:
            return FixPlan(status="source_dependency", reason=reason,
                           source_dependency=source_dependency, hypothesis=hypothesis,
                           fix_direction=fix_direction, repair_action=repair_action, target=target,
                           evidence=evidence, expected_effect=expected_effect, matrix_rank=matrix_rank,
                           intervention_target=intervention_target, evidence_gate=evidence_gate,
                           selection_origin=selection_origin)
        return FixPlan(status="cannot_solve", reason=reason, hypothesis=hypothesis,
                       fix_direction=fix_direction, repair_action=repair_action, target=target,
                       evidence=evidence, expected_effect=expected_effect, matrix_rank=matrix_rank,
                       intervention_target=intervention_target, evidence_gate=evidence_gate,
                       selection_origin=selection_origin)

    sysuse_match = re.search(r"sysuse\s*[:=]\s*([A-Za-z0-9_./:-]+)", text, re.IGNORECASE)
    sysuse = sysuse_match.group(1) if sysuse_match else None
    hypothesis = _extract_hypothesis(text)
    commands = _extract_lenient_commands_array(text)
    if commands:
        return FixPlan(status="fix", commands=commands, reason="Parsed command repair plan.",
                       sysuse=sysuse, hypothesis=hypothesis)
    if "cannot_solve" in lowered or "cannot resolve" in lowered or "cannot be resolved" in lowered:
        if sysuse:
            return FixPlan(status="fix", reason=text, sysuse=sysuse, hypothesis=hypothesis)
        return FixPlan(status="cannot_solve", reason=text, hypothesis=hypothesis)

    command_text = _strip_code_fence(text)
    commands = []
    for line in command_text.splitlines():
        candidate = line.strip().lstrip("- ").lstrip("$ ").strip()
        if _looks_like_command(candidate):
            commands.append(candidate)
    if commands or sysuse:
        return FixPlan(status="fix", commands=commands, reason="Parsed command repair plan.",
                       sysuse=sysuse, hypothesis=hypothesis)
    return FixPlan(status="cannot_solve", reason="No executable repair commands found in fixer output.",
                   hypothesis=hypothesis)


def bind_fix_plan_to_route(plan: FixPlan, diagnosis_route: Optional[DiagnosisRoute]) -> FixPlan:
    """Bind a concrete plan to the controller-selected matrix action.

    The matrix chooses the repair direction; Phase 3 is only allowed to fill in
    the repository-specific operation.  Missing metadata is filled for
    compatibility with older translators, but an explicit conflicting action
    is rejected rather than silently re-routed.
    """
    if diagnosis_route is None:
        return plan
    scope = diagnosis_route.repair_scope or {}
    # Compatibility for callers that construct a pre-matrix route manually.
    # Real routes always carry matrix_actions and matrix_key.
    if "matrix_actions" not in scope and "matrix_key" not in scope:
        return plan
    selected = scope.get("selected_matrix_action")
    if not isinstance(selected, dict) or not selected.get("action_id"):
        if plan.status == "cannot_solve":
            return plan
        return FixPlan(
            status="cannot_solve",
            reason="No offline matrix action or untried Build Plan alternative is available for this failure.",
            hypothesis=plan.hypothesis,
            fix_direction=plan.fix_direction,
            repair_action=plan.repair_action,
            target=plan.target,
            evidence=plan.evidence,
            expected_effect=plan.expected_effect,
        )

    expected_action = str(selected.get("action_id") or "")
    expected_direction = str(selected.get("fix_direction") or "")
    expected_target = str(selected.get("intervention_target") or "")
    expected_gate = str(selected.get("evidence_gate") or "")
    expected_origin = str(selected.get("selection_tier") or "direct")
    supplied_action = str(plan.repair_action or "").strip()
    supplied_direction = str(plan.fix_direction or "").strip()
    if supplied_action and supplied_action != expected_action:
        return FixPlan(
            status="cannot_solve",
            reason=f"Plan selected repair_action={supplied_action!r}, expected {expected_action!r}.",
            hypothesis=plan.hypothesis,
            fix_direction=supplied_direction,
            repair_action=supplied_action,
            target=plan.target,
            evidence=plan.evidence,
            expected_effect=plan.expected_effect,
        )
    if supplied_direction and supplied_direction != expected_direction:
        return FixPlan(
            status="cannot_solve",
            reason=f"Plan selected fix_direction={supplied_direction!r}, expected {expected_direction!r}.",
            hypothesis=plan.hypothesis,
            fix_direction=supplied_direction,
            repair_action=expected_action,
            target=plan.target,
            evidence=plan.evidence,
            expected_effect=plan.expected_effect,
        )
    if plan.intervention_target and plan.intervention_target != expected_target:
        plan.status = "cannot_solve"
        plan.reason = (
            f"Plan selected intervention_target={plan.intervention_target!r}, "
            f"expected {expected_target!r}."
        )
        return plan
    if plan.evidence_gate and plan.evidence_gate != expected_gate:
        plan.status = "cannot_solve"
        plan.reason = f"Plan selected evidence_gate={plan.evidence_gate!r}, expected {expected_gate!r}."
        return plan
    if plan.selection_origin and plan.selection_origin != expected_origin:
        plan.status = "cannot_solve"
        plan.reason = (
            f"Plan selected selection_origin={plan.selection_origin!r}, "
            f"expected {expected_origin!r}."
        )
        return plan

    plan.fix_direction = expected_direction
    plan.repair_action = expected_action
    plan.intervention_target = expected_target
    plan.evidence_gate = expected_gate
    plan.selection_origin = expected_origin
    plan.matrix_action_id = str(selected.get("number") or "")
    try:
        plan.matrix_rank = int(selected.get("route_rank") or selected.get("matrix_rank") or 99)
    except (TypeError, ValueError):
        plan.matrix_rank = 99
    if not plan.target:
        plan.target = expected_target or diagnosis_route.invocation_role

    # These operation fields are typed controller decisions, not optional LLM
    # suggestions. Reject mixed shapes and clear their executable payload.
    invalid_reason = ""
    if expected_action == "switch_system_release":
        if not plan.sysuse:
            invalid_reason = "The selected system-release action has no target suite."
        elif plan.commands or plan.source_dependency:
            invalid_reason = "A system-release action must contain only a target suite."
    elif expected_action == "build_dependency_source":
        if not plan.source_dependency:
            invalid_reason = "The selected source-dependency action has no structured dependency plan."
        elif plan.commands or plan.sysuse:
            invalid_reason = "A source-dependency action must contain only a structured dependency plan."
    elif plan.sysuse:
        invalid_reason = "A sysuse switch was emitted for a non-system-release matrix action."
    elif plan.source_dependency:
        invalid_reason = "A source dependency was emitted for a non-source-realization matrix action."
    if invalid_reason:
        return FixPlan(
            status="cannot_solve",
            reason=invalid_reason,
            hypothesis=plan.hypothesis,
            fix_direction=expected_direction,
            repair_action=expected_action,
            target=plan.target,
            evidence=plan.evidence,
            expected_effect=plan.expected_effect,
            matrix_rank=plan.matrix_rank,
            matrix_action_id=plan.matrix_action_id,
            intervention_target=expected_target,
            evidence_gate=expected_gate,
            selection_origin=expected_origin,
        )
    return plan


class BuildAutomation:
    def __init__(self, compile_path: str, commit: str, sysuse: str, ablation_mode: str = "none"):
        self.compile_path = compile_path
        self.commit = commit
        self.initial_sysuse = sysuse
        self.sysuse = sysuse
        self.phase1_selected_sysuse = ""
        self.first_executed_sysuse = ""
        self.ablation_mode = ablation_mode
        self.agent_log_dir = compile_path.replace("/env_mapping", "/env_mapping/agent_logs") + f"/{commit}"
        self.docker_log_dir = os.path.join(self.agent_log_dir, "docker_commands")
        self.build_plan_path = os.path.join(self.agent_log_dir, "build_plan.json")
        self.last_build_log_path = ""
        self.reposummary = ""
        self.summary = ""
        self.script_content = ""
        self.errorlog = ""
        self.stdout = ""
        self.if_done = False
        self.timeout = False
        self.cannot_solve = 0
        self.no_plan = 0
        self.plan_error = ""
        self.plan_source = ""
        self.readrepo_attempts: List[dict[str, Any]] = []
        self.phase1_status = "not_started"
        self.phase1_errors: List[dict[str, str]] = []
        self.phase1_producers: Dict[str, dict[str, Any]] = {}
        self.phase1_extractor_attempts: List[dict[str, Any]] = []
        self.provider_error = 0
        self.no_command = 0
        self.error_summary = ["Init With No Error"]
        self.docs = ""
        self.urlsummaries = {}
        self.files = []
        self.phase2_project_info = ""
        self.build_plan_state: dict = {}
        self.latest_failure_frame: dict = {}
        self.latest_diagnosis_route: dict = {}
        self.attempt_history: List[dict] = []
        self.fix_history = []
        self.submodule_preflight: List[dict] = []
        self.dependency_selected_sysuse = ""
        os.makedirs(self.agent_log_dir, exist_ok=True)
        os.makedirs(self.docker_log_dir, exist_ok=True)

        self.shell_agent = None
        self.container_repo_root = DEFAULT_WORK_DIR
        self.host_install_path = os.path.join(self.compile_path, self.commit, "install")
        self.container_install_path = f"{DEFAULT_WORK_DIR}/{self.commit}/install"
        self.install_path = self.container_install_path
        self.build_repo_agent = None
        self._phase2_attempt_state: Dict[str, bool] = {"failed": False}
        self.error_summary_agent = None
        self._sysuse_switched_from = None
        # Source deps cloned by Phase 1 (from source_dep_resolver). Each entry
        # is a ClonedSourceDep dict that gets merged into `source_dep_plan`
        # for Phase 2 to consume.
        self.cloned_source_dep_plan: List[dict] = []
        # Extra docker volumes contributed by source deps:
        # list of (host_path, container_path) tuples.
        self.extra_volumes: List[tuple] = []
        # Card-driven repair metrics — surfaced in the final stats dump so we
        # can evaluate the ROI of the card path vs the LLM Phase 3 fallback.
        self.card_match_count = 0
        self.diagnosis_card_match_count = 0
        self.diagnosis_card_category_support_count = 0
        self.card_translate_count = 0
        self.card_succeeded = 0
        self.fallback_to_llm = 0
        self.max_card_candidates = 4
        # Tracks whether the fix applied in the previous loop iteration came
        # from the card-driven path. Used to credit `card_succeeded` when the
        # next Phase 2 build succeeds.
        self._last_fix_was_card_driven = False
        # Card IDs whose strategies were already tried and failed. Cleared on
        # sysuse switch (the new environment may legitimately retry them).
        self._tried_card_ids: set[str] = set()
        self._tried_card_keys: set[str] = set()
        self._exhausted_matrix_actions: Dict[str, set[str]] = {}
        self._exhausted_operation_fingerprints: Dict[str, set[str]] = {}
        self._provider_evidence_attempts: set[tuple[str, str, int, str, str]] = set()
        self._translation_retry_counts: Dict[str, int] = {}
        self._pending_typed_repair: Optional[Dict[str, Any]] = None
        self._matrix_empty_identity = ""
        self._consecutive_matrix_empty = 0
        self._incomplete_attempts = 0
        self._container_generation = 0
        self._phase2_continuation_counts: Dict[str, int] = {}
        self._phase2_continuation_note = ""

    async def initialize_docker(self, extra_volumes: Optional[List[tuple]] = None):
        """Start the docker container, optionally mounting extra volumes.

        Args:
            extra_volumes: Optional list of (host_path, container_path) tuples
                in addition to compile_path → /work. Used by source_dep_resolver
                to mount the clone root at /source_deps.
        """
        self.shell_agent = await init_docker_agent(
            self.sysuse, self.compile_path,
            log_dir=self.docker_log_dir,
            extra_volumes=extra_volumes,
        )
        if not self.first_executed_sysuse:
            self.first_executed_sysuse = normalize_sysuse(self.sysuse)
        self._container_generation += 1

    async def initialize_agents(self):
        """Initialize phase-specific agents after repository summary is available."""
        if self.shell_agent is None:
            await self.initialize_docker()
        self.phase2_project_info = self._build_phase2_project_info()
        if self.no_plan:
            return
        self.build_repo_agent = await self._create_phase2_agent()
        self.error_summary_agent = None

    async def _create_phase2_agent(self):
        args = (
            self.container_repo_root,
            self.sysuse,
            self._phase2_execution_prompt_info(),
            self.container_install_path,
            self.shell_agent,
        )
        try:
            return await ConstructScriptAgentFactory.create(
                *args,
                attempt_state=self._phase2_attempt_state,
            )
        except TypeError as exc:
            if "attempt_state" not in str(exc):
                raise
            # Keep test doubles and third-party wrappers written against the
            # pre-latch factory signature working.
            return await ConstructScriptAgentFactory.create(*args)

    def _phase2_execution_prompt_info(self) -> str:
        """Project the controller state down to the selected executable plan."""
        project_info = _extract_json_object(self.phase2_project_info or "") or {}
        apt_install_plan = [
            str(item) for item in project_info.get("apt_install_plan") or [] if str(item)
        ]
        setup_command_plan = [
            {
                "name": str(item.get("name") or ""),
                "commands": [str(command) for command in item.get("commands") or [] if str(command)],
            }
            for item in project_info.get("setup_command_plan") or []
            if isinstance(item, dict) and item.get("commands")
        ]
        source_dep_fields = (
            "name",
            "clone_status",
            "setup_commands",
            "fetch_commands",
            "container_source_path",
            "build_order",
            "env_exports",
            "install_prefix",
            "apt_packages",
        )
        source_dep_plan = []
        for item in project_info.get("source_dep_plan") or []:
            if not isinstance(item, dict) or not _source_dependency_has_executable_contract(item):
                continue
            source_dep_plan.append({
                key: deepcopy(item[key])
                for key in source_dep_fields
                if item.get(key) not in (None, "", [])
            })
        main_build_steps = []
        for item in project_info.get("main_build_steps") or []:
            if not isinstance(item, dict):
                continue
            command = str(item.get("command") or "")
            relative_workdir = str(item.get("workdir") or ".").strip()
            if not command:
                continue
            if relative_workdir in {"", ".", "./"}:
                workdir = self.container_repo_root
            elif relative_workdir.startswith("/"):
                workdir = relative_workdir
            else:
                workdir = f"{self.container_repo_root.rstrip('/')}/{relative_workdir.removeprefix('./')}"
            main_build_steps.append({
                "command": command,
                "workdir": workdir,
                "stage": str(item.get("stage") or ""),
            })
        if not main_build_steps:
            default_workdir = str(project_info.get("build_workdir") or self.container_repo_root)
            if not default_workdir.startswith("/"):
                default_workdir = (
                    self.container_repo_root
                    if default_workdir in {"", ".", "./"}
                    else f"{self.container_repo_root.rstrip('/')}/{default_workdir.removeprefix('./')}"
                )
            main_build_steps = [
                {"command": str(item), "workdir": default_workdir, "stage": ""}
                for item in project_info.get("main_build_plan") or []
                if str(item).strip()
            ]
        payload: dict[str, Any] = {"main_build_steps": main_build_steps}
        if apt_install_plan:
            payload["apt_install_plan"] = apt_install_plan
        if setup_command_plan:
            payload["setup_command_plan"] = setup_command_plan
        if source_dep_plan:
            payload["source_dep_plan"] = source_dep_plan
        if project_info.get("source_deps_root") not in (None, "", [], {}):
            payload["source_deps_root"] = deepcopy(project_info["source_deps_root"])
        return json.dumps(payload, ensure_ascii=False, indent=2)

    async def switch_sysuse(self, new_sysuse: str):
        if not new_sysuse or normalize_sysuse(new_sysuse) == normalize_sysuse(self.sysuse):
            return True
        inventory = self.build_plan_state.get("dependency_inventory") or {}
        selected_entry_id = str((self.build_plan_state.get("selected") or {}).get("build_entry_id") or "")
        binding = bind_required_packages(inventory, new_sysuse, selected_entry_id)
        if not binding["complete"]:
            self.build_plan_state.setdefault("environment_rejections", []).append({
                "sysuse": normalize_sysuse(new_sysuse),
                "reason": "switched_dependency_provider_unresolved",
                "unresolved_dependencies": binding["unresolved_dependencies"][:8],
            })
            self.build_plan_state["environment_rejections"] = self.build_plan_state["environment_rejections"][-16:]
            # Match initial binding semantics: a partial provider set is still
            # executable. The new suite may be required for a toolchain floor,
            # while the missing provider needs live Phase-2 evidence to choose
            # an alternative. Do not discard that environment before it runs.
        old_sysuse = self.sysuse
        if self.shell_agent is None:
            self.sysuse = new_sysuse
            try:
                await self.initialize_docker()
            except Exception:
                self.sysuse = old_sysuse
                raise
        else:
            await self.shell_agent.restart(new_sysuse, self.compile_path)
            self.sysuse = new_sysuse
            self._container_generation += 1
        bound_packages = binding["packages"]
        self.build_plan_state = rebind_environment(
            self.build_plan_state,
            sysuse=normalize_sysuse(self.sysuse),
            container_generation=self._container_generation,
            packages=bound_packages,
            repository_profile_hash=repository_profile_hash(),
        )
        self.build_plan_state.setdefault("selected", {})["unresolved_dependencies"] = list(
            binding["unresolved_dependencies"][:8]
        )
        project_info = _extract_json_object(self.phase2_project_info or "") or {}
        project_info["apt_packages"] = [item["name"] for item in bound_packages]
        project_info["apt_install_plan"] = [item["name"] for item in bound_packages]
        project_info["bound_package_providers"] = bound_packages
        project_info["unresolved_dependencies"] = list(binding["unresolved_dependencies"][:8])
        project_info["repository_profile_hash"] = repository_profile_hash()
        project_info["resolved_repository_profile_hash"] = resolved_repository_profile_hash(self.sysuse)
        project_info["repository_profile_verified"] = metadata_matches_profile(self.sysuse)
        self.phase2_project_info = json.dumps(project_info, ensure_ascii=False, indent=2)
        await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
        self._sysuse_switched_from = old_sysuse
        # New environment — previously-tried cards may legitimately apply
        # again (different toolchain / packages), so clear the dedup set.
        self._tried_card_ids.clear()
        self._tried_card_keys.clear()
        return True

    async def cleanup(self):
        if self.shell_agent is not None:
            await self.shell_agent.cleanup()

    def save_result(self, filename, content):
        try:
            os.makedirs(self.agent_log_dir, exist_ok=True)
            with open(os.path.join(self.agent_log_dir, filename), "w", encoding="utf-8") as f:
                f.write(str(content))
            return (True, "")
        except Exception as e:
            return (False, str(e))

    def _build_phase2_project_info(self) -> str:
        """Augment Phase 1 JSON with an execution-oriented view for Phase 2."""
        parsed = _extract_json_object(self.reposummary)
        if not parsed:
            self.no_plan = 1
            self.plan_error = "ReadRepo plan is unavailable to Phase 2."
            return ""

        project_info = dict(parsed)
        phase1_setup_commands = [
            deepcopy(item)
            for item in project_info.get("setup_commands") or []
            if isinstance(item, dict) and item.get("commands")
        ]
        unbound_source_deps = [
            deepcopy(item)
            for item in project_info.get("unbound_source_deps") or []
            if isinstance(item, dict)
        ]
        build_file_path = str(project_info.get("build_file_path") or "").strip()
        build_workdir = os.path.dirname(build_file_path) if build_file_path else ""
        build_workdir = build_workdir or "."

        raw_build_commands = [
            str(command).strip()
            for command in (project_info.get("build_commands") or [])
            if str(command).strip()
        ]
        readrepo_build_entry_candidates = [
            item for item in (project_info.get("build_entry_candidates") or [])
            if isinstance(item, dict)
        ]
        readrepo_command_candidates = [
            item for item in (project_info.get("command_candidates") or [])
            if isinstance(item, dict) and str(item.get("command") or "").strip()
        ]
        scanner_build_entry_candidates = []
        scanner_command_candidates = []
        repo_context_path = os.path.join(self.agent_log_dir, "repo_context.json")
        if os.path.isfile(repo_context_path):
            try:
                with open(repo_context_path, encoding="utf-8") as fh:
                    repo_context = json.load(fh)
                if isinstance(repo_context, dict):
                    if isinstance(repo_context.get("dependency_projects"), list):
                        project_info["dependency_projects"] = repo_context.get("dependency_projects")
                    if isinstance(repo_context.get("dependency_inventory"), dict):
                        project_info["dependency_inventory"] = repo_context.get("dependency_inventory")
                    scanner_build_entry_candidates.extend(
                        item for item in (repo_context.get("build_entry_candidates") or [])
                        if isinstance(item, dict)
                    )
                    scanner_command_candidates.extend(
                        item for item in (repo_context.get("command_candidates") or [])
                        if isinstance(item, dict) and str(item.get("command") or "").strip()
                    )
            except Exception:
                pass
        # Scanner evidence augments the ReadRepo catalog. It must not replace
        # the controller-selected entry after ReadRepo has made its choice.
        raw_build_entry_candidates = list(readrepo_build_entry_candidates)
        if not raw_build_entry_candidates:
            raw_build_entry_candidates.extend(scanner_build_entry_candidates)
        raw_command_candidates = list(readrepo_command_candidates)
        if not raw_command_candidates:
            raw_command_candidates.extend(scanner_command_candidates)
        grouped_commands = {
            str(command).strip()
            for item in raw_build_entry_candidates
            for command in (item.get("command_sequence") or [item.get("command") or ""])
            if str(command).strip()
        }
        for item in raw_command_candidates:
            command = str(item.get("command") or "").strip()
            if not command or command in grouped_commands:
                continue
            source = str(item.get("source") or "inferred")
            path = str(item.get("path") or "").strip()
            raw_build_entry_candidates.append({
                "entry_path": path if source in {"script_detected", "project_script"} else "",
                "evidence_paths": [path] if path else [],
                "source": source,
                "confidence": item.get("confidence") or "unknown",
                "family": item.get("family") or "",
                "command_sequence": [command],
            })
        if raw_build_commands and not raw_build_entry_candidates:
            raw_build_entry_candidates.append({
                "source": "readrepo_explicit",
                "confidence": "medium",
                "command_sequence": raw_build_commands,
            })
            grouped_commands.update(raw_build_commands)
        ranked_build_entries = rank_build_entry_candidates(
            raw_build_entry_candidates,
            repo_path=self.compile_path,
            target_commit=self.commit,
        )
        ranked_build_entries = [materialize_build_entry(item) for item in ranked_build_entries]
        current_selected_id = str((self.build_plan_state.get("selected") or {}).get("build_entry_id") or "").strip()
        requested_entry_id = current_selected_id or str(project_info.get("selected_build_entry_id") or "").strip()
        if not requested_entry_id:
            requested = project_info.get("selected_build_entry")
            if isinstance(requested, dict):
                requested_entry_id = str(requested.get("entry_id") or "").strip()
        if requested_entry_id:
            requested_entry = next(
                (
                    item for item in ranked_build_entries
                    if str(item.get("entry_id") or "") == requested_entry_id
                    and item.get("eligible", True)
                ),
                None,
            )
            if requested_entry is not None:
                for item in ranked_build_entries:
                    if item is not requested_entry and item.get("status") == "selected":
                        item["status"] = "candidate"
                requested_entry["status"] = "selected"
        selected_build_entry = next(
            (item for item in ranked_build_entries if item.get("status") == "selected"),
            None,
        )
        if ranked_build_entries and selected_build_entry is None:
            raw_build_commands = []
        if selected_build_entry:
            project_info["selected_build_entry"] = selected_build_entry
            project_info["build_entry_candidates"] = ranked_build_entries
            raw_build_commands = list(selected_build_entry.get("command_sequence") or raw_build_commands)
        build_commands = self._normalize_main_build_commands(raw_build_commands, build_workdir)
        main_build_steps: list[dict[str, Any]] = []
        command_candidates = []
        if ranked_build_entries:
            command_candidates = [
                {
                    "entry_id": item.get("entry_id") or "",
                    "command": (item.get("command_sequence") or [""])[0],
                    "command_sequence": list(item.get("command_sequence") or []),
                    "command_steps": deepcopy(item.get("command_steps") or []),
                    "source": item.get("source") or "unknown",
                    "confidence": item.get("confidence") or "unknown",
                    "path": item.get("entry_path") or "",
                    "evidence_paths": list(item.get("evidence_paths") or []),
                    "family": item.get("build_system") or "",
                    "scope": item.get("scope") or "unknown",
                    "root_evidence": list(item.get("root_evidence") or [])[:4],
                    "entry_preflight": deepcopy(item.get("entry_preflight") or {}),
                    "rank": item.get("rank"),
                }
                for item in ranked_build_entries
                if item.get("command_sequence") and item.get("eligible", True)
            ]
        else:
            seen_candidate_commands = set()
            for item in raw_command_candidates:
                normalized_command = self._normalize_main_build_commands([str(item.get("command") or "")], build_workdir)
                if not normalized_command:
                    continue
                key = normalized_command[0].strip().lower()
                if key in seen_candidate_commands:
                    continue
                seen_candidate_commands.add(key)
                candidate_workdir = str(item.get("workdir") or build_workdir)
                command_candidates.append({
                    "command": normalized_command[0],
                    "command_steps": [{
                        "command": normalized_command[0],
                        "workdir": candidate_workdir,
                    }],
                    "workdir": candidate_workdir,
                    "source": item.get("source") or "unknown",
                    "confidence": item.get("confidence") or "unknown",
                    "path": item.get("path") or "",
                    "family": item.get("family") or "",
                })
        if selected_build_entry and selected_build_entry.get("command_sequence"):
            build_commands = list(selected_build_entry.get("command_sequence") or [])
            main_build_steps = deepcopy(selected_build_entry.get("command_steps") or [])
            if main_build_steps:
                build_workdir = str(main_build_steps[0].get("workdir") or build_workdir)
        if not main_build_steps:
            main_build_steps = [
                {"command": command, "workdir": build_workdir}
                for command in build_commands
                if str(command).strip()
            ]
        apt_packages = [
            str(package).strip()
            for package in (project_info.get("apt_packages") or [])
            if str(package).strip()
        ]

        normalized_source_deps = []
        for raw_dep in project_info.get("source_deps") or []:
            normalized = _normalize_source_dependency_plan(raw_dep)
            if normalized is None:
                continue
            if _source_dependency_has_executable_contract(normalized):
                normalized_source_deps.append(normalized)
            else:
                unbound_source_deps.append({
                    **deepcopy(normalized),
                    "status": "unbound",
                    "reason": "source dependency has no explicit build command sequence",
                })

        detected_submodule_plan = _build_submodule_plan(self.compile_path, self.submodule_preflight)
        submodule_plan = detected_submodule_plan or list(project_info.get("submodule_plan") or [])
        existing_source_keys = {
            (
                str(item.get("name") or "").strip().lower(),
                str(item.get("submodule_path") or item.get("source_path") or "").strip(),
            )
            for item in normalized_source_deps
            if isinstance(item, dict)
        }
        for item in submodule_plan:
            if not isinstance(item, dict):
                continue
            source_dep = _source_dep_from_submodule(item, self.container_install_path)
            if not source_dep:
                continue
            if not _source_dependency_has_executable_contract(source_dep):
                unbound_source_deps.append({
                    **source_dep,
                    "status": "unbound",
                    "reason": "submodule has no explicit source build command sequence",
                })
                continue
            key = (source_dep["name"].lower(), source_dep["submodule_path"])
            if key not in existing_source_keys:
                normalized_source_deps.append(source_dep)
                existing_source_keys.add(key)

        # Merge Phase-1-cloned source deps (from source_dep_resolver) into the
        # project_info source_dep_plan, and accumulate their apt_packages into
        # apt_install_plan so Phase 2 installs everything in one shot.
        for cloned in self.cloned_source_dep_plan:
            if not isinstance(cloned, dict):
                continue
            if cloned.get("clone_status") != "cloned" or not _source_dependency_has_executable_contract(cloned):
                unbound_source_deps.append({
                    **deepcopy(cloned),
                    "status": "unbound",
                    "reason": "source dependency clone or build contract is not executable",
                })
                continue
            name = str(cloned.get("name") or "").strip()
            if not name:
                continue
            key = (name.lower(), str(cloned.get("source_path") or "").strip())
            if key in existing_source_keys:
                continue
            # Convert ClonedSourceDep → _normalize_source_dependency_plan-compatible
            entry = _normalize_source_dependency_plan({
                "name": name,
                "reason": _format_cloned_dep_reason(cloned),
                "apt_alternative": None,
                "source_path": cloned.get("source_path") or "",
                "submodule_path": None,
                "build_order": list(cloned.get("build_order") or []),
                "setup_commands": list(cloned.get("setup_commands") or []),
                "apt_packages": list(cloned.get("apt_packages") or []),
                "install_prefix": cloned.get("install_prefix") or self.container_install_path,
                "metadata_path": None,
                "instructions": cloned.get("instructions") or cloned.get("notes") or "",
                "try_apt_first": False,
            }) or {"name": name}
            entry.update({
                # Source-dep specific extras surfaced to Phase 2:
                "clone_status": cloned.get("clone_status") or "unknown",
                "host_source_path": cloned.get("host_source_path") or "",
                "selected_ref": cloned.get("selected_ref") or "",
                "depth": int(cloned.get("depth") or 0),
                "transitive_deps": list(cloned.get("transitive_deps") or []),
                "registry_notes": cloned.get("registry_notes") or "",
                "checkout_before_project_date": bool(cloned.get("checkout_before_project_date")),
                "checkout_branch": cloned.get("checkout_branch") or "",
            })
            normalized_source_deps.append(entry)
            existing_source_keys.add(key)
            # Accumulate dep's own apt_packages into the main apt_install_plan
            for pkg in (cloned.get("apt_packages") or []):
                pkg = str(pkg).strip()
                if pkg and pkg not in apt_packages:
                    apt_packages.append(pkg)
            # Also recurse into transitive_deps
            for tdep in (cloned.get("transitive_deps") or []):
                if not isinstance(tdep, dict):
                    continue
                if tdep.get("clone_status") != "cloned" or not _source_dependency_has_executable_contract(tdep):
                    unbound_source_deps.append({
                        **deepcopy(tdep),
                        "status": "unbound",
                        "reason": "transitive source dependency clone or build contract is not executable",
                    })
                    continue
                tname = str(tdep.get("name") or "").strip()
                if not tname:
                    continue
                tkey = (tname.lower(), str(tdep.get("source_path") or "").strip())
                if tkey in existing_source_keys:
                    continue
                t_entry = _normalize_source_dependency_plan({
                    "name": tname,
                    "reason": f"transitive source dep of {name}",
                    "apt_alternative": None,
                    "source_path": tdep.get("source_path") or "",
                    "submodule_path": None,
                    "build_order": list(tdep.get("build_order") or []),
                    "setup_commands": list(tdep.get("setup_commands") or []),
                    "apt_packages": list(tdep.get("apt_packages") or []),
                    "install_prefix": tdep.get("install_prefix") or self.container_install_path,
                    "metadata_path": None,
                    "instructions": tdep.get("instructions") or tdep.get("notes") or "",
                    "try_apt_first": False,
                }) or {"name": tname}
                t_entry.update({
                    "clone_status": tdep.get("clone_status") or "unknown",
                    "depth": int(tdep.get("depth") or 1),
                })
                normalized_source_deps.append(t_entry)
                existing_source_keys.add(tkey)
                for pkg in (tdep.get("apt_packages") or []):
                    pkg = str(pkg).strip()
                    if pkg and pkg not in apt_packages:
                        apt_packages.append(pkg)

        # Build transitive source dependencies before their parents.  The
        # resolver records depth explicitly, so a stable descending sort keeps
        # sibling discovery order while honoring dependency-first execution.
        normalized_source_deps.sort(
            key=lambda item: int(item.get("depth") or 0) if isinstance(item, dict) else 0,
            reverse=True,
        )

        toolchain_profile = project_info.get("toolchain_profile")
        if not isinstance(toolchain_profile, dict):
            toolchain_profile = _infer_toolchain_profile(self.compile_path, project_info)
        target_scope = project_info.get("target_scope")
        if not isinstance(target_scope, dict):
            target_scope = _infer_target_scope(project_info, toolchain_profile, self.compile_path)
        optional_feature_plan = project_info.get("optional_feature_plan")
        if not isinstance(optional_feature_plan, list):
            optional_feature_plan = _infer_optional_feature_plan(self.compile_path)

        submodule_execution_contract = _build_submodule_execution_contract(submodule_plan)
        project_info["build_workdir"] = build_workdir
        project_info["submodule_plan"] = submodule_plan
        project_info["submodules_required"] = any(
            isinstance(item, dict) and item.get("role") in {"main_integrated", "prebuild_required"}
            for item in submodule_plan
        )
        project_info["submodules_available"] = submodule_execution_contract["required_submodules_present"]
        project_info["submodule_execution_contract"] = submodule_execution_contract
        project_info["apt_install_plan"] = apt_packages
        project_info["setup_command_plan"] = phase1_setup_commands
        project_info["source_dep_plan"] = normalized_source_deps
        project_info["unbound_source_deps"] = unbound_source_deps
        project_info["main_build_plan"] = build_commands
        project_info["main_build_steps"] = main_build_steps
        project_info["command_candidates"] = command_candidates
        project_info["build_entry_candidates"] = ranked_build_entries
        if selected_build_entry:
            project_info["selected_build_entry"] = selected_build_entry
        project_info["toolchain_profile"] = toolchain_profile
        project_info["target_scope"] = target_scope
        project_info["repo_root"] = self.container_repo_root
        project_commit_date = _get_commit_date(self.compile_path, self.commit)
        project_info["project_commit_date"] = project_commit_date
        project_info["sysuse_candidate_plan"] = build_sysuse_candidate_plan(
            repo_path=self.compile_path,
            current_sysuse=self.sysuse,
            commit_date=project_commit_date,
            selected_distro=self.dependency_selected_sysuse,
        )
        candidate_releases = project_info.get("candidate_releases") or project_info.get("build_configuration_candidates") or []
        dependency_projects = project_info.get("dependency_projects") or []
        if dependency_projects and candidate_releases:
            configuration = select_build_configuration(
                dependency_projects,
                candidate_releases,
                build_entry_id=(selected_build_entry or {}).get("entry_id", ""),
                commit_date=project_commit_date,
            )
            project_info["build_configuration"] = configuration
            selected_configuration = configuration.get("selected") if isinstance(configuration, dict) else None
            if isinstance(selected_configuration, dict) and normalize_sysuse(selected_configuration.get("sysuse")) == normalize_sysuse(self.sysuse):
                project_info["selected_realization_methods"] = selected_configuration.get("selected_realization_methods") or {}
        self.build_plan_state = build_initial_plan(
            project_info,
            current_sysuse=self.sysuse,
            existing_plan=self.build_plan_state,
        )
        initial_binding = bind_required_packages(
            self.build_plan_state.get("dependency_inventory") or {},
            self.sysuse,
            str((self.build_plan_state.get("selected") or {}).get("build_entry_id") or ""),
        )
        # A partial provider binding is still executable: keep the providers
        # proven in this suite and leave unresolved groups for the normal
        # build/repair loop.  Treating one unavailable provider as ``no_plan``
        # discarded a valid entry before Phase 2 could produce the diagnostic
        # needed to select another provider or suite.
        if initial_binding["unresolved_dependencies"]:
            self.build_plan_state.setdefault("environment_rejections", []).append({
                "sysuse": normalize_sysuse(self.sysuse),
                "reason": "initial_dependency_provider_unresolved",
                "unresolved_dependencies": initial_binding["unresolved_dependencies"][:8],
            })
            self.build_plan_state["environment_rejections"] = self.build_plan_state[
                "environment_rejections"
            ][-16:]
        self.build_plan_state = rebind_environment(
            self.build_plan_state,
            sysuse=normalize_sysuse(self.sysuse),
            container_generation=self._container_generation,
            packages=initial_binding["packages"],
            repository_profile_hash=repository_profile_hash(),
        )
        self.build_plan_state["selected"]["repository_profile_hash"] = repository_profile_hash()
        self.build_plan_state["selected"]["resolved_repository_profile_hash"] = resolved_repository_profile_hash(self.sysuse)
        self.build_plan_state["selected"]["repository_profile_verified"] = metadata_matches_profile(self.sysuse)
        self.build_plan_state["selected"]["bound_package_providers"] = initial_binding["packages"]
        self.build_plan_state["selected"]["unresolved_dependencies"] = initial_binding["unresolved_dependencies"][:8]
        bound_candidates = [
            item for item in self.build_plan_state.get("system_dep_candidates") or []
            if isinstance(item, dict)
            and item.get("status") == "selected"
            and item.get("name")
        ]
        bound_packages = [
            {
                "name": str(item.get("name") or ""),
                "suite": normalize_sysuse(self.sysuse),
                "version": str(item.get("version") or ""),
                "source_repo": str(item.get("source_repo") or ""),
                "component": str(item.get("component") or ""),
                "dependency_id": str(item.get("dependency_id") or ""),
            }
            for item in bound_candidates
        ]
        project_info["apt_packages"] = [item["name"] for item in bound_packages]
        project_info["apt_install_plan"] = [item["name"] for item in bound_packages]
        project_info["bound_package_providers"] = bound_packages
        project_info["unresolved_source_packages"] = list(
            self.build_plan_state.get("unresolved_source_packages") or []
        )
        project_info["repository_profile_hash"] = repository_profile_hash()
        project_info["resolved_repository_profile_hash"] = resolved_repository_profile_hash(self.sysuse)
        project_info["repository_profile_verified"] = metadata_matches_profile(self.sysuse)
        try:
            save_build_plan(self.build_plan_state, self.build_plan_path)
        except OSError:
            pass
        project_info["build_plan"] = compact_build_plan_for_prompt(self.build_plan_state)
        # optional_feature_plan: only surface the disable flags (compact),
        # not the full plan entries (evidence/component bloat the prompt).
        if optional_feature_plan:
            project_info["optional_feature_plan"] = [
                item.get("disable_flag") for item in optional_feature_plan
                if isinstance(item, dict) and item.get("disable_flag")
            ]
        # build_options from build_hints.json (pre-extracted disable flags)
        hints_file = os.path.join(self.agent_log_dir, "build_hints.json")
        if os.path.isfile(hints_file):
            try:
                with open(hints_file) as f:
                    bh = json.load(f)
                if bh.get("disable_flags"):
                    project_info["build_options"] = [
                        d["flag"] if isinstance(d, dict) else str(d)
                        for d in bh["disable_flags"]
                    ]
                if bh.get("bundled_libraries") and "bundled_libraries" not in project_info:
                    project_info["bundled_libraries"] = [
                        d["name"] if isinstance(d, dict) else str(d)
                        for d in bh["bundled_libraries"]
                    ]
            except Exception:
                pass
        # Surface the source-dep mount root so Phase 2 knows where to find deps.
        if self.extra_volumes:
            for _host, container in self.extra_volumes:
                if container.endswith("/source_deps") or container == "/source_deps":
                    project_info["source_deps_root"] = container
                    break
        return json.dumps(project_info, indent=2, ensure_ascii=False)

    def _normalize_main_build_commands(self, commands: List[str], build_workdir: str) -> List[str]:
        """Keep Phase 1 command text intact; workdir travels as separate state."""
        del build_workdir
        return [str(command) for command in commands if str(command).strip()]

    def _recent_command_summary(self, limit: int = 8) -> str:
        if self.shell_agent is None:
            return "[]"
        records = self._build_command_records()[-limit:]
        compact = []
        for item in records:
            compact.append(
                {
                    "command": item.get("command"),
                    "status": item.get("status"),
                    "exit_code": item.get("exit_code"),
                    "log_path": item.get("log_path"),
                }
            )
        return json.dumps(compact, indent=2, ensure_ascii=False)

    def _detect_command_loop(self) -> Optional[str]:
        """Detect if the agent is stuck in a command repetition loop.

        Checks the last N build commands for repeated identical commands.
        Returns a warning string to inject into the Phase 2 prompt if a
        loop is detected, or None if no loop.
        """
        import hashlib
        records = self._build_command_records()
        if len(records) < 6:
            return None
        # Hash each command (normalized: strip whitespace, lowercase)
        recent = records[-12:]
        hashes = []
        for r in recent:
            cmd = str(r.get("command", "")).strip().lower()
            # Remove volatile parts (paths, version numbers) to catch
            # "same intent, different args" loops.
            cmd = re.sub(r"/tmp/\S+", "/tmp/X", cmd)
            cmd = re.sub(r"/work\S*", "/work", cmd)
            h = hashlib.md5(cmd.encode()).hexdigest()[:8]
            hashes.append(h)
        # Check if any single hash appears >=3 times in recent commands
        from collections import Counter
        counts = Counter(hashes)
        most_common_hash, most_count = counts.most_common(1)[0]
        if most_count >= 3:
            # Find the repeating command
            for r, h in zip(recent, hashes):
                if h == most_common_hash:
                    return (
                        f"The selected build command '{str(r.get('command',''))[:80]}...' has been "
                        f"executed {most_count} times without success. Do not execute it again; "
                        f"the controller will advance the repair queue."
                    )
        return None

    def _latest_diagnosis_route_object(self) -> Optional[DiagnosisRoute]:
        snapshot = self.latest_diagnosis_route
        if not isinstance(snapshot, dict) or not snapshot.get("repair_scope"):
            return None
        fields = DiagnosisRoute.__dataclass_fields__
        return DiagnosisRoute(**{
            name: deepcopy(snapshot[name])
            for name in fields
            if name in snapshot
        })

    async def _advance_repeated_command(self, loop_hint: str) -> str:
        """Advance the current repair queue instead of asking Phase 2 to repeat."""
        route = self._latest_diagnosis_route_object()
        failed_command = self._latest_failed_command_record()
        if route is None or failed_command is None:
            return "terminal"
        error_text = str(
            (self.latest_failure_frame or {}).get("primary_failure_text")
            or self._last_error_content("")
            or loop_hint
        )
        self.error_summary.append("Repeated Phase 2 command detected; controller is advancing the repair queue.")
        while (route.repair_scope or {}).get("selected_matrix_action"):
            result = await self._concretize_selected_matrix_action(
                route,
                error_text=error_text,
                failed_command=failed_command,
                recent_commands=self._recent_command_summary(),
                failed_phase=route.failed_phase or "unknown",
            )
            if result.get("status") == "provider_error":
                self._handle_provider_failure(str(result.get("raw") or ""), "Repair planner")
                return "provider_error"
            if result.get("status") == "applied":
                self.latest_diagnosis_route = route.to_dict()
                return "applied"
            advance = await self._advance_after_selected_action_failure(
                route,
                error_text=error_text,
                failed_command=failed_command,
            )
            if advance == "fallback":
                self.latest_diagnosis_route = route.to_dict()
                return "applied"
            if advance != "matrix_action":
                break
        self.latest_diagnosis_route = route.to_dict()
        return "terminal"

    def _phase2_bounded_alternatives(self, limit: int = 5) -> dict[str, Any]:
        """Small alternative set for a controller-authorized Phase 2 continuation."""
        plan = self.build_plan_state or {}

        def compact(items: Any, fields: tuple[str, ...], item_limit: int = limit) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("status") or "") in {"selected", "ineligible", "failed"}:
                    continue
                reduced = {
                    field_name: deepcopy(item[field_name])
                    for field_name in fields
                    if item.get(field_name) not in (None, "", [], {})
                }
                if reduced:
                    result.append(reduced)
                if len(result) >= item_limit:
                    break
            return result

        alternatives = plan.get("alternatives") or {}
        return {
            "build_entries": compact(
                alternatives.get("build_entries") or plan.get("entrypoint_candidates"),
                ("entry_id", "path", "source", "confidence", "scope", "command_sequence", "command_steps", "workdir"),
            ),
            "sysuses": compact(
                alternatives.get("sysuses") or plan.get("sysuse_candidates"),
                ("candidate_id", "sysuse", "source", "reason", "evidence"),
            ),
            "system_deps": compact(
                plan.get("system_dep_candidates"),
                ("candidate_id", "name", "kind", "status", "provider_evidence_id", "entity"),
            ),
            "source_deps": compact(
                plan.get("source_dep_candidates"),
                ("candidate_id", "source_dep_id", "name", "status", "container_source_path", "build_order", "env_exports"),
            ),
            "generators": compact(
                plan.get("generator_candidates"),
                ("generator_id", "candidate_id", "command", "source", "status", "evidence"),
            ),
            "targets": compact(
                plan.get("target_candidates"),
                ("target_id", "candidate_id", "name", "entry_id", "source", "evidence"),
            ),
        }

    async def _request_phase2_bounded_continuation(
        self,
        route: Optional[DiagnosisRoute],
        *,
        error_text: str,
        failed_command: Optional[dict],
        reason: str,
    ) -> bool:
        """Allow Phase 2 at most two controller-bounded continuations per failure."""
        failure_identity = (
            self._failure_scope_key((route or DiagnosisRoute()).failure_identity)
            if route is not None
            else ""
        )
        if not failure_identity:
            failure_identity = _stable_signature({
                "reason": reason,
                "failed_command": (failed_command or {}).get("command", ""),
            })
        attempts = self._phase2_continuation_counts.get(failure_identity, 0)
        if attempts >= 2:
            return False

        self._phase2_continuation_counts[failure_identity] = attempts + 1
        self._phase2_continuation_note = (
            f"selected controller repairs are exhausted ({reason}). "
            "Use only supplied current-run evidence and the allowed alternatives. "
            "Choose one materially different executable action, or stop with Error occurred. "
            "Do not repeat the failed command, invent package names, invent suites, or invent URLs."
        )
        self._record_fix_attempt(
            error_text=error_text,
            solution_kind="phase2_bounded_continuation",
            solution_detail=self._phase2_continuation_note,
            result="retry",
            failed_command=failed_command,
            hypothesis=f"phase2_continuation_{reason}",
        )
        self.build_plan_state = set_pending_fix_plan(self.build_plan_state, [])
        await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
        return True

    @staticmethod
    def _phase2_command_makes_progress(command: str) -> bool:
        return bool(_PHASE2_PROGRESS_COMMAND_RE.search(str(command or "")))

    @staticmethod
    def _phase2_command_is_low_progress(command: str) -> bool:
        return bool(_PHASE2_LOW_PROGRESS_COMMAND_RE.search(str(command or "")))

    def _phase2_stall_reason(self, attempt_records: list[dict[str, Any]]) -> str:
        commands = [
            str(item.get("command") or "").strip()
            for item in attempt_records or []
            if isinstance(item, dict) and str(item.get("command") or "").strip()
        ]
        if len(commands) < 8:
            return ""
        if any(self._phase2_command_makes_progress(command) for command in commands):
            return ""
        low_progress_count = sum(self._phase2_command_is_low_progress(command) for command in commands)
        if low_progress_count >= 6:
            return f"phase2 executed {low_progress_count} inspection/probe commands without setup or build progress"
        return ""

    @staticmethod
    def _command_timeout_is_terminal(record: dict[str, Any]) -> bool:
        command = str((record or {}).get("command") or "")
        return not bool(_PHASE2_NONTERMINAL_TIMEOUT_RE.search(command))

    def _failure_prompt_context(
        self,
        route: Optional[DiagnosisRoute],
        error_text: str,
        failed_command: Optional[dict],
    ) -> dict[str, Any]:
        frame = self.latest_failure_frame or {}
        evidence = _diagnosis_evidence_bundle(frame)
        if not evidence and str(error_text or "").strip():
            evidence = [{"id": "E1", "source": "failed_output", "text": str(error_text)[-1200:]}]
        payload: dict[str, Any] = {
            "failure_identity": str(((route or DiagnosisRoute()).failure_identity or {}).get("identity_key") or ""),
            "failed_command": str((failed_command or {}).get("command") or ""),
            "evidence": evidence,
        }
        if route is not None:
            payload.update({
                "observed_phase": route.failed_phase,
                "error_type": route.canonical_error_type,
                "entity": deepcopy(route.entity or {}),
            })
        return payload

    def _current_selection_prompt_context(self) -> dict[str, Any]:
        selected = self.build_plan_state.get("selected") or {}
        return {
            "sysuse": normalize_sysuse(self.sysuse),
            **{
                key: deepcopy(selected[key])
                for key in ("build_entry_id", "generator_id", "target_id")
                if selected.get(key)
            },
        }

    def _repair_prompt_context(
        self,
        route: DiagnosisRoute,
        error_text: str,
        failed_command: Optional[dict],
        selected_action: dict[str, Any],
        candidate_catalog: dict[str, Any],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "failure": self._failure_prompt_context(route, error_text, failed_command),
            "selected_action": _compact_selected_action(selected_action),
            "candidate_catalog": self._repair_prompt_candidate_catalog(candidate_catalog),
        }
        execution = self._build_card_translation_context(
            failed_command,
            selected_action,
        ).get("current_execution")
        if execution:
            context["current_execution"] = execution
        return context

    def _alternative_prompt_context(
        self,
        route: DiagnosisRoute,
        error_text: str,
        failed_command: Optional[dict],
        alternatives: List[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "failure": self._failure_prompt_context(route, error_text, failed_command),
            "current_selection": self._current_selection_prompt_context(),
            "alternatives": alternatives,
        }

    def _emergency_prompt_context(
        self,
        error_text: str,
        failed_command: Optional[dict],
        route: Optional[DiagnosisRoute] = None,
    ) -> dict[str, Any]:
        return {
            "failure": self._failure_prompt_context(route, error_text, failed_command),
            "current_selection": self._current_selection_prompt_context(),
            "untried_alternatives": self._available_build_plan_alternatives(route)[:8],
            "prior_repairs": [
                {
                    key: item.get(key)
                    for key in ("solution_kind", "result", "action_signature")
                    if item.get(key) not in (None, "")
                }
                for item in self.fix_history[-6:]
                if isinstance(item, dict)
            ],
        }

    def _build_command_records(self) -> List[dict]:
        if self.shell_agent is None:
            return []
        return [
            item
            for item in self.shell_agent.get_record()
            if item.get("origin") != "repair_diagnostic"
            and str(item.get("status") or "").lower() not in {"control", "skipped", "blocked"}
        ]

    def _latest_build_command_record(self) -> Optional[dict]:
        records = self._build_command_records()
        return records[-1] if records else None

    def _card_hypothesis_already_failed(self, card_id: str) -> bool:
        """Check whether this card was already tried and failed.

        Two sources of truth:
          1. `_tried_card_ids` set — populated when Phase 2 exhausted a
             strategy list without success (precise, O(1) lookup).
          2. `fix_history` — for hypotheses recorded before the
             `_tried_card_ids` mechanism was added.
        """
        if card_id in self._tried_card_ids:
            return True
        target = f"card_{card_id}"
        for h in self.fix_history:
            if target in str(h.get("hypothesis") or "") and h.get("result") in ("failed", "final"):
                return True
        return False

    def _card_failure_key(self, card_id: str, failure_identity: str = "") -> str:
        return f"{failure_identity}:{card_id}" if failure_identity else str(card_id or "")

    @staticmethod
    def _failure_scope_key(identity: Any) -> str:
        """Scope retry state to semantic failure, environment, and occurrence."""
        if isinstance(identity, dict):
            parts = (
                identity.get("identity_key"),
                identity.get("context_id"),
                identity.get("occurrence_signature"),
            )
            return "|".join(str(part or "") for part in parts).strip("|")
        return str(identity or "")

    def _card_hypothesis_failed_for(self, card_id: str, failure_identity: str = "") -> bool:
        if failure_identity:
            return self._card_failure_key(card_id, failure_identity) in self._tried_card_keys
        return self._card_hypothesis_already_failed(card_id)

    def _build_card_translation_context(
        self,
        failed_command: Optional[dict],
        selected_matrix_action: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Expose only context needed to concretize the selected card action."""
        project_info = _extract_json_object(self.phase2_project_info) or {}
        selected_action = _compact_selected_action(selected_matrix_action)
        operation_kinds = set(selected_action.get("operation_schemas") or {})
        context: dict[str, Any] = {
            "selected_action": selected_action,
            "failed_command": str((failed_command or {}).get("command") or ""),
        }
        if operation_kinds & {"build_plan_patch", "build_file_patch", "tool_wrapper"}:
            context["current_execution"] = {
                "build_system": project_info.get("build_system") or "unknown",
                "build_workdir": project_info.get("build_workdir") or self.container_repo_root,
                "main_build_plan": list(project_info.get("main_build_plan") or [])[:8],
                "build_options": list(project_info.get("build_options") or [])[:8],
                "repair_overrides": deepcopy(project_info.get("repair_overrides") or {}),
            }
        return context

    def _dependency_inventory_provider_matches(self, entity: dict[str, Any]) -> list[dict[str, Any]]:
        kind = str(entity.get("kind") or "")
        name = str(entity.get("name") or "").strip()
        if kind != "build_macro" or not re.fullmatch(r"(?:AX|AC|AM|LT|PKG)_[A-Za-z0-9_]+", name):
            return []
        macro = name.lower()
        macro_file = f"{macro}.m4"
        inventory = self.build_plan_state.get("dependency_inventory") or {}
        rows = inventory.get("dependencies") if isinstance(inventory, dict) else []
        matches: list[dict[str, Any]] = []
        seen_packages: set[str] = set()
        for dependency in rows or []:
            if not isinstance(dependency, dict):
                continue
            evidence_items = [
                item for item in dependency.get("evidence") or []
                if isinstance(item, dict)
            ]
            searchable = " ".join(
                [
                    str((dependency.get("entity") or {}).get("name") or ""),
                    str(dependency.get("dependency_id") or ""),
                    *[
                        f"{item.get('path') or ''} {item.get('excerpt') or ''}"
                        for item in evidence_items
                    ],
                ]
            ).lower()
            if macro not in searchable and macro_file not in searchable:
                continue
            evidence_path = next(
                (
                    str(item.get("path") or item.get("id") or "").strip()
                    for item in evidence_items
                    if any(
                        needle in f"{item.get('path') or ''} {item.get('excerpt') or ''}".lower()
                        for needle in (macro, macro_file)
                    )
                ),
                str(dependency.get("dependency_id") or "dependency_inventory"),
            )
            rank = 0 if macro_file in searchable else 1
            for provider in dependency.get("providers") or []:
                if not isinstance(provider, dict) or provider.get("kind") != "apt":
                    continue
                package = str(provider.get("name") or "").strip()
                package_key = package.lower()
                if not package or package_key in seen_packages:
                    continue
                seen_packages.add(package_key)
                matches.append({
                    "path": evidence_path,
                    "package": package,
                    "query": name,
                    "sysuse": normalize_sysuse(self.sysuse),
                    "architecture": "amd64",
                    "source": "phase1_dependency_inventory",
                    "source_repo": str(provider.get("source_repo") or ""),
                    "match_quality": "phase1_macro_provider_evidence",
                    "match_rank": rank,
                })
        return matches[:20]

    def _provider_probe(self, entity: dict[str, Any]) -> tuple[str, list[str]]:
        kind = str(entity.get("kind") or "")
        name = str(entity.get("name") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_+./:-]+", name):
            return "", []
        quoted = shlex.quote(name)
        if not name:
            return "", []
        if kind == "build_tool":
            lower_name = name.lower()
            lower_quoted = shlex.quote(lower_name)
            commands = [f"command -v -- {quoted} >/dev/null 2>&1"]
            if lower_name != name:
                commands.append(f"command -v -- {lower_quoted} >/dev/null 2>&1")
            return " || ".join(commands), [f"usr/bin/{lower_name}", f"bin/{lower_name}"]
        if kind == "build_macro":
            if not re.fullmatch(r"(?:AX|AC|AM|LT|PKG)_[A-Za-z0-9_]+", name):
                return "", []
            macro_file = f"{name.lower()}.m4"
            macro_pattern = shlex.quote(r"(^|[^A-Za-z0-9_])" + re.escape(name) + r"([^A-Za-z0-9_]|$)")
            macro_file_quoted = shlex.quote(macro_file)
            roots = "/usr/share/aclocal /usr/share/aclocal-* /usr/share/gnulib/m4 /usr/share/*/m4"
            command = (
                f"find {roots} -type f -iname {macro_file_quoted} -print -quit 2>/dev/null | grep -q . "
                f"|| grep -R -E {macro_pattern} {roots} >/dev/null 2>&1"
            )
            return command, [f"usr/share/aclocal/{macro_file}", macro_file, name]
        if kind == "header":
            command = f"find /usr/include -type f -path '*{quoted}' -print -quit 2>/dev/null | grep -q ."
            return command, [name]
        if kind == "pkg_config_module":
            return f"pkg-config --exists {quoted}", [f"{name}.pc"]
        if kind == "cmake_package":
            command = f"find /usr -type f \\( -name '{name}Config.cmake' -o -name '{name.lower()}-config.cmake' \\) -print -quit 2>/dev/null | grep -q ."
            return command, [f"{name}Config.cmake", f"{name.lower()}-config.cmake"]
        if kind == "library":
            library = name[3:] if name.startswith("lib") else name
            return f"ldconfig -p 2>/dev/null | grep -Eq 'lib{re.escape(library)}\\.so'", [f"lib{library}.so"]
        return "", []

    async def _populate_provider_evidence(
        self,
        route: DiagnosisRoute,
        *,
        include_conditional: bool = False,
    ) -> None:
        scope = route.repair_scope or {}
        actions = [
            item for item in scope.get("matrix_actions") or []
            if isinstance(item, dict)
        ]
        if not actions and isinstance(scope.get("selected_matrix_action"), dict):
            actions = [scope["selected_matrix_action"]]
        if include_conditional:
            actions.extend(
                item for item in scope.get("conditional_actions") or []
                if isinstance(item, dict)
            )
        if not any(
            "system_package_selection" in (action.get("operation_kinds") or [])
            for action in actions
        ):
            return
        failure_identity = str((route.failure_identity or {}).get("identity_key") or "")
        entity = route.entity or {}
        version_relation = dict(route.version_relation or {})
        if version_relation and not entity.get("name"):
            entity = {"kind": "build_tool", "name": version_relation.get("subject") or route.failed_tool}
        probe_command, queries = self._provider_probe(entity)
        if not failure_identity or not probe_command or not queries or self.shell_agent is None:
            return
        attempt_key = (
            failure_identity,
            normalize_sysuse(self.sysuse),
            self._container_generation,
            str(entity.get("kind") or ""),
            str(entity.get("name") or "").lower(),
        )
        if attempt_key in self._provider_evidence_attempts:
            return
        self._provider_evidence_attempts.add(attempt_key)
        try:
            probe = getattr(self.shell_agent, "probe_command", None)
            if not callable(probe):
                logging.getLogger("agent_build").warning("Provider precondition probe unavailable; failing closed")
                return
            pre_result = await probe(probe_command)
        except Exception as exc:
            logging.getLogger("agent_build").warning("Provider precondition probe failed closed: %s", exc)
            return
        exit_code = pre_result.get("exit_code")
        observed_unsatisfied = pre_result.get("status") != "success" or exit_code not in {0, "0"}
        effective_probe_command = probe_command
        observed_version = ""
        if version_relation:
            tool_name = str(entity.get("name") or "")
            version_command = f"{shlex.quote(tool_name)} --version"
            version_result = await probe(version_command)
            version_text = str(version_result.get("stdout") or version_result.get("output") or "")
            version_match = re.search(r"[0-9]+(?:\.[0-9A-Za-z+~:-]+)+", version_text)
            observed_version = version_match.group(0) if version_match else ""
            observed_unsatisfied = not (
                version_result.get("status") == "success"
                and version_result.get("exit_code") in {0, "0"}
                and version_satisfies(
                    observed_version,
                    str(version_relation.get("required_operator") or ">="),
                    str(version_relation.get("required_version") or ""),
                )
            )
            effective_probe_command = version_command
            exit_code = version_result.get("exit_code")
        if not observed_unsatisfied:
            self.build_plan_state.setdefault("provider_probe_events", []).append({
                "failure_identity": failure_identity,
                "environment": {
                    "sysuse": normalize_sysuse(self.sysuse),
                    "container_generation": self._container_generation,
                },
                "entity": deepcopy(entity),
                "status": "satisfied_pending_rebuild",
                "command": effective_probe_command,
                "observed_version": observed_version,
            })
            self.build_plan_state["provider_probe_events"] = self.build_plan_state["provider_probe_events"][-16:]
            return
        matches: list[dict[str, Any]] = self._dependency_inventory_provider_matches(entity)
        for query in queries:
            try:
                result = await Search_PkgFile_Matches(query, self.sysuse)
            except Exception as exc:
                logging.getLogger("agent_build").warning("Provider file lookup failed closed: %s", exc)
                continue
            if result.get("status") == "success":
                matches.extend(item for item in result.get("matches") or [] if isinstance(item, dict))
        if matches:
            ranked_matches = [
                (
                    int(item["match_rank"])
                    if str(item.get("match_rank", "")).isdigit()
                    else 99,
                    item,
                )
                for item in matches
            ]
            best_rank = min(rank for rank, _ in ranked_matches)
            matches = [
                item for rank, item in ranked_matches
                if rank == best_rank
            ]
        packages = list(dict.fromkeys(str(item.get("package") or "").strip() for item in matches if item.get("package")))
        try:
            verification = (
                await Get_False_Pkgname(packages, self.sysuse)
                if packages else {"status": "success", "invalid": []}
            )
        except Exception as exc:
            logging.getLogger("agent_build").warning("Provider availability check failed closed: %s", exc)
            verification = {"status": "error", "invalid": packages}
        invalid = {
            str(item).lower() for item in verification.get("invalid") or []
        } if verification.get("status") == "success" else {item.lower() for item in packages}
        seen = {
            str(item.get("evidence_id") or "")
            for item in self.build_plan_state.get("provider_evidence") or []
            if isinstance(item, dict)
        }
        current_environment = {
            "sysuse": normalize_sysuse(self.sysuse),
            "architecture": "amd64",
            "container_generation": self._container_generation,
        }
        existing_packages = {
            str((item.get("provider") or {}).get("package") or "").lower()
            for item in self.build_plan_state.get("provider_evidence") or []
            if isinstance(item, dict)
            and item.get("failure_identity") == failure_identity
            and (item.get("environment") or {}) == current_environment
            and item.get("available")
            and (item.get("availability_probe") or {}).get("status") == "available"
        }
        retired_packages = {
            str(item).lower() for item in self.build_plan_state.get("retired_system_packages") or []
        }
        seen_packages: set[str] = set()
        availability_probes = 0
        for match in matches:
            package = str(match.get("package") or "").strip()
            package_key = package.lower()
            if (
                not package
                or package_key in invalid
                or package_key in seen_packages
                or package_key in existing_packages
                or package_key in retired_packages
            ):
                continue
            seen_packages.add(package_key)
            if availability_probes >= 8:
                break
            availability_probes += 1
            availability_command = f"apt-cache show --no-all-versions {shlex.quote(package)} >/dev/null 2>&1"
            try:
                probe = getattr(self.shell_agent, "probe_command", None)
                if not callable(probe):
                    logging.getLogger("agent_build").warning("Package availability probe unavailable; failing closed")
                    continue
                availability_result = await probe(availability_command)
            except Exception as exc:
                logging.getLogger("agent_build").warning("Container package availability probe failed closed: %s", exc)
                continue
            availability_exit = availability_result.get("exit_code")
            if availability_result.get("status") != "success" or availability_exit not in {0, "0"}:
                continue
            version_fact = best_satisfying_package_version(
                package,
                self.sysuse,
                str(version_relation.get("required_operator") or ""),
                str(version_relation.get("required_version") or ""),
            )
            if version_relation.get("required_version") and not version_fact:
                continue
            evidence = make_provider_evidence(
                failure_identity=failure_identity,
                entity_kind=str(entity.get("kind") or ""),
                entity_name=str(entity.get("name") or ""),
                sysuse=normalize_sysuse(self.sysuse),
                architecture="amd64",
                container_generation=self._container_generation,
                package=package,
                provider_artifact=str(match.get("path") or ""),
                provider_source=str(match.get("source") or "clean_pkgpath"),
                available=True,
                precondition_probe={
                    "command": effective_probe_command,
                    "exit_code": exit_code,
                    "observed_missing": not bool(version_relation),
                    "observed_unsatisfied": bool(version_relation),
                    "observed_version": observed_version,
                },
                availability_probe={
                    "command": availability_command,
                    "exit_code": availability_exit,
                    "status": "available",
                    "candidate_version": str(version_fact.get("version") or ""),
                    "source_repo": str(version_fact.get("source_repo") or match.get("source") or ""),
                    "repository_profile_hash": repository_profile_hash(),
                },
                postcondition_probe={
                    "command": (
                        f"{shlex.quote(str(entity.get('name') or ''))} --version"
                        if version_relation and entity.get("name")
                        else probe_command
                    ),
                    "status": "pending",
                },
                version_relation=version_relation,
            ).to_dict()
            if evidence["evidence_id"] in seen:
                continue
            seen.add(evidence["evidence_id"])
            self.build_plan_state = register_provider_evidence(self.build_plan_state, evidence)
            self.build_plan_state.setdefault("system_dep_candidates", []).append({
                "candidate_id": f"package:{hashlib.sha256(package.encode()).hexdigest()[:12]}",
                "name": package,
                "kind": "apt_package_provider",
                "status": "candidate",
                "source": "provider_evidence",
                "provider_evidence_id": evidence["evidence_id"],
                "entity": deepcopy(entity),
            })

    def _repair_candidate_catalog(
        self,
        route: Optional[DiagnosisRoute] = None,
        operation_kinds: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        identity = str(((route or DiagnosisRoute()).failure_identity or {}).get("identity_key") or "")
        if operation_kinds is None:
            selected_action = ((route or DiagnosisRoute()).repair_scope or {}).get("selected_matrix_action") or {}
            operation_kinds = list(selected_action.get("operation_kinds") or [])
        kinds = {str(item) for item in operation_kinds or [] if item}
        catalog: dict[str, Any] = {
            "route_evidence_refs": [
                str(item.get("id") or "")
                for item in (self.latest_failure_frame or {}).get("evidence_bundle") or []
                if isinstance(item, dict) and item.get("id")
            ][:8],
            "exhausted_operation_fingerprints": sorted(
                self._exhausted_operation_fingerprints.get(identity, set())
            ),
            "execution_context": self._repair_execution_context(self._latest_failed_command_record()),
        }
        if not kinds:
            return catalog
        if "system_package_selection" in kinds:
            catalog["current_environment"] = {
                "sysuse": normalize_sysuse(self.sysuse),
                "architecture": "amd64",
                "container_generation": self._container_generation,
            }
            catalog["provider_evidence"] = [
                deepcopy(item) for item in self.build_plan_state.get("provider_evidence") or []
                if isinstance(item, dict)
                and item.get("failure_identity") == identity
                and (item.get("environment") or {}).get("sysuse") == normalize_sysuse(self.sysuse)
                and (item.get("environment") or {}).get("architecture") == "amd64"
                and (item.get("environment") or {}).get("container_generation") == self._container_generation
            ][-8:]
        if "source_dependency_selection" in kinds:
            source_candidates = []
            for item in self.build_plan_state.get("source_dep_candidates") or []:
                if not isinstance(item, dict) or item.get("status") in {"selected", "failed", "exhausted", "ineligible"}:
                    continue
                verification = self._source_dependency_verification(item)
                if not verification:
                    continue
                candidate = deepcopy(item)
                candidate["verification"] = verification
                source_candidates.append(candidate)
            catalog["source_dep_candidates"] = source_candidates[:8]
        if "generator_selection" in kinds:
            verification = self._generator_verification()
            catalog["generator_candidates"] = [
                {**deepcopy(item), "verification": verification}
                for item in self.build_plan_state.get("generator_candidates") or []
                if isinstance(item, dict)
                and item.get("status") not in {"selected", "failed", "exhausted", "ineligible"}
                and verification
            ][:8]
        if "build_entry_selection" in kinds:
            catalog["entrypoint_candidates"] = [
                deepcopy(item) for item in self.build_plan_state.get("entrypoint_candidates") or []
                if isinstance(item, dict) and item.get("eligible", True)
                and item.get("status") not in {"selected", "failed", "exhausted", "ineligible"}
            ][:8]
        if "build_target_selection" in kinds:
            catalog["target_candidates"] = [
                deepcopy(item) for item in self.build_plan_state.get("target_candidates") or []
                if isinstance(item, dict) and item.get("status") not in {"selected", "failed", "exhausted", "ineligible"}
            ][:8]
        if "system_release_selection" in kinds:
            release_candidates = [
                deepcopy(item) for item in self.build_plan_state.get("sysuse_candidates") or []
                if isinstance(item, dict) and item.get("status") not in {"current", "failed", "exhausted", "ineligible"}
            ]
            relation = dict((route or DiagnosisRoute()).version_relation or {})
            subject = str(relation.get("subject") or "").lower()
            if relation.get("required_version") and subject:
                package_aliases = {
                    "cmake": "cmake",
                    "meson": "meson",
                    "ninja": "ninja-build",
                    "pkg-config": "pkg-config",
                    "pkgconfig": "pkg-config",
                    "autoconf": "autoconf",
                    "automake": "automake",
                }
                package = package_aliases.get(subject, subject)
                release_candidates = [
                    item for item in release_candidates
                    if best_satisfying_package_version(
                        package,
                        str(item.get("sysuse") or ""),
                        str(relation.get("required_operator") or ">="),
                        str(relation.get("required_version") or ""),
                    )
                ]
            catalog["sysuse_candidates"] = release_candidates[:8]
        return catalog

    @staticmethod
    def _safe_container_probe_path(value: Any) -> str:
        path = str(value or "").strip().replace("\\", "/")
        if not path:
            return ""
        if not path.startswith("/"):
            path = f"{DEFAULT_WORK_DIR}/{path.lstrip('./')}"
        path = os.path.normpath(path)
        if path in {DEFAULT_WORK_DIR, SOURCE_DEP_ROOT}:
            return ""
        if path.startswith(DEFAULT_WORK_DIR + "/") or path.startswith(SOURCE_DEP_ROOT + "/"):
            return path
        return ""

    def _generator_verification(self) -> dict[str, Any]:
        """Provide a controller-checkable generator outcome for this failure."""
        for raw_path in (self.latest_failure_frame or {}).get("missing_paths") or []:
            path = self._safe_container_probe_path(raw_path)
            if path:
                return {
                    "kind": "generated_file",
                    "path": path,
                    "command": f"test -e {shlex.quote(path)}",
                }
        for name in ("configure.ac", "configure.in"):
            if (
                os.path.isfile(os.path.join(self.compile_path, name))
                and not os.path.isfile(os.path.join(self.compile_path, "configure"))
            ):
                path = f"{DEFAULT_WORK_DIR}/configure"
                return {
                    "kind": "generated_file",
                    "path": path,
                    "command": f"test -e {shlex.quote(path)}",
                }
        return {}

    def _source_dependency_verification(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Require a cloned external source or an in-repository build path."""
        path = self._safe_container_probe_path(
            candidate.get("container_source_path")
            or candidate.get("source_path")
            or candidate.get("submodule_path")
        )
        if not path or not _source_dependency_has_executable_contract(candidate):
            return {}
        external_source = path.startswith(SOURCE_DEP_ROOT + "/")
        clone_status = str(candidate.get("clone_status") or "")
        if external_source and clone_status != "cloned":
            return {}
        return {
            "kind": "source_dependency_path",
            "path": path,
            "command": f"test -d {shlex.quote(path)}",
            "clone_status": clone_status or "in_repository",
            "checkout": str(candidate.get("selected_ref") or candidate.get("revision") or ""),
        }

    @staticmethod
    def _repair_prompt_candidate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
        """Remove controller-only candidate metadata before an LLM call."""
        fields = {
            "provider_evidence": (
                "evidence_id", "failure_identity", "entity", "environment", "provider",
                "available", "precondition_probe", "availability_probe", "postcondition_probe",
            ),
            "source_dep_candidates": (
                "candidate_id", "source_dep_id", "name", "source", "status",
                "container_source_path", "source_path", "submodule_path", "build_order", "env_exports",
            ),
            "generator_candidates": (
                "generator_id", "candidate_id", "command", "source", "status", "evidence",
            ),
            "entrypoint_candidates": (
                "entry_id", "candidate_id", "path", "entry_path", "build_system",
                "command_sequence", "source", "confidence", "scope", "root_evidence",
                "entry_preflight", "evidence_paths",
            ),
            "target_candidates": (
                "target_id", "candidate_id", "name", "entry_id", "source", "evidence",
            ),
            "sysuse_candidates": (
                "candidate_id", "sysuse", "source", "reason", "evidence",
            ),
        }
        result = {
            key: deepcopy(catalog[key])
            for key in ("current_environment", "route_evidence_refs", "exhausted_operation_fingerprints")
            if catalog.get(key) not in (None, "", [], {})
        }
        for key, allowed_fields in fields.items():
            if key not in catalog:
                continue
            result[key] = [
                {
                    field_name: deepcopy(item[field_name])
                    for field_name in allowed_fields
                    if item.get(field_name) not in (None, "", [], {})
                }
                for item in catalog.get(key) or []
                if isinstance(item, dict)
            ]
        return result

    async def _apply_typed_repair(self, decision: RepairDecision) -> bool:
        identity = decision.failure_identity
        if decision.fingerprint in self._exhausted_operation_fingerprints.get(identity, set()):
            return False
        operation = decision.operation
        self.build_plan_state = record_repair_decision(
            self.build_plan_state,
            decision.to_dict(),
            state="proposed",
        )
        if operation.get("kind") == "system_release_selection":
            requested_sysuse = str(operation.get("sysuse") or "")
            if not requested_sysuse or normalize_sysuse(requested_sysuse) == normalize_sysuse(self.sysuse):
                self.build_plan_state = record_repair_decision(
                    self.build_plan_state, decision.to_dict(), state="apply_failed",
                )
                return False
        applied_patch = False
        if operation.get("kind") == "build_file_patch":
            applied_patch = self._apply_build_file_patch(operation, reverse=False)
            if not applied_patch:
                self.build_plan_state = record_repair_decision(
                    self.build_plan_state, decision.to_dict(), state="apply_failed",
                )
                return False
        if operation.get("kind") == "system_release_selection":
            new_sysuse = str(operation.get("sysuse") or "")
            if not await self.switch_sysuse(new_sysuse):
                self.build_plan_state = record_repair_decision(
                    self.build_plan_state, decision.to_dict(), state="apply_failed",
                )
                return False
        self.build_plan_state = apply_repair_operation_to_plan(self.build_plan_state, operation)
        self.build_plan_state = record_repair_decision(
            self.build_plan_state,
            decision.to_dict(),
            state="applied_pending_rebuild",
        )
        self._pending_typed_repair = {
            "decision": decision.to_dict(),
            "build_file_patch_applied": applied_patch,
            "record_boundary": len(self._build_command_records()),
            "container_generation": self._container_generation,
            "origin_failure_frame": deepcopy(self.latest_failure_frame),
        }
        if operation.get("kind") != "system_release_selection":
            await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
        return True

    def _pending_repair_records(self, pending: dict[str, Any]) -> list[dict[str, Any]]:
        records = self._build_command_records()
        if int(pending.get("container_generation") or -1) != self._container_generation:
            return records
        boundary = max(0, int(pending.get("record_boundary") or 0))
        return records[boundary:]

    def _is_pipeline_record(self, record: dict[str, Any]) -> bool:
        command = re.sub(r"\s+", " ", str(record.get("command") or "").strip())
        if not command or str(record.get("status") or "") in {"control", "skipped", "blocked"}:
            return False
        selected_commands = {
            re.sub(r"\s+", " ", str(item).strip())
            for item in (self.build_plan_state.get("selected") or {}).get("command_sequence") or []
            if str(item).strip()
        }
        if command in selected_commands:
            return True
        return bool(re.search(
            r"(?:^|\s)(?:\./configure|autoreconf|cmake|meson|make|gmake|ninja|scons|"
            r"(?:\./)?waf|xmake|qmake|rake|gradle|gradlew|gn|gyp|gcc|g\+\+|clang|clang\+\+|cc|c\+\+)(?:\s|$)",
            command,
            re.I,
        ))

    def _repair_effect_observed(
        self,
        operation: dict[str, Any],
        records: list[dict[str, Any]],
        postcondition_satisfied: Optional[bool],
    ) -> bool:
        kind = str(operation.get("kind") or "")
        commands = [str(item.get("command") or "") for item in records]
        if kind == "system_package_selection":
            package = str(operation.get("package") or "")
            applied = False
            for command, record in zip(commands, records):
                try:
                    tokens = shlex.split(command)
                except ValueError:
                    tokens = []
                if (
                    re.search(r"\bapt(?:-get)?\s+install\b", command)
                    and package in tokens
                    and str(record.get("exit_code")) == "0"
                ):
                    applied = True
                    break
            return applied and postcondition_satisfied is not False
        if kind == "generator_selection":
            generator_id = str(operation.get("generator_id") or "")
            selected_candidate = next((
                item for item in self.build_plan_state.get("generator_candidates") or []
                if isinstance(item, dict) and generator_id in {
                    str(item.get("generator_id") or ""),
                    str(item.get("candidate_id") or ""),
                }
            ), {})
            selected = str(selected_candidate.get("command") or "")
            applied = any(
                command.strip() == selected.strip() and str(record.get("exit_code")) == "0"
                for command, record in zip(commands, records)
            )
            return applied and postcondition_satisfied is True
        if kind == "source_dependency_selection":
            source_id = str(operation.get("source_dependency_id") or "")
            candidate = next((
                item for item in self.build_plan_state.get("source_dep_candidates") or []
                if isinstance(item, dict) and source_id in {
                    str(item.get("source_dep_id") or ""),
                    str(item.get("candidate_id") or ""),
                    str(item.get("name") or ""),
                }
            ), {})
            expected = {
                str(item).strip() for item in candidate.get("build_order") or [] if str(item).strip()
            }
            applied = bool(expected) and any(command.strip() in expected for command in commands)
            return applied and postcondition_satisfied is True
        if kind == "build_plan_patch":
            changes = operation.get("changes") or {}
            command_text = "\n".join(commands)
            checks: list[bool] = []
            for field_name in ("configure_args", "compile_flags", "link_flags"):
                values = [str(item) for item in changes.get(field_name) or [] if str(item)]
                if values:
                    checks.append(all(value in command_text for value in values))
            environment = changes.get("environment") or {}
            if environment:
                checks.append(all(
                    re.search(rf"(?:^|\s)(?:export\s+)?{re.escape(str(name))}=", command_text, re.M)
                    for name in environment
                ))
            if changes.get("workdir"):
                requested = str(changes["workdir"]).strip().replace("\\", "/")
                requested = requested if requested.startswith("/") else f"{DEFAULT_WORK_DIR}/{requested.removeprefix('./')}"
                checks.append(any(
                    os.path.normpath(str(record.get("workdir") or record.get("cwd") or ""))
                    == os.path.normpath(requested)
                    for record in records
                ))
            return bool(checks) and all(checks)
        if kind == "build_file_patch":
            return bool(records)
        if kind == "build_entry_selection":
            return str((self.build_plan_state.get("selected") or {}).get("build_entry_id") or "") == str(operation.get("entry_id") or "")
        if kind == "build_target_selection":
            return str((self.build_plan_state.get("selected") or {}).get("target_id") or "") == str(operation.get("target_id") or "")
        if kind == "system_release_selection":
            return normalize_sysuse(self.sysuse) == normalize_sysuse(operation.get("sysuse") or "")
        return False

    def _apply_build_file_patch(self, operation: dict[str, Any], *, reverse: bool) -> bool:
        path = str(operation.get("path") or "")
        patch_text = str(operation.get("patch") or "")
        repo_root = os.path.realpath(self.compile_path)
        target_path = os.path.join(repo_root, path)
        if os.path.islink(target_path) or not os.path.isfile(target_path):
            return False
        if os.path.commonpath([repo_root, os.path.realpath(target_path)]) != repo_root:
            return False
        touched = {
            line[4:].split("\t", 1)[0].removeprefix("a/").removeprefix("b/")
            for line in patch_text.splitlines()
            if line.startswith(("--- ", "+++ ")) and line[4:].split("\t", 1)[0] != "/dev/null"
        }
        if touched != {path}:
            return False
        args = ["patch", "--batch", "--silent", "-p1"]
        if reverse:
            args.insert(1, "--reverse")
        result = _subprocess.run(
            args,
            cwd=self.compile_path,
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=20,
        )
        return result.returncode == 0

    async def _settle_pending_typed_repair(
        self,
        next_failure_identity: str = "",
        *,
        build_succeeded: bool = False,
    ) -> None:
        pending = self._pending_typed_repair
        if not pending:
            return
        decision = pending.get("decision") or {}
        identity = str(decision.get("failure_identity") or "")
        same_failure = bool(next_failure_identity and next_failure_identity == identity)
        operation = decision.get("operation") or {}
        postcondition_failed = False
        postcondition_satisfied: Optional[bool] = None
        if operation.get("kind") == "system_package_selection":
            evidence_id = str(operation.get("provider_evidence_id") or "")
            evidence = next(
                (item for item in self.build_plan_state.get("provider_evidence") or [] if item.get("evidence_id") == evidence_id),
                None,
            )
            command = str(((evidence or {}).get("postcondition_probe") or {}).get("command") or "")
            if command and self.shell_agent is not None:
                probe = getattr(self.shell_agent, "probe_command", None)
                if not callable(probe):
                    postcondition_failed = True
                    result = {"status": "failed", "exit_code": None}
                else:
                    result = await probe(command)
                postcondition_satisfied = (
                    result.get("status") == "success"
                    and result.get("exit_code") in {0, "0"}
                )
                relation = dict((evidence or {}).get("version_relation") or {})
                if postcondition_satisfied and relation.get("required_version"):
                    version_match = re.search(
                        r"[0-9]+(?:\.[0-9A-Za-z+~:-]+)+",
                        str(result.get("stdout") or result.get("output") or ""),
                    )
                    observed = version_match.group(0) if version_match else ""
                    postcondition_satisfied = version_satisfies(
                        observed,
                        str(relation.get("required_operator") or ">="),
                        str(relation.get("required_version") or ""),
                    )
                    (evidence or {}).setdefault("postcondition_probe", {})["observed_version"] = observed
                postcondition_failed = not build_succeeded and not postcondition_satisfied
                (evidence or {}).setdefault("postcondition_probe", {}).update({
                    "status": "satisfied" if postcondition_satisfied else "failed",
                    "exit_code": result.get("exit_code"),
                })
                if postcondition_failed:
                    outcome = "failed_postcondition"
                    for candidate in self.build_plan_state.get("system_dep_candidates") or []:
                        if candidate.get("provider_evidence_id") == evidence_id:
                            candidate["status"] = "failed"
        elif operation.get("kind") in {"generator_selection", "source_dependency_selection"}:
            verification = operation.get("verification") or {}
            command = str(verification.get("command") or "")
            if command and self.shell_agent is not None:
                probe = getattr(self.shell_agent, "probe_command", None)
                result = await probe(command) if callable(probe) else {"status": "failed", "exit_code": None}
                postcondition_satisfied = (
                    result.get("status") == "success"
                    and result.get("exit_code") in {0, "0"}
                )
            else:
                result = {"status": "failed", "exit_code": None}
                postcondition_satisfied = False
            postcondition_failed = not build_succeeded and not postcondition_satisfied
            if postcondition_failed:
                outcome = "failed_postcondition"
                candidate_key = (
                    "generator_id" if operation.get("kind") == "generator_selection"
                    else "source_dependency_id"
                )
                candidate_id = str(operation.get(candidate_key) or "")
                candidate_list = (
                    self.build_plan_state.get("generator_candidates")
                    if operation.get("kind") == "generator_selection"
                    else self.build_plan_state.get("source_dep_candidates")
                ) or []
                for candidate in candidate_list:
                    if not isinstance(candidate, dict):
                        continue
                    candidate_ids = {
                        str(candidate.get("generator_id") or candidate.get("candidate_id") or ""),
                        str(candidate.get("source_dep_id") or candidate.get("candidate_id") or candidate.get("name") or ""),
                    }
                    if candidate_id in candidate_ids:
                        candidate["status"] = "failed"
        records = self._pending_repair_records(pending)
        causal_build = any(self._is_pipeline_record(record) for record in records)
        effect_observed = self._repair_effect_observed(operation, records, postcondition_satisfied)
        if postcondition_failed:
            outcome = "failed_postcondition"
        elif not causal_build:
            outcome = "pending_rebuild"
        elif build_succeeded:
            outcome = "succeeded" if effect_observed else "inconclusive"
        elif same_failure:
            outcome = "failed_after_rebuild"
        elif effect_observed:
            outcome = "progressed"
        else:
            outcome = "inconclusive"
        if outcome in {"failed_after_rebuild", "failed_postcondition"}:
            fingerprint = str(decision.get("fingerprint") or "")
            if fingerprint:
                self._exhausted_operation_fingerprints.setdefault(identity, set()).add(fingerprint)
            if pending.get("build_file_patch_applied"):
                self._apply_build_file_patch(operation, reverse=True)
        self.build_plan_state = record_repair_outcome(
            self.build_plan_state,
            fingerprint=str(decision.get("fingerprint") or ""),
            failure_identity=identity,
            outcome=outcome,
            next_failure_identity=next_failure_identity,
            operation=operation,
            details={
                "causal_build": causal_build,
                "effect_observed": effect_observed,
                "postcondition_satisfied": postcondition_satisfied,
                "record_count": len(records),
            },
        )
        self.build_plan_state = record_repair_decision(
            self.build_plan_state,
            decision,
            state=outcome,
        )
        if outcome != "pending_rebuild":
            self._pending_typed_repair = None

    async def _abandon_pending_typed_repair(self, reason: str) -> None:
        """Settle an applied repair when execution exits before another diagnosis."""
        if self.if_done:
            return
        pending = self._pending_typed_repair
        if not pending:
            return
        decision = pending.get("decision") or {}
        operation = decision.get("operation") or {}
        records = self._pending_repair_records(pending)
        if pending.get("build_file_patch_applied") and records:
            self._apply_build_file_patch(operation, reverse=True)
        outcome = reason if records else "pending_rebuild"
        self.build_plan_state = record_repair_outcome(
            self.build_plan_state,
            fingerprint=str(decision.get("fingerprint") or ""),
            failure_identity=str(decision.get("failure_identity") or ""),
            outcome=outcome,
            operation=operation,
            details={"reason": reason, "record_count": len(records)},
        )
        self.build_plan_state = record_repair_decision(
            self.build_plan_state,
            decision,
            state=outcome,
        )
        if records:
            self._pending_typed_repair = None

    async def _try_card_driven_repair(
        self,
        error_lines: str,
        failed_command: Optional[dict],
        recent_commands: str,
        failed_phase: str = "unknown",
        diagnosis_route: Optional[DiagnosisRoute] = None,
    ) -> Optional[Dict[str, Any]]:
        """Translate the best matching card into one validated typed decision."""
        line_list = [l.strip() for l in (error_lines or "").splitlines() if l.strip()]
        if not line_list:
            return None

        phase_for_kb = failed_phase if failed_phase in {"setup", "bootstrap", "configure", "compile", "link", "unknown"} else "unknown"
        if diagnosis_route is None:
            return None
        await self._populate_provider_evidence(diagnosis_route)
        route_matrix_actions = [
            item
            for item in (diagnosis_route.repair_scope or {}).get("matrix_actions") or []
            if isinstance(item, dict) and item.get("action_id")
        ]
        if not route_matrix_actions:
            return None
        route_action_ids = [
            str(item.get("action_id") or "")
            for item in route_matrix_actions
            if isinstance(item, dict) and item.get("action_id")
        ]
        candidates = _policy_kb().select(
            error_lines=line_list,
            sysuse=self.sysuse,
            failed_phase=phase_for_kb,
            symptom_family=(diagnosis_route.symptom_family if diagnosis_route else None),
            candidate_decisions=(diagnosis_route.candidate_decisions if diagnosis_route else None),
            allowed_revision_types=(
                (diagnosis_route.repair_scope or {}).get("allowed_revision_types")
                if diagnosis_route
                else None
            ),
            matrix_actions=route_action_ids,
            matrix_action_specs=route_matrix_actions,
            provenance_probe=(
                (diagnosis_route.repair_scope or {}).get("provenance_probe")
                if diagnosis_route
                else None
            ),
            top_k=15,
        )
        if not candidates:
            return None

        route_by_action = {
            str(item.get("action_id") or ""): item
            for item in route_matrix_actions
            if isinstance(item, dict) and item.get("action_id")
        }
        for card in candidates:
            supported = [
                route_by_action.get(str(item.get("matrix_action") or ""))
                for item in card.get("matrix_actions") or []
                if isinstance(item, dict)
            ]
            supported = [item for item in supported if item]
            supported.sort(key=lambda item: int(item.get("route_rank") or item.get("matrix_rank") or 99))
            card["selected_matrix_action"] = supported[0] if supported else {}
        candidates.sort(
            key=lambda card: (
                int((card.get("selected_matrix_action") or {}).get("route_rank") or
                    (card.get("selected_matrix_action") or {}).get("matrix_rank") or 99),
                int(card.get("effective_cost", card.get("cost", 99))),
            )
        )

        self.card_match_count += len(candidates)

        # Filter out cards already tried and failed
        failure_identity = self._failure_scope_key(diagnosis_route.failure_identity)
        fresh = [
            c for c in candidates
            if not self._card_hypothesis_failed_for(c.get("id") or "", failure_identity)
        ]
        if not fresh:
            return None
        fresh = fresh[: self.max_card_candidates]
        # Translation failure is local to a card, not to the selected matrix
        # action. Try the next matching card before invoking the generic
        # concretizer for the same action.
        for card in fresh:
            cid = card.get("id") or "?"
            selected_matrix_action = card.get("selected_matrix_action") or {}
            ctx = self._build_card_translation_context(failed_command, selected_matrix_action)
            ctx["failure"] = self._failure_prompt_context(
                diagnosis_route,
                error_lines,
                failed_command,
            )
            ctx["selected_card_actions"] = [
                {
                    "type": item.get("card_action_type") or item.get("type") or "",
                    "hint": item.get("hint") or "",
                }
                for item in card.get("matrix_actions") or []
                if isinstance(item, dict)
                and item.get("matrix_action") == selected_matrix_action.get("action_id")
                and item.get("executable", True)
            ]
            if not ctx["selected_card_actions"]:
                continue
            candidate_catalog = self._repair_candidate_catalog(
                diagnosis_route,
                list(selected_matrix_action.get("operation_kinds") or []),
            )
            ctx["candidate_catalog"] = self._repair_prompt_candidate_catalog(candidate_catalog)
            ctx["failure_identity"] = str((diagnosis_route.failure_identity or {}).get("identity_key") or "")
            try:
                raw = await translate_card_to_fix_plan(card, ctx)
                if _looks_like_provider_error(raw):
                    return {"status": "provider_error", "raw": raw}
                decision = parse_repair_decision(
                    raw,
                    selected_action=selected_matrix_action,
                    candidate_catalog=candidate_catalog,
                    failure_identity=ctx["failure_identity"],
                    selection_origin="policy_card",
                    card_id=cid,
                    failure_context={
                        **(diagnosis_route.failure_identity or {}),
                        "failure_mechanism": diagnosis_route.failure_mechanism,
                        "version_relation": diagnosis_route.version_relation,
                    },
                )
            except Exception as exc:
                decision = RepairDecision(status="cannot_solve", reason=f"translation error: {exc}")

            if decision.status == "cannot_solve":
                self._tried_card_keys.add(self._card_failure_key(cid, failure_identity))
                self._record_fix_attempt(
                    error_text=error_lines,
                    solution_kind="card_translation_cannot_apply",
                    solution_detail=f"card {cid}: {decision.reason}",
                    result="failed",
                    failed_command=failed_command,
                    hypothesis=f"card_{cid}",
                )
                continue

            self.card_translate_count += 1
            self.error_summary.append(
                f"[card-driven] {cid} (cost={card.get('cost', '?')}): {card.get('summary', '')[:120]}"
            )
            self.build_plan_state = set_pending_fix_plan(self.build_plan_state, [decision.to_dict()])
            return {"repair_decision": decision, "card_id": cid}
        self.build_plan_state = set_pending_fix_plan(self.build_plan_state, [])
        return None

    def _fix_plan_action_signature(
        self,
        plan: FixPlan,
        failed_command: Optional[dict],
    ) -> str:
        selected = self.build_plan_state.get("selected") or {}
        identity = self.latest_diagnosis_route.get("failure_identity") or {}
        source_env = [
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "container_source_path": str(item.get("container_source_path") or item.get("source_path") or ""),
                "env_exports": list(item.get("env_exports") or []),
            }
            for item in self.build_plan_state.get("source_dep_candidates") or []
            if isinstance(item, dict) and item.get("status") == "selected"
        ]
        payload: Dict[str, Any] = {
            "current_sysuse": normalize_sysuse(self.sysuse),
            "build_entry_id": selected.get("build_entry_id") or "",
            "command_sequence": [
                re.sub(r"\s+", " ", str(command).strip())
                for command in selected.get("command_sequence") or []
            ],
            "realization_methods": selected.get("realization_methods") or {},
            "status": plan.status,
            "repair_action": plan.repair_action or "",
            "fix_direction": plan.fix_direction or "",
            "intervention_target": plan.intervention_target or "",
            "evidence_gate": plan.evidence_gate or "",
            "selection_origin": plan.selection_origin or "",
            "failure_identity": str(identity.get("identity_key") or ""),
            "failed_command": re.sub(r"\s+", " ", str((failed_command or {}).get("command") or "").strip()),
            "environment": {
                "repair_overrides": self.build_plan_state.get("repair_overrides") or {},
                "source_dependencies": source_env,
            },
        }
        if plan.sysuse and not plan.commands:
            payload.update({"kind": "sysuse", "target_sysuse": normalize_sysuse(plan.sysuse)})
        elif plan.commands:
            payload.update({
                "kind": "commands",
                "commands": [re.sub(r"\s+", " ", command.strip()) for command in plan.commands],
            })
        elif plan.source_dependency:
            normalized_dep = _normalize_source_dependency_plan(plan.source_dependency) or plan.source_dependency
            payload.update({
                "kind": "source_dependency",
                "name": str(normalized_dep.get("name") or "").lower(),
                "source_path": normalized_dep.get("container_source_path") or normalized_dep.get("source_path") or "",
                "build_order": normalized_dep.get("build_order") or [],
            })
        else:
            payload.update({"kind": "terminal", "hypothesis": plan.hypothesis or ""})
        return _stable_signature(payload)

    def _repair_execution_context(self, failed_command: Optional[dict]) -> dict[str, Any]:
        """Return controller-owned state that determines a repair's command effect."""
        selected = self.build_plan_state.get("selected") or {}
        source_env = [
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "container_source_path": str(item.get("container_source_path") or item.get("source_path") or ""),
                "env_exports": list(item.get("env_exports") or []),
            }
            for item in self.build_plan_state.get("source_dep_candidates") or []
            if isinstance(item, dict) and item.get("status") == "selected"
        ]
        return {
            "sysuse": normalize_sysuse(self.sysuse),
            "build_entry_id": str(selected.get("build_entry_id") or ""),
            "command_sequence": [
                re.sub(r"\s+", " ", str(command).strip())
                for command in selected.get("command_sequence") or []
            ],
            "failed_command": re.sub(r"\s+", " ", str((failed_command or {}).get("command") or "").strip()),
            "repair_overrides": deepcopy(self.build_plan_state.get("repair_overrides") or {}),
            "source_dependencies": source_env,
        }

    def _fallback_action_signature(
        self,
        solution_kind: str,
        solution_detail: str,
        failed_command: Optional[dict],
    ) -> str:
        return _stable_signature({
            "current_sysuse": normalize_sysuse(self.sysuse),
            "kind": solution_kind,
            "solution": solution_detail or "",
            "failed_command": (failed_command or {}).get("command", ""),
        })

    def _has_prior_equivalent_fix(
        self,
        plan: FixPlan,
        failed_command: Optional[dict],
    ) -> bool:
        signature = self._fix_plan_action_signature(plan, failed_command)
        current_sysuse = normalize_sysuse(self.sysuse)
        for item in self.fix_history[:-1]:
            if item.get("result") not in {"proposed", "failed", "final"}:
                continue
            if item.get("sysuse") != current_sysuse:
                continue
            if item.get("action_signature") == signature:
                return True
        return False

    def _record_fix_attempt(
        self,
        error_text: str,
        solution_kind: str,
        solution_detail: str,
        result: str,
        failure: str = "",
        classification_category: Optional[str] = None,
        classification_summary: str = "",
        failed_command: Optional[dict] = None,
        hypothesis: str = "",
        action_signature: str = "",
        repair_action_override: str = "",
        fix_direction_override: str = "",
    ) -> None:
        current_sysuse = normalize_sysuse(self.sysuse)
        route_scope = self.latest_diagnosis_route.get("repair_scope") or {}
        selected_matrix_action = (
            route_scope.get("selected_matrix_action")
            or route_scope.get("selected_fallback_action")
            or {}
        )
        matrix_key = str(route_scope.get("matrix_key") or "")
        identity_payload = self.latest_diagnosis_route.get("failure_identity") or {}
        failure_identity = str(identity_payload.get("identity_key") or "")
        failure_scope = self._failure_scope_key(identity_payload)
        repair_action = str(
            repair_action_override
            or selected_matrix_action.get("action_id")
            or self.latest_diagnosis_route.get("repair_action")
            or ""
        )
        fix_direction = str(fix_direction_override or selected_matrix_action.get("fix_direction") or "")
        intervention_target = str(selected_matrix_action.get("intervention_target") or "")
        evidence_gate = str(selected_matrix_action.get("evidence_gate") or "")
        selection_origin = str(selected_matrix_action.get("selection_tier") or "")
        action_signature = action_signature or self._fallback_action_signature(
            solution_kind,
            solution_detail,
            failed_command,
        )
        self.fix_history.append(
            {
                "error": (error_text or "").strip()[:400],
                "failed_command": failed_command or {},
                "sysuse": current_sysuse,
                "classification_category": classification_category,
                "classification_summary": (classification_summary or "").strip()[:400],
                "diagnosis_route": self.latest_diagnosis_route or {},
                "solution_kind": solution_kind,
                "solution": (solution_detail or "").strip()[:800],
                "result": result,
                "failure": (failure or "").strip()[:400],
                "hypothesis": hypothesis or "",
                "action_signature": action_signature,
                "matrix_key": matrix_key,
                "failure_identity": failure_identity,
                "failure_context_id": str(identity_payload.get("context_id") or ""),
                "occurrence_signature": str(identity_payload.get("occurrence_signature") or ""),
                "failure_scope": failure_scope,
                "repair_action": repair_action,
                "fix_direction": fix_direction,
                "intervention_target": intervention_target,
                "evidence_gate": evidence_gate,
                "selection_origin": selection_origin,
            }
        )
        self.attempt_history.append(
            make_attempt_history_record(
                solution_kind=solution_kind,
                result=result,
                hypothesis=hypothesis,
                failed_command=failed_command,
                sysuse=current_sysuse,
                action_signature=action_signature,
            )
        )
        self.attempt_history = self.attempt_history[-12:]
        self.build_plan_state = record_build_attempt(
            self.build_plan_state,
            {
                "kind": solution_kind,
                "result": result,
                "hypothesis": hypothesis or "",
                "command": (failed_command or {}).get("command", ""),
                "sysuse": current_sysuse,
                "action_signature": action_signature,
                "failure": (failure or "").strip()[:400],
                "matrix_key": matrix_key,
                "failure_identity": failure_identity,
                "failure_context_id": str(identity_payload.get("context_id") or ""),
                "occurrence_signature": str(identity_payload.get("occurrence_signature") or ""),
                "failure_scope": failure_scope,
                "repair_action": repair_action,
                "fix_direction": fix_direction,
                "intervention_target": intervention_target,
                "evidence_gate": evidence_gate,
                "selection_origin": selection_origin,
            },
        )
        try:
            save_build_plan(self.build_plan_state, self.build_plan_path)
        except OSError:
            pass

    def _format_all_command_records(self) -> str:
        if self.shell_agent is None:
            return ""
        chunks = []
        for item in self._build_command_records():
            chunks.append(
                "\n".join(
                    [
                        f"Command: {item.get('command')}",
                        f"Status: {item.get('status')}",
                        f"Exit code: {item.get('exit_code')}",
                        f"Full log: {item.get('log_path')}",
                        "Stdout:",
                        str(item.get("stdout", "")),
                        "Stderr:",
                        str(item.get("stderr", "")),
                        "Excerpt:",
                        str(item.get("output", "")),
                    ]
                )
            )
        return "\n\n".join(chunks)

    def write_run_result(self, *, elapsed_sec: Optional[float] = None) -> str:
        """Persist one compact machine-readable outcome for batch reconciliation."""
        if self.if_done:
            status = "done"
        elif self.provider_error:
            status = "provider_error"
        elif self.no_command:
            status = "no_command"
        elif self.timeout:
            status = "timeout"
        elif self.no_plan:
            status = "no_plan"
        elif self.cannot_solve:
            status = "cannot_solve"
        else:
            status = "error"
        records = self._build_command_records()
        payload = {
            "schema_version": 2,
            "status": status,
            "agent_mode": os.environ.get(_AGENT_MODE_ENV, _DEFAULT_AGENT_MODE),
            "plan_error": self.plan_error,
            "plan_source": self.plan_source,
            "readrepo_attempts": self.readrepo_attempts,
            "phase1": {
                "status": self.phase1_status,
                "errors": self.phase1_errors,
                "producers": self.phase1_producers,
                "extractor_attempts": self.phase1_extractor_attempts,
            },
            "initial_sysuse": normalize_sysuse(self.initial_sysuse),
            "phase1_selected_sysuse": normalize_sysuse(self.phase1_selected_sysuse),
            "first_executed_sysuse": normalize_sysuse(self.first_executed_sysuse),
            "final_sysuse": normalize_sysuse(self.sysuse),
            "elapsed_sec": round(elapsed_sec, 2) if elapsed_sec is not None else None,
            "attempts": {
                "phase2_builds": sum(
                    1 for item in self.build_plan_state.get("attempts") or []
                    if isinstance(item, dict) and item.get("kind") == "build"
                ),
                "commands": len(records),
                "repairs": len(self.fix_history),
            },
            "failure_frame": compact_failure_frame_for_prompt(self.latest_failure_frame) if self.latest_failure_frame else {},
            "diagnosis_route": self.latest_diagnosis_route or {},
            "metrics": {
                "card_match_count": self.card_match_count,
                "diagnosis_card_match_count": self.diagnosis_card_match_count,
                "diagnosis_card_category_support_count": self.diagnosis_card_category_support_count,
                "card_translate_count": self.card_translate_count,
                "card_succeeded": self.card_succeeded,
                "fallback_to_llm": self.fallback_to_llm,
            },
            "artifacts": {
                "build_plan": self.build_plan_path,
                "docker_log_dir": self.docker_log_dir,
                "command_log": self.last_build_log_path,
            },
            "fix_history": self.fix_history[-12:],
        }
        path = os.path.join(self.agent_log_dir, "run_result.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except OSError:
            pass
        return status

    def _last_error_content(self, agent_output: str) -> str:
        if self.shell_agent is None:
            return agent_output
        failed = self._latest_failed_command_raw_record()
        if failed:
            return str(failed.get("stderr") or failed.get("output") or agent_output)
        return agent_output

    def _latest_failed_command_raw_record(self) -> Optional[dict]:
        for item in reversed(self._build_command_records()):
            if item.get("status") not in {"failed", "timeout"}:
                continue
            return item
        return None

    def _latest_failed_command_record(self) -> Optional[dict]:
        records = self._build_command_records()
        for index, item in reversed(list(enumerate(records, 1))):
            if item.get("status") not in {"failed", "timeout"}:
                continue
            return self._compact_command_record(item, index)
        return None

    def _compact_command_record(self, item: dict, index: int) -> dict:
        return {
            "index": index,
            "command": item.get("command", ""),
            "status": item.get("status", ""),
            "exit_code": item.get("exit_code"),
            "log_path": item.get("log_path", ""),
            "stderr_tail": str(item.get("stderr") or "")[-800:],
            "output_tail": str(item.get("output") or "")[-800:],
        }

    def _rule_based_success_verdict(self, records: List[dict]) -> Optional[dict]:
        success_markers = (
            "Compilation succeeded",
            "Built target",
            "BUILD SUCCESS",
            "Successfully built",
            "Linking C executable",
            "Linking CXX executable",
            "Linking C shared library",
            "Linking CXX shared library",
        )
        artifact_re = re.compile(r"(\S+\.(?:so|a|o|node)|\b[a-zA-Z0-9_.+-]+(?:_test|test)\b)")
        build_cmd_re = re.compile(
            r"\b(?:make|ninja|cmake\s+--build|meson\s+compile|rake\s+compile|gcc|g\+\+|clang|clang\+\+|node-waf\s+.*build)\b",
            re.IGNORECASE,
        )
        for item in reversed(records[-16:]):
            if str(item.get("exit_code")) not in ("0", "None"):
                continue
            command = str(item.get("command") or "")
            if not build_cmd_re.search(command):
                continue
            output = "\n".join(str(item.get(field) or "") for field in ("stdout", "stderr", "output"))
            if any(marker in output for marker in success_markers):
                return {"success": True, "status": "done", "reason": "Rule-based success marker from build output."}
            if artifact_re.search(output):
                return {"success": True, "status": "done", "reason": "Rule-based artifact evidence from successful build command."}
        return None

    def _failed_command_fix_history_summary(self, current_failed_command: Optional[dict]) -> str:
        payload = {
            "current_failed_command": current_failed_command or {},
            "previous_failed_commands_and_fixes": [
                {
                    "error": item.get("error", ""),
                    "failed_command": item.get("failed_command", {}),
                    "classification_category": item.get("classification_category"),
                    "classification_summary": item.get("classification_summary", ""),
                    "solution_kind": item.get("solution_kind", ""),
                    "solution": item.get("solution", ""),
                    "result": item.get("result", ""),
                    "failure": item.get("failure", ""),
                    "sysuse": item.get("sysuse", ""),
                    "action_signature": item.get("action_signature", ""),
                }
                for item in self.fix_history[-8:]
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _extract_error_lines(self, error_content: str) -> str:
        # Direct stderr truncation — no LLM call
        return error_content[-4000:] if len(error_content) > 4000 else error_content

        # --- Original LLM-based extraction (disabled) ---
        # try:
        #     extracted = ask_ds(
        #         system_prompt=EXTRACT_ERROR_LINE_SYS,
        #         user_prompt=EXTRACT_ERROR_LINE_USR.format(error_content=error_content),
        #     )
        #     if extracted and not str(extracted).startswith("Error:"):
        #         return str(extracted)
        # except Exception as exc:
        #     logging.getLogger("agent_build").warning("Error extraction failed: %s", exc)
        # return error_content[-12000:]

    def _validate_build_success(self) -> dict:
        """Use LLM to verify whether compilation actually succeeded based on command records."""
        if self.shell_agent is None:
            return {"success": False, "status": "no_command", "reason": "No shell agent available"}
        records = self._build_command_records()
        if not records:
            return {"success": False, "status": "no_command", "reason": "No command records available"}

        rule_verdict = self._rule_based_success_verdict(records)
        if rule_verdict:
            return rule_verdict

        # Rule-based guard: if every "build" command (configure/cmake/meson/make) in the
        # last K records is syntax-only or a no-output check, reject without LLM. These
        # invocations do not produce binaries/libraries and must not count as success
        # even when their exit code is 0. Common offenders observed in the wild:
        #   - `gcc -fsyntax-only` / `g++ -fsyntax-only`
        #   - `make compile` (mongoose-style: runs `-fsyntax-only` per source)
        #   - `make check` / `make test` returning non-zero on a sub-target but the
        #     overall exit code is somehow 0 — these are tests, not the build itself.
        # We only reject when there is NO accompanying real build command (configure/cmake
        # setup + ninja/make with non-syntax-only output).
        syntax_only_re = re.compile(r"-fsyntax-only|--syntax-only|\bcheck\b|\btest\b|^\s*make\s+(compile|check|test)\b", re.IGNORECASE)
        build_re = re.compile(
            r"\b(?:configure|cmake|meson\s+setup|meson\s+compile|ninja|make\s+-j|make\b(?!.*\b(?:compile|check|test)\b)"
            r"|gem\s+install|cargo\s+build|cargo\s+install|npm\s+install|npm\s+run\s+build"
            r"|pip\s+install|go\s+build|rustc|stack\s+build|cabal\s+(build|install)"
            r"|dpkg-buildpackage|debuild|rake|mvn\s+compile|gradle\s+(build|assemble))",
            re.IGNORECASE,
        )
        last_k = records[-12:]
        has_syntax_only_only = any(syntax_only_re.search(str(r.get("command", ""))) for r in last_k)
        has_real_build = any(build_re.search(str(r.get("command", ""))) and not syntax_only_re.search(str(r.get("command", ""))) for r in last_k)
        if has_syntax_only_only and not has_real_build:
            return {
                "success": False,
                "status": "rejected",
                "reason": "Last commands are syntax-only / test-only (e.g. -fsyntax-only or `make compile/check/test`) and no real build command was executed; no binary/library artifact could have been produced.",
            }

        compact = []
        for item in records:
            compact.append({
                "command": item.get("command"),
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
            })
        user_prompt = json.dumps(compact, indent=2, ensure_ascii=False)
        try:
            result = ask_ds(
                system_prompt=VALIDATION_SYSTEM,
                user_prompt=user_prompt,
            )
            parsed = _extract_json_object(result)
            if parsed and "success" in parsed:
                parsed.setdefault("status", "done" if parsed.get("success") else "provider_error")
                return parsed
            return {"success": False, "status": "provider_error", "reason": f"Validation parse failed: {result[:200]}"}
        except Exception as e:
            logging.getLogger("agent_build").warning("Validation call failed: %s", e)
            return {"success": False, "status": "provider_error", "reason": "Validation unavailable"}

    def _handle_provider_failure(self, output: str, source: str) -> None:
        verdict = self._validate_build_success()
        status = verdict.get("status")
        if verdict.get("success"):
            self.if_done = True
        elif _looks_like_provider_error(output):
            self.provider_error = 1
        elif status == "no_command":
            self.no_command = 1
        else:
            self.provider_error = 1
        recorded_status = "done" if verdict.get("success") else "provider_error" if _looks_like_provider_error(output) else status
        self.error_summary.append(
            f"{source} provider failure.\n{output}\nValidation: {verdict.get('reason', '')}"
        )
        self._record_fix_attempt(
            error_text=self._last_error_content(output),
            solution_kind=f"{source.lower().replace(' ', '_')}_provider_failure",
            solution_detail="Validation-driven provider failure resolution.",
            result=recorded_status or "provider_error",
            failure=output,
            failed_command=self._latest_failed_command_record(),
        )

    async def _register_source_dependency_plan(self, source_dependency: dict) -> bool:
        """Add a Phase3 source dependency plan to Phase2 project_info for the next executor turn."""
        plan = _normalize_source_dependency_plan(source_dependency)
        if not plan:
            return False
        if not plan.get("build_order"):
            return False
        verification = self._source_dependency_verification(plan)
        if not verification or self.shell_agent is None:
            return False
        probe = getattr(self.shell_agent, "probe_command", None)
        if not callable(probe):
            return False
        verification_result = await probe(str(verification.get("command") or ""))
        if (
            verification_result.get("status") != "success"
            or verification_result.get("exit_code") not in {0, "0"}
        ):
            return False
        plan_identity = (
            plan["name"].strip().lower(),
            str(plan.get("repo_url") or plan.get("source_path") or plan.get("submodule_path") or ""),
        )
        already_selected = any(
            isinstance(item, dict)
            and item.get("status") == "selected"
            and (
                str(item.get("name") or "").strip().lower(),
                str(item.get("repo_url") or item.get("source_path") or item.get("submodule_path") or ""),
            ) == plan_identity
            for item in self.build_plan_state.get("source_dep_candidates") or []
        )
        if already_selected:
            return False
        self.build_plan_state = add_source_dependency_candidate(self.build_plan_state, plan)
        selected_id = str(next(
            (
                item.get("candidate_id") or ""
                for item in self.build_plan_state.get("source_dep_candidates") or []
                if isinstance(item, dict)
                and str(item.get("name") or "").strip().lower() == plan["name"].lower()
                and str(item.get("repo_url") or item.get("source_path") or "")
                == str(plan.get("repo_url") or plan.get("source_path") or "")
            ),
            "",
        ))
        if not selected_id:
            return False
        self.build_plan_state = apply_repair_operation_to_plan(
            self.build_plan_state,
            {"kind": "source_dependency_selection", "source_dependency_id": selected_id},
        )
        await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
        return True

    async def _sync_build_plan_to_phase2_agent(self, recreate_agent: bool = False) -> None:
        """Refresh controller state, then recreate the executor from its narrow projection."""
        project_info = _extract_json_object(self.phase2_project_info or "")
        if not project_info:
            return
        selected = self.build_plan_state.get("selected") or {}
        selected_entry = next(
            (
                item for item in self.build_plan_state.get("entrypoint_candidates") or []
                if isinstance(item, dict)
                and item.get("entry_id") == selected.get("build_entry_id")
                and item.get("eligible", True)
            ),
            None,
        )
        if selected_entry:
            project_info["main_build_plan"] = list(selected.get("command_sequence") or selected_entry.get("command_sequence") or [])
            project_info["main_build_steps"] = deepcopy(
                selected.get("command_steps") or selected_entry.get("command_steps") or []
            )
            project_info["selected_build_entry"] = deepcopy(selected_entry)
        selected_generator = next(
            (
                item for item in self.build_plan_state.get("generator_candidates") or []
                if isinstance(item, dict) and item.get("status") == "selected"
            ),
            None,
        )
        if selected_generator and selected_generator.get("command"):
            command = str(selected_generator["command"])
            project_info["main_build_plan"] = list(dict.fromkeys(
                [command] + list(project_info.get("main_build_plan") or [])
            ))
        selected_packages = [
            str(item.get("name") or "")
            for item in self.build_plan_state.get("system_dep_candidates") or []
            if isinstance(item, dict) and item.get("status") == "selected" and item.get("name")
        ]
        retired_packages = {
            str(item).lower() for item in self.build_plan_state.get("retired_system_packages") or []
        }
        project_info["apt_install_plan"] = list(dict.fromkeys(
            [
                str(item) for item in project_info.get("apt_install_plan") or []
                if str(item).lower() not in retired_packages
            ] + selected_packages
        ))
        selected_source_deps = [
            {
                key: deepcopy(item[key])
                for key in (
                    "name", "repo_url", "clone_status", "setup_commands",
                    "fetch_commands", "container_source_path", "source_path",
                    "submodule_path", "build_order", "env_exports",
                    "install_prefix", "metadata_path",
                )
                if item.get(key) not in (None, "", [])
            }
            for item in self.build_plan_state.get("source_dep_candidates") or []
            if isinstance(item, dict)
            and item.get("status") == "selected"
            and _source_dependency_has_executable_contract(item)
        ]
        project_info["source_dep_plan"] = selected_source_deps
        project_info["unresolved_source_packages"] = list(
            self.build_plan_state.get("unresolved_source_packages") or []
        )
        overrides = deepcopy(self.build_plan_state.get("repair_overrides") or {})
        if overrides:
            project_info["repair_overrides"] = overrides
            project_info["build_options"] = list(dict.fromkeys(
                list(project_info.get("build_options") or [])
                + list(overrides.get("configure_args") or [])
            ))
            for key in ("compile_flags", "link_flags", "environment", "workdir"):
                if key in overrides:
                    project_info[key] = deepcopy(overrides[key])
            if overrides.get("workdir"):
                workdir = str(overrides["workdir"])
                project_info["build_workdir"] = workdir
                existing_steps = project_info.get("main_build_steps") or [
                    {"command": command}
                    for command in project_info.get("main_build_plan") or []
                    if str(command).strip()
                ]
                project_info["main_build_steps"] = [
                    {**deepcopy(step), "workdir": workdir}
                    for step in existing_steps
                    if isinstance(step, dict) and str(step.get("command") or "").strip()
                ]
        project_info["build_plan"] = compact_build_plan_for_prompt(self.build_plan_state)
        self.phase2_project_info = json.dumps(project_info, indent=2, ensure_ascii=False)
        try:
            save_build_plan(self.build_plan_state, self.build_plan_path)
        except OSError:
            pass
        if recreate_agent and self.shell_agent is not None:
            self.build_repo_agent = await self._create_phase2_agent()

    def _available_build_plan_alternatives(self, route: Optional[DiagnosisRoute] = None) -> list[dict[str, Any]]:
        """Return compact, untried alternatives that the controller can apply."""
        current_sysuse = normalize_sysuse(self.sysuse)
        alternatives = self.build_plan_state.get("alternatives") or {}
        candidates: list[dict[str, Any]] = []
        fallback_ids = {
            str(item.get("action_id") or "")
            for item in ((route.repair_scope or {}).get("fallback_actions") if route else []) or []
            if isinstance(item, dict)
        }
        allow_entry_fallback = route is None or "switch_build_entry" in fallback_ids
        allow_target_fallback = route is None or "switch_build_target" in fallback_ids

        for item in alternatives.get("build_entries") or []:
            if not allow_entry_fallback:
                break
            if not isinstance(item, dict):
                continue
            if item.get("status") == "ineligible" or not item.get("eligible", True):
                continue
            entry_id = str(item.get("entry_id") or "").strip()
            command_sequence = [
                str(command).strip()
                for command in item.get("command_sequence") or []
                if str(command).strip()
            ]
            failed_sysuses = {
                normalize_sysuse(value)
                for value in item.get("failed_sysuses") or []
                if value
            }
            if not entry_id or not command_sequence or current_sysuse in failed_sysuses:
                continue
            preflight_ok, _ = _preflight_build_entry(item, self.compile_path)
            if not preflight_ok:
                item["status"] = "ineligible"
                item["eligible"] = False
                continue
            candidates.append({
                "alternative_id": f"build_entry:{entry_id}",
                "kind": "build_entry",
                "entry_id": entry_id,
                "path": item.get("path") or "",
                "build_system": item.get("build_system") or "unknown",
                "source": item.get("source") or "unknown",
                "confidence": item.get("confidence") or "unknown",
                "rank": item.get("rank"),
                "command_sequence": command_sequence[:6],
                "evidence_paths": list(item.get("evidence_paths") or [])[:4],
            })

        allow_sysuse_fallback = bool(
            route
            and any(
                str(item.get("action_id") or "") == "switch_system_release"
                for item in (route.repair_scope or {}).get("fallback_actions") or []
                if isinstance(item, dict)
            )
        )
        tried_sysuses = {
            normalize_sysuse(item.get("sysuse"))
            for item in self.build_plan_state.get("attempts") or []
            if isinstance(item, dict)
            and item.get("kind") == "build"
            and item.get("result") == "failed"
            and item.get("sysuse")
        }
        tried_sysuses.add(current_sysuse)
        for item in alternatives.get("sysuses") or []:
            if not allow_sysuse_fallback:
                continue
            if not isinstance(item, dict) or not item.get("sysuse"):
                continue
            suite = normalize_sysuse(item.get("sysuse"))
            if not suite or suite in tried_sysuses:
                continue
            candidates.append({
                "alternative_id": f"sysuse:{suite}",
                "kind": "sysuse",
                "sysuse": suite,
                "source": item.get("source") or "build_plan",
                "reason": item.get("reason") or "",
            })
        for item in self.build_plan_state.get("target_candidates") or []:
            if not allow_target_fallback:
                break
            if not isinstance(item, dict) or item.get("status") in {"selected", "failed", "exhausted"}:
                continue
            target_id = str(item.get("target_id") or "")
            if target_id:
                candidates.append({
                    "alternative_id": f"build_target:{target_id}",
                    "kind": "build_target",
                    "target_id": target_id,
                    "name": item.get("name") or "",
                    "entry_id": item.get("entry_id") or "",
                    "source": item.get("source") or "build_plan",
                    "evidence": item.get("evidence") or "",
                })
        return candidates

    async def _switch_to_next_build_entry(
        self,
        preferred_entry_id: str = "",
        prefer_existing_generated: bool = False,
    ) -> Optional[dict]:
        selected = self.build_plan_state.get("selected") or {}
        current_id = str(selected.get("build_entry_id") or "")
        current_sysuse = normalize_sysuse(self.sysuse)
        candidates = sorted(
            [item for item in self.build_plan_state.get("entrypoint_candidates") or [] if isinstance(item, dict)],
            key=lambda item: (
                0 if prefer_existing_generated and self._entry_has_existing_artifact(item, "configure") else 1,
                int(item.get("rank")) if item.get("rank") is not None else 10**6,
                str(item.get("entry_id") or item.get("command") or ""),
            ),
        )
        next_entry = None
        for item in candidates:
            if (
                str(item.get("entry_id") or "") == current_id
                or (preferred_entry_id and str(item.get("entry_id") or "") != preferred_entry_id)
                or item.get("status") in {"selected", "ineligible"}
                or not item.get("eligible", True)
                or current_sysuse in {
                    normalize_sysuse(value)
                    for value in item.get("failed_sysuses") or []
                    if value
                }
                or not item.get("command_sequence")
            ):
                continue
            preflight_ok, preflight_reason = _preflight_build_entry(item, self.compile_path)
            if not preflight_ok:
                item["status"] = "ineligible"
                item["eligible"] = False
                item.setdefault("eligibility_reasons", []).append(preflight_reason)
                continue
            next_entry = item
            break
        if next_entry is None:
            return None

        self.build_plan_state = select_build_entry_in_plan(
            self.build_plan_state,
            str(next_entry.get("entry_id") or ""),
        )
        project_info = _extract_json_object(self.phase2_project_info or "") or {}
        project_info["main_build_plan"] = list(next_entry.get("command_sequence") or [])
        project_info["selected_build_entry"] = dict(next_entry)
        project_info["build_entry_candidates"] = list(self.build_plan_state.get("entrypoint_candidates") or [])
        self.phase2_project_info = json.dumps(project_info, indent=2, ensure_ascii=False)
        await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
        return next_entry

    def _entry_has_existing_artifact(self, entry: dict[str, Any], filename: str) -> bool:
        path = str(entry.get("path") or entry.get("entry_path") or "").strip()
        path = path.removeprefix("/work/").lstrip("/")
        if not path:
            path = filename
        if Path(path).name != filename:
            return False
        return (Path(self.compile_path) / path).is_file()

    async def _switch_to_next_sysuse_candidate(
        self,
        preferred_sysuse: str = "",
        route: Optional[DiagnosisRoute] = None,
    ) -> Optional[str]:
        """Switch to the next untried suite already present in the Build Plan."""
        tried = {
            normalize_sysuse(item.get("sysuse"))
            for item in self.build_plan_state.get("attempts") or []
            if isinstance(item, dict)
            and item.get("kind") == "build"
            and item.get("result") == "failed"
            and item.get("sysuse")
        }
        tried.add(normalize_sysuse(self.sysuse))
        candidate = next(
            (
                normalize_sysuse(item.get("sysuse"))
                for item in self.build_plan_state.get("sysuse_candidates") or []
                if isinstance(item, dict)
                and item.get("sysuse")
                and (
                    not preferred_sysuse
                    or normalize_sysuse(item.get("sysuse")) == normalize_sysuse(preferred_sysuse)
                )
                and normalize_sysuse(item.get("sysuse")) not in tried
                and self._sysuse_candidate_supports_route(normalize_sysuse(item.get("sysuse")), route)
            ),
            "",
        )
        if not candidate:
            return None
        switched = await self.switch_sysuse(candidate)
        return candidate if switched else None

    def _sysuse_candidate_supports_route(
        self,
        sysuse: str,
        route: Optional[DiagnosisRoute],
    ) -> bool:
        relation = (route.version_relation if route else {}) or {}
        if not relation.get("required_version"):
            return True
        subject = str(relation.get("subject") or (route.failed_tool if route else "") or "").lower()
        package = {
            "meson": "meson",
            "cmake": "cmake",
            "autoconf": "autoconf",
            "automake": "automake",
            "gcc": "gcc",
            "g++": "g++",
            "clang": "clang",
        }.get(subject, subject)
        if not package:
            return False
        return bool(best_satisfying_package_version(
            package,
            sysuse,
            str(relation.get("required_operator") or ">="),
            str(relation.get("required_version") or ""),
        ))

    async def _switch_to_next_build_target(self, preferred_target_id: str = "") -> Optional[dict]:
        target = next(
            (
                item for item in self.build_plan_state.get("target_candidates") or []
                if isinstance(item, dict)
                and item.get("status") not in {"selected", "failed", "exhausted"}
                and (
                    not preferred_target_id
                    or str(item.get("target_id") or "") == preferred_target_id
                )
            ),
            None,
        )
        if not target:
            return None
        self.build_plan_state = apply_repair_operation_to_plan(self.build_plan_state, {
            "kind": "build_target_selection",
            "target_id": target.get("target_id") or "",
        })
        await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
        return target

    async def _try_next_route_fallback(self, route: DiagnosisRoute) -> Optional[dict]:
        """Apply one matrix-declared fallback backed by a live Build Plan candidate."""
        scope = route.repair_scope or {}
        for action in scope.get("fallback_actions") or []:
            if not isinstance(action, dict):
                continue
            action_id = str(action.get("action_id") or "")
            detail: Any = None
            solution_kind = ""
            if action_id == "switch_build_entry":
                prefer_existing = route.symptom_family in {"generated_file_missing", "source_path_missing"}
                detail = await self._switch_to_next_build_entry(prefer_existing_generated=prefer_existing)
                solution_kind = "build_plan_fallback_entry"
            elif action_id == "switch_build_target":
                detail = await self._switch_to_next_build_target()
                solution_kind = "build_plan_fallback_target"
            elif action_id == "switch_system_release":
                detail = await self._switch_to_next_sysuse_candidate(route=route)
                solution_kind = "build_plan_fallback_sysuse"
            if not detail:
                continue
            scope["selected_fallback_action"] = dict(action)
            route.repair_scope = scope
            self.latest_diagnosis_route = route.to_dict()
            return {
                "action": action,
                "solution_kind": solution_kind,
                "detail": detail,
            }
        return None

    async def _apply_next_route_fallback(
        self,
        route: DiagnosisRoute,
        *,
        error_text: str,
        failed_command: Optional[dict],
    ) -> bool:
        """Apply and record one executable fallback from the current route."""
        fallback = await self._try_next_route_fallback(route)
        if not fallback:
            return False
        action = fallback["action"]
        detail = fallback["detail"]
        self._record_fix_attempt(
            error_text=error_text,
            solution_kind=fallback["solution_kind"],
            solution_detail=(
                json.dumps(detail, ensure_ascii=False)
                if isinstance(detail, dict)
                else f"switch to {detail}"
            ),
            result="applied",
            classification_category=route.category,
            classification_summary=route.summary,
            failed_command=failed_command,
            hypothesis="matrix_fallback_build_plan_alternative",
            repair_action_override=str(action.get("action_id") or ""),
            fix_direction_override=str(action.get("fix_direction") or ""),
        )
        return True

    async def _explore_matrix_empty_alternative(
        self,
        route: DiagnosisRoute,
        *,
        error_text: str,
        failed_command: Optional[dict],
        recent_commands: str,
    ) -> dict[str, Any]:
        candidates = self._available_build_plan_alternatives(route)
        if not candidates:
            return {"status": "no_alternative", "reason": "Build Plan has no untried alternatives."}

        task = MATRIX_EMPTY_EXPLORATION_TASK.format(
            exploration_context=json.dumps(
                self._alternative_prompt_context(route, error_text, failed_command, candidates),
                ensure_ascii=False,
                indent=2,
            ),
        )
        explorer = await BuildPlanAlternativeAgentFactory.create(
            self.sysuse,
            log_dir=self.docker_log_dir,
            shell_manager=self.shell_agent,
        )
        self.fallback_to_llm += 1
        raw = await call_error_agent(task, explorer)
        if _looks_like_provider_error(raw):
            return {"status": "provider_error", "raw": raw}

        selection = _parse_build_plan_alternative_selection(
            raw,
            {str(item.get("alternative_id") or "") for item in candidates},
        )
        if not selection:
            return {"status": "invalid", "reason": "Explorer returned an invalid alternative selection."}
        if selection["status"] == "cannot_solve":
            return {"status": "cannot_solve", "reason": selection.get("reason") or "No supported alternative."}

        candidate = next(
            item for item in candidates
            if item.get("alternative_id") == selection["alternative_id"]
        )
        action_id = ""
        fix_direction = ""
        detail: Any = None
        solution_kind = ""
        if candidate["kind"] == "build_entry":
            action_id = "switch_build_entry"
            fix_direction = "build_entry"
            detail = await self._switch_to_next_build_entry(
                candidate["entry_id"],
                prefer_existing_generated=route.symptom_family in {"generated_file_missing", "source_path_missing"},
            )
            solution_kind = "build_plan_llm_alternative_entry"
        elif candidate["kind"] == "sysuse":
            action_id = "switch_system_release"
            fix_direction = "system_release"
            detail = await self._switch_to_next_sysuse_candidate(candidate["sysuse"], route=route)
            solution_kind = "build_plan_llm_alternative_sysuse"
        elif candidate["kind"] == "build_target":
            action_id = "switch_build_target"
            fix_direction = "build_target"
            detail = await self._switch_to_next_build_target(candidate["target_id"])
            solution_kind = "build_plan_llm_alternative_target"
        if not detail:
            return {"status": "stale", "reason": "Selected alternative is no longer applicable."}

        scope = route.repair_scope or {}
        scope["selected_fallback_action"] = {
            "action_id": action_id,
            "fix_direction": fix_direction,
            "selection_tier": "matrix_empty_llm_alternative",
        }
        route.repair_scope = scope
        self.latest_diagnosis_route = route.to_dict()
        self._record_fix_attempt(
            error_text=error_text,
            solution_kind=solution_kind,
            solution_detail=json.dumps(
                {
                    "alternative_id": candidate["alternative_id"],
                    "detail": detail,
                    "reason": selection.get("reason") or "",
                    "evidence": selection.get("evidence") or [],
                },
                ensure_ascii=False,
            ),
            result="applied",
            classification_category=route.category,
            classification_summary=route.summary,
            failed_command=failed_command,
            hypothesis="matrix_empty_build_plan_exploration",
            repair_action_override=action_id,
            fix_direction_override=fix_direction,
        )
        return {"status": "applied", "candidate": candidate}

    async def _concretize_selected_matrix_action(
        self,
        route: DiagnosisRoute,
        *,
        error_text: str,
        failed_command: Optional[dict],
        recent_commands: str,
        failed_phase: str,
    ) -> dict[str, Any]:
        """Concretize and apply one selected Matrix action without rebuilding."""
        failure_identity = str((route.failure_identity or {}).get("identity_key") or "")
        card_plan: Optional[Dict[str, Any]] = None
        if not os.environ.get("THEAUTO_NO_CARD_DRIVEN"):
            try:
                card_plan = await self._try_card_driven_repair(
                    error_lines=error_text,
                    failed_command=failed_command,
                    recent_commands=recent_commands,
                    failed_phase=failed_phase,
                    diagnosis_route=route,
                )
            except Exception as exc:
                print(f"[CardDriven] card-driven repair raised ({exc}); falling back to LLM Phase 3")

        if card_plan is not None and card_plan.get("status") == "provider_error":
            return card_plan

        card_driven = bool(
            card_plan is not None
            and isinstance(card_plan.get("repair_decision"), RepairDecision)
        )
        if card_driven:
            repair_decision = card_plan["repair_decision"]
        else:
            self.fallback_to_llm += 1
            await self._populate_provider_evidence(route)
            selected_action = (route.repair_scope or {}).get("selected_matrix_action") or {}
            candidate_catalog = self._repair_candidate_catalog(
                route,
                list(selected_action.get("operation_kinds") or []),
            )
            required_candidate_fields = {
                "system_package_selection": "provider_evidence",
                "source_dependency_selection": "source_dep_candidates",
                "generator_selection": "generator_candidates",
                "build_entry_selection": "entrypoint_candidates",
                "build_target_selection": "target_candidates",
                "system_release_selection": "sysuse_candidates",
            }
            required_fields = [
                required_candidate_fields[kind]
                for kind in selected_action.get("operation_kinds") or []
                if kind in required_candidate_fields
            ]
            if required_fields and not any(candidate_catalog.get(field) for field in required_fields):
                reason = "producer_no_candidate: " + ",".join(required_fields)
                self._record_fix_attempt(
                    error_text=error_text,
                    solution_kind="producer_no_candidate",
                    solution_detail=reason,
                    result="failed",
                    classification_category=route.category,
                    classification_summary=route.summary,
                    failed_command=failed_command,
                    hypothesis=f"producer_no_candidate_{failure_identity}",
                    repair_action_override=selected_action.get("action_id"),
                )
                self._last_candidate_failure_semantic = False
                return {"status": "unavailable", "reason": reason, "semantic_candidate": False}
            task = ERROR_SUMMARY_TASK.format(
                repair_context=json.dumps(
                    self._repair_prompt_context(
                        route,
                        error_text,
                        failed_command,
                        selected_action,
                        candidate_catalog,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            agent = await ErrorSummaryAgentFactory.create(
                self.sysuse,
                log_dir=self.docker_log_dir,
                shell_manager=self.shell_agent,
                operation_kinds=list(selected_action.get("operation_kinds") or []),
            )
            repair_decision = None
            validation_error = ""
            for translation_attempt in range(2):
                retry_task = task
                if validation_error:
                    retry_task += (
                        "\nThe controller rejected the previous response: "
                        f"{validation_error}. Return one corrected object using the same "
                        "selected action and supplied candidates."
                    )
                raw = await call_error_agent(retry_task, agent)
                if _looks_like_provider_error(raw):
                    return {"status": "provider_error", "raw": raw}
                try:
                    repair_decision = parse_repair_decision(
                        raw,
                        selected_action=selected_action,
                        candidate_catalog=candidate_catalog,
                        failure_identity=failure_identity,
                        selection_origin="phase3_concretizer",
                        failure_context={
                            **(route.failure_identity or {}),
                            "failure_mechanism": route.failure_mechanism,
                            "version_relation": route.version_relation,
                        },
                    )
                    break
                except InvalidRepairDecision as exc:
                    validation_error = str(exc)
                    if translation_attempt == 1:
                        repair_decision = RepairDecision(
                            status="cannot_solve",
                            reason=f"invalid typed repair decision: {validation_error}",
                        )
            assert repair_decision is not None

        if repair_decision.status != "repair":
            self._last_candidate_failure_semantic = "translation_unavailable"
            self.build_plan_state = record_candidate_outcome(
                self.build_plan_state,
                operation=repair_decision.operation,
                outcome="translation_unavailable",
                failure_identity=failure_identity,
            )

        self.error_summary.append(repair_decision.reason or "typed repair decision")
        self._record_fix_attempt(
            error_text=error_text,
            solution_kind=f"typed_{repair_decision.operation.get('kind') or 'cannot_solve'}",
            solution_detail=json.dumps(repair_decision.to_dict(), ensure_ascii=False),
            result="proposed" if repair_decision.status == "repair" else "failed",
            classification_category=route.category,
            classification_summary=route.summary,
            failed_command=failed_command,
            hypothesis=f"typed_{repair_decision.fingerprint or 'cannot_solve'}",
            action_signature=repair_decision.fingerprint,
            repair_action_override=repair_decision.matrix_action,
            fix_direction_override=repair_decision.fix_direction,
        )
        if repair_decision.status == "repair":
            if await self._apply_typed_repair(repair_decision):
                self._last_fix_was_card_driven = card_driven
                self.build_plan_state = set_pending_fix_plan(self.build_plan_state, [])
                return {"status": "applied", "decision": repair_decision}
            if repair_decision.fingerprint:
                self._exhausted_operation_fingerprints.setdefault(failure_identity, set()).add(
                    repair_decision.fingerprint
                )
            # The operation was concrete but could not be applied (for
            # example, a patch did not match the current file). Keep the
            # Matrix action alive so another producer candidate can be
            # translated. Only an empty candidate catalog retires the action.
            self._last_candidate_failure_semantic = "retry_candidate"
            self.build_plan_state = record_candidate_outcome(
                self.build_plan_state,
                operation=repair_decision.operation,
                outcome="apply_failed",
                failure_identity=failure_identity,
            )
        # Invalid JSON or an omitted evidence reference is a translation
        # failure, not evidence that this Matrix action has no viable repair.
        # A concrete operation that could not be applied keeps its earlier
        # ``retry_candidate`` state so another candidate remains possible.
        self._last_fix_was_card_driven = False
        return {"status": "unavailable", "decision": repair_decision}

    async def _advance_after_selected_action_failure(
        self,
        route: DiagnosisRoute,
        *,
        error_text: str,
        failed_command: Optional[dict],
    ) -> str:
        """Retire the selected action and advance without another build guess."""
        candidate_failure = getattr(self, "_last_candidate_failure_semantic", True)
        if candidate_failure == "retry_candidate":
            self._last_candidate_failure_semantic = True
            return "matrix_action"
        if candidate_failure == "translation_unavailable":
            scope_key = self._failure_scope_key(route.failure_identity)
            action_id = str(((route.repair_scope or {}).get("selected_matrix_action") or {}).get("action_id") or "")
            retry_key = f"{scope_key}:{action_id}"
            attempts = self._translation_retry_counts.get(retry_key, 0) + 1
            self._translation_retry_counts[retry_key] = attempts
            self._last_candidate_failure_semantic = True
            # One controller-level retry is enough because each concretizer
            # request already gets an in-request validation retry. Do not
            # retire the semantic action merely because the model response was
            # malformed; terminal state remains explicitly translation-bound.
            return "matrix_action" if attempts <= 1 else "terminal"
        if candidate_failure is False:
            # This producer/card could not yield a concrete operation. Retire
            # only the selected Matrix action, then try the remaining actions
            # and fallback pool before declaring the failure unsolved.
            self._last_candidate_failure_semantic = True
        if self._exhaust_selected_matrix_action(route):
            return "matrix_action"
        if await self._apply_next_route_fallback(
            route,
            error_text=error_text,
            failed_command=failed_command,
        ):
            return "fallback"
        return "terminal"

    def _apply_exhausted_matrix_actions(self, route: DiagnosisRoute) -> DiagnosisRoute:
        scope = route.repair_scope or {}
        identity_key = self._failure_scope_key(route.failure_identity)
        exhausted = self._exhausted_matrix_actions.get(identity_key, set())
        actions = [
            item
            for item in scope.get("matrix_actions") or []
            if isinstance(item, dict) and item.get("action_id") not in exhausted
        ]
        scope["matrix_actions"] = actions
        scope["selected_matrix_action"] = actions[0] if actions else None
        scope["matrix_exhausted_actions"] = sorted(exhausted)
        scope["matrix_empty"] = not bool(actions)
        route.repair_scope = scope
        route.candidate_decisions = list(dict.fromkeys(
            str(item.get("fix_direction") or "")
            for item in actions
            if item.get("fix_direction")
        ))
        return route

    def _exhaust_selected_matrix_action(self, route: DiagnosisRoute) -> bool:
        scope = route.repair_scope or {}
        selected = scope.get("selected_matrix_action") or {}
        identity_key = self._failure_scope_key(route.failure_identity)
        action_id = str(selected.get("action_id") or "")
        if not identity_key or not action_id:
            return False
        self._exhausted_matrix_actions.setdefault(identity_key, set()).add(action_id)
        self._apply_exhausted_matrix_actions(route)
        self.latest_diagnosis_route = route.to_dict()
        return bool((route.repair_scope or {}).get("selected_matrix_action"))

    def _retire_previous_repair_after_build_failure(self, next_failure_identity: str = "") -> None:
        """Convert the last applied repair into feedback for card/action routing."""
        previous = next(
            (
                item
                for item in reversed(self.fix_history)
                if item.get("result") == "applied" and item.get("repair_action")
            ),
            None,
        )
        if not previous or previous.get("outcome_checked"):
            return
        previous["outcome_checked"] = True
        previous_identity = str(previous.get("failure_identity") or "")
        previous_scope = str(previous.get("failure_scope") or previous_identity)
        next_scope = str(next_failure_identity or "")
        same_failure = bool(
            not next_scope
            or previous_scope == next_scope
            or previous_identity == next_scope
            or (previous_identity and next_scope.startswith(previous_identity + "|"))
        )
        previous["result"] = "failed_after_rebuild" if same_failure else "progressed"
        if not same_failure:
            return
        hypothesis = str(previous.get("hypothesis") or "")
        if hypothesis.startswith("card_"):
            card_id = hypothesis[5:]
            self._tried_card_keys.add(self._card_failure_key(card_id, previous_scope))
            return
        identity_key = previous_scope
        repair_action = str(previous.get("repair_action") or "")
        if identity_key and repair_action:
            self._exhausted_matrix_actions.setdefault(identity_key, set()).add(repair_action)

    async def _execute_fix_commands(self, commands: List[str]) -> dict:
        if self.shell_agent is None:
            raise RuntimeError("Docker shell agent is not initialized")
        executed = []
        failed_record = None
        for command in commands:
            if is_package_install_command(command):
                failed_record = {
                    "command": command,
                    "status": "blocked",
                    "exit_code": None,
                    "stderr": "Package installs require current ProviderEvidence and are forbidden in generic repair.",
                }
                executed.append(failed_record)
                break
            result = await self.shell_agent.execute_command(command)
            record = dict(result or {})
            record.setdefault("command", command)
            executed.append(record)
            if record.get("status") == "timeout":
                self.timeout = True
            if record.get("status") != "success":
                failed_record = record
                self.error_summary.append(
                    f"Repair command failed: {command}\n"
                    f"stderr:\n{record.get('stderr') or ''}\n"
                    f"output:\n{record.get('output') or ''}\n"
                    f"log_path: {record.get('log_path')}"
                )
                break
        return {
            "success": failed_record is None and len(executed) == len(commands),
            "executed": executed,
            "failed_command": failed_record,
        }

    async def try_compile_cmd(self, max_rounds: int = 100, max_total_commands: int = 100):
        """Run Phase2 build, invoke Phase3 fixer on failure, then retry linearly.

        max_rounds is interpreted as the maximum number of *Phase 2 build attempts*
        (i.e. real call_build_agent invocations). Sysuse switches, source-dep plan
        registrations, and repair command executions do NOT consume a
        round on their own — they trigger another phase 2 build under the new
        environment, and that next build is what counts toward the budget. Only
        terminal conditions (cannot_solve without any actionable fix, success,
        timeout, provider errors, no_command) break the loop.
        """
        if self.no_plan:
            return 0
        if self.build_repo_agent is None:
            await self.initialize_agents()
            if self.no_plan:
                return 0

        phase2_tooling_errors = 0
        phase2_no_command_errors = 0
        phase2_build_count = 0
        while True:
            if self.shell_agent and len(self._build_command_records()) >= max_total_commands:
                logging.getLogger("agent_build").info(
                    "Total command limit reached (%d/%d)", len(self._build_command_records()), max_total_commands
                )
                break
            if phase2_build_count >= max_rounds:
                logging.getLogger("agent_build").info(
                    "Phase2 build budget exhausted (%d/%d phase2 builds)",
                    phase2_build_count, max_rounds,
                )
                break
            phase2_build_count += 1
            self._phase2_attempt_state.clear()
            self._phase2_attempt_state["failed"] = False

            pre_command_count = len(self._build_command_records()) if self.shell_agent is not None else 0
            execution_state = {
                "current_sysuse": self.sysuse,
                "environment_recreated": False,
                "repeat_guard": "",
            }
            if self._sysuse_switched_from:
                execution_state["environment_recreated"] = True
                self._sysuse_switched_from = None
            if self._phase2_continuation_note:
                execution_state["bounded_continuation"] = {
                    "instruction": self._phase2_continuation_note,
                    "allowed_alternatives": self._phase2_bounded_alternatives(),
                }
                self._phase2_continuation_note = ""

            # A repeated command is a controller event. Do not hand the same
            # command back to Phase 2 and rely on a prompt warning.
            loop_hint = self._detect_command_loop()
            # A typed repair is only a plan mutation until Phase 2 executes
            # it. Let that first changed-environment attempt run once.
            if loop_hint and not self._pending_typed_repair:
                advance = await self._advance_repeated_command(loop_hint)
                if advance == "applied":
                    phase2_build_count -= 1
                    continue
                if advance == "provider_error":
                    await self._abandon_pending_typed_repair("provider_error")
                    break
                if await self._request_phase2_bounded_continuation(
                    self._latest_diagnosis_route_object(),
                    error_text=loop_hint,
                    failed_command=self._latest_failed_command_record(),
                    reason="repeat_guard_exhausted",
                ):
                    phase2_build_count -= 1
                    continue
                self.cannot_solve = 1
                self.error_summary.append("Repeated build command has no remaining matrix action or Build Plan alternative.")
                await self._abandon_pending_typed_repair("repeat_guard_exhausted")
                break

            output = await call_build_agent(
                SHELL_INTERACT_USR.format(
                    inputed_cmds=self._recent_command_summary(),
                    execution_state=json.dumps(execution_state, ensure_ascii=False, indent=2),
                ),
                self.build_repo_agent,
            )
            post_command_count = len(self._build_command_records()) if self.shell_agent is not None else 0
            attempt_records = (
                self._build_command_records()[pre_command_count:post_command_count]
                if self.shell_agent is not None and post_command_count >= pre_command_count
                else []
            )
            attempt_failed_record = next(
                (
                    item for item in reversed(attempt_records)
                    if item.get("status") in {"failed", "timeout"}
                    or str(item.get("exit_code")) not in {"0", "None"}
                ),
                None,
            )
            if _looks_like_provider_error(output):
                self._handle_provider_failure(output, "Build executor")
                await self._abandon_pending_typed_repair("provider_error")
                break

            if post_command_count == pre_command_count and "Compilation succeeded" not in output and output.strip() != "done":
                phase2_no_command_errors += 1
                detail = "Build executor returned without executing a Docker command."
                self._record_fix_attempt(
                    error_text=output,
                    solution_kind="phase2_no_command_retry",
                    solution_detail=detail,
                    result="retry" if phase2_no_command_errors <= 1 else "failed",
                    failed_command=self._latest_failed_command_record(),
                )
                if phase2_no_command_errors <= 1:
                    # The executor may have emitted an invalid final response
                    # before using a tool. Recreate its scratchpad and give the
                    # same controller-selected plan one bounded retry.
                    self.build_repo_agent = await self._create_phase2_agent()
                    phase2_build_count -= 1
                    continue

                next_entry = await self._switch_to_next_build_entry()
                if next_entry:
                    self._record_fix_attempt(
                        error_text=output,
                        solution_kind="no_command_build_entry_fallback",
                        solution_detail=json.dumps(next_entry, ensure_ascii=False),
                        result="applied",
                        failed_command=self._latest_failed_command_record(),
                    )
                    phase2_no_command_errors = 0
                    phase2_build_count -= 1
                    continue

                if await self._request_phase2_bounded_continuation(
                    None,
                    error_text=output,
                    failed_command=self._latest_failed_command_record(),
                    reason="no_command_exhausted",
                ):
                    phase2_build_count -= 1
                    continue

                self.no_command = 1
                self.error_summary.append(
                    "Build executor produced no command after a bounded retry, "
                    "and no untried build entry remains."
                )
                await self._abandon_pending_typed_repair("no_command")
                break

            if post_command_count > pre_command_count:
                phase2_no_command_errors = 0

            if _looks_like_phase2_retryable_error(output):
                phase2_tooling_errors += 1
                self.error_summary.append(
                    "Build executor agent/tooling failure. Retry the build executor directly instead of invoking the repair planner.\n"
                    f"{output}"
                )
                self._record_fix_attempt(
                    error_text=self._last_error_content(output),
                    solution_kind="build_executor_tooling_retry",
                    solution_detail="Retry build executor after tool/validation failure.",
                    result="failed" if phase2_tooling_errors >= 2 else "retry",
                    failure=output,
                    failed_command=self._latest_failed_command_record(),
                )
                if phase2_tooling_errors >= 2:
                    self.cannot_solve = 1
                    await self._abandon_pending_typed_repair("tooling_error")
                    break
                # Tooling retry does NOT consume a real phase2 build budget — refund it.
                phase2_build_count -= 1
                continue
            phase2_tooling_errors = 0

            if "Compilation succeeded" in output or output.strip() == "done":
                latest = self._latest_build_command_record()
                if latest and str(latest.get("exit_code")) not in ("0", "None"):
                    self.error_summary.append(
                        f"Agent claimed success but last command exit_code={latest.get('exit_code')}. Continuing."
                    )
                else:
                    verdict = self._validate_build_success()
                    if verdict.get("success"):
                        # Credit the card-driven path if the previous iteration
                        # applied a card-derived fix and this Phase 2 build now
                        # succeeds off the back of it.
                        if self._last_fix_was_card_driven:
                            self.card_succeeded += 1
                        await self._settle_pending_typed_repair(build_succeeded=True)
                        success_records = self._build_command_records()[pre_command_count:post_command_count]
                        self.build_plan_state = record_build_attempt(
                            self.build_plan_state,
                            {
                                "kind": "build",
                                "result": "done",
                                "sysuse": normalize_sysuse(self.sysuse),
                                "selected": dict(self.build_plan_state.get("selected") or {}),
                                "commands": [
                                    {
                                        "command": item.get("command", ""),
                                        "status": item.get("status", ""),
                                        "exit_code": item.get("exit_code"),
                                        "log_path": item.get("log_path", ""),
                                    }
                                    for item in success_records
                                ],
                                "validation": verdict,
                            },
                        )
                        try:
                            save_build_plan(self.build_plan_state, self.build_plan_path)
                        except OSError:
                            pass
                        self.if_done = True
                        break
                    else:
                        self.error_summary.append(
                            f"Validation rejected: {verdict.get('reason', 'unknown')}. Retrying."
                        )

            stall_reason = self._phase2_stall_reason(attempt_records)
            if stall_reason:
                self._record_fix_attempt(
                    error_text=output,
                    solution_kind="phase2_stall_guard",
                    solution_detail=stall_reason,
                    result="failed",
                    failed_command=self._latest_failed_command_record(),
                    hypothesis="phase2_stall",
                )
                next_entry = await self._switch_to_next_build_entry()
                if next_entry:
                    self._record_fix_attempt(
                        error_text=output,
                        solution_kind="phase2_stall_build_entry_fallback",
                        solution_detail=json.dumps(next_entry, ensure_ascii=False),
                        result="applied",
                        failed_command=self._latest_failed_command_record(),
                        hypothesis="phase2_stall_build_entry",
                    )
                    phase2_build_count -= 1
                    continue
                if await self._request_phase2_bounded_continuation(
                    None,
                    error_text=output,
                    failed_command=self._latest_failed_command_record(),
                    reason="phase2_stall_exhausted",
                ):
                    phase2_build_count -= 1
                    continue
                self.cannot_solve = 1
                self.error_summary.append(stall_reason)
                await self._abandon_pending_typed_repair("phase2_stall_guard")
                break

            latest = self._latest_build_command_record()
            if latest and latest.get("status") == "timeout":
                if self._command_timeout_is_terminal(latest):
                    self.timeout = True
                    await self._abandon_pending_typed_repair("timeout")
                    break
                self.error_summary.append(
                    "Non-terminal setup/configure command timed out; routing it as build evidence."
                )

            error_content = str(
                (attempt_failed_record or {}).get("stderr")
                or (attempt_failed_record or {}).get("output")
                or output
            )
            if self.ablation_mode == "no_phase3":
                self._retire_previous_repair_after_build_failure()
            command_records = self._build_command_records() if self.shell_agent is not None else []
            selected_steps = list((self.build_plan_state.get("selected") or {}).get("command_steps") or [])
            self.latest_failure_frame = extract_failure_frame_from_records(
                attempt_records,
                fallback_text=error_content,
                selected_command_steps=selected_steps,
            )
            self.latest_failure_frame["sysuse"] = normalize_sysuse(self.sysuse)
            self.latest_failure_frame["container_generation"] = self._container_generation
            if self.latest_failure_frame.get("attempt_status") == "incomplete_attempt":
                self._incomplete_attempts += 1
                self._record_fix_attempt(
                    error_text="",
                    solution_kind="incomplete_attempt",
                    solution_detail=json.dumps(
                        self.latest_failure_frame.get("evidence_requests") or [],
                        ensure_ascii=False,
                    ),
                    result="retry" if self._incomplete_attempts < 2 else "failed",
                    failed_command=self.latest_failure_frame.get("primary_command"),
                )
                if self._incomplete_attempts >= 2:
                    self.cannot_solve = 1
                    self.error_summary.append(
                        "Build command records contain no non-zero failure and no verified build postcondition."
                    )
                    await self._abandon_pending_typed_repair("incomplete_attempt")
                    break
                continue
            self._incomplete_attempts = 0
            failed_command = (
                self.latest_failure_frame.get("primary_command")
                or self.latest_failure_frame.get("failed_command")
                or {}
            )
            extract_error_lines = render_failure_frame_text(self.latest_failure_frame, error_content)
            self.build_plan_state = update_build_plan_from_failure(
                self.build_plan_state,
                self.latest_failure_frame,
                failed_command,
            )
            self.build_plan_state = record_build_attempt(
                self.build_plan_state,
                {
                    "kind": "build",
                    "result": "failed",
                    "sysuse": normalize_sysuse(self.sysuse),
                    "selected": dict(self.build_plan_state.get("selected") or {}),
                    "commands": [
                        {
                            "command": item.get("command", ""),
                            "status": item.get("status", ""),
                            "exit_code": item.get("exit_code"),
                            "log_path": item.get("log_path", ""),
                        }
                        for item in attempt_records
                    ],
                    "failure_frame": compact_failure_frame_for_prompt(self.latest_failure_frame),
                },
            )
            await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
            recent_commands = self._recent_command_summary()
            self.latest_diagnosis_route = {}

            if self.ablation_mode == "no_phase3":
                generic_phase3_task = GENERIC_PHASE3_TASK.format(
                    repair_context=json.dumps(
                        self._emergency_prompt_context(
                            extract_error_lines,
                            failed_command,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                generic_phase3_agent = await GenericPhase3AgentFactory.create(
                    self.container_repo_root,
                    self.sysuse,
                    self.shell_agent,
                )
                fixer_output = await call_generic_agent(generic_phase3_task, generic_phase3_agent)
                if _looks_like_provider_error(fixer_output):
                    self._handle_provider_failure(fixer_output, "Generic repair planner")
                    break
                fix_plan = parse_fix_plan(fixer_output)
                diagnosis_route = DiagnosisRoute()
                self._last_fix_was_card_driven = False
                self.error_summary.append(fix_plan.reason or fixer_output)
                proposed_detail = (
                    f"sysuse={fix_plan.sysuse or 'null'}; commands={fix_plan.commands}"
                    if fix_plan.commands or fix_plan.sysuse
                    else f"source_dependency={fix_plan.source_dependency}"
                    if fix_plan.source_dependency
                    else (fix_plan.reason or fixer_output)
                )
                self._record_fix_attempt(
                    error_text=extract_error_lines,
                    solution_kind=(
                        "sysuse_switch" if fix_plan.sysuse and not fix_plan.commands
                        else "repair_commands" if fix_plan.commands
                        else "source_dependency" if fix_plan.source_dependency
                        else "cannot_solve"
                    ),
                    solution_detail=proposed_detail,
                    result="proposed",
                    classification_category=diagnosis_route.category,
                    classification_summary=diagnosis_route.summary,
                    failed_command=failed_command,
                    hypothesis=fix_plan.hypothesis,
                    action_signature=self._fix_plan_action_signature(fix_plan, failed_command),
                )
            else:
                # Rules and policy-card patterns classify before matrix lookup;
                # the constrained LLM is used only for unknown/conflicting types.
                rule_route = route_failure(
                    failure_frame=self.latest_failure_frame,
                    failed_command=failed_command,
                    build_plan=compact_build_plan_for_prompt(self.build_plan_state),
                )
                diagnosis_cards: List[dict[str, Any]] = []
                if not os.environ.get("THEAUTO_NO_CARD_DRIVEN"):
                    try:
                        causal_error_lines = [
                            str(item.get("text") or "")
                            for item in _diagnosis_evidence_bundle(self.latest_failure_frame)
                            if str(item.get("text") or "").strip()
                        ]
                        diagnosis_cards = _policy_kb().diagnose(
                            error_lines=causal_error_lines,
                            sysuse=self.sysuse,
                            failed_phase=rule_route.failed_phase,
                            top_k=15,
                        )
                    except Exception as exc:
                        print(f"[DiagnosisFusion] card diagnosis raised ({exc}); continuing with rules")
                diagnosis_route = fuse_diagnosis(
                    self.latest_failure_frame,
                    failed_command,
                    self.build_plan_state,
                    diagnosis_cards,
                )
                if diagnosis_route.needs_llm_arbitration:
                    arbitration_raw, arbitration = await _request_diagnosis_arbitration(
                        self.latest_failure_frame,
                        diagnosis_route,
                    )
                    if _looks_like_provider_error(arbitration_raw):
                        self._handle_provider_failure(arbitration_raw, "Diagnosis arbitrator")
                        break
                    diagnosis_route = fuse_diagnosis(
                        self.latest_failure_frame,
                        failed_command,
                        self.build_plan_state,
                        diagnosis_cards,
                        llm_result=(
                            arbitration
                            if arbitration is not None
                            else {
                                "classification_status": "insufficient_evidence",
                                "failure_domain": "unknown",
                                "canonical_error_type": "unknown",
                                "confidence": "low",
                                "reason": "invalid strict JSON arbitration output",
                            }
                        ),
                    )
                self.diagnosis_card_match_count += len(diagnosis_cards)
                if diagnosis_route.card_category_support is True:
                    self.diagnosis_card_category_support_count += 1
                print(
                    "[DiagnosisFusion] "
                    f"rule={rule_route.canonical_error_type} cards={len(diagnosis_cards)} "
                    f"source={diagnosis_route.classification_source} "
                    f"final={diagnosis_route.canonical_error_type}"
                )
                diagnosis_probe = run_provenance_probes(
                    diagnosis_route,
                    self.latest_failure_frame,
                    self.build_plan_state,
                    self.compile_path,
                    self._build_command_records(),
                )
                if diagnosis_probe.get("provider_lookup_required"):
                    await self._populate_provider_evidence(
                        diagnosis_route,
                        include_conditional=True,
                    )
                    diagnosis_probe = run_provenance_probes(
                        diagnosis_route,
                        self.latest_failure_frame,
                        self.build_plan_state,
                        self.compile_path,
                        self._build_command_records(),
                    )
                diagnosis_route = apply_probe_refinement(diagnosis_route, diagnosis_probe)
                failure_identity = str((diagnosis_route.failure_identity or {}).get("identity_key") or "")
                satisfied_event = next((
                    item for item in reversed(self.build_plan_state.get("provider_probe_events") or [])
                    if isinstance(item, dict)
                    and item.get("failure_identity") == failure_identity
                    and item.get("status") == "satisfied_pending_rebuild"
                ), None)
                if satisfied_event is not None:
                    satisfied_event["status"] = "rebuild_requested"
                    await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
                    continue
                failure_scope = self._failure_scope_key(diagnosis_route.failure_identity)
                await self._settle_pending_typed_repair(failure_identity)
                if self._pending_typed_repair:
                    await self._sync_build_plan_to_phase2_agent(recreate_agent=True)
                    continue
                self._retire_previous_repair_after_build_failure(failure_scope)
                diagnosis_route = self._apply_exhausted_matrix_actions(diagnosis_route)
                self.latest_diagnosis_route = diagnosis_route.to_dict()
                failed_phase = diagnosis_route.failed_phase or "unknown"
                if diagnosis_route.classification_status == "out_of_matrix":
                    if failure_scope != self._matrix_empty_identity:
                        self._matrix_empty_identity = failure_scope
                        self._consecutive_matrix_empty = 0
                    self._consecutive_matrix_empty += 1
                    exploration = await self._explore_matrix_empty_alternative(
                        diagnosis_route,
                        error_text=extract_error_lines,
                        failed_command=failed_command,
                        recent_commands=recent_commands,
                    )
                    if exploration.get("status") == "provider_error":
                        self._handle_provider_failure(str(exploration.get("raw") or ""), "Build Plan alternative explorer")
                        await self._abandon_pending_typed_repair("provider_error")
                        break
                    if exploration.get("status") == "applied":
                        continue
                    if self._consecutive_matrix_empty >= 2:
                        if await self._request_phase2_bounded_continuation(
                            diagnosis_route,
                            error_text=extract_error_lines,
                            failed_command=failed_command,
                            reason="out_of_matrix_exhausted",
                        ):
                            phase2_build_count -= 1
                            continue
                        self.cannot_solve = 1
                        reason = (
                            "Failure is outside the repair matrix and no supported Build Plan "
                            f"alternative remains (domain={diagnosis_route.failure_domain})."
                        )
                        self.error_summary.append(reason)
                        await self._abandon_pending_typed_repair("matrix_exhausted")
                        break
                    self.error_summary.append("out_of_matrix: first evidence/provenance exploration produced no operation; retrying once")
                    continue
                if (diagnosis_route.repair_scope or {}).get("matrix_empty"):
                    if failure_scope != self._matrix_empty_identity:
                        self._matrix_empty_identity = failure_scope
                        self._consecutive_matrix_empty = 0
                    self._consecutive_matrix_empty += 1
                    diagnosis_route.repair_scope["consecutive_matrix_empty"] = self._consecutive_matrix_empty
                    self.latest_diagnosis_route = diagnosis_route.to_dict()
                    exploration = await self._explore_matrix_empty_alternative(
                        diagnosis_route,
                        error_text=extract_error_lines,
                        failed_command=failed_command,
                        recent_commands=recent_commands,
                    )
                    if exploration.get("status") == "provider_error":
                        self._handle_provider_failure(
                            str(exploration.get("raw") or ""),
                            "Build Plan alternative explorer",
                        )
                        await self._abandon_pending_typed_repair("provider_error")
                        break
                    if exploration.get("status") == "applied":
                        continue
                    exploration_reason = str(
                        exploration.get("reason")
                        or "Build Plan alternative exploration produced no executable selection."
                    )
                    if self._consecutive_matrix_empty >= 2:
                        if await self._request_phase2_bounded_continuation(
                            diagnosis_route,
                            error_text=extract_error_lines,
                            failed_command=failed_command,
                            reason="matrix_empty_exhausted",
                        ):
                            phase2_build_count -= 1
                            continue
                        self.cannot_solve = 1
                        self.error_summary.append(
                            f"Offline matrix and Build Plan alternatives are exhausted for "
                            f"{diagnosis_route.failed_phase}/{diagnosis_route.canonical_error_type}. "
                            f"{exploration_reason}"
                        )
                        await self._abandon_pending_typed_repair("matrix_exhausted")
                        break
                    self.error_summary.append("matrix_empty: first alternative/provenance exploration produced no operation; retrying once")
                    continue
                self._consecutive_matrix_empty = 0
                self._matrix_empty_identity = ""
                # Walk the remaining actions for this exact failure without an
                # unchanged Phase 2 rebuild between failed concretizations.
                repair_applied = False
                while (diagnosis_route.repair_scope or {}).get("selected_matrix_action"):
                    result = await self._concretize_selected_matrix_action(
                        diagnosis_route,
                        error_text=extract_error_lines,
                        failed_command=failed_command,
                        recent_commands=recent_commands,
                        failed_phase=failed_phase,
                    )
                    if result.get("status") == "provider_error":
                        self._handle_provider_failure(str(result.get("raw") or ""), "Repair planner")
                        await self._abandon_pending_typed_repair("provider_error")
                        break
                    if result.get("status") == "applied":
                        repair_applied = True
                        break
                    advance = await self._advance_after_selected_action_failure(
                        diagnosis_route,
                        error_text=extract_error_lines,
                        failed_command=failed_command,
                    )
                    if advance == "matrix_action":
                        continue
                    if advance == "fallback":
                        self._last_fix_was_card_driven = False
                        repair_applied = True
                    break

                if self.provider_error:
                    await self._abandon_pending_typed_repair("provider_error")
                    break
                if repair_applied:
                    continue
                self._last_fix_was_card_driven = False
                if await self._request_phase2_bounded_continuation(
                    diagnosis_route,
                    error_text=extract_error_lines,
                    failed_command=failed_command,
                    reason="matrix_actions_exhausted",
                ):
                    phase2_build_count -= 1
                    continue
                self.cannot_solve = 1
                self.error_summary.append(
                    "Selected Matrix actions and supported Build Plan alternatives are exhausted."
                )
                await self._abandon_pending_typed_repair("matrix_exhausted")
                break

            # The remaining command-oriented path is reserved for no_phase3.
            switched_sysuse = False
            pending_sysuse = fix_plan.sysuse if fix_plan.sysuse and not fix_plan.commands else None

            if pending_sysuse and normalize_sysuse(pending_sysuse) != normalize_sysuse(self.sysuse):
                # Guard against infinite sysuse loops: if we already attempted a
                # switch to this exact sysuse, treat as cannot_solve instead of
                # re-switching.
                already_tried = any(
                    h.get("solution_kind") == "sysuse_switch"
                    and normalize_sysuse(
                        h.get("solution_detail", "").replace("switch to ", "")
                    )
                    == normalize_sysuse(pending_sysuse)
                    for h in self.fix_history
                )
                if already_tried:
                    self.cannot_solve = 1
                    self.error_summary.append(
                        f"sysuse {pending_sysuse} already tried; cannot_solve"
                    )
                    self._record_fix_attempt(
                        error_text=extract_error_lines,
                        solution_kind="cannot_solve",
                        solution_detail=f"sysuse {pending_sysuse} already tried",
                        result="final",
                        failed_command=failed_command,
                        hypothesis=fix_plan.hypothesis,
                        action_signature=self._fix_plan_action_signature(fix_plan, failed_command),
                    )
                    await self._abandon_pending_typed_repair("repeated_sysuse")
                    break
                switched_sysuse = bool(await self.switch_sysuse(pending_sysuse))
                if not switched_sysuse:
                    self.error_summary.append(
                        f"sysuse {pending_sysuse} rejected: required dependency provider unavailable"
                    )
                    continue
                self._record_fix_attempt(
                    error_text=extract_error_lines,
                    solution_kind="sysuse_switch",
                    solution_detail=f"switch to {pending_sysuse}",
                    result="applied",
                    failed_command=failed_command,
                    hypothesis=fix_plan.hypothesis,
                    action_signature=self._fix_plan_action_signature(fix_plan, failed_command),
                )
                # After a sysuse switch, do NOT process commands/source-deps in
                # this iteration — the next loop iteration's Phase 2 build will
                # run fresh in the new container.
                continue

            # Hard dedup: stop only for materially identical actions under the
            # same sysuse. The same hypothesis label may still be valid after a
            # suite switch or with a different command/source-dep action.
            if self._has_prior_equivalent_fix(fix_plan, failed_command):
                advance = await self._advance_after_selected_action_failure(
                    diagnosis_route,
                    error_text=extract_error_lines,
                    failed_command=failed_command,
                )
                if advance in {"matrix_action", "fallback"}:
                    phase2_build_count -= 1
                    self.error_summary.append(
                        "Equivalent repair already failed; advancing to the next selected action."
                    )
                    continue
                self.cannot_solve = 1
                self.error_summary.append(
                    "Equivalent repair action already tried under the same selected build decisions. Stopping."
                )
                await self._abandon_pending_typed_repair("duplicate_repair")
                break

            if fix_plan.source_dependency:
                registered = await self._register_source_dependency_plan(fix_plan.source_dependency)
                self._record_fix_attempt(
                    error_text=extract_error_lines,
                    solution_kind="source_dependency",
                    solution_detail=json.dumps(fix_plan.source_dependency, ensure_ascii=False),
                    result="applied" if registered else "failed",
                    failed_command=failed_command,
                    hypothesis=fix_plan.hypothesis,
                    action_signature=self._fix_plan_action_signature(fix_plan, failed_command),
                )
                if registered and not fix_plan.commands:
                    # Source dep plan registered into project_info; phase2 will pick
                    # it up on the next build. Do NOT consume a round for the
                    # registration itself — the next loop iteration's phase2 build
                    # is what counts.
                    continue

            if fix_plan.status == "cannot_solve" and not switched_sysuse and not fix_plan.commands:
                advance = await self._advance_after_selected_action_failure(
                    diagnosis_route,
                    error_text=extract_error_lines,
                    failed_command=failed_command,
                )
                if advance in {"matrix_action", "fallback"}:
                    phase2_build_count -= 1
                    self.error_summary.append(
                        "Selected matrix action could not be concretized; advancing to the next action."
                    )
                    continue
                self.cannot_solve = 1
                self._record_fix_attempt(
                    error_text=extract_error_lines,
                    solution_kind="cannot_solve",
                    solution_detail=fix_plan.reason or "[no reason]",
                    result="final",
                    failed_command=failed_command,
                    hypothesis=fix_plan.hypothesis,
                    action_signature=self._fix_plan_action_signature(fix_plan, failed_command),
                    repair_action_override=fix_plan.repair_action,
                    fix_direction_override=fix_plan.fix_direction,
                )
                await self._abandon_pending_typed_repair("cannot_solve")
                break

            if fix_plan.commands:
                while fix_plan.commands:
                    execution = await self._execute_fix_commands(list(fix_plan.commands))
                    repair_failed_command = execution.get("failed_command") or failed_command
                    repair_failure = str(
                        (execution.get("failed_command") or {}).get("stderr")
                        or (execution.get("failed_command") or {}).get("output")
                        or ""
                    )
                    self._record_fix_attempt(
                        error_text=extract_error_lines,
                        solution_kind="repair_commands",
                        solution_detail="\n".join(fix_plan.commands),
                        result="applied" if execution.get("success") else "failed",
                        failure=repair_failure,
                        failed_command=repair_failed_command,
                        hypothesis=fix_plan.hypothesis,
                        action_signature=self._fix_plan_action_signature(fix_plan, repair_failed_command),
                    )
                    self.build_plan_state = set_pending_fix_plan(self.build_plan_state, [])
                    if execution.get("success"):
                        break
                    if self.timeout:
                        break
                    if fix_plan.hypothesis.startswith("card_"):
                        self._tried_card_ids.add(fix_plan.hypothesis[5:])

                    next_card_plan = None
                    if (
                        fix_plan.hypothesis.startswith("card_")
                        and not os.environ.get("THEAUTO_NO_CARD_DRIVEN")
                    ):
                        retry = await self._try_card_driven_repair(
                            error_lines=extract_error_lines,
                            failed_command=failed_command,
                            recent_commands=recent_commands,
                            failed_phase=failed_phase,
                            diagnosis_route=diagnosis_route,
                        )
                        in_container = (retry or {}).get("in_container_strategies") or []
                        next_card_plan = in_container[0] if in_container else None
                    if next_card_plan and next_card_plan.commands:
                        fix_plan = next_card_plan
                        continue

                    advance = await self._advance_after_selected_action_failure(
                        diagnosis_route,
                        error_text=extract_error_lines,
                        failed_command=repair_failed_command,
                    )
                    if advance in {"matrix_action", "fallback"}:
                        phase2_build_count -= 1
                    else:
                        self.cannot_solve = 1
                    break

                if self.timeout or self.cannot_solve:
                    break
                # A successful repair or an action/fallback transition both
                # require another Phase 2 build; failed repair commands do not.
                continue
            elif not switched_sysuse:
                self.cannot_solve = 1
                await self._abandon_pending_typed_repair("cannot_solve")
                break

        if self._pending_typed_repair and not self.if_done:
            await self._abandon_pending_typed_repair("controller_exit")
        if not self.if_done and not self.timeout and not self.cannot_solve and not self.provider_error and not self.no_command:
            self.cannot_solve = 1
        return len(self._build_command_records()) if self.shell_agent is not None else 0


def _resolve_ablation_mode(ablation: str = "") -> str:
    mode = (ablation or os.environ.get("THEAUTO_ABLATION") or "none").strip().lower()
    if mode not in _ABLATION_MODES:
        raise ValueError(f"unsupported ablation mode: {mode}")
    return mode


async def Start(compile_path_append, sysuse, max_turns: int = 5, ablation: str = ""):
    compile_path, commit = compile_path_append.split("##")
    owner = compile_path.split("/")[-1]
    repo = compile_path.split("/")[-2]
    Start_LLM_Logging(f"{owner}##{repo}##{commit}")
    starttime = time.time()
    ablation_mode = _resolve_ablation_mode(ablation)
    print(f"[Ablation] mode={ablation_mode}")

    # Auto-select sysuse from commit date if "auto" or empty
    if not sysuse or sysuse.lower() == "auto":
        sysuse = _commit_date_to_sysuse(compile_path, commit)
        print(f"[AutoSysuse] Selected {sysuse} based on commit date")

    # === Phase-0: Network pre-flight ===
    # Abort early if github.com / TLS endpoints are unreachable. Many build
    # failures (cargo/gem/git submodule) are caused by TLS/DNS issues that
    # no amount of agent retrying can fix. Skip with THEAUTO_SKIP_NETWORK_CHECK=1.
    if not os.environ.get("THEAUTO_SKIP_NETWORK_CHECK"):
        reachable, reason = _check_network_reachable()
        if not reachable:
            elapsed = round(time.time() - starttime, 2)
            print(f"[NetworkPreFlight] github.com unreachable: {reason}; aborting task")
            return f"network_failure {elapsed} try: 0"
        if reason:
            print(f"[NetworkPreFlight] {reason}")

    BTarget = BuildAutomation(compile_path, commit, sysuse, ablation_mode=ablation_mode)
    print(compile_path)
    try:
        commit_date = _get_commit_date(compile_path, commit)
        BTarget.submodule_preflight = preflight_git_submodules(
            compile_path,
            BTarget.agent_log_dir,
            project_commit=commit,
            project_commit_date=commit_date,
        )
        host_clone_root = source_dep_resolver_compute_host_clone_root(compile_path)
        if ablation_mode == "no_phase1":
            BTarget.phase1_status = "skipped"
            BTarget.docs = ""
            BTarget.urlsummaries = {}
            BTarget.cloned_source_dep_plan = []
            BTarget.pending_source_dep_candidates = []
            BTarget.dependency_selected_sysuse = ""
            print("[Ablation] no_phase1: skipping enhanced deps, doc scanner, source dep resolver, and ReadRepo")
        else:
            enhanced_dep_save_path = os.path.join(BTarget.agent_log_dir, "enhanced_dep_result.json")
            BTarget.phase1_status = "running"
            extractor_started_at = time.time()
            try:
                enhanced_result = await get_enhanced_dependencies(
                    compile_path,
                    BTarget.agent_log_dir,
                    enhanced_dep_save_path,
                    debian_version=BTarget.sysuse,
                )
            except Exception as exc:
                enhanced_result = (
                    [],
                    {"schema_version": 2, "status": "error", "error": f"{type(exc).__name__}: {exc}"},
                    enhanced_dep_save_path,
                    None,
                )
                BTarget.phase1_errors.append({
                    "producer": "enhanced_dep_extractor",
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                })
            _deps, _loaded_result, _dep_source_path, selected_distro = enhanced_result
            extractor_project_status = (
                _project_extractor_status(_loaded_result, compile_path)
                if isinstance(_loaded_result, dict)
                else {}
            )
            extractor_status = str(
                extractor_project_status.get("status")
                or (_loaded_result or {}).get("status")
                or "missing"
            )
            extractor_usable = isinstance(_loaded_result, dict) and _enhanced_result_usable(
                _loaded_result,
                compile_path,
            )
            extractor_attempt_status = (
                "ok"
                if extractor_usable
                else "error" if extractor_status in {"error", "failed", "timeout", "missing"}
                else "degraded"
            )
            BTarget.phase1_extractor_attempts.append({
                "attempt": 1,
                "status": extractor_attempt_status,
                "producer_status": extractor_status,
                "elapsed_sec": round(time.time() - extractor_started_at, 2),
                "result_path": _dep_source_path or enhanced_dep_save_path,
                "error": str(
                    extractor_project_status.get("error")
                    or (_loaded_result or {}).get("error")
                    or ""
                )[:500],
            })
            BTarget.dependency_selected_sysuse = selected_distro or ""
            if selected_distro and normalize_sysuse(selected_distro) != normalize_sysuse(BTarget.sysuse):
                old_sysuse = BTarget.sysuse
                BTarget.sysuse = f"debian:{normalize_sysuse(selected_distro)}"
                BTarget.phase1_selected_sysuse = normalize_sysuse(BTarget.sysuse)
                BTarget._sysuse_switched_from = old_sysuse
                print(f"[EnhancedDepExtractor] Selected sysuse {BTarget.sysuse} from dependency constraints")
            elif selected_distro:
                BTarget.phase1_selected_sysuse = normalize_sysuse(BTarget.sysuse)

            # Run doc+scanner BEFORE docker init — it does not need docker, and we
            # need its `artifact_summaries` (LLM-extracted source_deps) to drive
            # source_dep_resolver below.
            try:
                await get_relateds_from_docs_and_scanner(BTarget, enhanced_result=enhanced_result)
            except Exception as exc:
                BTarget.phase1_status = "error"
                BTarget.phase1_errors.append({
                    "producer": "phase1_orchestrator",
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                })
                BTarget.plan_error = f"Phase 1 failed: {type(exc).__name__}: {exc}"
                raise
            producer_states = {
                str(item.get("status") or "")
                for item in BTarget.phase1_producers.values()
                if isinstance(item, dict)
            }
            for producer, producer_result in BTarget.phase1_producers.items():
                if not isinstance(producer_result, dict):
                    continue
                if producer_result.get("status") not in {"degraded", "error"}:
                    continue
                BTarget.phase1_errors.append({
                    "producer": str(producer),
                    "type": str(producer_result.get("status") or "error"),
                    "message": str(producer_result.get("error") or "producer returned no usable result")[:500],
                })
            BTarget.phase1_status = "degraded" if producer_states.intersection({"degraded", "error"}) else "ok"

            # ===== Phase-1 source dependency resolver =====
            # Clone external source deps that apt cannot satisfy, run mini-Phase-1
            # on each to extract build_commands, then mount the clone root into
            # the container so Phase 2 can build them in order.
            try:
                cargo_refs: list[dict] = []
                build_hints_path = os.path.join(BTarget.agent_log_dir, "build_hints.json")
                if os.path.isfile(build_hints_path):
                    try:
                        with open(build_hints_path) as f:
                            cargo_refs = (json.load(f) or {}).get("cargo_git_refs") or []
                    except Exception:
                        pass
                doc_source_deps: list[dict] = []
                for s in (BTarget.urlsummaries.get("artifact_summaries") or []):
                    for sd in (s.get("source_deps") or []):
                        doc_source_deps.append(sd)
                source_dep_candidates = source_dep_resolver_collect_candidates(
                    repo_path=compile_path,
                    doc_source_deps=doc_source_deps,
                    cargo_refs=cargo_refs,
                )
                # Do not clone source dependencies before ReadRepo has chosen
                # an entry. Keep only candidates with an executable contract;
                # unresolved clues remain in the inventory for Phase 3.
                candidates = [
                    candidate for candidate in source_dep_candidates
                    if candidate.clone_url
                    and _source_dependency_has_executable_contract(candidate)
                ]
                unbound_candidates = [
                    candidate for candidate in source_dep_candidates
                    if candidate.clone_url
                    and not _source_dependency_has_executable_contract(candidate)
                ]
                repo_context = BTarget.urlsummaries.get("repo_context") or {}
                known_unbound = {
                    (
                        str(item.get("name") or "").lower(),
                        str(item.get("url") or item.get("repo_url") or "").lower(),
                    )
                    for item in repo_context.get("source_deps") or []
                    if isinstance(item, dict)
                }
                for candidate in unbound_candidates:
                    key = (candidate.name.lower(), candidate.clone_url.lower())
                    if key in known_unbound:
                        continue
                    repo_context.setdefault("source_deps", []).append({
                        "name": candidate.name,
                        "url": candidate.clone_url,
                        "build_commands": [],
                        "status": "unbound",
                        "reason": "source dependency has no explicit build command sequence",
                        "origin": candidate.origin,
                    })
                    known_unbound.add(key)
                if candidates:
                    print(
                        f"[SourceDepResolver] {len(candidates)} candidate(s): "
                        f"{[c.name for c in candidates]}"
                    )
                    BTarget.pending_source_dep_candidates = candidates
                    BTarget.phase1_producers["source_dep_resolver"] = {
                        "status": "pending",
                        "error": "",
                        "candidates": len(candidates),
                        "unbound_candidates": len(unbound_candidates),
                    }
                else:
                    print("[SourceDepResolver] no candidates; skipping")
                    BTarget.pending_source_dep_candidates = []
                    BTarget.phase1_producers["source_dep_resolver"] = {
                        "status": "ok",
                        "error": "",
                        "candidates": 0,
                        "unbound_candidates": len(unbound_candidates),
                    }
            except Exception as e:
                # Source-dep resolution is best-effort — never let it abort the build.
                print(f"[SourceDepResolver] error (continuing without source deps): {e}")
                BTarget.cloned_source_dep_plan = []
                BTarget.pending_source_dep_candidates = []
                BTarget.phase1_producers["source_dep_resolver"] = {
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}"[:500],
                }
                BTarget.phase1_errors.append({
                    "producer": "source_dep_resolver",
                    "type": type(e).__name__,
                    "message": str(e)[:500],
                })
                BTarget.phase1_status = "degraded"
                host_clone_root = source_dep_resolver_compute_host_clone_root(compile_path)

        # Keep the source-dependency mount stable across retries. Executable
        # dependency plans resolved below refer to paths under /source_deps.
        os.makedirs(host_clone_root, exist_ok=True)
        BTarget.extra_volumes = [(host_clone_root, "/source_deps")]

        await BTarget.initialize_docker(extra_volumes=BTarget.extra_volumes or None)
        print("docker shell_agent inited")

        # Legacy sysuse (etch/lenny/squeeze/wheezy/jessie) apt repos have
        # expired GPG keys / valid-until dates. Disable the check before any
        # apt-get install runs, otherwise every install fails with
        # KEYEXPIRED / Release file expired.
        _legacy_codenames = {"etch", "lenny", "squeeze", "wheezy", "jessie"}
        if normalize_sysuse(BTarget.sysuse) in _legacy_codenames:
            await BTarget.shell_agent.execute_command(
                "echo 'Acquire::Check-Valid-Until \"false\";' > /etc/apt/apt.conf.d/99no-valid-until"
            )
            await BTarget.shell_agent.execute_command("apt-get update -qq")
            print(f"[LegacyAptFix] Disabled Check-Valid-Until for {BTarget.sysuse}")

        if ablation_mode == "no_phase1":
            generic_repo_agent = await GenericRepoExplorerAgentFactory.create(
                BTarget.container_repo_root,
                BTarget.sysuse,
                BTarget.shell_agent,
            )
            generic_repo_task = GENERIC_REPO_EXPLORER_TASK.format(
                project_path=BTarget.container_repo_root,
                commit_date=commit_date,
                submodule_info=json.dumps(
                    _compact_submodule_prompt_state(BTarget.submodule_preflight),
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            summary = await call_generic_agent(generic_repo_task, generic_repo_agent)
            BTarget.plan_source = "generic_ablation"
        else:
            repo_context = BTarget.urlsummaries.get("repo_context") or {}
            artifact_summaries = BTarget.urlsummaries.get("artifact_summaries") or []
            await _resolve_debian_package_alternatives(BTarget, artifact_summaries)
            catalog = _build_readrepo_catalog(
                repo_context,
                artifact_summaries,
                repo_path=BTarget.compile_path,
                target_commit=BTarget.commit,
            )
            provider_failure = ""
            if not catalog.get("entries"):
                inventory = repo_context.get("manifest_inventory") or {}
                if inventory.get("files"):
                    try:
                        zero_entry_agent = await ZeroEntryReadRepoAgentFactory.create(
                            BTarget.compile_path,
                            inventory,
                        )
                    except Exception as exc:
                        provider_failure = str(exc)
                        BTarget.readrepo_attempts.append({
                            "attempt": 0,
                            "kind": "zero_entry_discovery",
                            "status": "provider_error",
                            "finish_reason": "",
                            "response_chars": 0,
                            "error": provider_failure[:500],
                        })
                    else:
                        retry_note = ""
                        for attempt_no in range(1, 3):
                            task = ZERO_ENTRY_READ_REPO_TASK.format(
                                project_path=BTarget.compile_path,
                                manifest_inventory=json.dumps(inventory, indent=2, ensure_ascii=False),
                            )
                            if retry_note:
                                task += (
                                    "\nThe controller rejected the previous response: "
                                    f"{retry_note}. Return only corrected raw JSON."
                                )
                            try:
                                raw_discovery = await zero_entry_agent.run(task)
                            except Exception as exc:
                                provider_failure = str(exc)
                                BTarget.readrepo_attempts.append({
                                    "attempt": attempt_no,
                                    "kind": "zero_entry_discovery",
                                    "status": "provider_error",
                                    "finish_reason": "",
                                    "response_chars": 0,
                                    "error": provider_failure[:500],
                                })
                                break
                            BTarget.save_result(
                                f"zero_entry_readrepo_attempt_{attempt_no}.txt",
                                str(raw_discovery),
                            )
                            if _looks_like_provider_error(raw_discovery):
                                provider_failure = str(raw_discovery)
                                BTarget.readrepo_attempts.append({
                                    "attempt": attempt_no,
                                    "kind": "zero_entry_discovery",
                                    "status": "provider_error",
                                    "finish_reason": "",
                                    "response_chars": len(str(raw_discovery)),
                                    "error": provider_failure[:500],
                                })
                                break
                            discovered, discovery_error = _parse_zero_entry_candidates(
                                raw_discovery,
                                inventory,
                                BTarget.compile_path,
                            )
                            status = "selected" if discovered else discovery_error
                            BTarget.readrepo_attempts.append({
                                "attempt": attempt_no,
                                "kind": "zero_entry_discovery",
                                "status": status,
                                "finish_reason": "",
                                "response_chars": len(str(raw_discovery)),
                                "error": "" if discovered else discovery_error[:500],
                            })
                            if discovered:
                                repo_context.setdefault("build_entry_candidates", []).extend(discovered)
                                catalog = _build_readrepo_catalog(
                                    repo_context,
                                    artifact_summaries,
                                    repo_path=BTarget.compile_path,
                                    target_commit=BTarget.commit,
                                )
                                break
                            if discovery_error == "cannot_select":
                                break
                            retry_note = discovery_error
            baseline_plan = _materialize_readrepo_plan(repo_context, catalog)
            baseline_valid, baseline_error = _validate_readrepo_plan(
                json.dumps(baseline_plan, ensure_ascii=False)
            )
            selected_plan: Optional[dict[str, Any]] = None
            selection_declined = False

            entry_count = len(catalog.get("entries") or [])
            # Even a single executable entry can have competing dependency,
            # setup, or source-dependency evidence. Let ReadRepo compose a
            # bounded plan whenever Phase 1 found an entry; the validated
            # baseline remains available if the model declines or is invalid.
            if entry_count > 0:
                import models as _models
                from langchain_core.messages import HumanMessage, SystemMessage

                try:
                    readrepo_model = _models.get_chat_model(
                        "readrepo",
                        temperature=0,
                        top_p=None,
                        max_tokens=4096,
                    )
                except Exception as exc:
                    provider_failure = str(exc)
                    BTarget.readrepo_attempts.append({
                        "attempt": 0,
                        "status": "provider_error",
                        "finish_reason": "",
                        "response_chars": 0,
                        "error": provider_failure[:500],
                    })
                else:
                    retry_note = ""
                    for attempt_no in range(1, 3):
                        readrepo_task = READ_REPO_TASK.format(
                            project_path=BTarget.compile_path,
                            sysuse=BTarget.sysuse,
                            commit_date=commit_date,
                            candidate_catalog=json.dumps(catalog.get("prompt") or {}, indent=2, ensure_ascii=False),
                            submodule_info=json.dumps(
                                _compact_submodule_prompt_state(BTarget.submodule_preflight),
                                indent=2,
                                ensure_ascii=False,
                            ),
                        )
                        if retry_note:
                            readrepo_task += (
                                "\nThe previous response was rejected by the controller: "
                                f"{retry_note}. Return only the required compact JSON object."
                            )
                        try:
                            readrepo_resp = await readrepo_model.ainvoke([
                                SystemMessage(content=READREPO_SYSTEM.format()),
                                HumanMessage(content=readrepo_task),
                            ])
                            raw_summary = (
                                readrepo_resp.content
                                if hasattr(readrepo_resp, "content")
                                else str(readrepo_resp)
                            )
                            response_metadata = getattr(readrepo_resp, "response_metadata", {}) or {}
                            finish_reason = str(
                                response_metadata.get("finish_reason")
                                or response_metadata.get("stop_reason")
                                or ""
                            )
                        except Exception as exc:
                            provider_failure = str(exc)
                            BTarget.readrepo_attempts.append({
                                "attempt": attempt_no,
                                "status": "provider_error",
                                "finish_reason": "",
                                "response_chars": 0,
                                "error": provider_failure[:500],
                            })
                            break

                        BTarget.save_result(f"readrepo_attempt_{attempt_no}.txt", raw_summary)
                        if _looks_like_provider_error(raw_summary):
                            provider_failure = str(raw_summary)
                            BTarget.readrepo_attempts.append({
                                "attempt": attempt_no,
                                "status": "provider_error",
                                "finish_reason": finish_reason,
                                "response_chars": len(str(raw_summary)),
                                "error": provider_failure[:500],
                            })
                            break

                        selection, selection_error = _parse_readrepo_selection(raw_summary, catalog)
                        if selection is None:
                            BTarget.readrepo_attempts.append({
                                "attempt": attempt_no,
                                "status": "invalid",
                                "finish_reason": finish_reason,
                                "response_chars": len(str(raw_summary)),
                                "error": selection_error,
                            })
                            retry_note = selection_error
                            continue

                        if selection.get("status") == "cannot_select":
                            selection_declined = True
                            BTarget.plan_source = "readrepo_declined"
                            BTarget.plan_error = str(selection.get("reason") or "ReadRepo found no supported build entry.")
                            BTarget.readrepo_attempts.append({
                                "attempt": attempt_no,
                                "status": "cannot_select",
                                "finish_reason": finish_reason,
                                "response_chars": len(str(raw_summary)),
                                "error": BTarget.plan_error[:500],
                            })
                            break
                        candidate_plan = _materialize_readrepo_plan(repo_context, catalog, selection)
                        validated_plan, validation_error = _validate_readrepo_plan(
                            json.dumps(candidate_plan, ensure_ascii=False)
                        )
                        if validated_plan is None:
                            BTarget.readrepo_attempts.append({
                                "attempt": attempt_no,
                                "status": "invalid_plan",
                                "finish_reason": finish_reason,
                                "response_chars": len(str(raw_summary)),
                                "error": validation_error,
                            })
                            retry_note = validation_error
                            continue

                        BTarget.readrepo_attempts.append({
                            "attempt": attempt_no,
                            "status": selection.get("status") or "selected",
                            "finish_reason": finish_reason,
                            "response_chars": len(str(raw_summary)),
                            "error": "",
                        })
                        selected_plan = validated_plan
                        BTarget.plan_source = (
                            "readrepo_selected"
                            if selection.get("status") == "selected"
                            else "deterministic_declined"
                        )
                        break
            else:
                BTarget.plan_source = "deterministic_no_entry"

            if selected_plan is None and baseline_valid is not None:
                selected_plan = baseline_valid
                if selection_declined:
                    BTarget.plan_source = "deterministic_declined_fallback"
                elif provider_failure:
                    BTarget.plan_source = "deterministic_provider_fallback"
                else:
                    BTarget.plan_source = "deterministic_invalid_fallback"
                print(
                    f"[ReadRepo] using deterministic fallback ({BTarget.plan_source}); "
                    f"entries={len(catalog.get('entries') or [])}"
                )
                # The rejected model arbitration is retained in
                # readrepo_attempts; it is not a plan error once the validated
                # controller baseline is selected.
                BTarget.plan_error = ""

            BTarget.save_result(
                "readrepo_attempts.json",
                json.dumps(BTarget.readrepo_attempts, indent=2, ensure_ascii=False),
            )
            if selected_plan is None:
                source_dep_state = BTarget.phase1_producers.get("source_dep_resolver") or {}
                if source_dep_state.get("status") == "pending":
                    BTarget.phase1_producers["source_dep_resolver"] = {
                        **source_dep_state,
                        "status": "skipped",
                        "error": "no selected build entry",
                    }
                BTarget.plan_error = BTarget.plan_error or baseline_error or "ReadRepo and repo_context returned no executable build plan."
                if provider_failure:
                    BTarget.provider_error = 1
                    BTarget.summary = provider_failure
                    BTarget.reposummary = provider_failure
                    BTarget.save_result("readrepo.txt", provider_failure)
                    during_time = time.time() - starttime
                    return f"provider_error {during_time} try: 0"
                BTarget.no_plan = 1
                BTarget.summary = ""
                BTarget.reposummary = ""
                BTarget.save_result("readrepo.txt", "")
                print(f"ReadRepo/no_plan: {BTarget.plan_error}")
                during_time = time.time() - starttime
                return f"no_plan {during_time} try: 0"
            summary = json.dumps(selected_plan, ensure_ascii=False)
        if ablation_mode == "no_phase1" and _looks_like_provider_error(summary):
            BTarget.provider_error = 1
            BTarget.summary = summary
            BTarget.reposummary = summary
            success, error = BTarget.save_result("readrepo.txt", BTarget.summary)
            if not success:
                return f"error {error}"
            during_time = time.time() - starttime
            return f"provider_error {during_time} try: 0"
        BTarget.summary = summary
        BTarget.reposummary = summary

        print("BTarget.summary:", summary)
        success, error = BTarget.save_result("readrepo.txt", BTarget.summary)
        if not success:
            return f"error {error}"
        if ablation_mode != "no_phase1":
            validated_plan, plan_error = _validate_readrepo_plan(summary)
            if validated_plan is None:
                BTarget.no_plan = 1
                BTarget.plan_error = plan_error
                print(f"ReadRepo/no_plan: {plan_error}")
                during_time = time.time() - starttime
                return f"no_plan {during_time} try: 0"
            BTarget.reposummary = json.dumps(validated_plan, ensure_ascii=False)
        pending_source_candidates = list(getattr(BTarget, "pending_source_dep_candidates", []) or [])
        if pending_source_candidates:
            try:
                selected_payload = _extract_json_object(BTarget.reposummary) or {}
                selected_source_deps = [
                    item for item in selected_payload.get("source_deps") or []
                    if isinstance(item, dict)
                ]
                selected_names = {
                    str(item.get("name") or "").strip().lower()
                    for item in selected_source_deps
                    if str(item.get("name") or "").strip()
                }
                selected_urls = {
                    str(item.get("url") or item.get("repo_url") or "").strip().lower()
                    for item in selected_source_deps
                    if str(item.get("url") or item.get("repo_url") or "").strip()
                }
                pending_source_candidates = [
                    item for item in pending_source_candidates
                    if item.key() in selected_names
                    or str(item.clone_url or "").strip().lower() in selected_urls
                ]
                if not pending_source_candidates:
                    print("[SourceDepResolver] ReadRepo-selected entry has no source dependency candidate")
                    BTarget.cloned_source_dep_plan = []
                    source_dep_state = BTarget.phase1_producers.get("source_dep_resolver") or {}
                    BTarget.phase1_producers["source_dep_resolver"] = {
                        **source_dep_state,
                        "status": "skipped",
                        "error": "selected build entry does not bind a source dependency candidate",
                        "selected": 0,
                    }
                else:
                    print(f"[SourceDepResolver] resolving {len(pending_source_candidates)} selected-entry candidate(s)")
                    BTarget.cloned_source_dep_plan = await source_dep_resolver_recursive(
                        candidates=pending_source_candidates,
                        host_clone_root=host_clone_root,
                        sysuse=BTarget.sysuse,
                        max_depth=2,
                        project_commit_date=commit_date,
                    )
                    unusable_deps = [
                        item for item in BTarget.cloned_source_dep_plan
                        if not isinstance(item, dict)
                        or item.get("clone_status") != "cloned"
                        or not _source_dependency_has_executable_contract(item)
                    ]
                    BTarget.phase1_producers["source_dep_resolver"] = {
                        "status": "degraded" if unusable_deps else "ok",
                        "error": (
                            f"{len(unusable_deps)} selected source dependencies lack a usable clone/build contract"
                            if unusable_deps else ""
                        ),
                        "candidates": len(pending_source_candidates),
                        "resolved": len(BTarget.cloned_source_dep_plan) - len(unusable_deps),
                    }
                    if unusable_deps:
                        BTarget.phase1_status = "degraded"
                        BTarget.phase1_errors.append({
                            "producer": "source_dep_resolver",
                            "type": "degraded",
                            "message": BTarget.phase1_producers["source_dep_resolver"]["error"],
                        })
                    with open(os.path.join(BTarget.agent_log_dir, "source_dep_plan.json"), "w", encoding="utf-8") as f:
                        json.dump(BTarget.cloned_source_dep_plan, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                print(f"[SourceDepResolver] selected-entry resolution failed: {exc}")
                BTarget.cloned_source_dep_plan = []
                BTarget.phase1_producers["source_dep_resolver"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
                BTarget.phase1_errors.append({
                    "producer": "source_dep_resolver",
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                })
                BTarget.phase1_status = "degraded"
        print("ReadRepo/generic repo分析完成：", compile_path)

        await BTarget.initialize_agents()
        try_time = await BTarget.try_compile_cmd(max_rounds=max_turns)
        build_log_name = f"BuildLog_{try_time}.txt"
        BTarget.last_build_log_path = os.path.join(BTarget.agent_log_dir, build_log_name)
        success, error = BTarget.save_result(build_log_name, BTarget._format_all_command_records())
        if not success:
            return f"error {error}"
        during_time = time.time() - starttime

        AgentTimer.print_summary()
        AgentTimer.save_to_file(BTarget.agent_log_dir)

        # Save untracked LLM usage (DocScanner, artifact summarizer, dep extractor)
        from llm.ask_llm import llm_usage_tracker
        llm_summary = llm_usage_tracker.summary()
        if llm_summary["total_calls"] > 0:
            import json as _json
            with open(os.path.join(BTarget.agent_log_dir, "llm_usage.json"), "w") as f:
                _json.dump(llm_summary, f, indent=2)
            print(f"LLM usage (untracked by AgentTimer): {llm_summary['total_calls']} calls, "
                  f"{llm_summary['total_tokens']} tokens, {llm_summary['total_elapsed']:.1f}s")
        llm_usage_tracker.reset()

        # Card-driven path metrics — used to evaluate ROI of the card path vs
        # the LLM Phase 3 fallback. Persisted to card_driven_stats.json so
        # batch runners can aggregate across builds.
        card_stats = {
            "card_match_count": BTarget.card_match_count,
            "diagnosis_card_match_count": BTarget.diagnosis_card_match_count,
            "diagnosis_card_category_support_count": BTarget.diagnosis_card_category_support_count,
            "card_translate_count": BTarget.card_translate_count,
            "card_succeeded": BTarget.card_succeeded,
            "fallback_to_llm": BTarget.fallback_to_llm,
        }
        try:
            import json as _json_card
            with open(os.path.join(BTarget.agent_log_dir, "card_driven_stats.json"), "w") as _f:
                _json_card.dump(card_stats, _f, indent=2)
        except OSError:
            pass
        print(
            f"[CardDriven] stats: match={card_stats['card_match_count']} "
            f"diagnosis_match={card_stats['diagnosis_card_match_count']} "
            f"category_support={card_stats['diagnosis_card_category_support_count']} "
            f"translate={card_stats['card_translate_count']} "
            f"succeeded={card_stats['card_succeeded']} "
            f"fallback_to_llm={card_stats['fallback_to_llm']}"
        )

        if BTarget.if_done:
            return f"done {during_time} try: {try_time}"
        if BTarget.provider_error:
            return f"provider_error {during_time} try: {try_time}"
        if BTarget.no_command:
            return f"no_command {during_time} try: {try_time}"
        if BTarget.timeout:
            return f"timeout {during_time} try: {try_time}"
        if BTarget.no_plan:
            return f"no_plan {during_time} try: {try_time}"
        if BTarget.cannot_solve:
            return f"cannot_solve {during_time} try: {try_time}"
        return f"error {during_time} try: {try_time}"
    except Exception as exc:
        if BTarget.phase1_status in {"not_started", "running"}:
            BTarget.phase1_status = "error"
            BTarget.phase1_errors.append({
                "producer": "phase1_orchestrator",
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            })
            if not BTarget.plan_error:
                BTarget.plan_error = f"Phase 1 failed: {type(exc).__name__}: {exc}"
        logging.exception("Build workflow failed")
        return f"error {time.time() - starttime} try: 0"
    finally:
        BTarget.write_run_result(elapsed_sec=time.time() - starttime)
        await BTarget.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folderpath", help="要处理的文件夹路径")
    parser.add_argument("sysuse", help="使用系统版本")
    parser.add_argument("--max-turns", type=int, default=5, help="最大构建/修复轮数")
    parser.add_argument("--ablation", choices=sorted(_ABLATION_MODES), default="", help="消融模式")
    parser.add_argument(
        "--agent-mode",
        choices=_VALID_AGENT_MODES,
        default=os.environ.get(_AGENT_MODE_ENV, _DEFAULT_AGENT_MODE),
        help="Agent execution mode. Use json-planner to avoid provider-native tool calling.",
    )
    args = parser.parse_args()
    os.environ[_AGENT_MODE_ENV] = args.agent_mode
    result = asyncio.run(Start(args.folderpath, args.sysuse, max_turns=args.max_turns, ablation=args.ablation))

    print(result)
    if result.startswith("done "):
        exit(0)
    if result.startswith("provider_error"):
        exit(5)
    if result.startswith("no_command"):
        exit(6)
    if result.startswith("no_plan"):
        exit(1)
    if result.startswith("network_failure"):
        exit(7)
    if result.startswith("timeout"):
        exit(2)
    if result.startswith("error ") or result.startswith("cannot_solve "):
        exit(1)
    exit(3)
