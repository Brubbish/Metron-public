"""
batch_compile_main.py
---------------------
Batch build runner for the enriched JSON dataset.
Parses JSON entries, prepares repos (copy + checkout), and invokes agent_build_main.py.
"""

import os
import json
import subprocess
import time
import argparse
import logging
import signal
import shutil
import sys
import multiprocessing
import concurrent.futures
import atexit
import re
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

WORK_DIR = os.environ.get("METRON_WORK_DIR", os.path.join(os.path.dirname(__file__), "data", "collects"))
RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
CODES_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_TIMEOUT = 3600  # 1 hour per build
_ACTIVE_BUILD_GROUPS = None
_CLEANUP_DONE = False

# Agent writes its per-commit artifacts (enhanced_dep_result.json,
# docker_commands/, BuildLog_*.txt, etc.) into a subdir named after the
# commit SHA. See run_agent_compile_batch.py for rationale.
_COMMIT_SHA_DIR_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _ignore_agent_artifacts(_directory, names):
    return [n for n in names if _COMMIT_SHA_DIR_RE.match(n)]


def _cleanup_stale_containers():
    """Kill any leftover theauto-build-* containers."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=theauto-build-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    containers = [c.strip() for c in result.stdout.splitlines() if c.strip()]
    if containers:
        subprocess.run(["docker", "rm", "-f"] + containers, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Cleaned {len(containers)} stale containers")


def _cleanup_registered_build_groups():
    """Terminate any tracked agent_build_main process groups."""
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    if _ACTIVE_BUILD_GROUPS is None:
        return
    try:
        items = list(_ACTIVE_BUILD_GROUPS.items())
    except Exception:
        return
    for _, pgid in items:
        if not pgid:
            continue
        try:
            os.killpg(int(pgid), signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception as exc:
            logger.warning(f"Failed to SIGTERM process group {pgid}: {exc}")
    time.sleep(2)
    for _, pgid in items:
        if not pgid:
            continue
        try:
            os.killpg(int(pgid), signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception:
            continue


def _handle_parent_shutdown(signum, _frame):
    logger.warning(f"Received signal {signum}, requesting shutdown...")
    raise KeyboardInterrupt


def prepare_repo(entry: dict) -> str:
    """Copy repo to work dir and checkout the target commit.

    Uses {WORK_DIR}/{name}/{real_name}_{commit_id[:8]} to avoid conflicts
    between different commits of the same repo.
    Always does a full reset even if directory exists.
    """
    name = entry["name"]
    real_name = entry["real_name"]
    commit_id = entry["commit_id"]
    local_path = entry["local_path"]

    target_dir = os.path.join(WORK_DIR, name, f"{real_name}_{commit_id[:8]}")

    if os.path.isdir(target_dir):
        logger.info(f"Removing existing {target_dir}")
        shutil.rmtree(target_dir, ignore_errors=True)

    if not os.path.isdir(local_path):
        logger.error(f"Source repo not found: {local_path}")
        return ""

    logger.info(f"Copying {local_path} -> {target_dir}")
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    shutil.copytree(local_path, target_dir, symlinks=True, ignore_dangling_symlinks=True,
                    ignore=_ignore_agent_artifacts)

    if not _git_reset(target_dir, commit_id):
        return ""
    return target_dir


def _git_reset(repo_path: str, commit: str) -> bool:
    """Reset repo to specific commit and clean all untracked files."""
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
            cwd=repo_path, check=True,
            capture_output=True, text=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        ).stdout.strip()
        subprocess.run(
            ["git", "checkout", "--force", "--detach", resolved],
            cwd=repo_path, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        subprocess.run(
            ["git", "clean", "-ffdx"],
            cwd=repo_path, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        subprocess.run(
            ["git", "submodule", "deinit", "-f", "--all"],
            cwd=repo_path,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info(f"Reset {os.path.basename(repo_path)} to {resolved[:10]}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git reset failed for {repo_path}: {e.stderr if hasattr(e, 'stderr') else e}")
        return False


def run_build(target_path: str, commit: str, log_path: str, active_groups=None, job_key: str = "") -> tuple:
    """Run agent_build_main.py and return (exit_code, elapsed_seconds)."""
    args = [
        sys.executable,
        os.path.join(CODES_DIR, "agent_build_main.py"),
        f"{target_path}##{commit}",
        "auto",
    ]

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as logf:
        start = time.time()
        p = subprocess.Popen(
            args,
            stdout=logf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            cwd=CODES_DIR,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        registry_key = None
        if active_groups is not None:
            registry_key = job_key or f"{os.path.basename(target_path)}##{commit[:10]}"
            active_groups[registry_key] = p.pid
        try:
            retcode = p.wait(timeout=BUILD_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning(f"Build timed out after {BUILD_TIMEOUT}s, killing...")
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            retcode = 2  # timeout
        except KeyboardInterrupt:
            os.killpg(p.pid, signal.SIGTERM)
            raise
        finally:
            if active_groups is not None and registry_key is not None:
                active_groups.pop(registry_key, None)
        elapsed = time.time() - start
        logf.write(f"\n--- elapsed: {elapsed:.1f}s, exit_code: {retcode} ---\n")

    return retcode, elapsed


def status_from_retcode(retcode: int) -> str:
    return {0: "done", 1: "error", 2: "timeout", 4: "except_error", 5: "provider_error", 6: "no_command"}.get(retcode, f"unknown_{retcode}")


def process_entry(entry: dict, record_file: str, date_str: str, lock=None, logs_dir: str = None, active_groups=None) -> str:
    """Process a single entry. Returns status string."""
    logs_dir = logs_dir or LOGS_DIR
    name = entry["name"]
    real_name = entry["real_name"]
    commit_id = entry["commit_id"]
    key = f"{name}##{real_name}##{commit_id[:10]}"

    start_time = datetime.now().isoformat(timespec='seconds')
    logger.info(f"Processing {key}")

    target_path = prepare_repo(entry)
    if not target_path:
        _record(record_file, f"{key} prepare_error start={start_time}", lock)
        return "prepare_error"

    log_path = os.path.join(logs_dir, f"{date_str}_{name}_{real_name}_{commit_id[:8]}.log")
    try:
        retcode, elapsed = run_build(
            target_path,
            commit_id,
            log_path,
            active_groups=active_groups,
            job_key=key,
        )
    except KeyboardInterrupt:
        _record(record_file, f"{key} interrupted start={start_time}", lock)
        raise

    end_time = datetime.now().isoformat(timespec='seconds')
    status = status_from_retcode(retcode)
    logger.info(f"  {key} -> {status} ({elapsed:.0f}s)")
    _record(record_file, f"{key} {status} {elapsed:.0f}s start={start_time} end={end_time}", lock)
    return status


def _record(filepath: str, line: str, lock=None):
    if lock:
        lock.acquire()
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    finally:
        if lock:
            lock.release()


def _worker(args):
    """Worker function for multiprocessing."""
    entry, record_file, date_str, lock, logs_dir, active_groups = args
    try:
        return process_entry(entry, record_file, date_str, lock, logs_dir=logs_dir, active_groups=active_groups)
    except KeyboardInterrupt:
        return "interrupted"
    except Exception as e:
        logger.exception(f"Worker exception: {e}")
        return f"exception: {e}"


def main():
    global BUILD_TIMEOUT, _ACTIVE_BUILD_GROUPS, _CLEANUP_DONE
    parser = argparse.ArgumentParser(description="Batch build runner for enriched JSON dataset")
    parser.add_argument("json_path", help="Path to enriched JSON file (list of entries)")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=BUILD_TIMEOUT, help="Per-build timeout in seconds")
    parser.add_argument("--skip-existing", action="store_true", help="Skip entries already in records")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip stale container cleanup at start")
    parser.add_argument("--record-date", type=str, default=None,
                        help="Override YYMMDD record filename suffix (e.g. 260527)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for records and logs. Defaults to Codes/records and Codes/logs")
    args = parser.parse_args()

    BUILD_TIMEOUT = args.timeout

    if args.output_dir:
        records_dir = os.path.join(args.output_dir, "records")
        logs_dir = os.path.join(args.output_dir, "logs")
    else:
        records_dir = RECORDS_DIR
        logs_dir = LOGS_DIR

    manager = multiprocessing.Manager() if args.workers > 1 else None
    _ACTIVE_BUILD_GROUPS = manager.dict() if manager is not None else {}
    _CLEANUP_DONE = False
    atexit.register(_cleanup_registered_build_groups)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _handle_parent_shutdown)

    # Clean stale containers from previous runs
    if not args.no_cleanup:
        _cleanup_stale_containers()

    with open(args.json_path, encoding="utf-8") as f:
        entries = json.load(f)

    entries = entries[args.start:args.end]
    non_c_count = sum(1 for e in entries if not e.get("is_c_cpp", True))
    entries = [e for e in entries if e.get("is_c_cpp", True)]
    logger.info(f"Loaded {len(entries)} C/C++ entries (skipped {non_c_count} non-C/C++)")

    date_str = args.record_date or time.strftime("%y%m%d", time.localtime())
    record_file = os.path.join(records_dir, f"record_{date_str}.txt")
    os.makedirs(records_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Filter out already-recorded entries
    if args.skip_existing and os.path.isfile(record_file):
        with open(record_file) as rf:
            existing = rf.read()
        entries = [
            e for e in entries
            if (
                f"{e['name']}##{e['real_name']}##{e['commit_id'][:10]}" not in existing
                and f"{e['name']}##{e['real_name']}##{e['commit_id'][:8]}" not in existing
            )
        ]
        logger.info(f"After skip-existing filter: {len(entries)} entries remaining")

    pool = None
    try:
        if args.workers <= 1:
            # Serial execution (no lock needed)
            for i, entry in enumerate(entries):
                logger.info(f"[{i+1}/{len(entries)}]")
                process_entry(entry, record_file, date_str, logs_dir=logs_dir, active_groups=_ACTIVE_BUILD_GROUPS)
        else:
            # Parallel execution with lock
            lock = manager.Lock()
            worker_args = [(entry, record_file, date_str, lock, logs_dir, _ACTIVE_BUILD_GROUPS) for entry in entries]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker, wa): wa[0] for wa in worker_args}
                done_count = 0
                for future in concurrent.futures.as_completed(futures):
                    done_count += 1
                    entry = futures[future]
                    try:
                        status = future.result()
                        logger.info(f"[{done_count}/{len(entries)}] {entry['name']}/{entry['real_name']} -> {status}")
                    except Exception:
                        logger.exception(f"Future exception for {entry['name']}/{entry['real_name']}")
        logger.info("Batch complete.")
    except KeyboardInterrupt:
        logger.warning("Interrupted — shutting down workers...")
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        _cleanup_registered_build_groups()
        _cleanup_stale_containers()
        if manager is not None:
            manager.shutdown()


if __name__ == "__main__":
    main()
