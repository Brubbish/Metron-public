import json
import os
from pathlib import Path

from batch.run_agent_compile_batch import (
    CompileTarget,
    ModelApiKeyRotator,
    build_child_env,
    clear_previous_run_artifacts,
    filter_skip_done,
    get_system_by_date,
    get_system_by_year,
    load_done_keys,
    load_provider_error_keys,
    load_targets,
    reconcile_records,
    resolve_ablation_mode,
    run_one,
    safe_name,
    status_from_returncode,
)
import batch.run_agent_compile_batch as batch


def test_load_targets_records_parse_list_supports_offset_limit(tmp_path):
    payload = [
        {"name": "o1", "real_name": "r1", "local_path": "/repo/one", "commit_id": "a" * 40},
        {"name": "o2", "real_name": "r2", "local_path": "/repo/two", "commit_id": "b" * 40},
        {"name": "bad"},
    ]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    targets = load_targets(str(path), offset=1, limit=1)

    assert len(targets) == 1
    assert targets[0].owner == "o2"
    assert targets[0].repo == "r2"
    assert targets[0].local_path == "/repo/two"
    assert targets[0].commit == "b" * 40
    assert targets[0].key == f"o2##r2##{'b' * 40}"


def test_load_targets_legacy_nested_json(tmp_path):
    payload = {"owner": {"repo": {"c" * 40: {"vuln": []}}}}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    targets = load_targets(str(path))

    assert len(targets) == 1
    assert targets[0].owner == "owner"
    assert targets[0].repo == "repo"
    default_env_mapping = os.path.join(os.path.dirname(batch.__file__), "data", "env_mapping")
    assert targets[0].local_path == os.path.join(os.environ.get("METRON_ENV_MAPPING", default_env_mapping), "owner/repo")
    assert targets[0].commit == "c" * 40


def test_build_child_env_model_overrides_all_phases(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_PHASE2", raising=False)
    monkeypatch.setenv("THEAUTO_LLM_API_KEY_OVERRIDE", "inherited-key")

    env = build_child_env("gemma4_silicon")

    for phase in ("DEFAULT", "DIRECT", "READREPO", "ARTIFACT_SUMMARIZER", "PHASE2", "PHASE3", "CHECK"):
        assert env[f"LLM_MODEL_{phase}"] == "gemma4_silicon"
    assert env["THEAUTO_AGENT_MODE"] == "tool-calling"
    assert "THEAUTO_LLM_API_KEY_OVERRIDE" not in env

    selected_env = build_child_env("gemma4_silicon", "selected-key", "json-planner")
    assert selected_env["THEAUTO_LLM_API_KEY_OVERRIDE"] == "selected-key"
    assert selected_env["THEAUTO_AGENT_MODE"] == "json-planner"


def test_model_api_key_rotator_round_robins():
    rotator = ModelApiKeyRotator(["key-a", "key-b"])

    assert [rotator.next_key() for _ in range(5)] == [
        (0, "key-a"),
        (1, "key-b"),
        (0, "key-a"),
        (1, "key-b"),
        (0, "key-a"),
    ]
    assert rotator.key_count == 2


def test_status_from_returncode_uses_agent_exit_contract():
    assert status_from_returncode(0) == "done"
    assert status_from_returncode(1) == "error"
    assert status_from_returncode(2) == "timeout"
    assert status_from_returncode(5) == "provider_error"
    assert status_from_returncode(6) == "no_command"
    assert status_from_returncode(9) == "retcode_9"


def test_skip_done_filters_existing_successful_keys(tmp_path):
    payload = [
        {"name": "o1", "real_name": "r1", "local_path": "/repo/one", "commit_id": "a" * 40},
        {"name": "o2", "real_name": "r2", "local_path": "/repo/two", "commit_id": "b" * 40},
    ]
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps({"key": f"o1##r1##{'a' * 40}", "status": "done"}),
                json.dumps({"key": f"o2##r2##{'b' * 40}", "status": "error"}),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    targets = load_targets(str(input_path))
    done_keys = load_done_keys(records_path)
    remaining = filter_skip_done(targets, done_keys)

    assert done_keys == {f"o1##r1##{'a' * 40}"}
    assert [target.key for target in remaining] == [f"o2##r2##{'b' * 40}"]


def test_load_provider_error_keys_uses_done_wins_and_latest_non_done(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps({"key": "done-after-provider", "status": "provider_error"}),
                json.dumps({"key": "done-after-provider", "status": "done"}),
                json.dumps({"key": "provider-after-error", "status": "error"}),
                json.dumps({"key": "provider-after-error", "status": "provider_error"}),
                json.dumps({"key": "cannot-after-provider", "status": "provider_error"}),
                json.dumps({"key": "cannot-after-provider", "status": "cannot_solve"}),
                json.dumps({"key": "provider-after-done", "status": "done"}),
                json.dumps({"key": "provider-after-done", "status": "provider_error"}),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    assert load_provider_error_keys(records_path) == {"provider-after-error"}


def test_clear_previous_run_artifacts_removes_worktree_and_log(tmp_path):
    worktree = tmp_path / "worktree"
    log_path = tmp_path / "logs" / "case.log"
    worktree.mkdir()
    (worktree / "old.txt").write_text("old", encoding="utf-8")
    log_path.parent.mkdir()
    log_path.write_text("old log", encoding="utf-8")

    clear_previous_run_artifacts(worktree, log_path)

    assert not worktree.exists()
    assert not log_path.exists()


def test_skill_date_mapping_and_safe_name():
    assert get_system_by_date("2023-06-09") == "bullseye"
    assert get_system_by_date("2023-06-10") == "bookworm"
    assert get_system_by_date("2025-08-08") == "bookworm"
    assert get_system_by_date("2025-08-09") == "trixie"
    assert get_system_by_year(2024) == "bookworm"
    assert safe_name("owner/repo name") == "owner_repo_name"


def test_batch_ablation_env_fallback(monkeypatch):
    monkeypatch.setenv("THEAUTO_ABLATION", "no_phase1")

    assert resolve_ablation_mode("") == "no_phase1"
    assert resolve_ablation_mode("none") == "none"


def test_run_one_forwards_non_default_ablation(monkeypatch, tmp_path):
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    launched = {}

    class FakeProc:
        pid = 12345

        def wait(self, timeout=None):
            return 1

    def fake_copy_repo(src, dst):
        Path(dst).mkdir(parents=True)

    def fake_popen(argv, stdout=None, stderr=None, preexec_fn=None, env=None):
        launched["argv"] = argv
        launched["env"] = env
        return FakeProc()

    written = []

    monkeypatch.setattr(batch, "get_commit_date", lambda repo, commit: "2024-01-01")
    monkeypatch.setattr(
        batch,
        "source_commit_state",
        lambda repo, commit: {
            "commit_exists": True,
            "checkout_fetch_attempted": False,
            "checkout_fetch_error": "",
            "checkout_preflight_error": "",
        },
    )
    monkeypatch.setattr(batch, "copy_repo", fake_copy_repo)
    monkeypatch.setattr(batch, "checkout_worktree", lambda worktree, commit: None)
    monkeypatch.setattr(batch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(batch, "append_jsonl", lambda record_path, result, record_lock: written.append(result))

    target = CompileTarget(0, "owner", "repo", str(source_repo), "a" * 40)
    rotator = ModelApiKeyRotator(["primary-secret", "backup-secret"])
    result = run_one(
        target,
        output_root=tmp_path / "out",
        source_locks={},
        record_path=tmp_path / "records.jsonl",
        record_lock=batch.threading.Lock(),
        timeout_sec=30,
        max_turns=2,
        model="configured-model",
        sysuse_override=None,
        ablation="no_phase1",
        uv_bin="uv",
        batch_argv=["runner"],
        batch_args={"ablation": "no_phase1"},
        api_key_rotator=rotator,
    )

    assert result["ablation"] == "no_phase1"
    assert result["agent_mode"] == "tool-calling"
    assert result["status"] == "error"
    assert launched["argv"] == result["child_argv"]
    assert launched["argv"][launched["argv"].index("--ablation") + 1] == "no_phase1"
    assert launched["argv"][launched["argv"].index("--agent-mode") + 1] == "tool-calling"
    assert launched["env"]["THEAUTO_LLM_API_KEY_OVERRIDE"] == "primary-secret"
    assert launched["env"]["THEAUTO_AGENT_MODE"] == "tool-calling"
    assert result["resolved_model"] == "configured-model"
    assert result["api_key_index"] == 0
    assert result["api_key_count"] == 2
    assert "primary-secret" not in json.dumps(result)
    assert "backup-secret" not in json.dumps(result)
    assert written[-1]["ablation"] == "no_phase1"


def test_run_one_forwards_default_ablation_over_env(monkeypatch, tmp_path):
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    launched = {}

    class FakeProc:
        pid = 12345

        def wait(self, timeout=None):
            return 1

    def fake_copy_repo(src, dst):
        Path(dst).mkdir(parents=True)

    def fake_popen(argv, stdout=None, stderr=None, preexec_fn=None, env=None):
        launched["argv"] = argv
        launched["env"] = env
        return FakeProc()

    monkeypatch.setenv("THEAUTO_ABLATION", "no_phase1")
    monkeypatch.setattr(batch, "get_commit_date", lambda repo, commit: "2024-01-01")
    monkeypatch.setattr(
        batch,
        "source_commit_state",
        lambda repo, commit: {
            "commit_exists": True,
            "checkout_fetch_attempted": False,
            "checkout_fetch_error": "",
            "checkout_preflight_error": "",
        },
    )
    monkeypatch.setattr(batch, "copy_repo", fake_copy_repo)
    monkeypatch.setattr(batch, "checkout_worktree", lambda worktree, commit: None)
    monkeypatch.setattr(batch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(batch, "append_jsonl", lambda record_path, result, record_lock: None)

    target = CompileTarget(0, "owner", "repo", str(source_repo), "a" * 40)
    result = run_one(
        target,
        output_root=tmp_path / "out",
        source_locks={},
        record_path=tmp_path / "records.jsonl",
        record_lock=batch.threading.Lock(),
        timeout_sec=30,
        max_turns=2,
        model=None,
        sysuse_override=None,
        ablation="none",
        uv_bin="uv",
        batch_argv=["runner"],
        batch_args={"ablation": "none"},
    )

    assert result["ablation"] == "none"
    assert launched["argv"][launched["argv"].index("--ablation") + 1] == "none"
    assert launched["argv"][launched["argv"].index("--agent-mode") + 1] == "tool-calling"


def test_run_one_marks_missing_commit_as_checkout_exception(monkeypatch, tmp_path):
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    written = []

    monkeypatch.setattr(
        batch,
        "source_commit_state",
        lambda repo, commit: {
            "commit_exists": False,
            "checkout_fetch_attempted": True,
            "checkout_fetch_error": "fetch failed",
            "checkout_preflight_error": "missing object",
        },
    )
    monkeypatch.setattr(batch, "append_jsonl", lambda record_path, result, record_lock: written.append(result))

    target = CompileTarget(0, "owner", "repo", str(source_repo), "a" * 40)
    result = run_one(
        target,
        output_root=tmp_path / "out",
        source_locks={},
        record_path=tmp_path / "records.jsonl",
        record_lock=batch.threading.Lock(),
        timeout_sec=30,
        max_turns=2,
        model=None,
        sysuse_override=None,
        ablation="none",
        uv_bin="uv",
        batch_argv=["runner"],
        batch_args={"ablation": "none"},
    )

    assert result["status"] == "exception"
    assert result["exception_stage"] == "checkout"
    assert result["checkout_fetch_attempted"] is True
    assert "commit not found" in result["error"]
    assert "child_argv" not in result
    assert written[-1]["status"] == "exception"


def test_empty_batch_summary_records_ablation(monkeypatch, tmp_path):
    input_path = tmp_path / "empty.json"
    input_path.write_text("[]", encoding="utf-8")
    output_root = tmp_path / "out"

    monkeypatch.setattr(
        batch,
        "parse_args",
        lambda: batch.argparse.Namespace(
            input_json=str(input_path),
            jobs=1,
            offset=0,
            limit=None,
            output_root=str(output_root),
            timeout_sec=3600,
            max_turns=60,
            model=None,
            agent_mode="tool-calling",
            sysuse=None,
            ablation="no_phase3",
            only_c_cpp=False,
            skip_done=False,
            skip_existing=False,
            only_provider_error=False,
            reconcile_records=False,
            uv_bin="uv",
        ),
    )

    assert batch.main() == 1
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "schema_version": 2,
        "total": 0,
        "success": 0,
        "failed": 0,
        "by_status": {},
        "attempt_records": 0,
        "ablation": "no_phase3",
    }


def test_only_provider_error_rejects_skip_existing(monkeypatch):
    monkeypatch.setattr(
        batch,
        "parse_args",
        lambda: batch.argparse.Namespace(
            only_provider_error=True,
            skip_existing=True,
            ablation="none",
        ),
    )

    assert batch.main() == 2


def test_only_provider_error_empty_selection_exits_zero(monkeypatch, tmp_path):
    key = f"o##r##{'a' * 40}"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps([{"name": "o", "real_name": "r", "local_path": "/repo/one", "commit_id": "a" * 40}]),
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "records.jsonl").write_text(json.dumps({"key": key, "status": "error"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        batch,
        "parse_args",
        lambda: batch.argparse.Namespace(
            input_json=str(input_path),
            jobs=1,
            offset=0,
            limit=None,
            output_root=str(output_root),
            timeout_sec=3600,
            max_turns=60,
            model=None,
            sysuse=None,
            ablation="none",
            only_c_cpp=False,
            skip_done=False,
            skip_existing=False,
            only_provider_error=True,
            reconcile_records=False,
            uv_bin="uv",
        ),
    )

    assert batch.main() == 0
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["by_status"] == {"error": 1}


def test_reconcile_records_done_wins_and_log_proven_success(tmp_path):
    records = tmp_path / "records.jsonl"
    summary = tmp_path / "summary.json"
    log = tmp_path / "build.log"
    log.write_text("Compilation succeeded\n", encoding="utf-8")
    records.write_text(
        "\n".join(
            [
                json.dumps({"key": "a", "status": "error", "returncode": 1, "log_path": str(log)}),
                json.dumps({"key": "b", "status": "error", "returncode": 1}),
                json.dumps({"key": "b", "status": "done", "returncode": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = reconcile_records(records, summary)
    out = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]

    assert stats == {"total": 2, "done": 2, "reconciled": 1}
    assert out[0]["key"] == "a"
    assert out[0]["status"] == "done"
    assert out[0]["reconciled_from_log"] is True
