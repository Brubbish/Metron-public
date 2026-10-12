from core.build_plan_state import (
    MAX_PROMPT_ITEMS,
    build_initial_plan,
    compact_build_plan_for_prompt,
    compact_failure_frame_for_prompt,
    extract_failure_frame,
    update_build_plan_from_failure,
)


def test_failure_frame_updates_plan_from_current_attempt_only():
    project_info = {
        "build_workdir": ".",
        "main_build_plan": ["cmake -B build", "cmake --build build"],
        "command_candidates": [
            {
                "command": "./Config",
                "source": "script_detected",
                "confidence": "high",
                "path": "Config",
            }
        ],
        "apt_install_plan": ["cmake"],
        "source_dep_plan": [
            {
                "name": "foo",
                "source_path": "/work/commit1/source_deps/foo",
                "clone_status": "cloned",
            }
        ],
        "sysuse_candidate_plan": [
            {"sysuse": "bookworm", "reason": "current selected sysuse"},
            {"sysuse": "trixie", "reason": "newer compiler probe"},
        ],
    }
    plan = build_initial_plan(project_info, current_sysuse="bookworm")
    assert any(
        item["command"] == "./Config"
        and item["source"] == "script_detected"
        and item["confidence"] == "high"
        for item in plan["entrypoint_candidates"]
    )
    failed_command = {"command": "cmake --build build", "status": "failed", "exit_code": 2}
    current_error = """
Package 'libfoo', required by 'virtual:world', not found
fatal error: third_party/sdk/include/sdk.h: No such file or directory
/usr/bin/ld: cannot find -lbar
"""

    frame = extract_failure_frame(current_error, failed_command)
    updated = update_build_plan_from_failure(plan, frame, failed_command)

    assert frame["error_class"] == "missing_package_metadata"
    assert frame["missing_pkg_configs"] == ["libfoo"]
    assert frame["missing_headers"] == ["third_party/sdk/include/sdk.h"]
    assert frame["missing_libraries"] == ["bar"]
    assert any(dep["name"] == "libfoo" and dep["source"] == "current_failure" for dep in updated["system_dep_candidates"])
    assert any(dep["name"] == "bar" and dep["kind"] == "link_library" for dep in updated["system_dep_candidates"])
    assert any(dep["name"] == "sdk" and dep["source"] == "current_failure" for dep in updated["source_dep_candidates"])
    assert updated["failed_attempts"][-1]["command"] == "cmake --build build"
    assert updated["entrypoint_candidates"][0]["status"] == "failed"
    assert updated["entrypoint_candidates"][0]["command_sequence"] == [
        "cmake -B build",
        "cmake --build build",
    ]


def test_compact_prompt_views_limit_items():
    plan = {
        "entrypoint_candidates": [{"command": f"make target-{i}"} for i in range(10)],
        "sysuse_candidates": [{"sysuse": f"suite-{i}"} for i in range(10)],
        "system_dep_candidates": [{"name": f"lib-{i}"} for i in range(10)],
        "source_dep_candidates": [{"name": f"dep-{i}"} for i in range(10)],
        "bootstrap_candidates": [{"command": f"./bootstrap-{i}.sh"} for i in range(10)],
        "failed_attempts": [{"command": f"cmd-{i}"} for i in range(10)],
        "next_action_hints": [f"hint-{i}" for i in range(10)],
    }
    frame = {
        "error_class": "missing_dependency",
        "failed_command": {"command": "make"},
        "missing_headers": [f"h{i}.h" for i in range(10)],
        "missing_libraries": [f"lib{i}" for i in range(10)],
        "missing_pkg_configs": [f"pkg{i}" for i in range(10)],
        "missing_cmake_packages": [f"cmake{i}" for i in range(10)],
        "missing_paths": [f"path/{i}" for i in range(10)],
        "toolchain_errors": [f"tool-{i}" for i in range(10)],
        "bootstrap_errors": [f"boot-{i}" for i in range(10)],
        "diagnose_tags": [f"tag-{i}" for i in range(10)],
        "evidence": [f"line-{i}" for i in range(10)],
    }

    compact_plan = compact_build_plan_for_prompt(plan)
    compact_frame = compact_failure_frame_for_prompt(frame)

    assert all(len(value) <= MAX_PROMPT_ITEMS for value in compact_plan.values() if isinstance(value, list))
    assert all(len(value) <= MAX_PROMPT_ITEMS for value in compact_frame.values() if isinstance(value, list))
    assert compact_plan["entrypoints"][0]["command"] == "make target-4"
    assert compact_frame["missing_headers"][0] == "h4.h"


def test_apt_conflict_failure_gets_runtime_package_tag_and_hint():
    failed_command = {"command": "apt-get install -y heimdal-dev libkrb5-dev", "status": "failed", "exit_code": 100}
    frame = extract_failure_frame(
        """
        The following packages have unmet dependencies:
          heimdal-dev : Conflicts: libkrb5-dev but it is not going to be installed
        E: Unable to correct problems, you have held broken packages.
        """,
        failed_command,
    )
    updated = update_build_plan_from_failure({}, frame, failed_command)

    assert frame["error_class"] == "package_install_failure"
    assert "runtime_package_conflict" in frame["diagnose_tags"]
    assert any("one provider" in hint for hint in updated["next_action_hints"])


def test_source_dependency_path_signal_gets_diagnose_tag():
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame(
        "fatal error: vendor/sdk/include/sdk.h: No such file or directory",
        failed_command,
    )

    assert "source_dependency_needed" in frame["diagnose_tags"]
