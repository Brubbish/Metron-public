from pathlib import Path

from core.build_plan_state import extract_failure_frame
from diagnosis.diagnosis_probes import apply_probe_refinement, run_provenance_probes
from diagnosis.diagnosis_router import route_failure


def test_probe_header_in_repo_marks_include_visibility(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    (include / "foo.h").write_text("#pragma once\n")
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame("fatal error: foo.h: No such file or directory", failed_command)
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(route, frame, {}, str(tmp_path), [])
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "repo_local_invisible"
    assert probe["confidence"] == "high"
    assert refined.candidate_decisions[0] == "generation"
    assert refined.repair_scope["provenance_probe"]["provenance"] == "repo_local_invisible"


def test_probe_vendored_missing_header_prefers_source_dependency(tmp_path):
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame(
        "fatal error: third_party/sdk/include/sdk.h: No such file or directory",
        failed_command,
    )
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(route, frame, {}, str(tmp_path), [])
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "project_managed"
    assert refined.candidate_decisions[0] == "source_dependency"
    assert "system_release" in probe["discouraged_decisions"]


def test_probe_vendored_header_not_shadowed_by_same_basename(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    (include / "sdk.h").write_text("#pragma once\n")
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame(
        "fatal error: third_party/sdk/include/sdk.h: No such file or directory",
        failed_command,
    )
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(route, frame, {}, str(tmp_path), [])
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "project_managed"
    assert refined.candidate_decisions[0] == "source_dependency"


def test_probe_local_library_prefers_link_command(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "libfoo.a").write_text("")
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame("/usr/bin/ld: cannot find -lfoo", failed_command)
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(
        route,
        frame,
        {},
        str(tmp_path),
        [{"command": "gcc main.o -lfoo"}],
    )
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "repo_local_invisible"
    assert refined.candidate_decisions[0] == "linker_configuration"


def test_probe_abi_symbol_prefers_system_release(tmp_path):
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame("undefined reference to `foo@GLIBCXX_3.4.29'", failed_command)
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(route, frame, {}, str(tmp_path), [])
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "system_or_external"
    assert probe["confidence"] == "high"
    assert refined.candidate_decisions[0] == "compiler_configuration"


def test_probe_generated_header_prefers_generation_step(tmp_path):
    Path(tmp_path / "CMakeLists.txt").write_text("configure_file(config.h.in config.h)\n")
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame("fatal error: config.h: No such file or directory", failed_command)
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(route, frame, {}, str(tmp_path), [])
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "generated"
    assert refined.candidate_decisions[0] == "generation"


def test_probe_generated_header_from_extension_hint(tmp_path):
    Path(tmp_path / "Makefile").write_text("generated.h: generated.h.in\n\t./gen > generated.h\n")
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame("fatal error: generated.h: No such file or directory", failed_command)
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(route, frame, {}, str(tmp_path), [])
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "generated"
    assert refined.candidate_decisions[0] == "generation"


def test_probe_repo_defined_symbol_prefers_build_target(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.c").write_text("void local_symbol(void) {}\n")
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame("undefined reference to `local_symbol'", failed_command)
    route = route_failure(frame, failed_command)

    probe = run_provenance_probes(route, frame, {}, str(tmp_path), [])
    refined = apply_probe_refinement(route, probe)

    assert probe["provenance"] == "project_source"
    assert refined.candidate_decisions[0] == "dependency_discovery"
