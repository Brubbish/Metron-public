from core.build_plan_state import extract_failure_frame
from diagnosis.diagnosis_router import route_failure


def test_route_header_not_found_to_dependency_and_compile_decisions():
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame(
        "fatal error: third_party/sdk/include/sdk.h: No such file or directory",
        failed_command,
    )

    route = route_failure(frame, failed_command, build_plan={"entrypoints": [{"command": "make"}]})

    assert route.failed_phase == "compile"
    assert route.symptom_family == "header_not_found"
    assert route.entity == {"kind": "header", "name": "third_party/sdk/include/sdk.h"}
    assert "source_dependency" in route.candidate_decisions
    assert "D7" in route.candidate_decisions
    assert any("exists under /work" in check for check in route.next_checks)
    assert "source_dependency_register" in route.repair_scope["allowed_revision_types"]


def test_route_apt_conflict_to_setup_phase_and_single_provider_revision():
    failed_command = {"command": "apt-get install -y heimdal-dev libkrb5-dev", "status": "failed", "exit_code": 100}
    frame = extract_failure_frame(
        "E: Unable to correct problems, you have held broken packages.",
        failed_command,
    )

    route = route_failure(frame, failed_command)

    assert route.failed_phase == "setup"
    assert route.symptom_family == "apt_install_failure"
    # FDM v5 cell (setup, T1a) orders D5, D7, D8, D9.
    assert route.candidate_decisions[:2] == ["system_package_provider", "D5"]
    assert "system_release" in route.candidate_decisions
    assert "apt_single_provider_reduction" in route.repair_scope["allowed_revision_types"]


def test_route_wrong_entrypoint_to_build_entry_scope():
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame(
        "make: *** No targets specified and no makefile found.  Stop.",
        failed_command,
    )

    route = route_failure(frame, failed_command)

    assert route.symptom_family == "wrong_entrypoint"
    # FDM v5 cell (compile, T4) orders D1, D7, D9.
    assert route.candidate_decisions[:2] == ["compiler_configuration", "D1"]
    assert "build_flag_adjustment" in route.repair_scope["allowed_revision_types"]


def test_route_undefined_reference_to_link_symbol_scope():
    failed_command = {"command": "make", "status": "failed", "exit_code": 2}
    frame = extract_failure_frame(
        "/usr/bin/ld: src/net.o: undefined reference to `SSL_library_init'",
        failed_command,
    )

    route = route_failure(frame, failed_command)

    assert route.failed_phase == "link"
    assert route.symptom_family == "unresolved_symbol"
    assert route.entity == {"kind": "function_symbol", "name": "SSL_library_init"}
    # FDM v5 cell (link, T3a) orders D1, D2, D4, D6, D7; the legacy
    # link_command label is still promoted ahead of the linker direction.
    assert route.candidate_decisions[:2] == ["link_command", "compiler_configuration"]
    assert "link_flag_adjustment" in route.repair_scope["allowed_revision_types"]
    assert any("symbol" in check for check in route.next_checks)
