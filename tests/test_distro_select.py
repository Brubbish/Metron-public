from core.enhanced_dep_extractor import distro_select as mod
from core.enhanced_dep_extractor.scanner import ProjectScan


class _Eval:
    def __init__(self, status, matched_package="", matched_version="", source_repo=""):
        self.status = status
        self.matched_package = matched_package
        self.matched_version = matched_version
        self.source_repo = source_repo


def _patch_package_constraint(monkeypatch, statuses_by_suite):
    constraint = object()
    monkeypatch.setattr(mod.suite_compat, "extract_version_constraints", lambda scan: [constraint])
    monkeypatch.setattr(
        mod.suite_compat,
        "_evaluate_constraint_for_suite",
        lambda _constraint, suite: _Eval(statuses_by_suite.get(suite, "unknown")),
    )
    monkeypatch.setattr(
        mod,
        "_constraint_label",
        lambda _constraint: "pkgconfig: foo >= 2.0 (meson.build:1)",
    )
    monkeypatch.setattr(
        mod,
        "_constraint_detail",
        lambda _constraint, label: {
            "kind": "package",
            "label": label,
            "dep_system": "pkgconfig",
            "dep_name": "foo",
        },
    )
    monkeypatch.setattr(mod.sysuse_select, "extract_signals", lambda project_path: object())
    monkeypatch.setattr(
        mod.sysuse_select,
        "decide_sysuse",
        lambda signals: type("Decision", (), {"codename": "", "constraints": []})(),
    )


def test_keeps_baseline_when_constraints_pass(monkeypatch):
    _patch_package_constraint(monkeypatch, {"bookworm": "hard_pass", "trixie": "hard_pass"})

    decision = mod.select_debian_version("/repo", ProjectScan(), "bookworm")

    assert decision.selected_debian_version == "bookworm"
    assert decision.selection_source == "baseline_satisfies_constraints"


def test_upgrades_to_nearest_passing_suite(monkeypatch):
    _patch_package_constraint(monkeypatch, {
        "bullseye": "hard_fail",
        "bookworm": "hard_pass",
        "trixie": "hard_pass",
    })

    decision = mod.select_debian_version("/repo", ProjectScan(), "bullseye")

    assert decision.selected_debian_version == "bookworm"
    assert decision.selection_source == "constraints_upgrade"


def test_downgrades_to_nearest_passing_suite(monkeypatch):
    _patch_package_constraint(monkeypatch, {
        "bullseye": "hard_pass",
        "bookworm": "hard_fail",
        "trixie": "hard_fail",
    })

    decision = mod.select_debian_version("/repo", ProjectScan(), "bookworm")

    assert decision.selected_debian_version == "bullseye"
    assert decision.selection_source == "constraints_downgrade"


def test_falls_back_to_baseline_when_no_suite_satisfies(monkeypatch):
    _patch_package_constraint(monkeypatch, {
        "bullseye": "hard_fail",
        "bookworm": "hard_fail",
        "trixie": "hard_fail",
    })

    decision = mod.select_debian_version("/repo", ProjectScan(), "bookworm")

    assert decision.selected_debian_version == "bookworm"
    assert decision.selection_source == "baseline_after_unsatisfied_constraints"


def test_unknown_constraints_do_not_override_baseline(monkeypatch):
    _patch_package_constraint(monkeypatch, {})

    decision = mod.select_debian_version("/repo", ProjectScan(), "bookworm")

    assert decision.selected_debian_version == "bookworm"
    assert decision.selection_source == "baseline_satisfies_constraints"
    assert decision.unknown_constraints == ("pkgconfig: foo >= 2.0 (meson.build:1)",)


def test_constraint_details_include_next_satisfying_suite_versions(monkeypatch):
    constraint = object()
    versions = {
        "buster": _Eval("hard_pass", "libfoo-dev", "2.0-1", "buster/main"),
        "bullseye": _Eval("hard_pass", "libfoo-dev", "2.4-1", "bullseye/main"),
        "bookworm": _Eval("hard_pass", "libfoo-dev", "3.0-1", "bookworm/main"),
    }
    monkeypatch.setattr(mod.suite_compat, "extract_version_constraints", lambda scan: [constraint])
    monkeypatch.setattr(
        mod.suite_compat,
        "_evaluate_constraint_for_suite",
        lambda _constraint, suite: versions.get(suite, _Eval("hard_fail")),
    )
    monkeypatch.setattr(
        mod,
        "_constraint_label",
        lambda _constraint: "pkgconfig: foo >= 2.0 (meson.build:1)",
    )
    monkeypatch.setattr(
        mod,
        "_constraint_detail",
        lambda _constraint, label: {
            "kind": "package",
            "label": label,
            "dep_system": "pkgconfig",
            "dep_name": "foo",
        },
    )
    monkeypatch.setattr(
        mod.sysuse_select,
        "decide_sysuse",
        lambda signals: type("Decision", (), {"codename": "", "constraints": []})(),
    )

    decision = mod.select_debian_version("/repo", ProjectScan(), "buster")
    details = decision.as_constraint_details()

    assert details["minimum_satisfying_sysuse"] == "buster"
    assert details["next_satisfying_sysuse_candidates"][0]["sysuse"] == "bullseye"
    assert details["next_satisfying_sysuse_candidates"][0]["package_version_differences"] == [
        {
            "dep_system": "pkgconfig",
            "dep_name": "foo",
            "package": "libfoo-dev",
            "minimum_sysuse": "buster",
            "minimum_version": "2.0-1",
            "candidate_sysuse": "bullseye",
            "candidate_version": "2.4-1",
            "source_repo": "bullseye/main",
        }
    ]
