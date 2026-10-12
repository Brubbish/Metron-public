from pathlib import Path

from core.build_entry_selector import rank_build_entry_candidates
from core.build_family_resolver import resolve_build_families


def _touch(root: Path, relpath: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _inventory(*paths: str) -> dict:
    directories = {"."}
    for path in paths:
        parent = Path(path).parent.as_posix()
        if parent != ".":
            parts = parent.split("/")
            directories.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    return {
        "files": [{"path": path, "manifest": True, "size": 0} for path in paths],
        "directories": sorted(directories),
        "manifest_paths": list(paths),
    }


def test_qmake_family_candidate_is_eligible(tmp_path):
    _touch(tmp_path, "app.pro")
    _touch(tmp_path, ".qmake.conf")

    candidates = resolve_build_families(tmp_path, _inventory("app.pro", ".qmake.conf"))
    ranked = rank_build_entry_candidates(candidates, repo_path=str(tmp_path))

    selected = ranked[0]
    assert selected["family"] == "qmake"
    assert selected["eligible"] is True
    assert selected["command_sequence"] == ["qmake app.pro", "make"]


def test_python_configure_family_candidate_is_eligible(tmp_path):
    _touch(tmp_path, "configure.py")

    candidates = resolve_build_families(tmp_path, _inventory("configure.py"))
    ranked = rank_build_entry_candidates(candidates, repo_path=str(tmp_path))

    selected = ranked[0]
    assert selected["family"] == "python_configure"
    assert selected["eligible"] is True
    assert selected["command_sequence"] == ["python3 configure.py", "make"]


def test_gn_family_prefers_project_build_script_when_present(tmp_path):
    for relpath in (".gn", "BUILD.gn", "DEPS", "script/bootstrap.py", "script/build.py"):
        _touch(tmp_path, relpath)

    candidates = resolve_build_families(
        tmp_path,
        _inventory(".gn", "BUILD.gn", "DEPS", "script/bootstrap.py", "script/build.py"),
    )
    ranked = rank_build_entry_candidates(candidates, repo_path=str(tmp_path))

    selected = ranked[0]
    assert selected["family"] == "gn_ninja"
    assert selected["eligible"] is True
    assert selected["candidate_notes"] == ["requires_external_workspace_deps"]
    assert selected["command_sequence"] == ["python3 script/bootstrap.py", "python3 script/build.py"]


def test_workspace_make_candidate_allows_nested_workspace_root(tmp_path):
    _touch(tmp_path, "usr/src/Makefile")

    candidates = resolve_build_families(tmp_path, _inventory("usr/src/Makefile"))
    ranked = rank_build_entry_candidates(candidates, repo_path=str(tmp_path))

    selected = ranked[0]
    assert selected["family"] == "workspace_make"
    assert selected["scope"] == "workspace_root"
    assert selected["eligible"] is True
    assert selected["command_steps"][0]["workdir"] == "usr/src"


def test_family_resolver_keeps_test_only_marker_ineligible(tmp_path):
    _touch(tmp_path, "tests/helper.pro")

    candidates = resolve_build_families(tmp_path, _inventory("tests/helper.pro"))
    ranked = rank_build_entry_candidates(candidates, repo_path=str(tmp_path))

    selected = ranked[0]
    assert selected["scope"] == "test"
    assert selected["eligible"] is False


def test_manifest_inventory_recalls_depth_three_family_markers(tmp_path):
    from repo_parse.doc_scanner_parse import _collect_manifest_inventory

    _touch(tmp_path, "src/gui/app.pro")
    _touch(tmp_path, "usr/src/Makefile")
    _touch(tmp_path, "pkg/configure.py")

    inventory = _collect_manifest_inventory(str(tmp_path))

    assert "src/gui/app.pro" in inventory["manifest_paths"]
    assert "usr/src/Makefile" in inventory["manifest_paths"]
    assert "pkg/configure.py" in inventory["manifest_paths"]
