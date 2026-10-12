"""Tests for source_dep_resolver.

Covers:
- Version constraint parsing (PKG_CHECK_MODULES / meson / CMake)
- Semver selection (select_min_satisfying_tag)
- Candidate collection from 3 sources + dedup
- clone_one_dep against a real local git fixture
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.source_dep_resolver import (
    VersionConstraint,
    parse_version_constraint,
    parse_configure_ac_constraints,
    parse_meson_constraints,
    parse_cmake_constraints,
    select_min_satisfying_tag,
    _parse_semver,
    _satisfies,
    SourceDepCandidate,
    collect_source_dep_candidates,
    clone_one_dep,
    compute_host_clone_root,
    extract_version_constraints,
    _make_cloned_dep,
)


# -----------------------------
# Version constraint parsing
# -----------------------------

def test_parse_version_constraint_simple():
    vc = parse_version_constraint('libyang >= 2.1.128')
    assert vc is not None
    assert vc.pkg == 'libyang'
    assert vc.operator == '>='
    assert vc.version == '2.1.128'


def test_parse_version_constraint_no_version():
    vc = parse_version_constraint('foo')
    assert vc is not None
    assert vc.pkg == 'foo'
    assert vc.version is None


def test_parse_version_constraint_empty_returns_none():
    assert parse_version_constraint('') is None
    assert parse_version_constraint('   ') is None


def test_parse_configure_ac_constraints():
    text = """
    PKG_CHECK_MODULES([LIBYANG], [libyang >= 2.1.128])
    PKG_CHECK_MODULES([GLIB], [glib-2.0 >= 2.48.0 gobject-2.0 >= 2.48.0])
    PKG_CHECK_MODULES([FOO], [foo])
    """
    cs = parse_configure_ac_constraints(text)
    pkgs = [c.pkg for c in cs]
    assert 'libyang' in pkgs
    assert 'glib-2.0' in pkgs
    assert 'gobject-2.0' in pkgs
    assert 'foo' in pkgs
    ly = next(c for c in cs if c.pkg == 'libyang')
    assert ly.version == '2.1.128'


def test_parse_meson_constraints():
    text = """
    dependency('glib-2.0', version: '>= 2.48.0')
    dependency('libyang', version: '>= 2.1.128')
    """
    cs = parse_meson_constraints(text)
    pkgs = {c.pkg for c in cs}
    assert 'glib-2.0' in pkgs
    assert 'libyang' in pkgs


def test_parse_cmake_constraints():
    text = """
    find_package(libyang 2.1.128 REQUIRED)
    find_package(Boost 1.70 COMPONENTS filesystem system)
    """
    cs = parse_cmake_constraints(text)
    pkgs = {c.pkg for c in cs}
    assert 'libyang' in pkgs
    assert 'Boost' in pkgs


# -----------------------------
# Semver selection
# -----------------------------

def test_parse_semver_handles_v_prefix():
    assert _parse_semver('v2.1.128') == (2, 1, 128)
    assert _parse_semver('2.1.128') == (2, 1, 128)
    assert _parse_semver('v2.1') == (2, 1)
    assert _parse_semver('not-a-version') is None


def test_satisfies_basic():
    assert _satisfies((2, 1, 128), '>=', (2, 1, 128))
    assert _satisfies((2, 2, 0), '>=', (2, 1, 128))
    assert not _satisfies((2, 1, 30), '>=', (2, 1, 128))
    assert _satisfies((3, 0, 0), '>=', (2, 1, 128))  # cross-major still satisfies


def test_select_min_satisfying_tag_picks_minimum():
    tags = ['v1.0.0', 'v2.0.0', 'v2.1.30', 'v2.1.128', 'v2.2.0', 'v3.0.0']
    chosen = select_min_satisfying_tag(tags, '>=', '2.1.128')
    assert chosen == 'v2.1.128'


def test_select_min_satisfying_tag_returns_none_when_unsatisfiable():
    tags = ['v1.0.0', 'v2.0.0', 'v2.1.30']
    assert select_min_satisfying_tag(tags, '>=', '2.1.128') is None


def test_select_min_satisfying_tag_skips_unparseable():
    tags = ['latest', 'v2.1.128', 'release-foo', 'v3.0.0']
    chosen = select_min_satisfying_tag(tags, '>=', '2.1.128')
    assert chosen == 'v2.1.128'


# -----------------------------
# Candidate collection
# -----------------------------

def test_collect_cargo_candidates_picks_up_refs():
    from core.source_dep_resolver import _collect_cargo_candidates
    refs = [
        {'name': 'cairo-rs', 'url': 'https://github.com/foo/cairo.git', 'commit': 'abc123'},
        {'name': 'libc', 'url': '', 'commit': ''},  # filtered out (no url)
    ]
    out = _collect_cargo_candidates(refs)
    assert len(out) == 1
    assert out[0].name == 'cairo-rs'
    assert out[0].origin == 'cargo_lock'


def test_collect_doc_candidates_requires_url():
    from core.source_dep_resolver import _collect_doc_candidates
    docs = [
        {'name': 'libyang', 'url': 'https://github.com/CESNET/libyang.git', 'reason': 'doc says so'},
        {'name': 'foo', 'url': '', 'reason': 'no url'},  # filtered
    ]
    out = _collect_doc_candidates(docs)
    assert len(out) == 1
    assert out[0].name == 'libyang'
    assert out[0].clone_url.endswith('libyang.git')


def test_collect_configure_ac_candidates_needs_doc_url():
    from source_dep_resolver import _collect_configure_ac_candidates
    vc = VersionConstraint(pkg='libyang', operator='>=', version='2.1.128')
    docs = [
        {'name': 'libyang', 'url': 'https://github.com/CESNET/libyang.git'},
    ]
    out = _collect_configure_ac_candidates([vc], docs)
    assert len(out) == 1
    assert out[0].name == 'libyang'
    assert out[0].version_constraint.version == '2.1.128'

    # No doc URL → no candidate (we cannot invent URLs)
    out2 = _collect_configure_ac_candidates([vc], [])
    assert out2 == []


def test_collect_source_dep_candidates_dedupes_by_name(tmp_path):
    # Write a configure.ac so extract_version_constraints picks up libyang
    (tmp_path / 'configure.ac').write_text(
        'PKG_CHECK_MODULES([LIBYANG], [libyang >= 2.1.128])\n',
        encoding='utf-8',
    )
    cargo_refs = [
        {'name': 'libyang', 'url': 'https://github.com/CESNET/libyang.git', 'commit': 'x'},  # dup name
        {'name': 'cairo-rs', 'url': 'https://github.com/foo/cairo.git', 'commit': 'y'},
    ]
    docs = [
        {'name': 'libyang', 'url': 'https://github.com/CESNET/libyang.git', 'reason': 'doc says so'},
    ]
    out = collect_source_dep_candidates(str(tmp_path), docs, cargo_refs)
    # libyang appears in 3 sources but should be deduped to 1
    names = [c.name for c in out]
    assert names.count('libyang') == 1
    # cairo-rs is unique to cargo
    assert 'cairo-rs' in names


def test_collect_source_dep_candidates_excludes_submodule_only():
    """Verify the function signature does not take submodule plan."""
    import inspect
    sig = inspect.signature(collect_source_dep_candidates)
    assert 'submodule' not in sig.parameters


def test_cloned_dep_uses_documented_build_contract_at_every_depth():
    candidate = SourceDepCandidate(
        name="demo",
        clone_url="https://example.com/demo.git",
        origin="doc_extracted",
        setup_commands=["./bootstrap"],
        build_order=["./configure", "make"],
    )

    result = _make_cloned_dep(
        candidate,
        host_source_path="/tmp/demo",
        container_source_path="/source_deps/demo",
        clone_status="cloned",
        phase1={"build_commands": ["make install"]},
        depth=1,
    )

    assert result["phase1_status"] == "ok"
    assert result["setup_commands"] == ["./bootstrap"]
    assert result["build_order"] == ["./configure", "make", "make install"]


# -----------------------------
# clone_one_dep (uses local git fixture)
# -----------------------------

def _make_local_repo(path: Path) -> None:
    """Create a bare-ish git repo with tags at `path` for clone tests."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(['git', 'init', '-q'], cwd=str(path))
    subprocess.check_call(['git', 'config', 'user.email', 't@t'], cwd=str(path))
    subprocess.check_call(['git', 'config', 'user.name', 't'], cwd=str(path))
    (path / 'README.md').write_text('hello\n', encoding='utf-8')
    subprocess.check_call(['git', 'add', '.'], cwd=str(path))
    env = {**os.environ, 'GIT_AUTHOR_DATE': '2020-01-01T00:00:00', 'GIT_COMMITTER_DATE': '2020-01-01T00:00:00'}
    subprocess.check_call(['git', 'commit', '-q', '-m', 'init'], cwd=str(path), env=env)
    subprocess.check_call(['git', 'tag', 'v1.0.0'], cwd=str(path))
    (path / 'foo.txt').write_text('foo\n', encoding='utf-8')
    subprocess.check_call(['git', 'add', '.'], cwd=str(path))
    subprocess.check_call(['git', 'commit', '-q', '-m', 'second'], cwd=str(path), env=env)
    subprocess.check_call(['git', 'tag', 'v2.0.0'], cwd=str(path))


def test_clone_one_dep_success(tmp_path):
    src = tmp_path / 'src'
    _make_local_repo(src)
    dest = tmp_path / 'dest' / 'myrepo'
    candidate = SourceDepCandidate(
        name='myrepo',
        clone_url=str(src),
        origin='doc_extracted',
    )
    ok, msg = clone_one_dep(candidate, str(dest), timeout=30)
    assert ok, msg
    assert (dest / 'README.md').exists()
    # HEAD by default (no version constraint)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(dest), text=True).strip()
    assert head


def test_clone_one_dep_with_version_constraint(tmp_path):
    src = tmp_path / 'src'
    _make_local_repo(src)  # has v1.0.0 and v2.0.0
    dest = tmp_path / 'dest' / 'myrepo'
    candidate = SourceDepCandidate(
        name='myrepo',
        clone_url=str(src),
        origin='configure_ac',
        version_constraint=VersionConstraint(pkg='myrepo', operator='>=', version='1.5.0'),
    )
    ok, msg = clone_one_dep(candidate, str(dest), timeout=30)
    assert ok, msg
    # Should have checked out v2.0.0 (min satisfying >= 1.5.0)
    tag = subprocess.check_output(['git', 'describe', '--tags', 'HEAD'], cwd=str(dest), text=True).strip()
    assert tag == 'v2.0.0'


def test_clone_one_dep_bad_url_returns_false(tmp_path):
    dest = tmp_path / 'dest' / 'nope'
    candidate = SourceDepCandidate(
        name='nope',
        clone_url='/nonexistent/path/that/does/not/exist',
        origin='doc_extracted',
    )
    ok, msg = clone_one_dep(candidate, str(dest), timeout=10)
    assert not ok
    assert 'clone_status' not in msg  # this is just a status message
    assert 'failed' in msg.lower() or 'error' in msg.lower()


# -----------------------------
# compute_host_clone_root
# -----------------------------

def test_compute_host_clone_root_layout():
    compile_path = '/some/output_root/worktrees/00027_FRRouting__frr__b0800bfdf04b'
    clone_root = compute_host_clone_root(compile_path)
    assert clone_root == '/some/output_root/source_deps/00027_FRRouting__frr__b0800bfdf04b'


def test_compute_host_clone_root_preserves_worktree_name():
    compile_path = '/data/batch/worktrees/00001_owner__repo__abc123'
    clone_root = compute_host_clone_root(compile_path)
    assert clone_root.endswith('/source_deps/00001_owner__repo__abc123')


# -----------------------------
# extract_version_constraints (file-based)
# -----------------------------

def test_extract_version_constraints_reads_all_files(tmp_path):
    (tmp_path / 'configure.ac').write_text(
        'PKG_CHECK_MODULES([LIBYANG], [libyang >= 2.1.128])\n',
        encoding='utf-8',
    )
    (tmp_path / 'CMakeLists.txt').write_text(
        'find_package(Boost 1.70 COMPONENTS filesystem)\n',
        encoding='utf-8',
    )
    (tmp_path / 'meson.build').write_text(
        "dependency('glib-2.0', version: '>= 2.48.0')\n",
        encoding='utf-8',
    )
    cs = extract_version_constraints(str(tmp_path))
    pkgs = {c.pkg for c in cs}
    assert 'libyang' in pkgs
    assert 'Boost' in pkgs
    assert 'glib-2.0' in pkgs
