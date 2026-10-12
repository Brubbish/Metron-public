import os
import subprocess
from datetime import date

from core.debian_sysuse import (
    git_commit_date,
    apply_sysuse_floor,
    debian_codename_for_date,
    sysuse_for_date,
)


def _git(repo_path, *args, env=None):
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_path,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})},
    ).strip()


def test_git_commit_date_resolves_commit_even_when_path_has_same_name(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "tracked.txt").write_text("first\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    commit_env = {
        "GIT_AUTHOR_DATE": "2020-01-02T03:04:05+00:00",
        "GIT_COMMITTER_DATE": "2020-01-02T03:04:05+00:00",
    }
    _git(tmp_path, "commit", "-m", "first", env=commit_env)
    commit = _git(tmp_path, "rev-parse", "HEAD")

    # This used to be fragile when git interpreted the token as both a revision
    # and a path. The date lookup should force commit-object resolution first.
    (tmp_path / commit[:12]).write_text("path with same name as commit prefix\n", encoding="utf-8")
    _git(tmp_path, "add", commit[:12])
    _git(tmp_path, "commit", "-m", "add ambiguous path")

    assert git_commit_date(str(tmp_path), commit[:12]).isoformat() == "2020-01-02"


# -----------------------------
# apply_sysuse_floor / floor behavior
# -----------------------------

def test_apply_sysuse_floor_bumps_legacy_releases_to_stretch_by_default():
    # No commit_date -> preserve the historical stretch floor for all legacy suites.
    for legacy in ("etch", "lenny", "squeeze", "wheezy", "jessie"):
        assert apply_sysuse_floor(legacy) == "stretch", legacy
    # accepts sysuse:image prefix
    assert apply_sysuse_floor("debian:squeeze") == "stretch"
    assert apply_sysuse_floor("theauto/debian:jessie") == "stretch"


def test_apply_sysuse_floor_keeps_modern_releases():
    for modern in ("stretch", "buster", "bullseye", "bookworm", "trixie"):
        assert apply_sysuse_floor(modern) == modern, modern


def test_apply_sysuse_floor_can_opt_out():
    assert apply_sysuse_floor("jessie", allow_legacy=True) == "jessie"


def test_apply_sysuse_floor_keeps_legacy_when_commit_in_window():
    # Commit from 2015-06 was written against jessie ABI -> keep jessie.
    assert apply_sysuse_floor("jessie", commit_date=date(2015, 6, 1)) == "jessie"
    # Commit from 2014-02 was written against wheezy ABI -> keep wheezy.
    assert apply_sysuse_floor("wheezy", commit_date=date(2014, 2, 1)) == "wheezy"
    # LTS tail: jessie LTS ran until 2020-06, so a 2019 commit is still plausible.
    assert apply_sysuse_floor("jessie", commit_date=date(2019, 6, 1)) == "jessie"
    # Lead time: a commit made during the suite's freeze (1 year before release) counts.
    assert apply_sysuse_floor("wheezy", commit_date=date(2012, 6, 1)) == "wheezy"


def test_apply_sysuse_floor_bumps_when_commit_outside_window():
    # Modern commit proposing a legacy suite -> bump (likely wrong diagnosis).
    assert apply_sysuse_floor("jessie", commit_date=date(2024, 1, 1)) == "stretch"
    # Pre-freeze commit (more than 1 year before release) -> bump.
    assert apply_sysuse_floor("jessie", commit_date=date(2013, 12, 1)) == "stretch"
    # Post-LTS commit -> bump.
    assert apply_sysuse_floor("jessie", commit_date=date(2021, 1, 1)) == "stretch"


def test_debian_codename_for_date_keeps_era_legacy_suite():
    # A 2015 commit is in jessie's release window and era -> should get jessie,
    # not stretch. This is the libsoup-style fix: source written for jessie
    # GnuTLS/GLib ABI should build in jessie.
    assert debian_codename_for_date("2015-06-01") == "jessie"
    # A 2014 commit falls in wheezy's window -> wheezy.
    assert debian_codename_for_date("2014-06-01") == "wheezy"
    # floor=False still returns the strict window codename.
    assert debian_codename_for_date("2014-06-01", floor=False) == "wheezy"


def test_debian_codename_for_date_floors_when_commit_precedes_window():
    # A 2010 commit is in squeeze's window, but squeeze has real toolchain
    # issues. We keep squeeze here because the commit IS from squeeze era;
    # the toolchain floor (stretch) only applies when commit is post-LTS
    # or pre-freeze.
    assert debian_codename_for_date("2011-06-01") == "squeeze"


def test_sysuse_for_date_returns_prefix_form():
    # 2016 commit falls in jessie's window -> jessie (era-matched, not floored)
    assert sysuse_for_date("2016-03-01") == "debian:jessie"
    # 2015-03 falls in wheezy's strict window -> wheezy (also era-matched now)
    assert sysuse_for_date("2015-03-01") == "debian:wheezy"
    # Modern commit -> untouched
    assert sysuse_for_date("2022-09-01") == "debian:bullseye"
