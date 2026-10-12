import textwrap
from pathlib import Path

from core.enhanced_dep_extractor import build_conflicts_index as mod


def test_parse_pkginfo_merges_channels_and_drops_self_conflicts(tmp_path, monkeypatch):
    download_dir = tmp_path / "Download"
    download_dir.mkdir()

    (download_dir / "bookworm_main_amd64_pkginfo").write_text(
        textwrap.dedent(
            """\
            Package: alpha
            Version: 1.0-1
            Conflicts: beta, alpha, gamma (<< 2.0), alt-a | alt-b
            Provides: virt-a, alpha

            Package: gamma
            Version: 1.5-1

            Package: dup-provider
            Version: 1.0-1
            Provides: virt-shared, virt-shared
            """
        ),
        encoding="utf-8",
    )
    (download_dir / "bookworm-updates_main_amd64_pkginfo").write_text(
        textwrap.dedent(
            """\
            Package: beta
            Version: 3.0-1

            Package: helper
            Version: 1.0-1
            Provides: virt-shared
            """
        ),
        encoding="utf-8",
    )
    (download_dir / "bookworm-backports_main_amd64_pkginfo").write_text(
        textwrap.dedent(
            """\
            Package: gamma
            Version: 2.5-1~bpo
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "pkginfo_paths", lambda distro: sorted(download_dir.glob(f"{distro}*_amd64_pkginfo")))

    conflicts, provides, versions = mod._parse_pkginfo("bookworm")

    assert conflicts == {
        "alpha": ["alt-a", "alt-b", "beta", "gamma"],
    }
    assert provides == {
        "alpha": ["alpha", "virt-a"],
        "dup-provider": ["virt-shared"],
        "helper": ["virt-shared"],
    }
    assert versions["gamma"] == {"1.5-1", "2.5-1~bpo"}


def test_build_provides_groups_deduplicates_and_skips_self_members():
    groups = mod._build_provides_groups(
        {
            "alpha": ["virt-a", "alpha"],
            "beta": ["virt-a", "virt-a"],
            "gamma": ["virt-b"],
        }
    )

    assert groups == {
        "virt-a": ["alpha", "beta"],
    }


def test_build_per_distro_conflicts_index_wraps_each_suite_payload(monkeypatch):
    monkeypatch.setattr(mod, "_available_distros", lambda: ["bookworm", "trixie"])
    payload_by_distro = {
        "bookworm": (
            {"alpha": ["beta"]},
            {"pkg-a": ["virt-a"], "pkg-b": ["virt-a"], "pkg-c": ["virt-b"]},
            {},
        ),
        "trixie": ({}, {}, {}),
    }
    monkeypatch.setattr(mod, "_parse_pkginfo", lambda distro: payload_by_distro[distro])

    payloads = mod.build_per_distro_conflicts_index()

    assert payloads == {
        "bookworm": {
            "conflicts": {"alpha": ["beta"]},
            "provides_groups": {"virt-a": ["pkg-a", "pkg-b"]},
        },
        "trixie": {
            "conflicts": {},
            "provides_groups": {},
        },
    }


def test_comparison_operator_aliases_support_bare_lt_and_gt():
    versions = {
        "beta": {"1.0-1"},
        "gamma": {"3.0-1"},
    }

    assert mod._conflict_is_active("beta", "<", "2.0", versions) is True
    assert mod._conflict_is_active("gamma", ">", "2.0", versions) is True
    assert mod._conflict_is_active("gamma", "<", "2.0", versions) is False
