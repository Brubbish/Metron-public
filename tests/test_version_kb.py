import json

from rag.import_to_postgres import load_version_kb_rows
from rag.version_kb import (
    RepoTarget,
    build_binary_source_rows,
    build_source_version_rows,
    group_source_versions_by_distro,
    merge_binary_records,
    merge_source_records,
    normalize_source_field,
    parse_binary_packages_text,
    parse_source_label,
    parse_source_packages_text,
    resolve_query_candidates,
)


def test_normalize_source_field_splits_inline_version():
    source_pkg, source_version_hint, source_raw = normalize_source_field(
        "bash (5.2.15-2)",
        fallback_name="bash",
    )
    assert source_pkg == "bash"
    assert source_version_hint == "5.2.15-2"
    assert source_raw == "bash (5.2.15-2)"


def test_parse_binary_packages_text_extracts_source_mapping():
    text = """
Package: bash
Version: 5.2.15-2+b10
Source: bash (5.2.15-2)
Description: GNU Bourne Again SHell

Package: curl
Version: 7.88.1-10+deb12u14
Description: command line tool
""".strip()

    records = parse_binary_packages_text(text)

    assert records == [
        {
            "package": "bash",
            "binary_pkg": "bash",
            "description": "GNU Bourne Again SHell",
            "source": "bash",
            "source_pkg": "bash",
            "source_raw": "bash (5.2.15-2)",
            "source_version_hint": "5.2.15-2",
            "version": "5.2.15-2+b10",
        },
        {
            "package": "curl",
            "binary_pkg": "curl",
            "description": "command line tool",
            "source": "curl",
            "source_pkg": "curl",
            "source_raw": "",
            "source_version_hint": "",
            "version": "7.88.1-10+deb12u14",
        },
    ]


def test_parse_source_packages_text_extracts_binary_packages():
    text = """
Package: openssl
Binary: libssl-dev, libssl3, openssl
Version: 3.0.18-1~deb12u1
""".strip()

    records = parse_source_packages_text(text)

    assert records == [
        {
            "package": "openssl",
            "source_pkg": "openssl",
            "version": "3.0.18-1~deb12u1",
            "binary_packages": ["libssl-dev", "libssl3", "openssl"],
        }
    ]


def test_parse_source_label_splits_suite_repo_and_channel():
    assert parse_source_label("bookworm/main", "bookworm") == {
        "suite": "bookworm",
        "repo_name": "bookworm",
        "component": "main",
        "channel": "main",
        "is_primary": True,
        "source_repo": "bookworm/main",
    }
    assert parse_source_label("bookworm-updates/main", "bookworm") == {
        "suite": "bookworm",
        "repo_name": "bookworm-updates",
        "component": "main",
        "channel": "updates",
        "is_primary": False,
        "source_repo": "bookworm-updates/main",
    }
    assert parse_source_label("bookworm-backports/main", "bookworm") == {
        "suite": "bookworm",
        "repo_name": "bookworm-backports",
        "component": "main",
        "channel": "backports",
        "is_primary": False,
        "source_repo": "bookworm-backports/main",
    }


def test_build_rows_keep_source_versions_and_binary_mapping_separate():
    binary_target = RepoTarget(
        base_url="https://deb.debian.org/debian",
        suite="bookworm",
        repo_name="bookworm",
        channel="main",
        component="main",
        kind="binary",
        arch="amd64",
    )
    source_target = RepoTarget(
        base_url="https://deb.debian.org/debian",
        suite="bookworm",
        repo_name="bookworm",
        channel="main",
        component="main",
        kind="source",
    )

    binary_store = {}
    merge_binary_records(
        binary_store,
        [
            {
                "package": "bash",
                "binary_pkg": "bash",
                "description": "GNU Bourne Again SHell",
                "source": "bash",
                "source_pkg": "bash",
                "source_raw": "bash (5.2.15-2)",
                "source_version_hint": "5.2.15-2",
                "version": "5.2.15-2+b10",
            }
        ],
        binary_target,
    )

    source_store = {}
    merge_source_records(
        source_store,
        [
            {
                "package": "bash",
                "source_pkg": "bash",
                "version": "5.2.15-2",
                "binary_packages": ["bash"],
            }
        ],
        source_target,
    )

    source_rows = build_source_version_rows({"bookworm": source_store})
    binary_rows = build_binary_source_rows({"bookworm": binary_store})

    assert source_rows == [
        {
            "suite": "bookworm",
            "repo_name": "bookworm",
            "channel": "main",
            "component": "main",
            "source_pkg": "bash",
            "source_version": "5.2.15-2",
            "binary_packages": ["bash"],
            "is_primary": True,
            "version_origin": "sources",
        }
    ]
    assert binary_rows == [
        {
            "suite": "bookworm",
            "repo_name": "bookworm",
            "channel": "main",
            "component": "main",
            "arch": "amd64",
            "binary_pkg": "bash",
            "source_pkg": "bash",
            "description": "GNU Bourne Again SHell",
            "is_primary": True,
        }
    ]


def test_build_binary_source_rows_keeps_one_mapping_per_channel():
    pkg_payload = {
        "bookworm": {
            "libssl-dev": {
                "source_pkg": "openssl",
                "description": "Secure Sockets Layer toolkit",
                "version": [
                    {"version": "3.0.18-1~deb12u1", "source_repo": "bookworm/main"},
                    {"version": "3.0.17-1~deb12u2", "source_repo": "bookworm-updates/main"},
                    {"version": "3.0.18-1~deb12u1", "source_repo": "bookworm/main"},
                ],
            }
        }
    }

    rows = build_binary_source_rows(pkg_payload)

    assert rows == [
        {
            "suite": "bookworm",
            "repo_name": "bookworm",
            "channel": "main",
            "component": "main",
            "arch": "amd64",
            "binary_pkg": "libssl-dev",
            "source_pkg": "openssl",
            "description": "Secure Sockets Layer toolkit",
            "is_primary": True,
        },
        {
            "suite": "bookworm",
            "repo_name": "bookworm-updates",
            "channel": "updates",
            "component": "main",
            "arch": "amd64",
            "binary_pkg": "libssl-dev",
            "source_pkg": "openssl",
            "description": "Secure Sockets Layer toolkit",
            "is_primary": False,
        },
    ]


def test_load_version_kb_rows_builds_expected_row_counts(tmp_path):
    pkglist_dir = tmp_path / "pkglist"
    sourcelist_dir = tmp_path / "sourcelist"
    pkglist_dir.mkdir()
    sourcelist_dir.mkdir()

    (pkglist_dir / "bookworm_pkglist_amd64.json").write_text(
        json.dumps(
            {
                "curl": {
                    "source_pkg": "curl",
                    "description": "command line tool",
                    "version": [
                        {"version": "7.88.1-10+deb12u14", "source_repo": "bookworm/main"},
                        {"version": "7.88.1-10+deb12u15", "source_repo": "bookworm-updates/main"},
                    ],
                },
                "libssl-dev": {
                    "source_pkg": "openssl",
                    "description": "OpenSSL development files",
                    "version": [{"version": "3.0.18-1~deb12u1", "source_repo": "bookworm/main"}],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (pkglist_dir / "trixie_pkglist_amd64.json").write_text(
        json.dumps(
            {
                "curl": {
                    "source_pkg": "curl",
                    "description": "command line tool",
                    "version": [{"version": "8.14.1-2+deb13u2", "source_repo": "trixie/main"}],
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sourcelist_dir / "bookworm_sourcelist.json").write_text(
        json.dumps(
            {
                "curl": {
                    "version": [
                        {
                            "version": "7.88.1-10+deb12u14",
                            "source_repo": "bookworm/main",
                            "binary_packages": ["curl", "libcurl4"],
                        },
                        {
                            "version": "7.88.1-10+deb12u15",
                            "source_repo": "bookworm-updates/main",
                            "binary_packages": ["curl", "libcurl4"],
                        },
                    ]
                },
                "openssl": {
                    "version": [
                        {
                            "version": "3.0.18-1~deb12u1",
                            "source_repo": "bookworm/main",
                            "binary_packages": ["libssl-dev", "libssl3", "openssl"],
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sourcelist_dir / "trixie_sourcelist.json").write_text(
        json.dumps(
            {
                "curl": {
                    "version": [
                        {
                            "version": "8.14.1-2+deb13u2",
                            "source_repo": "trixie/main",
                            "binary_packages": ["curl", "libcurl4t64"],
                        }
                    ]
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    source_rows, binary_rows = load_version_kb_rows(str(tmp_path))

    assert len(source_rows) == 4
    assert len(binary_rows) == 4
    assert [row["repo_name"] for row in source_rows if row["source_pkg"] == "curl" and row["suite"] == "bookworm"] == [
        "bookworm",
        "bookworm-updates",
    ]
    assert {row["binary_pkg"]: row["source_pkg"] for row in binary_rows if row["suite"] == "bookworm" and row["repo_name"] == "bookworm"} == {
        "curl": "curl",
        "libssl-dev": "openssl",
    }


def test_resolve_query_candidates_prefers_source_exact_over_binary_exact():
    result = resolve_query_candidates(
        source_candidates=[
            {
                "matched_name": "curl",
                "resolved_source_pkg": "curl",
                "exact_match": True,
                "score": 1.0,
                "has_distro_hit": True,
            }
        ],
        binary_candidates=[
            {
                "matched_name": "curl",
                "resolved_source_pkg": "curl",
                "exact_match": True,
                "score": 1.0,
                "has_distro_hit": True,
            }
        ],
    )

    assert result == {
        "resolved_source_pkg": "curl",
        "match": {
            "matched_name": "curl",
            "matched_kind": "source",
            "match_type": "exact",
            "score": 1.0,
        },
    }


def test_resolve_query_candidates_maps_binary_exact_to_source_package():
    result = resolve_query_candidates(
        source_candidates=[],
        binary_candidates=[
            {
                "matched_name": "libssl-dev",
                "resolved_source_pkg": "openssl",
                "exact_match": True,
                "score": 1.0,
                "has_distro_hit": True,
            }
        ],
    )

    assert result == {
        "resolved_source_pkg": "openssl",
        "match": {
            "matched_name": "libssl-dev",
            "matched_kind": "binary",
            "match_type": "exact",
            "score": 1.0,
        },
    }


def test_resolve_query_candidates_prefers_distro_hit_within_same_match_tier():
    result = resolve_query_candidates(
        source_candidates=[
            {
                "matched_name": "curlplusplus",
                "resolved_source_pkg": "curlplusplus",
                "exact_match": False,
                "score": 0.91,
                "has_distro_hit": False,
            },
            {
                "matched_name": "curl",
                "resolved_source_pkg": "curl",
                "exact_match": False,
                "score": 0.85,
                "has_distro_hit": True,
            },
        ],
        binary_candidates=[],
        distro_filter="bookworm",
    )

    assert result == {
        "resolved_source_pkg": "curl",
        "match": {
            "matched_name": "curl",
            "matched_kind": "source",
            "match_type": "fuzzy",
            "score": 0.85,
        },
    }


def test_resolve_query_candidates_compares_fuzzy_score_across_source_and_binary():
    result = resolve_query_candidates(
        source_candidates=[
            {
                "matched_name": "libcue",
                "resolved_source_pkg": "libcue",
                "exact_match": False,
                "score": 0.5,
                "has_distro_hit": True,
            }
        ],
        binary_candidates=[
            {
                "matched_name": "libcurl4",
                "resolved_source_pkg": "curl",
                "exact_match": False,
                "score": 0.875,
                "has_distro_hit": True,
            }
        ],
        distro_filter="bookworm",
    )

    assert result == {
        "resolved_source_pkg": "curl",
        "match": {
            "matched_name": "libcurl4",
            "matched_kind": "binary",
            "match_type": "fuzzy",
            "score": 0.875,
        },
    }


def test_group_source_versions_by_distro_keeps_all_channels_sorted():
    rows = [
        {
            "suite": "bookworm",
            "repo_name": "bookworm-backports",
            "channel": "backports",
            "component": "main",
            "source_version": "8.14.1-2+deb13u2~bpo12+1",
            "binary_packages": ["curl"],
            "is_primary": False,
        },
        {
            "suite": "bookworm",
            "repo_name": "bookworm-updates",
            "channel": "updates",
            "component": "main",
            "source_version": "7.88.1-10+deb12u15",
            "binary_packages": ["curl"],
            "is_primary": False,
        },
        {
            "suite": "bookworm",
            "repo_name": "bookworm",
            "channel": "main",
            "component": "main",
            "source_version": "7.88.1-10+deb12u14",
            "binary_packages": ["curl"],
            "is_primary": True,
        },
        {
            "suite": "trixie",
            "repo_name": "trixie",
            "channel": "main",
            "component": "main",
            "source_version": "8.14.1-2+deb13u2",
            "binary_packages": ["curl"],
            "is_primary": True,
        },
    ]

    grouped = group_source_versions_by_distro(rows)

    assert list(grouped) == ["bookworm", "trixie"]
    assert [entry["repo_name"] for entry in grouped["bookworm"]] == [
        "bookworm",
        "bookworm-updates",
        "bookworm-backports",
    ]
    assert grouped["bookworm"][0]["version"] == "7.88.1-10+deb12u14"
    assert grouped["bookworm"][1]["version"] == "7.88.1-10+deb12u15"
    assert grouped["trixie"][0]["version"] == "8.14.1-2+deb13u2"
