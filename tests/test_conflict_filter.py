from core.enhanced_dep_extractor.conflict_filter import filter_conflicting_candidates
from core.enhanced_dep_extractor.conflict_filter import filter_against_header_conflicts


def test_filter_against_header_conflicts_drops_already_resolved_candidate():
    kept, dropped = filter_against_header_conflicts(
        header_packages={"cmake"},
        llm_candidates=["cmake", "meson"],
        conflicts_by_package={},
    )

    assert kept == ["meson"]
    assert dropped == [
        {
            "candidate": "cmake",
            "reason": "already_resolved",
            "matched_header_pkg": "cmake",
            "evidence_side": "header",
        }
    ]


def test_filter_against_header_conflicts_uses_candidate_side_conflicts():
    kept, dropped = filter_against_header_conflicts(
        header_packages={"libcurl4-gnutls-dev"},
        llm_candidates=["libcurl4-openssl-dev", "cmake"],
        conflicts_by_package={
            "libcurl4-openssl-dev": ["libcurl4-gnutls-dev"],
        },
    )

    assert kept == ["cmake"]
    assert dropped == [
        {
            "candidate": "libcurl4-openssl-dev",
            "reason": "direct_conflict",
            "matched_header_pkg": "libcurl4-gnutls-dev",
            "evidence_side": "candidate",
        }
    ]


def test_filter_against_header_conflicts_uses_header_side_conflicts():
    kept, dropped = filter_against_header_conflicts(
        header_packages={"bird2"},
        llm_candidates=["bird", "cmake"],
        conflicts_by_package={
            "bird2": ["bird"],
        },
    )

    assert kept == ["cmake"]
    assert dropped == [
        {
            "candidate": "bird",
            "reason": "direct_conflict",
            "matched_header_pkg": "bird2",
            "evidence_side": "header",
        }
    ]


def test_filter_against_header_conflicts_ignores_self_conflicts():
    kept, dropped = filter_against_header_conflicts(
        header_packages={"libantic-dev"},
        llm_candidates=["other-dev"],
        conflicts_by_package={
            "libantic-dev": ["libantic-dev"],
            "other-dev": ["other-dev"],
        },
    )

    assert kept == ["other-dev"]
    assert dropped == []


def test_filter_conflicting_candidates_only_uses_candidate_side_conflicts():
    kept, dropped = filter_conflicting_candidates(
        candidates=["libcurl4-openssl-dev", "cmake"],
        resolved_packages={"libcurl4-gnutls-dev"},
        conflicts_by_package={
            "libcurl4-openssl-dev": ["libcurl4-gnutls-dev", "libssl1.0-dev"],
            "cmake": [],
        },
    )

    assert kept == ["cmake"]
    assert dropped == [
        {
            "candidate": "libcurl4-openssl-dev",
            "conflicts_with": "libcurl4-gnutls-dev",
        }
    ]


def test_filter_conflicting_candidates_ignores_self_conflicts():
    kept, dropped = filter_conflicting_candidates(
        candidates=["libantic-dev"],
        resolved_packages={"libantic-dev"},
        conflicts_by_package={
            "libantic-dev": ["libantic-dev"],
        },
    )

    assert kept == ["libantic-dev"]
    assert dropped == []
