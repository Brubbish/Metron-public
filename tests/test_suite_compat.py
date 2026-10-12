from core.enhanced_dep_extractor import suite_compat as mod
from core.enhanced_dep_extractor.rules import RuleMatch


def test_source_repo_priority_prefers_main_then_updates_then_backports():
    priorities = [
        ("bookworm/main", mod._source_repo_priority("bookworm/main", "bookworm")),
        ("bookworm-updates/main", mod._source_repo_priority("bookworm-updates/main", "bookworm")),
        ("bookworm-proposed-updates/main", mod._source_repo_priority("bookworm-proposed-updates/main", "bookworm")),
        ("bookworm-backports/main", mod._source_repo_priority("bookworm-backports/main", "bookworm")),
    ]

    assert priorities[0][1] < priorities[1][1] < priorities[2][1] < priorities[3][1]


def test_comparison_plan_rejects_non_debian_semantics():
    assert mod._comparison_plan("~=", "1.2").supported is False
    assert mod._comparison_plan(">=", "1.*").supported is False


def test_comparison_plan_normalizes_bare_relations():
    assert mod._comparison_plan(">", "1.2").op == ">>"
    assert mod._comparison_plan("<", "1.2").op == "<<"


def test_compare_versions_handles_supported_and_unsupported_cases():
    assert mod._compare_versions("1.2.3-1", ">", "1.2.0") is True
    assert mod._compare_versions("1.2.3", "~=", "1.2") is False
    assert mod._compare_versions("1.2.3", ">=", "1.*") is False


def test_parse_clause_value_preserves_only_supported_literal_clauses():
    assert mod._parse_clause_value("'>= 1.2'") == [mod.VersionClause(">=", "1.2")]
    assert mod._parse_clause_value("'~= 1.2'") == [mod.VersionClause("~=", "1.2")]


def _cmake_match(matched_text, context="", file_path="CMakeLists.txt", line_num=1):
    return RuleMatch(
        file_path=file_path,
        line_num=line_num,
        rule_type="cmake:find_package_call",
        matched_text=matched_text,
        context=context,
        extracted_name="",
    )


def test_extract_cmake_constraints_uses_matched_text_not_context():
    # Regression: a greedy `\((.*)\)` over the full text (context + matched_text)
    # crossed statement boundaries and let `find_package(LibtorrentRasterbar
    # REQUIRED)` borrow Qt5's `5.2.0` from a later call in the context window.
    match = _cmake_match(
        matched_text="find_package(LibtorrentRasterbar REQUIRED)",
        context=(
            "1: find_package(LibtorrentRasterbar REQUIRED)\n"
            "2: \n"
            "3: find_package(Qt5 5.2.0 COMPONENTS ${QBT_QT_COMPONENTS} REQUIRED)"
        ),
    )
    # LibtorrentRasterbar has no version -> no constraint (NOT 5.2.0).
    assert mod._extract_cmake_constraints(match, mod._constraint_text(match, None)) == []


def test_extract_cmake_constraints_ignores_set_statements_in_context():
    # Regression: `set(CMAKE_CXX_STANDARD "11")` in context leaked the `11`
    # version into the find_package body, producing a bogus
    # `CMAKE_CXX_STANDARD_REQUIRED >= 11` constraint.
    match = _cmake_match(
        matched_text="find_package(LibtorrentRasterbar REQUIRED)",
        context=(
            '1: set(CMAKE_CXX_STANDARD_REQUIRED True)\n'
            '2: set(CMAKE_CXX_STANDARD "11")\n'
            '3: find_package(LibtorrentRasterbar REQUIRED)'
        ),
    )
    assert mod._extract_cmake_constraints(match, mod._constraint_text(match, None)) == []


def test_extract_cmake_constraints_keeps_real_version():
    match = _cmake_match(matched_text="find_package(Qt5 5.2.0 COMPONENTS Core Network REQUIRED)")
    constraints = mod._extract_cmake_constraints(match, mod._constraint_text(match, None))
    assert len(constraints) == 1
    c = constraints[0]
    assert c.dep_name == "Qt5"
    assert c.clauses == (mod.VersionClause(">=", "5.2.0"),)
    assert c.confidence == "medium"


def test_extract_cmake_constraints_handles_find_dependency():
    match = _cmake_match(
        matched_text="find_dependency(Boost 1.70)",
        line_num=5,
    )
    constraints = mod._extract_cmake_constraints(match, mod._constraint_text(match, None))
    assert len(constraints) == 1
    assert constraints[0].dep_name == "Boost"
    assert constraints[0].clauses == (mod.VersionClause(">=", "1.70"),)


def test_extract_cmake_constraints_unversioned_call_returns_empty():
    match = _cmake_match(matched_text="find_package(OpenSSL REQUIRED)")
    assert mod._extract_cmake_constraints(match, mod._constraint_text(match, None)) == []
