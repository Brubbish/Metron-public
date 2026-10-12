"""Error pattern fingerprint matcher for Phase 3 fixer.

Classifies error text against known error categories using keyword matching,
then returns structured fix suggestions from the local reference files.
"""

from __future__ import annotations

import os
import re
from typing import Optional

PATTERNS_DIR = os.path.join(os.path.dirname(__file__), "error_patterns")

AVAILABLE_CATEGORIES = [
    "missing_header",
    "undefined_symbol",
    "missing_build_tool",
    "pkgconfig_failure",
    "version_mismatch",
    "compiler_error",
    "permission_path",
    "git_submodule",
    "cannot_solve",
]

# Priority-ordered fingerprint table.
# First match wins — order matters (most specific first).
FINGERPRINT_TABLE: list[tuple[list[str], str, str]] = [
    # (symptom_keywords, category_name, filename)
    (["submodule", "fatal: repository", "Failed to clone"], "git_submodule", "git_submodule.md"),
    (["undefined reference", "cannot find -l", "ld returned 1"], "undefined_symbol", "undefined_symbol.md"),
    (["cannot find -l", "cannot find -l"], "undefined_symbol", "undefined_symbol.md"),
    (["fatal error:", "No such file or directory", "cannot find include"], "missing_header", "missing_header.md"),
    (["requires version", "version too old", "need >=", "found.*need"], "version_mismatch", "version_mismatch.md"),
    (["command not found", "No such file"], "missing_build_tool", "missing_build_tool.md"),
    (["Package '", "No package '", "pkg-config.*not found"], "pkgconfig_failure", "pkgconfig_failure.md"),
    (["undeclared identifier", "implicit declaration", "no member named"], "compiler_error", "compiler_error.md"),
    (["Permission denied", "cannot create directory"], "permission_path", "permission_path.md"),
]


def classify_error(error_text: str) -> Optional[dict]:
    """Match error_text against the fingerprint table.

    Returns the first matching category dict, or None if no match.
    """
    if not error_text:
        return None
    text_lower = error_text.lower()

    for keywords, category, filename in FINGERPRINT_TABLE:
        for kw in keywords:
            if kw.lower() in text_lower:
                return {
                    "category": category,
                    "filename": filename,
                    "matched_keyword": kw,
                }
    return None


def read_pattern_file(filename: str) -> Optional[str]:
    """Read a pattern file and return its content, or None if not found."""
    path = os.path.join(PATTERNS_DIR, filename)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def query_error_pattern(error_text: str) -> dict:
    """Main entry point: classify error text and return fix suggestions.

    Returns:
        {
            "status": "matched" | "no_match",
            "category": str,
            "symptoms": str,
            "fix_pattern": str,        # raw content of the matched pattern file
            "diagnosis": str,
            "cannot_solve_hint": str,  # content of cannot_solve.md if relevant
        }
    """
    match = classify_error(error_text)
    if not match:
        return {"status": "no_match"}

    content = read_pattern_file(match["filename"])
    cannot_solve = read_pattern_file("cannot_solve.md")

    result: dict = {
        "status": "matched",
        "category": match["category"],
        "matched_keyword": match["matched_keyword"],
        "fix_pattern": content or "",
    }

    if cannot_solve:
        result["cannot_solve_hint"] = cannot_solve

    return result


def lookup_category(category: str) -> str:
    """Directly look up a category by name. Returns md file content."""
    if category not in AVAILABLE_CATEGORIES:
        return f"Unknown category: {category}. Available: {', '.join(AVAILABLE_CATEGORIES)}"
    content = read_pattern_file(f"{category}.md")
    return content or f"No content found for category: {category}"
