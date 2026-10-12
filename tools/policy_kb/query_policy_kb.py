"""
Query_Policy_KB - Policy card retrieval engine for build error diagnosis.

Given error lines and build context, finds matching policy cards that explain
why old code fails on newer toolchains and suggests workarounds.
"""

import json
import re
import sys
import os
from typing import List, Dict, Optional, Any

# Import the mapping tables from the materialized fix-decision matrix so that
# the matrix is the single source of truth.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from diagnosis.fix_decision_matrix import SYMPTOM_PROBLEM_HINTS as _SYMPTOM_PROBLEM_HINTS
from diagnosis.fix_decision_matrix import REVISION_ACTION_HINTS as _REVISION_ACTION_HINTS
from diagnosis.fix_decision_matrix import SYMPTOM_ID_HINTS as _SYMPTOM_ID_HINTS
from diagnosis.fix_decision_matrix import CANONICAL_ERROR_TYPES, canonical_error_type, normalize_card_record, normalize_phase
from diagnosis.diagnosis_signals import (
    VERSION_REQUIREMENT_PATTERN,
    has_compiler_policy,
    has_missing_tool,
    has_tool_configuration_failure,
    is_wrong_entry_or_target,
)
from core.repair_plan import card_matches_route


# Cost of each action type — used for sorting candidate cards so that cheap
# fixes (build flags) are tried before expensive ones (sysuse switch, source
# build). Derived purely from the card's action.type field, no manual data.
ACTION_TYPE_COST = {
    "build_flag": 1,
    "env_var": 2,
    "build_system_edit": 3,
    "source_fix": 3,
    "source_edit": 3,
    "sysuse": 5,
    "source_build": 6,
    "pkg_install": 4,
    "feature_disable": 4,
    # Meta-card action types (see library_abi_drift_diagnostic and
    # toolchain_behavior_block cards). Ranked so cheap probes sort
    # before expensive ones.
    "diagnostic_flow": 1,   # guides LLM investigation, ~free
    "tool_shim": 2,         # write a wrapper script under /tmp/
    "alt_pkg": 4,           # apt install a versioned/alternate package
    "source_dep": 6,        # build dependency from source
}

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


_PROBLEM_CLASS_ERROR_TYPES = {
    "compiler_toolchain_version": {"compiler_policy_rejection", "api_or_toolchain_incompatibility"},
    "build_system_policy": {"missing_generated_artifacts", "build_input_parsing_failure", "compiler_policy_rejection"},
    "dependency_resolution": {"dependency_discovery_failure", "header_resolution_failure", "library_resolution_failure"},
    "dependency_resolution_change": {"dependency_discovery_failure", "api_or_toolchain_incompatibility"},
    "missing_include": {"header_resolution_failure"},
    "linker_error": {"unresolved_link_symbols"},
    "library_abi_drift": {"library_resolution_failure", "api_or_toolchain_incompatibility", "unresolved_link_symbols"},
    "library_path_change": {"library_resolution_failure"},
    "source_dependency": {"dependency_discovery_failure", "build_input_parsing_failure"},
    "wrong_entrypoint": {"incorrect_build_entry_or_target"},
    "filename_change": {"missing_generated_artifacts", "build_input_parsing_failure"},
    "filesystem_layout_change": {"build_input_parsing_failure", "api_or_toolchain_incompatibility"},
    "tool_unavailable": {"dependency_discovery_failure"},
    "unsupported_protocol": {"api_or_toolchain_incompatibility"},
    "unsupported_upgrade": {"api_or_toolchain_incompatibility"},
    "deprecation_warning": {"compiler_policy_rejection", "api_or_toolchain_incompatibility"},
    "dependency_loop": {"dependency_discovery_failure"},
    "multiarch_not_enabled": {"system_package_installation_failure"},
    "component_split": {"dependency_discovery_failure"},
    "held_package": {"system_package_installation_failure"},
    "file_conflict": {"system_package_installation_failure"},
    "package_removed": {"system_package_installation_failure"},
    "silent_hold_back": {"system_package_installation_failure"},
    "output_format_change": {"build_input_parsing_failure"},
    "benign_warning": {"compiler_policy_rejection"},
}

_ACTION_ERROR_TYPES = {
    "install_system_package": {
        "system_package_installation_failure", "dependency_discovery_failure",
        "header_resolution_failure", "library_resolution_failure",
    },
    "build_dependency_source": {
        "dependency_discovery_failure", "header_resolution_failure",
        "library_resolution_failure", "build_input_parsing_failure",
    },
    "execute_generator": {"missing_generated_artifacts", "build_input_parsing_failure"},
    "dependency_discovery_config": {
        "dependency_discovery_failure", "header_resolution_failure", "library_resolution_failure",
    },
    "linker_config": {"library_resolution_failure", "unresolved_link_symbols"},
    "switch_build_entry": {"incorrect_build_entry_or_target"},
    "switch_build_target": {"incorrect_build_entry_or_target"},
    "compiler_config": {"compiler_policy_rejection", "api_or_toolchain_incompatibility"},
}


def _to_tcodes(names: Any) -> set:
    """Normalize a legacy-row/T-code name collection to canonical T-codes."""
    normalized = set()
    for item in names or []:
        code = canonical_error_type(str(item))
        if code != "unknown":
            normalized.add(code)
    return normalized


# Legacy table values name pre-FDM-v5 matrix rows; normalize them once at
# import so every candidate set below is keyed by canonical T-codes.
_PROBLEM_CLASS_ERROR_TYPES = {
    problem_class: _to_tcodes(types)
    for problem_class, types in _PROBLEM_CLASS_ERROR_TYPES.items()
}

_ACTION_ERROR_TYPES = {
    action: _to_tcodes(types)
    for action, types in _ACTION_ERROR_TYPES.items()
}


def card_error_type_candidates(card: Dict[str, Any]) -> List[str]:
    """Map one normalized card to matrix error-type candidates."""
    explicit = _to_tcodes(card.get("canonical_error_types") or [])
    if explicit:
        return [item for item in CANONICAL_ERROR_TYPES if item in explicit]
    problem_types = set(_PROBLEM_CLASS_ERROR_TYPES.get(str(card.get("problem_class") or ""), set()))
    action_types: set[str] = set()
    for action in card.get("matrix_actions") or []:
        if not isinstance(action, dict):
            continue
        action_types.update(_ACTION_ERROR_TYPES.get(str(action.get("matrix_action") or ""), set()))
    candidates = problem_types | action_types
    return [item for item in CANONICAL_ERROR_TYPES if item in candidates]

# _REVISION_ACTION_HINTS, _SYMPTOM_PROBLEM_HINTS, and _SYMPTOM_ID_HINTS are
# imported from fix_decision_matrix at the top of this file.


def card_cost(card: dict) -> int:
    """Cheapest action cost across the card's actions (min, since a card may
    carry multiple actions and we want the cheapest viable path)."""
    types = [a.get("type") for a in card.get("actions", []) if isinstance(a, dict)]
    costs = [ACTION_TYPE_COST.get(t, 99) for t in types if t]
    return min(costs) if costs else 99


def _has_route_action_match(card: dict, matrix_actions: Optional[List[str]]) -> bool:
    """Whether a card explicitly supports one routed Matrix action.

    This is deliberately a ranking signal.  A routed action is the best
    current hypothesis, not proof that all other evidence-backed cards are
    inapplicable.
    """
    routed = {str(item) for item in matrix_actions or [] if item}
    if not routed:
        return False
    card_actions = {
        str(item.get("matrix_action") or "")
        for item in card.get("matrix_actions") or []
        if isinstance(item, dict)
    }
    return bool(routed & card_actions)


def route_alignment_score(
    card: dict,
    symptom_family: Optional[str] = None,
    candidate_decisions: Optional[List[str]] = None,
    allowed_revision_types: Optional[List[str]] = None,
    provenance_probe: Optional[Dict[str, Any]] = None,
    matrix_actions: Optional[List[str]] = None,
) -> int:
    """Conservative soft score for cards aligned with a diagnosis route.

    This does not filter cards. The current corpus is mostly retrieval-level,
    so the score combines explicit future `surfaces` metadata when present
    with small heuristics over problem_class/action/id fields.
    """
    if not symptom_family and not candidate_decisions and not allowed_revision_types and not provenance_probe and not matrix_actions:
        return 0

    score = 0
    symptom = symptom_family or ""
    decisions = set(candidate_decisions or [])
    revisions = set(allowed_revision_types or [])
    actions = {
        str(action.get("type") or "")
        for action in card.get("actions", [])
        if isinstance(action, dict)
    }
    problem_class = str(card.get("problem_class") or "")
    card_id = str(card.get("id") or "")
    card_matrix_actions = {
        str(item.get("matrix_action") or "")
        for item in card.get("matrix_actions") or []
        if isinstance(item, dict)
    }
    if matrix_actions:
        score += 5 * len(card_matrix_actions & set(matrix_actions))

    for surface in card.get("surfaces", []) or []:
        if not isinstance(surface, dict):
            continue
        if symptom and surface.get("symptom_family") == symptom:
            score += 4
        surface_decisions = set(surface.get("related_plan_decisions") or [])
        if decisions and surface_decisions:
            score += min(len(decisions & surface_decisions), 3)

    if symptom and problem_class in _SYMPTOM_PROBLEM_HINTS.get(symptom, set()):
        score += 2
    if symptom:
        for token in _SYMPTOM_ID_HINTS.get(symptom, ()):
            if token in card_id:
                score += 1

    for revision in revisions:
        if actions & _REVISION_ACTION_HINTS.get(revision, set()):
            score += 2

    if "source_dependency" in decisions and (problem_class == "source_dependency" or "source_dep" in actions):
        score += 3
    if "build_entry" in decisions and problem_class == "wrong_entrypoint":
        score += 3
    if "system_package_provider" in decisions and problem_class in {"dependency_resolution", "library_abi_drift"}:
        score += 2
    if "compiler_configuration" in decisions and actions & {"build_flag", "tool_shim"}:
        score += 1
    if "linker_configuration" in decisions and actions & {"build_flag", "tool_shim", "diagnostic_flow"}:
        score += 1
    if "system_release" in decisions and "sysuse" in actions:
        score += 1

    if provenance_probe:
        score += _probe_alignment_score(card, provenance_probe, actions, problem_class)

    return score


def _probe_alignment_score(
    card: dict,
    provenance_probe: Dict[str, Any],
    actions: set[str],
    problem_class: str,
) -> int:
    provenance = str(provenance_probe.get("provenance") or "")
    confidence = str(provenance_probe.get("confidence") or "low")
    preferred = set(provenance_probe.get("preferred_decisions") or [])
    discouraged = set(provenance_probe.get("discouraged_decisions") or [])
    if confidence == "low":
        return 0
    score = 0
    if provenance == "project_managed":
        if problem_class == "source_dependency" or "source_dep" in actions:
            score += 5
        if "system_release" in discouraged and "sysuse" in actions:
            score -= 3
        if "system_package_provider" in discouraged and problem_class == "dependency_resolution":
            score -= 2
    elif provenance == "repo_local_invisible":
        if actions & {"build_flag", "tool_shim", "diagnostic_flow"}:
            score += 3
        if "sysuse" in actions:
            score -= 2
    elif provenance == "generated":
        if problem_class in {"wrong_entrypoint", "build_system_policy"} or "diagnostic_flow" in actions:
            score += 3
        if "sysuse" in actions:
            score -= 2
    elif provenance == "link_composition":
        if actions & {"build_flag", "tool_shim", "diagnostic_flow"}:
            score += 3
    elif provenance == "system_or_external":
        if problem_class in {"dependency_resolution", "library_abi_drift", "compiler_toolchain_version"}:
            score += 2
        if "sysuse" in actions:
            score += 1
    if "source_dependency" in preferred and (problem_class == "source_dependency" or "source_dep" in actions):
        score += 2
    if "linker_configuration" in preferred and actions & {"build_flag", "tool_shim", "diagnostic_flow"}:
        score += 1
    return score


def parse_version(v: str) -> Optional[tuple]:
    """Parse version string to comparable tuple."""
    if v is None:
        return None
    if v.startswith("CMP"):
        return (int(v[3:]),)
    parts = re.findall(r'\d+', str(v))
    if not parts:
        return None
    return tuple(int(p) for p in parts)


def version_gte(sys_ver: str, card_vmin: str) -> bool:
    """Check if system version >= card's version_min."""
    sv = parse_version(sys_ver)
    cv = parse_version(card_vmin)
    if sv is None or cv is None:
        return False
    return sv >= cv


def _is_regex_pattern(pattern: str) -> bool:
    """Detect whether a pattern should be treated as regex (not literal).

    A pattern is regex if it uses:
    - alternation ``|`` (e.g. ``(a|b|c)``)
    - regex escape syntax (``\\[`` for literal bracket, ``\\d``, ``.*``, ``.+``,
      ``\\b``, ``\\.``)
    - a bare character class ``[...]`` (e.g. ``[A-Za-z0-9_+.-]+``)

    Patterns with a bare ``[`` that is not part of a character class (e.g.
    ``[-Werror=...]``) are detected too: the resulting regex compiles and
    behaves as a character class, which for these patterns is acceptable
    because the character class happens to match the same text.
    """
    if '|' in pattern:
        return True
    if any(token in pattern for token in (
        '.*', '.+', r'\d', r'\s', r'\w', r'\b', r'\.', r'\[', r'\]',
    )):
        return True
    # Bare [ (not preceded by backslash) → likely a character class.
    if re.search(r'(?<!\\)\[', pattern):
        return True
    return False


def _compile_pattern(pattern: str):
    """Compile a trigger/negative pattern into a regex.

    Regex patterns are compiled as-is.  Literal patterns are ``re.escape``-d
    so that ``(``, ``[``, ``C++`` etc. match as text.  On compile failure,
    falls back to ``re.escape`` (preserves substring semantics).
    """
    try:
        if _is_regex_pattern(pattern):
            return re.compile(pattern, re.IGNORECASE)
        return re.compile(re.escape(pattern), re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def pattern_match(pattern: str, lines: List[str]) -> bool:
    """Check if a pattern appears in any line.

    Regex patterns (``|``, escape syntax, bare character classes) use native
    regex.  Literal patterns are escaped so ``(``, ``[``, ``+`` match as text.
    All matching is case-insensitive.
    """
    compiled = _compile_pattern(pattern)
    return any(compiled.search(line) for line in lines)


def _diagnosis_context_supported(card: Dict[str, Any], error_lines: List[str]) -> bool:
    """Reject broad diagnosis-card hits that lack their mechanism context."""
    problem_class = str(card.get("problem_class") or "").strip().lower()
    text = "\n".join(str(line or "") for line in error_lines)
    if problem_class == "wrong_entrypoint":
        return is_wrong_entry_or_target(text)
    if problem_class == "source_dependency":
        return bool(re.search(
            r"(?:^|/|\b)(?:third_party|3rdparty|deps|vendor|external|submodules?)(?:/|\b)|"
            r"\b(?:subproject|wrap file|source dependency|vendored dependency)\b",
            text,
            re.I,
        ))
    if problem_class == "tool_unavailable":
        return has_missing_tool(text)
    if problem_class in {"compiler_toolchain_version", "deprecation_warning", "benign_warning"}:
        return bool(
            has_compiler_policy(text)
            or has_tool_configuration_failure(text)
            or re.search(VERSION_REQUIREMENT_PATTERN, text, re.I)
            or re.search(r"no member named|too (?:few|many) arguments to function", text, re.I)
        )
    return True


def _disambiguator_key(d: Dict[str, Any]) -> str:
    """Build a stable cache key for a disambiguator spec.

    Examples:
        {"probe": "header_available", "header": "grp.h"}
            -> "header_available:header=grp.h"
        {"probe": "tool_version", "tool": "gcc", "op": "gte", "value": "14"}
            -> "tool_version:op=gte|tool=gcc|value=14"
    """
    probe = d.get("probe", "")
    args = sorted(
        (k, v) for k, v in d.items() if k not in ("probe", "expect")
    )
    arg_str = "|".join(f"{k}={v}" for k, v in args)
    return f"{probe}:{arg_str}"


def _check_disambiguator(d: Dict[str, Any], probe_results: Dict[str, Any]) -> Optional[bool]:
    """Check one disambiguator against probe_results.

    Returns:
        True  — disambiguator passed (probe ran, condition holds)
        False — disambiguator failed (probe ran, condition does NOT hold)
        None  — probe result missing (don't drop card on missing data)
    """
    key = _disambiguator_key(d)
    expected = d.get("expect", True)
    actual = probe_results.get(key, None)
    if actual is None:
        return None

    # Version comparisons carry op + value
    op = d.get("op")
    if op and isinstance(actual, str):
        actual_tuple = parse_version(actual)
        target_tuple = parse_version(str(d.get("value", "")))
        if actual_tuple is None or target_tuple is None:
            return None
        if op == "gte":
            result = actual_tuple >= target_tuple
        elif op == "lt":
            result = actual_tuple < target_tuple
        elif op == "eq":
            result = actual_tuple == target_tuple
        else:
            return None
        return result == bool(expected)
    return bool(actual) == bool(expected)


class PolicyKB:
    def __init__(self, kb_dir: str = None):
        if kb_dir is None:
            kb_dir = os.path.dirname(os.path.abspath(__file__))

        # Load debian version mapping
        with open(os.path.join(kb_dir, 'debian_versions.json')) as f:
            self.debian_versions = json.load(f)

        # Generated cards remain immutable.  Small mechanism-level handcrafted
        # cards fill matrix actions that official tool release notes do not
        # cover, notably package realization and bootstrap prerequisites.
        cards_by_id: Dict[str, Dict[str, Any]] = {}
        for filename in ('cards.jsonl', 'cards_handcrafted.jsonl'):
            path = os.path.join(kb_dir, filename)
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    card = normalize_card_record(json.loads(line))
                    card["canonical_error_type_candidates"] = card_error_type_candidates(card)
                    card["canonical_error_types"] = list(card["canonical_error_type_candidates"])
                    cards_by_id[str(card.get('id') or '')] = card
        self.cards = [card for card_id, card in cards_by_id.items() if card_id]

    def get_sys_versions(self, sysuse: str) -> Dict[str, Optional[str]]:
        """Get tool versions for a given Debian release."""
        return self.debian_versions.get(sysuse, {})

    def query(self,
              error_lines: List[str],
              sysuse: str,
              failed_phase: str = "compile",
              project_language: List[str] = None,
              build_system: str = None,
              top_k: int = 3) -> Dict[str, Any]:
        """
        Query the policy KB for matching cards.

        Args:
            error_lines: List of error messages from build log
            sysuse: Debian codename (e.g. "trixie", "bookworm")
            failed_phase: "compile", "link", or "configure"
            project_language: ["c"], ["c++"], or ["c", "c++"]
            build_system: "cmake", "meson", "autotools", "make" (optional)
            top_k: Number of top matches to return

        Returns:
            {
                "status": "matched" | "no_match",
                "matches": [...],
                "rejected_matches": [...]
            }
        """
        if project_language is None:
            project_language = ["c", "c++"]
        failed_phase = normalize_phase(failed_phase)

        sys_versions = self.get_sys_versions(sysuse)
        if not sys_versions:
            return {"status": "no_match", "matches": [], "rejected_matches": [],
                    "error": f"Unknown sysuse: {sysuse}"}

        # Normalize error lines to lowercase for matching
        error_lines_lower = [line.lower() for line in error_lines]

        matches = []
        rejected = []

        for card in self.cards:
            # Step 1: Version filter
            tool = card["tool"]
            # Meta-cards (tool == "any") are not tied to a specific
            # toolchain version; they describe cross-cutting diagnostic
            # flows (library ABI drift, tool-shim wrapper). Skip the
            # version filter for them so they remain eligible on every
            # Debian release.
            if tool == "any":
                sys_ver = "any"
            else:
                sys_ver = sys_versions.get(tool)
                if sys_ver is None:
                    continue

            card_vmin = card.get("version_min") or "0"
            # Skip CMP-style version_min for cmake policies (always check patterns)
            if not str(card_vmin).startswith("CMP") and tool != "any":
                if not version_gte(sys_ver, card_vmin):
                    continue
                # Check version_max if set
                if card.get("version_max") and not version_gte(card["version_max"], sys_ver):
                    continue

            # Step 2: Phase + Language filter
            card_phases = set(card.get("matrix_stages") or card.get("affected_phases", []))
            if failed_phase not in card_phases:
                continue

            card_langs = card.get("affected_languages", [])
            if not any(lang in card_langs for lang in project_language):
                continue

            # Step 3: Negative pattern check
            neg_hit = False
            for neg in card.get("negative_patterns", []):
                if pattern_match(neg, error_lines):
                    neg_hit = True
                    rejected.append({
                        "policy_id": card["id"],
                        "reason": f"negative pattern matched: '{neg}'"
                    })
                    break
            if neg_hit:
                continue

            # Step 4: Positive pattern match
            match_count = 0
            matched_patterns = []
            for pattern in card.get("trigger_patterns", []):
                if pattern_match(pattern, error_lines):
                    match_count += 1
                    matched_patterns.append(pattern)

            if match_count == 0:
                continue

            # Score: match_count * confidence_weight * version_proximity_bonus
            # Prefer cards whose version_min is closest to (but not exceeding) current version
            # This ensures gcc13 card ranks above gcc4.3 card on trixie (GCC 14)
            conf_weight = {"high": 3, "medium": 2, "low": 1}.get(card.get("confidence", "medium"), 2)

            # Version proximity: higher version_min = more specific = better match
            card_vmin_tuple = parse_version(card_vmin) or (0,)
            sys_ver_tuple = parse_version(sys_ver) or (99,)
            # Bonus: ratio of card_vmin to sys_ver (0.0 to 1.0)
            # e.g., gcc13 on gcc14 system = 13/14 = 0.93; gcc4.3 on gcc14 = 4.3/14 = 0.31
            if sys_ver_tuple[0] > 0:
                proximity = card_vmin_tuple[0] / sys_ver_tuple[0]
            else:
                proximity = 0.5
            proximity_bonus = 1.0 + proximity  # range 1.0 to 2.0

            # Pattern specificity: longer patterns are more specific
            avg_pattern_len = sum(len(p) for p in matched_patterns) / max(len(matched_patterns), 1)
            specificity_bonus = min(avg_pattern_len / 30.0, 2.0)  # cap at 2.0

            score = match_count * conf_weight * proximity_bonus * specificity_bonus

            matches.append({
                "score": score,
                "policy_id": card["id"],
                "tool": card["tool"],
                "problem_class": card["problem_class"],
                "confidence": card["confidence"],
                "match_count": match_count,
                "matched_patterns": matched_patterns,
                "why": f"Matched {match_count}/{len(card['trigger_patterns'])} trigger patterns; "
                       f"current {tool} {sys_ver} >= version_min {card_vmin}",
                "summary": card["summary"],
                "mechanism": card["mechanism"],
                "actions": card["actions"],
                "non_solutions": card.get("non_solutions", []),
                "source_urls": card.get("source_urls", []),
                "matrix_stages": card.get("matrix_stages", []),
                "matrix_actions": card.get("matrix_actions", []),
            })

        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)

        # Remove internal score from output
        for m in matches:
            del m["score"]

        return {
            "status": "matched" if matches else "no_match",
            "matches": matches[:top_k],
            "rejected_matches": rejected[:5]
        }

    def select(self,
               error_lines: List[str],
               sysuse: str,
               failed_phase: str = "compile",
               project_language: Optional[List[str]] = None,
               top_k: int = 7,
               symptom_family: Optional[str] = None,
               candidate_decisions: Optional[List[str]] = None,
               allowed_revision_types: Optional[List[str]] = None,
               provenance_probe: Optional[Dict[str, Any]] = None,
               matrix_actions: Optional[List[str]] = None,
               matrix_action_specs: Optional[List[Dict[str, Any]]] = None,
               intervention_target: Optional[str] = None,
               diagnosis_only: bool = False,
               ) -> List[Dict[str, Any]]:
        """Return matching cards sorted by cost (cheapest first).

        Unlike `query()` (which returns top-3 by score and is meant for LLM
        context), `select()` is for the programmatic card-driven repair path.
        It returns the candidate list (capped at `top_k`) so the caller can
        probe-refine and try each one in cost order.

        Each returned card dict is the raw card plus a computed `cost` field.
        Disambiguator refinement is NOT applied here — call
        `refine_with_probes()` separately when len(result) > 1.
        """
        if project_language is None:
            project_language = ["c", "c++"]
        failed_phase = normalize_phase(failed_phase)

        sys_versions = self.get_sys_versions(sysuse)
        if not sys_versions:
            return []

        error_lines_lower = [line.lower() for line in error_lines]
        candidates: List[Dict[str, Any]] = []

        for card in self.cards:
            # Version filter (same logic as query())
            tool = card["tool"]
            # Meta-cards bypass the version filter (see query()).
            if tool == "any":
                sys_ver = "any"
            else:
                sys_ver = sys_versions.get(tool)
                if sys_ver is None:
                    continue
            card_vmin = card.get("version_min") or "0"
            if not str(card_vmin).startswith("CMP") and tool != "any":
                if not version_gte(sys_ver, card_vmin):
                    continue
                if card.get("version_max") and not version_gte(card["version_max"], sys_ver):
                    continue

            card_phases = set(card.get("matrix_stages") or card.get("affected_phases", []))
            if failed_phase not in card_phases and not (diagnosis_only and failed_phase == "unknown"):
                continue

            card_langs = card.get("affected_languages", [])
            if not any(lang in card_langs for lang in project_language):
                continue

            if not diagnosis_only and (candidate_decisions or allowed_revision_types or matrix_actions) and not card_matches_route(
                card,
                candidate_decisions=candidate_decisions,
                allowed_revision_types=allowed_revision_types,
                matrix_actions=matrix_actions,
                matrix_action_specs=matrix_action_specs,
                intervention_target=intervention_target,
                failed_phase=failed_phase,
                provenance_probe=provenance_probe,
                problem_hints=_SYMPTOM_PROBLEM_HINTS,
                revision_hints=_REVISION_ACTION_HINTS,
            ):
                continue

            # Negative patterns
            neg_hit = any(
                pattern_match(neg, error_lines_lower)
                for neg in card.get("negative_patterns", [])
            )
            if neg_hit:
                continue

            # Positive patterns — require at least one match
            matched = [
                p for p in card.get("trigger_patterns", [])
                if pattern_match(p, error_lines_lower)
            ]
            if not matched:
                continue
            if diagnosis_only and not _diagnosis_context_supported(card, error_lines):
                continue

            card_out = dict(card)
            # Generic mechanism cards provide a fallback, while repository-
            # specific records should win when both explain the same failure.
            source_penalty = 12 if card.get("provenance") == "handcrafted_mechanism" else 0
            card_out["cost"] = card_cost(card) + source_penalty
            card_out["matched_patterns"] = matched
            card_out["route_score"] = route_alignment_score(
                card_out,
                symptom_family=symptom_family,
                candidate_decisions=candidate_decisions,
                allowed_revision_types=allowed_revision_types,
                provenance_probe=provenance_probe,
                matrix_actions=matrix_actions,
            )
            card_out["route_action_match"] = _has_route_action_match(
                card_out, matrix_actions
            )
            card_out["effective_cost"] = max(0, card_out["cost"] - card_out["route_score"])
            candidates.append(card_out)

        # Exact Matrix action agreement ranks first, but is intentionally not
        # an eligibility filter: a differently-directed card can still be the
        # only evidenced explanation for the current failure.
        candidates.sort(
            key=lambda c: (
                -int(bool(c["route_action_match"])),
                c["effective_cost"],
                -c["route_score"],
                c["cost"],
                -_CONFIDENCE_RANK.get(c.get("confidence", "medium"), 2),
            )
        )
        return candidates[:top_k]

    def diagnose(
        self,
        error_lines: List[str],
        sysuse: str,
        failed_phase: str = "unknown",
        project_language: Optional[List[str]] = None,
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """Match card patterns before matrix routing without selecting repairs."""
        return self.select(
            error_lines=error_lines,
            sysuse=sysuse,
            failed_phase=failed_phase,
            project_language=project_language,
            top_k=top_k,
            diagnosis_only=True,
        )

    @staticmethod
    def refine_with_probes(candidates: List[Dict[str, Any]],
                           probe_results: Dict[str, Any],
                           ) -> List[Dict[str, Any]]:
        """Filter candidates by their `disambiguators` against probe results.

        - Cards WITHOUT `disambiguators` are always kept (they don't claim
          any environment precondition).
        - Cards WITH `disambiguators` are kept iff ALL disambiguators pass.
        - If a probe result is missing (None), that disambiguator is treated
          as PASS (conservative — don't drop a candidate because a probe
          didn't run).
        - If the refined list is empty (every conflicting card got dropped),
          the caller should fall back to the original candidate list and
          try them in cost order without probe refinement.
        """
        refined: List[Dict[str, Any]] = []
        for card in candidates:
            dis = card.get("disambiguators")
            if not dis:
                refined.append(card)
                continue
            keep = True
            for d in dis:
                result = _check_disambiguator(d, probe_results)
                if result is False:  # explicit fail
                    keep = False
                    break
                # None (missing probe) or True → continue
            if keep:
                refined.append(card)
        return refined

    def list_applicable(self,
                        sysuse: str,
                        failed_phase: str = None,
                        project_language: List[str] = None,
                        tool_filter: str = None) -> Dict[str, Any]:
        """
        List all policy cards applicable to the current sysuse.
        No error_lines needed — returns all cards whose version_min <= current tool version.

        Use this to give the agent a "cheat sheet" of all known policy changes
        that could affect builds on this system.

        Args:
            sysuse: Debian codename (e.g. "trixie")
            failed_phase: Optional filter by phase ("compile", "link", "configure")
            project_language: Optional filter by language (["c"], ["c++"], etc.)
            tool_filter: Optional filter by tool name ("gcc", "clang", "cmake", etc.)

        Returns:
            {
                "sysuse": "trixie",
                "tool_versions": {"gcc": "14", ...},
                "applicable_cards": [
                    {"id": "...", "tool": "...", "version_min": "...", "summary": "...", "actions": [...]}
                ]
            }
        """
        if project_language is None:
            project_language = ["c", "c++"]

        sys_versions = self.get_sys_versions(sysuse)
        if not sys_versions:
            return {"error": f"Unknown sysuse: {sysuse}"}

        applicable = []

        for card in self.cards:
            tool = card["tool"]

            # Tool filter (meta-cards with tool="any" pass through any filter)
            if tool_filter and tool != tool_filter and tool != "any":
                continue

            # Meta-cards bypass the version filter.
            if tool == "any":
                sys_ver = "any"
            else:
                sys_ver = sys_versions.get(tool)
                if sys_ver is None:
                    continue

            # Version filter
            card_vmin = card.get("version_min") or "0"
            if not str(card_vmin).startswith("CMP") and tool != "any":
                if not version_gte(sys_ver, card_vmin):
                    continue
                if card.get("version_max") and not version_gte(card["version_max"], sys_ver):
                    continue

            # Phase filter (optional)
            if failed_phase and failed_phase not in card.get("affected_phases", []):
                continue

            # Language filter
            card_langs = card.get("affected_languages", [])
            if not any(lang in card_langs for lang in project_language):
                continue

            applicable.append({
                "id": card["id"],
                "tool": card["tool"],
                "version_min": card["version_min"],
                "summary": card["summary"],
                "confidence": card["confidence"],
                "trigger_patterns": card["trigger_patterns"][:3],  # first 3 for brevity
                "actions": card["actions"]
            })

        # Sort by tool then version_min descending (newest first)
        applicable.sort(key=lambda x: (x["tool"], parse_version(x["version_min"]) or (0,)), reverse=True)

        return {
            "sysuse": sysuse,
            "tool_versions": {k: v for k, v in sys_versions.items() if v is not None},
            "total_applicable": len(applicable),
            "applicable_cards": applicable
        }


# CLI interface for testing
if __name__ == "__main__":
    import sys

    kb = PolicyKB()

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run built-in test cases
        print("=== Test 1: GCC 14 implicit function declaration ===")
        result = kb.query(
            error_lines=["error: implicit declaration of function 'strlen'"],
            sysuse="trixie",
            failed_phase="compile",
            project_language=["c"]
        )
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
        print()

        print("=== Test 2: GCC 13 header dependency (cstdint) ===")
        result = kb.query(
            error_lines=["error: 'uint16_t' was not declared in this scope"],
            sysuse="trixie",
            failed_phase="compile",
            project_language=["c++"]
        )
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
        print()

        print("=== Test 3: GCC 10 -fno-common ===")
        result = kb.query(
            error_lines=["multiple definition of `global_var'", "first defined here"],
            sysuse="bullseye",
            failed_phase="link",
            project_language=["c"]
        )
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
        print()

        print("=== Test 4: Same error on buster (GCC 8) - should NOT match ===")
        result = kb.query(
            error_lines=["error: 'uint16_t' was not declared in this scope"],
            sysuse="buster",
            failed_phase="compile",
            project_language=["c++"]
        )
        print(f"Status: {result['status']} (expected: no_match)")
        print()

        print("=== Test 5: pkg-config not found ===")
        result = kb.query(
            error_lines=["Package openssl was not found in the pkg-config search path"],
            sysuse="trixie",
            failed_phase="configure",
            project_language=["c"]
        )
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])

    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        # List all applicable cards for a sysuse
        sysuse = sys.argv[2] if len(sys.argv) > 2 else "trixie"
        phase = sys.argv[3] if len(sys.argv) > 3 else None
        lang = sys.argv[4].split(",") if len(sys.argv) > 4 else None
        tool = sys.argv[5] if len(sys.argv) > 5 else None

        result = kb.list_applicable(sysuse, failed_phase=phase, project_language=lang, tool_filter=tool)
        print(f"Sysuse: {result['sysuse']}")
        print(f"Tool versions: {result['tool_versions']}")
        print(f"Total applicable cards: {result['total_applicable']}")
        print()
        for card in result['applicable_cards']:
            print(f"  [{card['tool']}≥{card['version_min']}] {card['summary'][:80]}")
            if card['actions']:
                print(f"    → {card['actions'][0]['hint'][:60]}")
        print()
        print("Full JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:5000])

    else:
        # Interactive: read error lines from stdin
        print("Policy KB loaded:", len(kb.cards), "cards")
        print("Usage:")
        print("  python query_policy_kb.py --test")
        print("  python query_policy_kb.py --list <sysuse> [phase] [lang] [tool]")
        print("  echo 'error line' | python query_policy_kb.py <sysuse> <phase> <lang>")

        if len(sys.argv) >= 4:
            sysuse = sys.argv[1]
            phase = sys.argv[2]
            lang = sys.argv[3].split(",")
            error_lines = [line.strip() for line in sys.stdin if line.strip()]
            result = kb.query(error_lines, sysuse, phase, lang)
            print(json.dumps(result, indent=2, ensure_ascii=False))
