"""Fix Decision Matrix — FDM v5 (13 Error Types × 5 Build Phases × 9 Fix Directions).

Implements the paper's FDM v5 schema:
  - 13 Error Types (T1a–T8) organized into three groups
  - 5 Build Phases (Setup, Bootstrap, Configure, Compile, Link)
  - 9 Fix Directions (D1–D9) ordered by environment restoration cost

The matrix defines which fix directions are permissible for each
``(error_type, build_phase)`` cell, constraining the LLM repair agent.

Runtime symptom names (e.g. ``pkg_config_failure``) are normalized to
T-codes via ``canonical_error_type()``; legacy fix_direction names
(e.g. ``system_package_provider``) are normalized to D-codes via
``canonical_fix_direction()``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# Build Phases
# ---------------------------------------------------------------------------

BUILD_PHASES = ("setup", "bootstrap", "configure", "compile", "link")
UNKNOWN_PHASE = "unknown"

# ---------------------------------------------------------------------------
# Error Types — 13 rows (FDM v5)
# ---------------------------------------------------------------------------

CANONICAL_ERROR_TYPES = (
    # Environment & Configuration
    "T1a", "T1b", "T1c",
    "T2",
    # Build Process
    "T3a", "T3b", "T3c", "T3d",
    "T4",
    # Source Code & ABI
    "T5", "T6", "T7", "T8",
)

ERROR_TYPE_NAMES: dict[str, str] = {
    "T1a": "Required Package Not Installed",
    "T1b": "Header File Not Found",
    "T1c": "Build Input Unavailable",
    "T2":  "Invalid Tool Option",
    "T3a": "Symbol Resolution Failure",
    "T3b": "Build Tool Error Report",
    "T3c": "Silent Misconfiguration",
    "T3d": "Platform-Dependent Behavior",
    "T4":  "Build Execution Defect",
    "T5":  "Source Code Error",
    "T6":  "Toolchain-Flagged Code Pattern",
    "T7":  "Unsafe Memory Access",
    "T8":  "ABI Incompatibility",
}

# Legacy symptom names still used by runtime callers.  They map to T-codes
# through SYMPTOM_TO_ROW below.
SYMPTOMS = (
    "apt_install_failure",
    "build_tool_not_found",
    "pkg_config_failure",
    "cmake_package_failure",
    "header_not_found",
    "library_not_found",
    "unresolved_symbol",
    "source_path_missing",
    "wrong_entrypoint",
    "generated_file_missing",
    "compiler_policy_error",
    "api_or_toolchain_mismatch",
)

# ---------------------------------------------------------------------------
# Fix Directions — 9 columns (FDM v5)
#
# D1→D9 ascending difficulty of restoring the build environment:
#   per-invocation flags → build tree → project files → user environment
#   → system state → system release
# ---------------------------------------------------------------------------

REPAIR_ACTIONS: dict[str, dict[str, Any]] = {
    "D1": {
        "number": 1,
        "label": "Adjust Compiler Behavior",
        "fix_direction": "D1",
        "intervention_target": "toolchain.configuration",
        "evidence_gate": "tool_configuration_signal",
        "fallback_only": False,
        "revision_types": ("build_flag_adjustment", "tool_shim"),
        "card_types": ("build_flag", "tool_shim", "build_system_edit"),
        "operation_kinds": ("build_plan_patch", "build_file_patch"),
    },
    "D2": {
        "number": 2,
        "label": "Adjust Linker Behavior",
        "fix_direction": "D2",
        "intervention_target": "link",
        "evidence_gate": "link_signal",
        "fallback_only": False,
        "revision_types": ("link_flag_adjustment", "link_order_adjustment"),
        "card_types": ("build_flag", "env_var", "build_system_edit", "tool_shim"),
        "operation_kinds": ("build_plan_patch", "build_file_patch"),
    },
    "D3": {
        "number": 3,
        "label": "Regenerate or Rebuild",
        "fix_direction": "D3",
        "intervention_target": "bootstrap",
        "evidence_gate": "generator_candidate",
        "fallback_only": False,
        "revision_types": ("known_generation_command",),
        "card_types": ("diagnostic_flow", "build_system_edit"),
        "operation_kinds": ("generator_selection",),
    },
    "D4": {
        "number": 4,
        "label": "Modify Build Definition",
        "fix_direction": "D4",
        "intervention_target": "configure",
        "evidence_gate": "discovery_evidence",
        "fallback_only": False,
        "revision_types": (
            "dependency_discovery_adjustment", "include_path_adjustment",
            "entrypoint_candidate_switch", "target_candidate_switch",
        ),
        "card_types": ("env_var", "build_flag", "build_system_edit", "source_dep"),
        "operation_kinds": ("build_plan_patch", "build_file_patch", "build_entry_selection", "build_target_selection"),
    },
    "D5": {
        "number": 5,
        "label": "Adjust Packaging Strategy",
        "fix_direction": "D5",
        "intervention_target": "dependency.package",
        "evidence_gate": "provider_candidate",
        "fallback_only": False,
        "revision_types": ("system_package_provider_switch", "apt_single_provider_reduction"),
        "card_types": ("pkg_install", "alt_pkg", "diagnostic_flow"),
        "operation_kinds": ("system_package_selection",),
    },
    "D6": {
        "number": 6,
        "label": "Adjust Execution Context",
        "fix_direction": "D6",
        "intervention_target": "environment",
        "evidence_gate": "tool_configuration_signal",
        "fallback_only": False,
        "revision_types": ("build_flag_adjustment", "tool_shim"),
        "card_types": ("env_var", "build_flag"),
        "operation_kinds": ("build_plan_patch",),
    },
    "D7": {
        "number": 7,
        "label": "Install or Locate Components",
        "fix_direction": "D7",
        "intervention_target": "dependency.source",
        "evidence_gate": "source_dependency_candidate",
        "fallback_only": False,
        "revision_types": ("source_dependency_register",),
        "card_types": ("source_dep", "source_build"),
        "operation_kinds": ("source_dependency_selection",),
    },
    "D8": {
        "number": 8,
        "label": "Manage System Packages",
        "fix_direction": "D8",
        "intervention_target": "dependency.package",
        "evidence_gate": "provider_candidate",
        "fallback_only": False,
        "revision_types": ("system_package_provider_switch", "apt_single_provider_reduction"),
        "card_types": ("pkg_install", "alt_pkg", "diagnostic_flow"),
        "operation_kinds": ("system_package_selection",),
    },
    "D9": {
        "number": 9,
        "label": "Switch System Release",
        "fix_direction": "D9",
        "intervention_target": "environment",
        "evidence_gate": "release_candidate",
        "fallback_only": True,
        "revision_types": ("candidate_release_switch",),
        "card_types": ("sysuse",),
        "operation_kinds": ("system_release_selection",),
    },
}

ACTION_ID_BY_NUMBER = {int(spec["number"]): action_id for action_id, spec in REPAIR_ACTIONS.items()}


# ---------------------------------------------------------------------------
# Compatibility: legacy direction names → D-codes
# ---------------------------------------------------------------------------

DIRECTION_COMPATIBILITY_ALIASES = {
    # Old fix_direction names used by runtime callers
    "system_package_provider": "D8",
    "source_dependency": "D7",
    "generation": "D3",
    "generation_step": "D3",
    "dependency_discovery": "D4",
    "linker_configuration": "D2",
    "link_command": "D2",
    "build_entry": "D4",
    "build_target": "D4",
    "system_release": "D9",
    "compiler_configuration": "D1",
    "compile_command": "D1",
    "toolchain": "D1",
}

# Reverse map: D-code → primary legacy name (for compatibility_directions)
_DCODE_TO_LEGACY: dict[str, str] = {
    "D1": "compiler_configuration",
    "D2": "linker_configuration",
    "D3": "generation",
    "D4": "dependency_discovery",
    "D5": "system_package_provider",
    "D6": "compiler_configuration",
    "D7": "source_dependency",
    "D8": "system_package_provider",
    "D9": "system_release",
}

# ---------------------------------------------------------------------------
# Symptom → T-code mapping
# ---------------------------------------------------------------------------

SYMPTOM_TO_ROW = {
    "apt_install_failure": "T1a",
    "build_tool_not_found": "T1a",
    "pkg_config_failure": "T1c",
    "cmake_package_failure": "T1c",
    "header_not_found": "T1b",
    "library_not_found": "T3a",
    "unresolved_symbol": "T3a",
    "source_path_missing": "T1c",
    "wrong_entrypoint": "T4",
    "generated_file_missing": "T1c",
    "compiler_policy_error": "T6",
    "api_or_toolchain_mismatch": "T8",
}

# Old canonical error type names → T-code (for persisted records)
_OLD_ROW_TO_TCODE = {
    "missing_generated_artifacts": "T1c",
    "system_package_installation_failure": "T1a",
    "dependency_discovery_failure": "T1c",
    "header_resolution_failure": "T1b",
    "library_resolution_failure": "T3a",
    "build_input_parsing_failure": "T1c",
    "incorrect_build_entry_or_target": "T4",
    "compiler_policy_rejection": "T6",
    "api_or_toolchain_incompatibility": "T8",
    "unresolved_link_symbols": "T3a",
}


SYMPTOM_PROBLEM_HINTS: dict[str, frozenset[str]] = {
    "apt_install_failure": frozenset({"dependency_resolution", "dependency_resolution_change"}),
    "build_tool_not_found": frozenset({"dependency_resolution", "source_dependency", "dependency_resolution_change"}),
    "pkg_config_failure": frozenset({"dependency_resolution", "source_dependency", "dependency_resolution_change"}),
    "cmake_package_failure": frozenset({"dependency_resolution", "source_dependency", "dependency_resolution_change"}),
    "header_not_found": frozenset({"missing_include", "dependency_resolution", "source_dependency"}),
    "library_not_found": frozenset({"dependency_resolution", "source_dependency", "library_abi_drift", "library_path_change"}),
    "source_path_missing": frozenset({"source_dependency", "wrong_entrypoint", "build_system_policy"}),
    "wrong_entrypoint": frozenset({"wrong_entrypoint"}),
    "generated_file_missing": frozenset({"wrong_entrypoint", "build_system_policy", "filename_change"}),
    "compiler_policy_error": frozenset({"compiler_toolchain_version", "missing_include", "build_system_policy"}),
    "api_or_toolchain_mismatch": frozenset({"compiler_toolchain_version", "library_abi_drift"}),
    "unresolved_symbol": frozenset({"compiler_toolchain_version", "library_abi_drift", "dependency_resolution", "linker_error"}),
}


# ---------------------------------------------------------------------------
# FDM v5 Matrix — 47 non-empty cells
#
# Each cell maps (phase, error_type) → ordered list of D-code directions.
# Source: fdm/fdm/matrix_v5.json (135 record-backed + 8 expert D9 edges)
# ---------------------------------------------------------------------------

_FDM_V5_CELLS: dict[tuple[str, str], tuple[str, ...]] = {
    # T1a: Required Package Not Installed
    ("setup",     "T1a"): ("D5", "D7", "D8", "D9"),
    ("bootstrap", "T1a"): ("D3",),
    ("configure", "T1a"): ("D4", "D6"),
    # T1b: Header File Not Found
    ("compile",   "T1b"): ("D3", "D7"),
    # T1c: Build Input Unavailable
    ("setup",     "T1c"): ("D1", "D2", "D3", "D7", "D8"),
    ("bootstrap", "T1c"): ("D3", "D4", "D7"),
    ("configure", "T1c"): ("D1", "D3", "D4", "D5", "D6", "D7"),
    ("compile",   "T1c"): ("D1", "D3", "D6", "D7"),
    ("link",      "T1c"): ("D2", "D3", "D4", "D7"),
    # T2: Invalid Tool Option
    ("setup",     "T2"):  ("D5",),
    ("bootstrap", "T2"):  ("D6",),
    ("configure", "T2"):  ("D2", "D7"),
    ("compile",   "T2"):  ("D1", "D3", "D7"),
    ("link",      "T2"):  ("D1", "D2", "D3", "D4", "D5", "D6"),
    # T3a: Symbol Resolution Failure
    ("setup",     "T3a"): ("D2", "D4", "D5"),
    ("bootstrap", "T3a"): ("D1", "D3", "D4", "D6", "D7", "D9"),
    ("configure", "T3a"): ("D1", "D3", "D4", "D5", "D6", "D7"),
    ("compile",   "T3a"): ("D1", "D3", "D4", "D6"),
    ("link",      "T3a"): ("D1", "D2", "D4", "D6", "D7"),
    # T3b: Build Tool Error Report
    ("bootstrap", "T3b"): ("D4",),
    ("configure", "T3b"): ("D2", "D3", "D4", "D5", "D6"),
    ("compile",   "T3b"): ("D3", "D4"),
    ("link",      "T3b"): ("D4", "D6"),
    # T3c: Silent Misconfiguration
    ("bootstrap", "T3c"): ("D4",),
    ("configure", "T3c"): ("D1", "D4", "D6", "D7"),
    ("compile",   "T3c"): ("D3", "D4", "D6"),
    ("link",      "T3c"): ("D4", "D7"),
    # T3d: Platform-Dependent Behavior
    ("bootstrap", "T3d"): ("D4",),
    ("configure", "T3d"): ("D4",),
    ("compile",   "T3d"): ("D3", "D4", "D6", "D7"),
    ("link",      "T3d"): ("D5",),
    # T4: Build Execution Defect
    ("bootstrap", "T4"):  ("D6",),
    ("configure", "T4"):  ("D1",),
    ("compile",   "T4"):  ("D1", "D7", "D9"),
    # T5: Source Code Error
    ("setup",     "T5"):  ("D4",),
    ("bootstrap", "T5"):  ("D4", "D6", "D7", "D9"),
    ("configure", "T5"):  ("D1", "D4", "D5", "D6", "D7"),
    ("compile",   "T5"):  ("D1", "D4", "D6", "D7", "D9"),
    ("link",      "T5"):  ("D2", "D4", "D5"),
    # T6: Toolchain-Flagged Code Pattern
    ("compile",   "T6"):  ("D1",),
    ("link",      "T6"):  ("D1", "D4", "D6"),
    # T7: Unsafe Memory Access
    ("compile",   "T7"):  ("D1", "D3", "D7", "D9"),
    ("link",      "T7"):  ("D1", "D3", "D7", "D9"),
    # T8: ABI Incompatibility
    ("bootstrap", "T8"):  ("D6",),
    ("configure", "T8"):  ("D1", "D4", "D6", "D7"),
    ("compile",   "T8"):  ("D1", "D3", "D6", "D9"),
    ("link",      "T8"):  ("D1", "D2", "D4", "D6"),
}


# Build the direct/conditional/fallback lookup tables from FDM cells.
# FDM v5 cells are flat — all listed directions are direct.  The old
# conditional/fallback distinction is subsumed by the FDM's cell structure.

_DIRECT_TABLE: dict[tuple[str, str], tuple[str, ...]] = dict(_FDM_V5_CELLS)
_CONDITIONAL_TABLE: dict[tuple[str, str], tuple[str, ...]] = {}
_FALLBACK_TABLE: dict[tuple[str, str], tuple[str, ...]] = {}


# ---------------------------------------------------------------------------
# FixMatrixCell and lookup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixMatrixCell:
    """A paper matrix cell: (error_type, phase) → ordered fix directions."""

    canonical_error_type: str
    stage: str
    actions: tuple[str, ...]
    fix_directions: tuple[str, ...]
    allowed_revisions: tuple[str, ...]
    problem_classes: frozenset[str]
    conditional_actions: tuple[str, ...] = ()
    fallback_actions: tuple[str, ...] = ()
    supporting_record_ids: tuple[str, ...] = ()


EMPTY_CELL = FixMatrixCell("unknown", "unknown", (), (), (), frozenset(), (), (), ())


def normalize_phase(phase: str) -> str:
    value = str(phase or "").strip().lower()
    if value == "build":
        return "compile"
    if value in {"other", "unknown", ""}:
        return UNKNOWN_PHASE
    return value if value in BUILD_PHASES else UNKNOWN_PHASE


def canonical_error_type(symptom: str) -> str:
    """Map a symptom or error-type name to a T-code.

    Accepts:
      - Old symptom names (e.g. ``header_not_found``)
      - Old canonical names (e.g. ``header_resolution_failure``)
      - T-codes directly (e.g. ``T1b``)
    """
    value = str(symptom or "").strip()
    lower = value.lower()
    # Already a T-code (preserve original case from CANONICAL_ERROR_TYPES)
    if value in CANONICAL_ERROR_TYPES:
        return value
    # Case-insensitive T-code match
    _TCODE_SET = {t.lower(): t for t in CANONICAL_ERROR_TYPES}
    if lower in _TCODE_SET:
        return _TCODE_SET[lower]
    # Symptom name
    if lower in SYMPTOM_TO_ROW:
        return SYMPTOM_TO_ROW[lower]
    # Old canonical name
    if lower in _OLD_ROW_TO_TCODE:
        return _OLD_ROW_TO_TCODE[lower]
    return "unknown"


def canonical_fix_direction(direction: str) -> str:
    """Return the D-code for a canonical or legacy direction name.

    Accepts D-codes directly, old fix_direction names, and legacy aliases.
    """
    value = str(direction or "").strip()
    upper = value.upper()
    # Already a D-code
    if upper in REPAIR_ACTIONS:
        return upper
    # Legacy name
    if value in DIRECTION_COMPATIBILITY_ALIASES:
        return DIRECTION_COMPATIBILITY_ALIASES[value]
    # Bare "toolchain" alias
    if value == "toolchain":
        return "D1"
    return value


def compatibility_directions(direction: str) -> tuple[str, ...]:
    """Return the legacy-first projection used by older route consumers."""
    canonical = canonical_fix_direction(direction)
    legacy = _DCODE_TO_LEGACY.get(canonical)
    return _unique((legacy, canonical)) if legacy else (canonical,)


def evidence_gate_satisfied(
    gate: str,
    *,
    provenance: str = "",
    failed_phase: str = "",
    preferred_decisions: Iterable[str] = (),
    fix_direction: str = "",
    evidence_facts: Optional[dict[str, Any]] = None,
) -> bool:
    """Evaluate one matrix evidence gate from the current route/probe facts."""
    normalized_gate = str(gate or "").strip().lower()
    normalized_phase = normalize_phase(failed_phase)
    normalized_provenance = str(provenance or "").strip().lower()
    preferred = {canonical_fix_direction(item) for item in preferred_decisions if item}
    direction = canonical_fix_direction(fix_direction)
    facts = evidence_facts or {}
    if facts:
        entity_name = str(facts.get("object_name") or facts.get("entity_name") or "").strip()
        provider_ready = bool(
            entity_name
            and facts.get("provider_candidate_ids")
        )
        source_ready = bool(facts.get("source_dependency_candidate_ids"))
        generator_ready = bool(facts.get("generator_candidate_ids"))
        discovery_ready = bool(facts.get("discovery_paths") or facts.get("discovery_flags"))
        link_ready = bool(facts.get("link_signal") or normalized_phase == "link")
        release_ready = bool(facts.get("release_candidate_ids"))
        alternative_ready = bool(
            facts.get("build_entry_candidate_ids")
            or facts.get("build_target_candidate_ids")
        )
        return {
            "provider_candidate": provider_ready,
            "source_dependency_candidate": source_ready,
            "generator_candidate": generator_ready,
            "discovery_evidence": discovery_ready,
            "link_signal": link_ready,
            "tool_configuration_signal": bool(facts.get("tool_configuration_signal")),
            "compiler_signal": bool(facts.get("tool_configuration_signal")),
            "release_candidate": release_ready,
            "build_plan_alternative": alternative_ready,
        }.get(normalized_gate, False)
    return {
        "provider_candidate": normalized_provenance == "system_or_external",
        "source_dependency_candidate": normalized_provenance == "project_managed",
        "generator_candidate": normalized_provenance == "generated",
        "discovery_evidence": normalized_provenance == "repo_local_invisible",
        "link_signal": normalized_provenance == "link_composition" or normalized_phase == "link",
        "tool_configuration_signal": (
            normalized_phase in {"configure", "compile", "link"}
            or (normalized_phase == "setup" and direction == "D1")
        ),
        "compiler_signal": (
            normalized_phase in {"configure", "compile", "link"}
            or (normalized_phase == "setup" and direction == "D1")
        ),
        "release_candidate": "D9" in preferred or normalized_provenance == "system_or_external",
        "build_plan_alternative": direction in preferred or "wrong_entrypoint" in preferred,
    }.get(normalized_gate, False)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _revisions_for_actions(actions: tuple[str, ...]) -> tuple[str, ...]:
    revisions: list[str] = []
    for action_id in actions:
        revisions.extend(REPAIR_ACTIONS.get(action_id, {}).get("revision_types", ()))
    return _unique(revisions)


def _directions_for_actions(actions: tuple[str, ...]) -> tuple[str, ...]:
    return _unique(REPAIR_ACTIONS.get(action_id, {}).get("fix_direction", "") for action_id in actions)


def lookup_cell(phase: str, symptom: str, supporting_record_ids: Iterable[str] = ()) -> FixMatrixCell:
    normalized_phase = normalize_phase(phase)
    row = canonical_error_type(symptom)
    actions = _DIRECT_TABLE.get((normalized_phase, row), ())
    conditional_actions = _CONDITIONAL_TABLE.get((normalized_phase, row), ())
    if row == "unknown":
        return EMPTY_CELL
    return FixMatrixCell(
        canonical_error_type=row,
        stage=normalized_phase,
        actions=actions,
        fix_directions=actions,  # In FDM v5, actions ARE the directions (D-codes)
        allowed_revisions=_revisions_for_actions(actions),
        problem_classes=SYMPTOM_PROBLEM_HINTS.get(symptom, frozenset()),
        conditional_actions=conditional_actions,
        fallback_actions=_FALLBACK_TABLE.get((normalized_phase, row), ()),
        supporting_record_ids=tuple(str(item) for item in supporting_record_ids if item),
    )


def repair_actions_for(phase: str, symptom: str) -> tuple[dict[str, Any], ...]:
    cell = lookup_cell(phase, symptom)
    return _repair_action_records(cell.actions, tier="direct")


def conditional_repair_actions_for(phase: str, symptom: str) -> tuple[dict[str, Any], ...]:
    cell = lookup_cell(phase, symptom)
    return _repair_action_records(cell.conditional_actions, tier="conditional")


def fallback_repair_actions_for(phase: str, symptom: str) -> tuple[dict[str, Any], ...]:
    cell = lookup_cell(phase, symptom)
    return _repair_action_records(cell.fallback_actions, tier="fallback")


def _repair_action_records(actions: Iterable[str], *, tier: str) -> tuple[dict[str, Any], ...]:
    records = []
    for index, action_id in enumerate(actions, 1):
        spec = REPAIR_ACTIONS[action_id]
        records.append(
            {
                "action_id": action_id,
                "number": spec["number"],
                "label": spec["label"],
                "fix_direction": spec["fix_direction"],
                "intervention_target": spec["intervention_target"],
                "evidence_gate": spec["evidence_gate"],
                "fallback_only": bool(spec.get("fallback_only")),
                "selection_tier": tier,
                "matrix_rank": index,
                "revision_types": list(spec["revision_types"]),
                "operation_kinds": list(spec.get("operation_kinds") or ()),
            }
        )
    return tuple(records)


def fix_directions_for(phase: str, symptom: str) -> tuple[str, ...]:
    """Compatibility projection: ordered unique directions for a cell."""
    return lookup_cell(phase, symptom).fix_directions


def allowed_revisions_for(phase: str, symptom: str) -> tuple[str, ...]:
    """Compatibility projection for older callers."""
    return lookup_cell(phase, symptom).allowed_revisions


def matrix_action_for_number(number: int) -> str:
    return ACTION_ID_BY_NUMBER[int(number)]


def _hint_text(card: dict[str, Any]) -> str:
    chunks = [str(card.get("summary") or ""), str(card.get("mechanism") or "")]
    for action in card.get("actions") or []:
        if isinstance(action, dict):
            chunks.append(str(action.get("hint") or ""))
    chunks.extend(str(item) for item in card.get("trigger_patterns") or [])
    return " ".join(chunks).lower()


_BOOTSTRAP_TOOLS = {"autoconf", "automake", "m4", "libtoolize", "aclocal", "autoheader", "autoreconf", "bison", "flex", "protoc", "protobuf", "moc", "uic"}
_PACKAGE_TOOLS = {"apt", "apt-get", "dpkg", "pip", "npm", "cargo", "gem"}
_SYSTEM_PACKAGE_TOOLS = {"apt", "apt-get", "dpkg"}


def normalize_card_stages(card: dict[str, Any]) -> tuple[str, ...]:
    """Normalize the observed invocation stage, preferring the card's tool."""
    tool = str(card.get("tool") or "").lower()
    if tool in _SYSTEM_PACKAGE_TOOLS:
        return ("setup",)
    if tool in _BOOTSTRAP_TOOLS:
        return ("bootstrap",)
    hint = _hint_text(card)
    if tool == "libtool" and re.search(
        r"libtoolize|autoreconf|ac_prog_libtool|lt_init|lt_with_ltdl|ltdl_init|"
        r"ac_config_macro|aclocal_amflags",
        hint,
    ):
        return ("bootstrap",)
    if tool == "gettext" and re.search(
        r"autopoint|gettextize|autoreconf|(?:am_gnu_gettext.*configure\.ac)|"
        r"(?:configure\.ac.*am_gnu_gettext)",
        hint,
    ):
        return ("bootstrap",)
    raw = set()
    for item in card.get("matrix_stages") or card.get("affected_phases") or []:
        phase = normalize_phase(item)
        raw.add(phase)
    if not raw:
        return (UNKNOWN_PHASE,)
    return tuple(phase for phase in (*BUILD_PHASES, UNKNOWN_PHASE) if phase in raw)


def _action_hint(action: dict[str, Any]) -> str:
    return str(action.get("hint") or "").lower()


def normalize_card_actions(card: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Normalize only explicit card-to-Matrix mappings.

    Natural-language hints remain useful retrieval evidence, but are not a
    controller contract and therefore cannot create executable directions.
    """
    normalized: list[dict[str, Any]] = []
    problem = str(card.get("problem_class") or "").lower()
    tool = str(card.get("tool") or "").lower()
    phases = set(normalize_card_stages(card))
    for action in card.get("actions") or []:
        if not isinstance(action, dict):
            continue
        explicit = action.get("matrix_action") or action.get("matrix_actions")
        candidates = _explicit_matrix_actions(explicit)
        action_type = str(action.get("type") or "").lower()
        hint = _action_hint(action)
        for candidate in candidates:
            # Resolve legacy action_id to D-code
            resolved = canonical_fix_direction(candidate) if candidate not in REPAIR_ACTIONS else candidate
            if resolved not in REPAIR_ACTIONS:
                continue
            spec = REPAIR_ACTIONS[resolved]
            normalized.append({
                "matrix_action": resolved,
                "action_id": spec["number"],
                "fix_direction": spec["fix_direction"],
                "intervention_target": action.get("intervention_target") or spec["intervention_target"],
                "evidence_gate": action.get("evidence_gate") or spec["evidence_gate"],
                "fallback_only": bool(action.get("fallback_only", spec.get("fallback_only"))),
                "requires_probe": bool(action.get("requires_probe", action_type in {"diagnostic_flow"})),
                "executable": action_type != "diagnostic_flow",
                "operation_kinds": list(spec.get("operation_kinds") or ()),
                "card_action_type": action_type,
                "hint": action.get("hint", ""),
                "phases": list(phases),
            })
    return tuple(normalized)


def _explicit_matrix_actions(explicit: Any) -> list[str]:
    if isinstance(explicit, str):
        return [explicit]
    if isinstance(explicit, dict):
        value = explicit.get("matrix_action")
        return [str(value)] if value else []
    if not isinstance(explicit, (list, tuple)):
        return []
    candidates = []
    for item in explicit:
        if isinstance(item, str):
            candidates.append(item)
        elif isinstance(item, dict) and item.get("matrix_action"):
            candidates.append(str(item["matrix_action"]))
    return candidates


def _infer_env_actions(hint: str, problem: str, tool: str) -> list[str]:
    if re.search(r"scrap(?:e|ing)|grep|error prefix|warning prefix|no.?color|filter.*warning|log parser", hint):
        return []
    if re.search(r"ld_library_path|ldflags|libs|rpath|linker|library path", hint):
        return ["D2"]
    if re.search(r"autoconf|automake|m4|aclocal|generator|autoreconf", hint):
        return ["D3"]
    if re.search(r"pkg.?config|cmake_prefix|include path|search path|metadata", hint):
        return ["D4"]
    if tool in _PACKAGE_TOOLS and problem not in {"output_format_change", "benign_warning"}:
        return ["D8"]
    return ["D4"]


def _infer_flag_actions(action_type: str, hint: str, problem: str, tool: str, phases: set[str]) -> list[str]:
    if problem in {"output_format_change", "benign_warning"}:
        return []
    if tool in {"ld", "binutils", "libtool"} or re.search(
        r"ldflags|libs|linker|link order|cannot find -l|-wl,|(?:^|\s)-l[a-z0-9_+.-]+(?:\s|$)",
        hint,
    ):
        return ["D2"]
    if re.search(r"pkg.?config|find_package|cmake_prefix|include path|include directory|library path|search path|metadata", hint):
        return ["D4"]
    if re.search(r"autoconf|automake|m4|aclocal|generator|autoreconf", hint):
        return ["D3"]
    if tool in _PACKAGE_TOOLS:
        return ["D8"]
    return ["D1"]


def _infer_edit_actions(hint: str, problem: str, tool: str, action_type: str, phases: set[str]) -> list[str]:
    if problem == "wrong_entrypoint":
        return ["D4"]
    if problem == "source_dependency":
        return ["D7"]
    if problem in {"output_format_change", "benign_warning"}:
        return []
    if re.search(r"\bapt(?:-get)?\b.*(?:--solver|\binstall\b|build-dep)|solver3|package provider", hint):
        return ["D8"]
    if re.search(r"--mode=(?:compile|link)", hint):
        actions = []
        if "compile" in phases or "--mode=compile" in hint:
            actions.append("D1")
        if "link" in phases or "--mode=link" in hint:
            actions.append("D2")
        return actions
    if action_type == "diagnostic_flow":
        if re.search(r"apt(?:-get)?\s+install|unable to locate package|no installation candidate|held broken|provider conflict|conflicts?", hint):
            return ["D8"]
        if problem == "library_abi_drift":
            return ["D8", "D7", "D9"]
        if problem == "compiler_toolchain_version":
            return ["D1"]
    if problem == "linker_error" or re.search(r"linker|link order|link line|ldflags|undefined reference|cannot find -l", hint):
        return ["D2"]
    if problem in {"dependency_resolution", "dependency_resolution_change", "library_path_change", "missing_include"} or re.search(r"pkg.?config|find_package|find_path|prefix|include path|library path|metadata|AC_CHECK_LIB", hint, re.I):
        return ["D4"]
    generator_command = re.search(
        r"autoreconf|autoconf|automake|aclocal|autoheader|libtoolize|config\.(?:sub|guess)",
        hint,
        re.I,
    )
    configure_macro = "configure.ac" in hint and re.search(
        r"\b(?:ac_[a-z0-9_]+|am_[a-z0-9_]+|lt_init|lt_with_ltdl|ltdl_init)\b",
        hint,
    )
    if generator_command or configure_macro:
        return ["D3"]
    if tool in _BOOTSTRAP_TOOLS:
        return ["D3"]
    if tool == "gettext" and re.search(r"autopoint|gettextize|autoreconf|configure\.ac", hint):
        return ["D3"]
    if tool == "cmake":
        return ["D4"]
    if tool == "meson":
        return ["D1"]
    if tool in _PACKAGE_TOOLS:
        return ["D8"]
    if "compiler" in problem or tool in {"gcc", "clang", "glibc", "libstdc++", "gnu-standards"} or "flag" in hint or "warning" in hint:
        return ["D1"]
    if action_type == "diagnostic_flow" and re.search(r"entry|target|wrapper|build plan|component", hint):
        return ["D4"]
    return []


def normalize_card_record(card: dict[str, Any]) -> dict[str, Any]:
    """Return a non-mutating card view with matrix metadata."""
    normalized = dict(card)
    normalized["matrix_schema_version"] = 3
    normalized["matrix_stages"] = list(normalize_card_stages(card))
    normalized["matrix_actions"] = list(normalize_card_actions(card))
    return normalized


# Legacy names consumed by query_policy_kb.py and older tests.  They now refer
# to D-code directions rather than old action_id keys.
REVISION_ACTION_HINTS = {
    revision: frozenset(spec["card_types"])
    for spec in REPAIR_ACTIONS.values()
    for revision in spec["revision_types"]
}

SYMPTOM_ID_HINTS: dict[str, tuple[str, ...]] = {
    "apt_install_failure": ("runtime_apt", "dependency", "package"),
    "source_path_missing": ("source_dependency", "wrong_entrypoint"),
    "wrong_entrypoint": ("wrong_entrypoint", "build_entry"),
    "unresolved_symbol": ("abi", "link", "glibc", "gcc"),
}
