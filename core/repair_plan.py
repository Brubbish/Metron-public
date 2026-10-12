"""Structured repair actions derived from deterministic diagnosis routes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable, Optional

from diagnosis.fix_decision_matrix import (
    REPAIR_ACTIONS,
    canonical_fix_direction,
    evidence_gate_satisfied,
    normalize_card_actions,
)


ACTION_SCOPE = {
    "build_flag": "command",
    "env_var": "environment",
    "build_system_edit": "build_system",
    "pkg_install": "dependency",
    "alt_pkg": "dependency",
    "source_dep": "dependency",
    "sysuse": "system_release",
    "diagnostic_flow": "diagnostic",
    "tool_shim": "toolchain",
}

ACTION_COST = {
    "build_flag": 1,
    "env_var": 2,
    "diagnostic_flow": 2,
    "tool_shim": 3,
    "build_system_edit": 3,
    "pkg_install": 4,
    "alt_pkg": 4,
    "sysuse": 5,
    "source_dep": 6,
}

REPAIR_DECISION_SCHEMA_VERSION = 3
OPERATION_KINDS = {
    "system_package_selection",
    "source_dependency_selection",
    "generator_selection",
    "build_entry_selection",
    "build_target_selection",
    "system_release_selection",
    "build_plan_patch",
    "build_file_patch",
    "tool_wrapper",
}
_OPERATION_PROMPT_SCHEMAS: dict[str, dict[str, Any]] = {
    "system_package_selection": {
        "required": ["kind", "package", "provider_evidence_id"],
        "properties": {
            "kind": "literal:system_package_selection",
            "package": "string from provider evidence",
            "provider_evidence_id": "supplied evidence ID",
        },
    },
    "source_dependency_selection": {
        "required": ["kind", "source_dependency_id"],
        "properties": {
            "kind": "literal:source_dependency_selection",
            "source_dependency_id": "supplied candidate ID",
        },
    },
    "generator_selection": {
        "required": ["kind", "generator_id"],
        "properties": {
            "kind": "literal:generator_selection",
            "generator_id": "supplied candidate ID",
        },
    },
    "build_entry_selection": {
        "required": ["kind", "entry_id"],
        "properties": {
            "kind": "literal:build_entry_selection",
            "entry_id": "supplied candidate ID",
        },
    },
    "build_target_selection": {
        "required": ["kind", "target_id"],
        "properties": {
            "kind": "literal:build_target_selection",
            "target_id": "supplied candidate ID",
        },
    },
    "system_release_selection": {
        "required": ["kind", "release_id"],
        "properties": {
            "kind": "literal:system_release_selection",
            "release_id": "supplied candidate ID",
        },
    },
    "build_plan_patch": {
        "required": ["kind", "changes"],
        "properties": {
            "kind": "literal:build_plan_patch",
            "changes": "object",
        },
        "changes_allowed": [
            "configure_args", "compile_flags", "link_flags", "environment", "workdir",
        ],
        "change_types": {
            "configure_args": "string array",
            "compile_flags": "string array",
            "link_flags": "string array",
            "environment": "string map",
            "workdir": "repository-relative string",
        },
    },
    "build_file_patch": {
        "required": ["kind", "path", "patch"],
        "properties": {
            "kind": "literal:build_file_patch",
            "path": "repository-relative string",
            "patch": "string",
        },
        "path_scope": "repository build configuration or build script",
        "patch_format": "one unified diff",
    },
    "tool_wrapper": {
        "required": ["kind", "tool", "wrapper"],
        "properties": {
            "kind": "literal:tool_wrapper",
            "tool": "tool name string",
            "wrapper": "object",
        },
        "wrapper_allowed": ["executable", "arguments", "environment"],
        "wrapper_types": {
            "executable": "string",
            "arguments": "string array",
            "environment": "string map",
        },
    },
}
_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp"}
_BUILD_FILE_NAMES = {
    "CMakeLists.txt", "Makefile", "Makefile.am", "Makefile.in",
    "configure.ac", "configure.in", "meson.build", "meson_options.txt",
    "configure", "bootstrap", "buildconf",
}
_BUILD_FILE_SUFFIXES = {".cmake", ".mk", ".m4", ".pro", ".pri", ".pc.in"}
_BUILD_SCRIPT_RE = re.compile(r"^(?:autogen|bootstrap|buildconf|configure|compile|build)(?:[-_.][A-Za-z0-9_.+-]+)?\.sh$", re.I)
_PLACEHOLDER_RE = re.compile(r"<[^>]+>|\{\{?[^{}]+\}?\}|\$\([^)]+\)")
_PACKAGE_RE = re.compile(r"^[a-z0-9][a-z0-9+.-]*(?::[a-z0-9][a-z0-9-]*)?$")
_UNSAFE_PLAN_VALUE_RE = re.compile(r"[\n\r\x00;&|><`]")
_WRAPPABLE_TOOL_RE = re.compile(
    r"^(?:gcc(?:-[0-9.]+)?|g\+\+(?:-[0-9.]+)?|clang(?:-[0-9.]+)?|"
    r"clang\+\+(?:-[0-9.]+)?|cc|c\+\+|ld(?:\.(?:gold|lld))?|gold|lld|"
    r"ar|ranlib|pkg-config|pkgconf)$",
    re.I,
)


@dataclass(frozen=True)
class ProviderEvidence:
    """A package-provider fact bound to one failure and container generation."""

    evidence_id: str
    failure_identity: str
    entity_kind: str
    entity_name: str
    sysuse: str
    architecture: str
    container_generation: int
    package: str
    provider_artifact: str
    provider_source: str
    available: bool
    precondition_probe: dict[str, Any]
    availability_probe: dict[str, Any]
    postcondition_probe: dict[str, Any]
    version_relation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "failure_identity": self.failure_identity,
            "entity": {"kind": self.entity_kind, "name": self.entity_name},
            "environment": {
                "sysuse": self.sysuse,
                "architecture": self.architecture,
                "container_generation": self.container_generation,
            },
            "provider": {
                "package": self.package,
                "artifact": self.provider_artifact,
                "source": self.provider_source,
            },
            "available": self.available,
            "precondition_probe": dict(self.precondition_probe),
            "availability_probe": dict(self.availability_probe),
            "postcondition_probe": dict(self.postcondition_probe),
            "version_relation": dict(self.version_relation),
        }


@dataclass(frozen=True)
class RepairDecision:
    """Validated controller-bound normal-flow repair decision."""

    status: str
    operation: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    expected_effect: str = ""
    reason: str = ""
    matrix_action: str = ""
    fix_direction: str = ""
    intervention_target: str = ""
    selection_origin: str = ""
    card_id: str = ""
    failure_identity: str = ""
    failure_context_id: str = ""
    occurrence_signature: str = ""
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPAIR_DECISION_SCHEMA_VERSION,
            "status": self.status,
            "operation": dict(self.operation),
            "evidence_refs": list(self.evidence_refs),
            "expected_effect": self.expected_effect,
            "reason": self.reason,
            "matrix_action": self.matrix_action,
            "fix_direction": self.fix_direction,
            "intervention_target": self.intervention_target,
            "selection_origin": self.selection_origin,
            "card_id": self.card_id,
            "failure_identity": self.failure_identity,
            "failure_context_id": self.failure_context_id,
            "occurrence_signature": self.occurrence_signature,
            "fingerprint": self.fingerprint,
        }


class InvalidRepairDecision(ValueError):
    pass


def operation_schemas_for_prompt(operation_kinds: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Expose only the typed operation contracts selected by the controller."""
    return {
        kind: dict(_OPERATION_PROMPT_SCHEMAS[kind])
        for kind in operation_kinds
        if kind in _OPERATION_PROMPT_SCHEMAS
    }


def make_provider_evidence(
    *,
    failure_identity: str,
    entity_kind: str,
    entity_name: str,
    sysuse: str,
    architecture: str,
    container_generation: int,
    package: str,
    provider_artifact: str,
    provider_source: str,
    available: bool,
    precondition_probe: dict[str, Any],
    postcondition_probe: dict[str, Any],
    availability_probe: Optional[dict[str, Any]] = None,
    version_relation: Optional[dict[str, Any]] = None,
) -> ProviderEvidence:
    payload = {
        "failure_identity": failure_identity,
        "entity_kind": entity_kind,
        "entity_name": entity_name,
        "sysuse": sysuse,
        "architecture": architecture,
        "container_generation": int(container_generation),
        "package": package,
        "provider_artifact": provider_artifact,
        "provider_source": provider_source,
        "version_relation": version_relation or {},
    }
    evidence_id = "provider:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return ProviderEvidence(
        evidence_id=evidence_id,
        failure_identity=str(failure_identity or ""),
        entity_kind=str(entity_kind or ""),
        entity_name=str(entity_name or ""),
        sysuse=str(sysuse or ""),
        architecture=str(architecture or "amd64"),
        container_generation=int(container_generation),
        package=str(package or ""),
        provider_artifact=str(provider_artifact or ""),
        provider_source=str(provider_source or ""),
        available=bool(available),
        precondition_probe=dict(precondition_probe or {}),
        availability_probe=dict(availability_probe or {}),
        postcondition_probe=dict(postcondition_probe or {}),
        version_relation=dict(version_relation or {}),
    )


def action_types(card: dict[str, Any]) -> set[str]:
    return {
        str(item.get("type") or "").strip()
        for item in card.get("actions") or []
        if isinstance(item, dict) and str(item.get("type") or "").strip()
    }


def card_matches_route(
    card: dict[str, Any],
    *,
    candidate_decisions: Optional[Iterable[str]] = None,
    allowed_revision_types: Optional[Iterable[str]] = None,
    matrix_actions: Optional[Iterable[str]] = None,
    matrix_action_specs: Optional[Iterable[dict[str, Any]]] = None,
    intervention_target: Optional[str] = None,
    failed_phase: Optional[str] = None,
    provenance_probe: Optional[dict[str, Any]] = None,
    problem_hints: Optional[dict[str, set[str]]] = None,
    revision_hints: Optional[dict[str, set[str]]] = None,
) -> bool:
    """Check whether a card has an intrinsically safe, evidenced action.

    The diagnosis route is a ranking prior, not an exclusive card contract.
    In particular, a card can explain the current failure while exposing a
    different Matrix action than the route's first-ranked action.  Rejecting
    such cards here collapses retrieval to exact action-ID equality and hides
    valid alternatives before the repair selector can rank them.

    Matrix action, direction, and intervention-target alignment therefore
    remain available to the caller as ranking information.  Eligibility here
    is limited to the card action's own phase and evidence gate.
    """
    decisions = {
        canonical_fix_direction(item)
        for item in candidate_decisions or []
        if item
    }
    revisions = {str(item) for item in allowed_revision_types or [] if item}
    normalized_actions = list(card.get("matrix_actions") or normalize_card_actions(card))
    card_directions = {
        str(item.get("fix_direction") or "")
        for item in normalized_actions
        if isinstance(item, dict) and item.get("fix_direction")
    }
    safe_actions = [
        item
        for item in normalized_actions
        if isinstance(item, dict)
        and _card_action_gate_passes(
            item,
            # Do not impose the route's action/target/gate on a card action.
            # Its own normalized Matrix metadata is the safety contract.
            route_action=None,
            intervention_target=None,
            failed_phase=failed_phase,
            provenance_probe=provenance_probe,
        )
    ]
    if normalized_actions and not safe_actions:
        return False
    if not decisions and not revisions:
        return bool(safe_actions) if normalized_actions else True

    types = action_types(card)
    problem_class = str(card.get("problem_class") or "")
    if not types:
        return False

    decision_fit = False
    if decisions & card_directions:
        decision_fit = True
    if "system_package_provider" in decisions and problem_class in {
        "dependency_resolution", "library_abi_drift"
    }:
        decision_fit = True
    if "source_dependency" in decisions and problem_class == "source_dependency":
        decision_fit = True
    if "build_entry" in decisions and problem_class == "wrong_entrypoint":
        decision_fit = True
    if "system_release" in decisions and "sysuse" in types:
        decision_fit = True

    revision_fit = False
    if revisions and revision_hints:
        for revision in revisions:
            if types & set(revision_hints.get(revision, set())):
                revision_fit = True
                break
    elif not revisions:
        revision_fit = True

    if normalized_actions:
        return bool(safe_actions)
    return decision_fit and revision_fit


def _card_action_gate_passes(
    action: dict[str, Any],
    *,
    route_action: Optional[dict[str, Any]],
    intervention_target: Optional[str],
    failed_phase: Optional[str],
    provenance_probe: Optional[dict[str, Any]],
) -> bool:
    expected = route_action or {}
    expected_direction = str(expected.get("fix_direction") or "")
    expected_target = str(expected.get("intervention_target") or intervention_target or "")
    expected_gate = str(expected.get("evidence_gate") or "")
    if route_action and bool(action.get("fallback_only")) != bool(expected.get("fallback_only")):
        return False
    if expected_direction and str(action.get("fix_direction") or "") != expected_direction:
        return False
    if expected_target and str(action.get("intervention_target") or "") != expected_target:
        return False
    if expected_gate and str(action.get("evidence_gate") or "") != expected_gate:
        return False
    if failed_phase:
        phases = set(action.get("phases") or [])
        if phases and failed_phase not in phases and "unknown" not in phases:
            return False
    probe = provenance_probe or {}
    preferred = probe.get("preferred_decisions") or []
    direction = canonical_fix_direction(action.get("fix_direction") or "")
    gate = str(action.get("evidence_gate") or expected_gate or "")
    provenance = str(probe.get("provenance") or "")
    return evidence_gate_satisfied(
        gate,
        provenance=provenance,
        failed_phase=failed_phase or "",
        preferred_decisions=preferred,
        fix_direction=direction,
        evidence_facts=probe,
    )


def parse_repair_decision(
    raw_output: str,
    *,
    selected_action: dict[str, Any],
    candidate_catalog: dict[str, Any],
    failure_identity: str,
    selection_origin: str = "phase3",
    card_id: str = "",
    failure_context: Optional[dict[str, Any]] = None,
) -> RepairDecision:
    """Parse and validate the normal repair protocol without command fallbacks."""
    payload = _strict_json_object(raw_output)
    status = str(payload.get("status") or "").strip().lower()
    reason = str(payload.get("reason") or "").strip()[:1000]
    if status == "cannot_solve":
        return RepairDecision(status=status, reason=reason or "No evidence-backed operation available.")
    if status not in {"repair", "fix"}:
        raise InvalidRepairDecision("status must be repair or cannot_solve")
    if "commands" in payload or "command" in payload:
        raise InvalidRepairDecision("normal repair decisions cannot contain raw commands")
    operation = payload.get("operation")
    if not isinstance(operation, dict):
        raise InvalidRepairDecision("operation must be an object")
    kind = str(operation.get("kind") or "").strip()
    allowed_kinds = {
        str(item) for item in selected_action.get("operation_kinds") or [] if item
    }
    if kind not in OPERATION_KINDS or (allowed_kinds and kind not in allowed_kinds):
        raise InvalidRepairDecision(f"operation kind {kind!r} is not allowed by the selected matrix action")
    placeholder_view = dict(operation)
    if kind == "build_file_patch":
        placeholder_view.pop("patch", None)
    if _contains_placeholder(placeholder_view):
        raise InvalidRepairDecision("operation contains an unresolved placeholder")
    normalized_operation, required_refs = _validate_operation(
        kind,
        operation,
        candidate_catalog,
        failure_identity,
    )
    evidence_refs = tuple(_string_values(payload.get("evidence_refs")))
    if not evidence_refs:
        raise InvalidRepairDecision("at least one supplied evidence reference is required")
    known_refs = _known_catalog_refs(candidate_catalog)
    unknown_refs = set(evidence_refs) - known_refs
    if unknown_refs:
        raise InvalidRepairDecision(f"unknown evidence refs: {', '.join(sorted(unknown_refs))}")
    if not required_refs.issubset(evidence_refs):
        missing = sorted(required_refs - set(evidence_refs))
        raise InvalidRepairDecision(f"operation is missing evidence refs: {', '.join(missing)}")
    expected_effect = str(payload.get("expected_effect") or "").strip()
    if not expected_effect:
        raise InvalidRepairDecision("expected_effect is required")
    matrix_action = str(selected_action.get("action_id") or selected_action.get("matrix_action") or "")
    direction = canonical_fix_direction(selected_action.get("fix_direction") or "")
    target = str(selected_action.get("intervention_target") or "")
    fingerprint = repair_operation_fingerprint(
        failure_identity=failure_identity,
        matrix_action=matrix_action,
        operation=normalized_operation,
        evidence_refs=evidence_refs,
        failure_context=failure_context,
        execution_context=candidate_catalog.get("execution_context") or {},
    )
    if fingerprint in {
        str(item) for item in candidate_catalog.get("exhausted_operation_fingerprints") or []
    }:
        raise InvalidRepairDecision("this concrete repair operation is already exhausted for the failure")
    return RepairDecision(
        status="repair",
        operation=normalized_operation,
        evidence_refs=evidence_refs,
        expected_effect=expected_effect[:1000],
        reason=reason,
        matrix_action=matrix_action,
        fix_direction=direction,
        intervention_target=target,
        selection_origin=selection_origin,
        card_id=card_id,
        failure_identity=failure_identity,
        failure_context_id=str((failure_context or {}).get("context_id") or ""),
        occurrence_signature=str((failure_context or {}).get("occurrence_signature") or ""),
        fingerprint=fingerprint,
    )


def repair_operation_fingerprint(
    *,
    failure_identity: str,
    matrix_action: str,
    operation: dict[str, Any],
    evidence_refs: Iterable[str],
    failure_context: Optional[dict[str, Any]] = None,
    execution_context: Optional[dict[str, Any]] = None,
) -> str:
    # Evidence references are validated above, but are deliberately excluded
    # from deduplication: citing an extra supporting fact must not make the same
    # concrete repair executable again for the same failure.
    payload = {
        "failure_identity": str(failure_identity or ""),
        "matrix_action": str(matrix_action or ""),
        "operation": operation,
        "failure_context": {
            "context_id": str((failure_context or {}).get("context_id") or ""),
            "occurrence_signature": str((failure_context or {}).get("occurrence_signature") or ""),
            "failure_mechanism": str((failure_context or {}).get("failure_mechanism") or ""),
            "version_relation": (failure_context or {}).get("version_relation") or {},
        },
        "execution_context": dict(execution_context or {}),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:20]


def _strict_json_object(raw_output: str) -> dict[str, Any]:
    text = str(raw_output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        payload = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                payload = value
                break
        if payload is None:
            raise InvalidRepairDecision(f"invalid repair JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidRepairDecision("repair output must be one JSON object")
    return payload


def _validate_operation(
    kind: str,
    operation: dict[str, Any],
    catalog: dict[str, Any],
    failure_identity: str,
) -> tuple[dict[str, Any], set[str]]:
    normalized = dict(operation)
    required_refs: set[str] = set()
    if kind == "system_package_selection":
        package = str(operation.get("package") or "").strip().lower()
        evidence_id = str(operation.get("provider_evidence_id") or "").strip()
        evidence = _catalog_record(catalog, "provider_evidence", evidence_id, "evidence_id")
        if not package or not _PACKAGE_RE.fullmatch(package):
            raise InvalidRepairDecision("system package name is invalid")
        provider = evidence.get("provider") or {}
        environment = evidence.get("environment") or {}
        current_environment = catalog.get("current_environment") or {}
        if str(provider.get("package") or "").lower() != package:
            raise InvalidRepairDecision("package does not match ProviderEvidence")
        if not evidence.get("available") or str(evidence.get("failure_identity") or "") != failure_identity:
            raise InvalidRepairDecision("ProviderEvidence is unavailable or belongs to another failure")
        if any(
            environment.get(key) != current_environment.get(key)
            for key in ("sysuse", "architecture", "container_generation")
        ):
            raise InvalidRepairDecision("ProviderEvidence belongs to another container environment")
        precondition = evidence.get("precondition_probe") or {}
        if not (precondition.get("observed_missing") or precondition.get("observed_unsatisfied")):
            raise InvalidRepairDecision("ProviderEvidence lacks an unsatisfied current-container precondition")
        availability_probe = evidence.get("availability_probe") or {}
        if (
            availability_probe.get("status") != "available"
            or availability_probe.get("exit_code") not in {0, "0"}
            or not availability_probe.get("command")
        ):
            raise InvalidRepairDecision("ProviderEvidence lacks current-container package availability")
        if not (evidence.get("postcondition_probe") or {}).get("command"):
            raise InvalidRepairDecision("ProviderEvidence lacks a postcondition probe")
        required_refs.add(evidence_id)
        normalized = {"kind": kind, "package": package, "provider_evidence_id": evidence_id}
    elif kind in {
        "source_dependency_selection", "generator_selection", "build_entry_selection",
        "build_target_selection", "system_release_selection",
    }:
        key_by_kind = {
            "source_dependency_selection": ("source_dependency_id", "source_dep_candidates", ("candidate_id", "source_dep_id", "name")),
            "generator_selection": ("generator_id", "generator_candidates", ("generator_id", "candidate_id")),
            "build_entry_selection": ("entry_id", "entrypoint_candidates", ("entry_id", "candidate_id")),
            "build_target_selection": ("target_id", "target_candidates", ("target_id", "candidate_id")),
            "system_release_selection": ("release_id", "sysuse_candidates", ("candidate_id", "sysuse")),
        }
        field_name, catalog_name, id_fields = key_by_kind[kind]
        candidate_id = str(operation.get(field_name) or "").strip()
        record = _catalog_record_multi(catalog, catalog_name, candidate_id, id_fields)
        if kind == "source_dependency_selection" and (
            not record.get("build_order")
            or not (record.get("container_source_path") or record.get("source_path") or record.get("submodule_path"))
        ):
            raise InvalidRepairDecision("source dependency candidate is not executable")
        if kind in {"source_dependency_selection", "generator_selection"}:
            verification = record.get("verification") or {}
            if not str(verification.get("command") or "").strip():
                raise InvalidRepairDecision(f"{kind} candidate lacks a controller-checkable postcondition")
        required_refs.add(candidate_id)
        normalized = {"kind": kind, field_name: candidate_id}
        if kind in {"source_dependency_selection", "generator_selection"}:
            normalized["verification"] = dict(record.get("verification") or {})
        if kind == "system_release_selection":
            normalized["sysuse"] = str(record.get("sysuse") or "")
    elif kind == "build_plan_patch":
        changes = operation.get("changes")
        allowed = {"configure_args", "compile_flags", "link_flags", "environment", "workdir"}
        if not isinstance(changes, dict) or not changes or set(changes) - allowed:
            raise InvalidRepairDecision("build_plan_patch contains unsupported fields")
        for field_name in ("configure_args", "compile_flags", "link_flags"):
            if field_name in changes and (
                not isinstance(changes[field_name], list)
                or not all(
                    isinstance(item, str)
                    and item.strip()
                    and not _UNSAFE_PLAN_VALUE_RE.search(item)
                    for item in changes[field_name]
                )
            ):
                raise InvalidRepairDecision(f"{field_name} must contain shell-control-free argument strings")
        if "environment" in changes and (
            not isinstance(changes["environment"], dict)
            or not all(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key))
                and isinstance(value, str)
                and not _UNSAFE_PLAN_VALUE_RE.search(value)
                for key, value in changes["environment"].items()
            )
        ):
            raise InvalidRepairDecision("environment must be a string map with valid variable names")
        if "workdir" in changes:
            workdir = PurePosixPath(str(changes["workdir"] or ""))
            if workdir.is_absolute() or ".." in workdir.parts:
                raise InvalidRepairDecision("workdir must remain inside the repository")
        normalized = {"kind": kind, "changes": changes}
    elif kind == "build_file_patch":
        path = str(operation.get("path") or "").strip()
        patch = str(operation.get("patch") or "")
        if not _is_build_file(path) or len(patch) > 12000 or not patch.strip():
            raise InvalidRepairDecision("build_file_patch must target a supported build file with a bounded patch")
        normalized = {"kind": kind, "path": path, "patch": patch}
    elif kind == "tool_wrapper":
        tool = str(operation.get("tool") or "").strip()
        wrapper = operation.get("wrapper")
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", tool) or not isinstance(wrapper, dict):
            raise InvalidRepairDecision("tool_wrapper requires a concrete tool and structured wrapper")
        if set(wrapper) - {"executable", "arguments", "environment"}:
            raise InvalidRepairDecision("tool_wrapper contains unsupported fields")
        if not isinstance(wrapper.get("executable"), str) or not wrapper.get("executable"):
            raise InvalidRepairDecision("tool_wrapper executable is required")
        executable = str(wrapper.get("executable") or "")
        executable_path = PurePosixPath(executable)
        if (
            not re.fullmatch(r"[A-Za-z0-9_./+-]+", executable)
            or ".." in executable_path.parts
            or not _WRAPPABLE_TOOL_RE.fullmatch(executable_path.name)
            or not _WRAPPABLE_TOOL_RE.fullmatch(tool)
        ):
            raise InvalidRepairDecision("tool_wrapper executable is invalid")
        if not isinstance(wrapper.get("arguments", []), list) or not all(
            isinstance(item, str) and not _UNSAFE_PLAN_VALUE_RE.search(item)
            for item in wrapper.get("arguments", [])
        ):
            raise InvalidRepairDecision("tool_wrapper arguments must be a string list")
        environment = wrapper.get("environment", {})
        if not isinstance(environment, dict) or not all(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key))
            and isinstance(value, str)
            and not _UNSAFE_PLAN_VALUE_RE.search(value)
            for key, value in environment.items()
        ):
            raise InvalidRepairDecision("tool_wrapper environment must be a safe string map")
        normalized = {"kind": kind, "tool": tool, "wrapper": wrapper}
    return normalized, required_refs


def _catalog_record(catalog: dict[str, Any], key: str, value: str, id_field: str) -> dict[str, Any]:
    return _catalog_record_multi(catalog, key, value, (id_field,))


def _catalog_record_multi(
    catalog: dict[str, Any], key: str, value: str, id_fields: tuple[str, ...]
) -> dict[str, Any]:
    if not value:
        raise InvalidRepairDecision(f"missing candidate reference for {key}")
    for item in catalog.get(key) or []:
        if isinstance(item, dict) and any(str(item.get(field) or "") == value for field in id_fields):
            if item.get("status") in {"failed", "exhausted", "ineligible", "selected", "current"}:
                break
            return item
    raise InvalidRepairDecision(f"unknown or unavailable candidate {value!r} in {key}")


def _known_catalog_refs(catalog: dict[str, Any]) -> set[str]:
    refs = {str(item) for item in catalog.get("route_evidence_refs") or [] if item}
    for key, fields in (
        ("provider_evidence", ("evidence_id",)),
        ("source_dep_candidates", ("candidate_id", "source_dep_id", "name")),
        ("generator_candidates", ("generator_id", "candidate_id")),
        ("entrypoint_candidates", ("entry_id", "candidate_id")),
        ("target_candidates", ("target_id", "candidate_id")),
        ("sysuse_candidates", ("candidate_id", "sysuse")),
    ):
        for item in catalog.get(key) or []:
            if not isinstance(item, dict):
                continue
            refs.update(str(item.get(field) or "") for field in fields if item.get(field))
    return refs


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.search(value))


def _string_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_build_file(path: str) -> bool:
    candidate = PurePosixPath(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts:
        return False
    if candidate.suffix.lower() in _SOURCE_SUFFIXES:
        return False
    return candidate.name in _BUILD_FILE_NAMES or bool(_BUILD_SCRIPT_RE.fullmatch(candidate.name)) or any(
        candidate.name.endswith(suffix) for suffix in _BUILD_FILE_SUFFIXES
    )


def action_fingerprint(
    *,
    sysuse: str,
    realization_methods: Any,
    build_entry_id: str,
    command_sequence: Iterable[str],
    operation: Any,
) -> str:
    payload = {
        "sysuse": str(sysuse or "").strip().lower(),
        "realization_methods": realization_methods or {},
        "build_entry_id": str(build_entry_id or "").strip(),
        "command_sequence": [re.sub(r"\s+", " ", str(item).strip()) for item in command_sequence or []],
        "operation": operation,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def make_fix_action(
    *,
    direction: str,
    target_decision: str,
    card: Optional[dict[str, Any]] = None,
    operation: Any = None,
    supporting_record_ids: Optional[list[str]] = None,
    sysuse: str = "",
    realization_methods: Any = None,
    build_entry_id: str = "",
    command_sequence: Optional[list[str]] = None,
    disruption: int = 0,
    risk: int = 0,
    matrix_action: str = "",
    matrix_rank: int = 99,
    intervention_target: str = "",
    evidence_gate: str = "",
    selection_origin: str = "",
) -> dict[str, Any]:
    card = card or {}
    types = sorted(action_types(card))
    cheapest_type = min(types, key=lambda item: ACTION_COST.get(item, 99)) if types else ""
    cost = ACTION_COST.get(cheapest_type, 99)
    action = {
        "action_id": str(card.get("id") or direction or "repair_action"),
        "fix_direction": direction,
        "target_decision": target_decision,
        "supporting_record_ids": list(supporting_record_ids or ([card.get("id")] if card.get("id") else [])),
        "operation": operation if operation is not None else {"kind": cheapest_type, "actions": card.get("actions", [])},
        "applicability": {"card_id": card.get("id", ""), "version_min": card.get("version_min", "")},
        "validation": {},
        "scope": ACTION_SCOPE.get(cheapest_type, "unknown"),
        "cost": cost,
        "disruption": disruption,
        "risk": risk,
        "card_id": card.get("id", ""),
        "summary": card.get("summary", ""),
        "matrix_action": matrix_action,
        "matrix_action_id": REPAIR_ACTIONS.get(matrix_action, {}).get("number"),
        "matrix_rank": matrix_rank,
        "intervention_target": intervention_target,
        "evidence_gate": evidence_gate,
        "selection_origin": selection_origin,
    }
    action["fingerprint"] = action_fingerprint(
        sysuse=sysuse,
        realization_methods=realization_methods,
        build_entry_id=build_entry_id,
        command_sequence=command_sequence or [],
        operation=action["operation"],
    )
    return action


def rank_fix_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        actions,
        key=lambda item: (
            int(item.get("matrix_rank", 99)),
            int(item.get("cost", 99)),
            int(item.get("disruption", 0)),
            int(item.get("risk", 0)),
            str(item.get("action_id") or ""),
        ),
    )
