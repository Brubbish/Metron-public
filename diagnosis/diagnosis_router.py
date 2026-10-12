"""Hybrid rule/card classification and routing to the offline matrix."""

from __future__ import annotations

import json
import hashlib
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from diagnosis.diagnosis_signals import (
    INFRASTRUCTURE_FAILURE_PATTERN,
    MISSING_HEADER_PATTERN,
    MISSING_TOOL_PATTERN,
    API_COMPATIBILITY_PATTERN,
    PACKAGE_MANAGER_FAILURE_PATTERN,
    TOOL_CONFIGURATION_PATTERN,
    VERSION_REQUIREMENT_PATTERN,
    has_causal_diagnostic,
    has_compiler_policy,
    has_missing_tool,
    has_tool_configuration_failure,
    extract_version_relation,
    is_wrong_entry_or_target,
)
from diagnosis.fix_decision_matrix import (
    CANONICAL_ERROR_TYPES,
    canonical_error_type,
    canonical_fix_direction,
    compatibility_directions,
    conditional_repair_actions_for,
    fallback_repair_actions_for,
    lookup_cell,
    repair_actions_for,
)


VALID_PHASES = {"setup", "bootstrap", "configure", "compile", "link", "other", "unknown"}
DIAGNOSIS_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class FailureIdentity:
    """Stable identity for one causal build failure."""

    phase: str
    error_type: str
    failed_tool: str
    invocation_role: str
    entity_kind: str
    entity_name: str
    failed_step_hash: str
    normalized_error_signature: str
    identity_key: str
    context_key: str = ""
    occurrence_signature: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "error_type": self.error_type,
            "failed_tool": self.failed_tool,
            "invocation_role": self.invocation_role,
            "entity_kind": self.entity_kind,
            "entity_name": self.entity_name,
            "failed_step_hash": self.failed_step_hash,
            "normalized_error_signature": self.normalized_error_signature,
            "identity_key": self.identity_key,
            "failure_id": self.identity_key,
            "context_id": self.context_key,
            "occurrence_signature": self.occurrence_signature,
        }


@dataclass
class DiagnosisRoute:
    failed_phase: str = "unknown"
    failed_tool: str = "unknown"
    invocation_role: str = "unknown"
    symptom_family: str = "unknown"
    canonical_error_type: str = "unknown"
    entity: dict[str, str] = field(default_factory=dict)
    candidate_decisions: list[str] = field(default_factory=list)
    next_checks: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    repair_scope: dict[str, Any] = field(default_factory=dict)
    classification_source: str = "rule"
    classification_confidence: str = "low"
    classification_status: str = "insufficient_evidence"
    failure_domain: str = "unknown"
    rule_candidates: list[dict[str, Any]] = field(default_factory=list)
    card_candidates: list[dict[str, Any]] = field(default_factory=list)
    rule_recalled_error_types: list[str] = field(default_factory=list)
    card_recalled_error_types: list[str] = field(default_factory=list)
    llm_allowed_error_types: list[str] = field(default_factory=list)
    card_category_support: Optional[bool] = None
    llm_arbitration: dict[str, Any] = field(default_factory=dict)
    needs_llm_arbitration: bool = False
    failed_segment: str = ""
    failure_identity: dict[str, str] = field(default_factory=dict)
    failure_mechanism: str = "unknown"
    version_relation: dict[str, str] = field(default_factory=dict)

    @property
    def observed_phase(self) -> str:
        return self.failed_phase

    @property
    def error_type(self) -> str:
        return self.canonical_error_type

    @property
    def category(self) -> str:
        if self.symptom_family in {"wrong_entrypoint", "generated_file_missing"}:
            return "build_system_mismatch"
        if self.symptom_family in {
            "header_not_found",
            "library_not_found",
            "pkg_config_failure",
            "cmake_package_failure",
            "source_path_missing",
            "apt_install_failure",
            "build_tool_not_found",
        }:
            return "dependency_issue"
        if self.symptom_family in {"compiler_policy_error", "api_or_toolchain_mismatch", "unresolved_symbol"}:
            return "toolchain_mismatch"
        return "unknown"

    @property
    def summary(self) -> str:
        return (
            f"Hybrid route: observed_phase={self.failed_phase}, "
            f"status={self.classification_status}, error_type={self.canonical_error_type}, "
            f"source={self.classification_source}, "
            f"decisions={self.candidate_decisions[:3]}"
        )

    def as_prompt_block(self) -> str:
        scope = self.repair_scope or {}
        selected = scope.get("selected_matrix_action") or {}
        payload = {
            "failure_identity": str((self.failure_identity or {}).get("identity_key") or ""),
            "observed_phase": self.failed_phase,
            "error_type": self.canonical_error_type,
            "entity": dict(self.entity or {}),
            "failure_mechanism": self.failure_mechanism,
            "version_relation": dict(self.version_relation or {}),
            "selected_action": {
                key: selected[key]
                for key in (
                    "action_id",
                    "repair_action",
                    "fix_direction",
                    "intervention_target",
                    "evidence_gate",
                    "operation_kinds",
                    "selection_tier",
                )
                if selected.get(key) not in (None, "", [])
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def to_dict(self) -> dict[str, Any]:
        scope = self.repair_scope or {}
        selected = scope.get("selected_matrix_action") or {}
        return {
            "diagnosis_schema_version": DIAGNOSIS_SCHEMA_VERSION,
            "summary": self.summary,
            "primary_hint": self.category,
            "observed_phase": self.failed_phase,
            "error_type": self.canonical_error_type,
            "direct_matrix_actions": list(scope.get("direct_matrix_actions") or []),
            "conditional_actions": list(scope.get("conditional_actions") or []),
            "intervention_target": selected.get("intervention_target") or "",
            "failed_phase": self.failed_phase,
            "failed_tool": self.failed_tool,
            "invocation_role": self.invocation_role,
            "symptom_family": self.symptom_family,
            "canonical_error_type": self.canonical_error_type,
            "entity": self.entity,
            "candidate_decisions": self.candidate_decisions,
            "next_checks": self.next_checks,
            "evidence": self.evidence,
            "classification_source": self.classification_source,
            "classification_confidence": self.classification_confidence,
            "classification_status": self.classification_status,
            "failure_domain": self.failure_domain,
            "rule_candidates": self.rule_candidates,
            "card_candidates": self.card_candidates,
            "rule_recalled_error_types": self.rule_recalled_error_types,
            "card_recalled_error_types": self.card_recalled_error_types,
            "llm_allowed_error_types": self.llm_allowed_error_types,
            "card_category_support": self.card_category_support,
            "llm_arbitration": self.llm_arbitration,
            "needs_llm_arbitration": self.needs_llm_arbitration,
            "failed_segment": self.failed_segment,
            "failure_identity": dict(self.failure_identity),
            "failure_mechanism": self.failure_mechanism,
            "version_relation": dict(self.version_relation or {}),
            "repair_scope": self.repair_scope,
        }


def route_failure(
    failure_frame: dict[str, Any],
    failed_command: Optional[dict[str, Any]] = None,
    build_plan: Optional[dict[str, Any]] = None,
    *,
    forced_error_type: str = "",
    forced_phase: str = "",
    classification_source: str = "rule",
    classification_confidence: str = "",
    forced_classification_status: str = "",
    forced_failure_domain: str = "",
) -> DiagnosisRoute:
    frame = failure_frame or {}
    command = str((failed_command or frame.get("primary_command") or frame.get("failed_command") or {}).get("command") or "")
    evidence = _evidence_lines_from_frame(frame)
    evidence_text = "\n".join(evidence)
    failed_segment = _select_failed_segment(command, evidence_text)
    tags = set(frame.get("primary_diagnose_tags") or []) | set(frame.get("diagnose_tags") or [])
    failed_tool = _infer_tool(failed_segment, evidence_text)
    failed_phase = _infer_phase(failed_segment, frame, build_plan or {}, failed_tool=failed_tool)
    if forced_phase in VALID_PHASES and forced_phase not in {"other", ""}:
        failed_phase = forced_phase
    invocation_role = _invocation_role(failed_phase, failed_tool)
    rule_candidates = _infer_rule_candidates(frame, tags, failed_phase, failed_tool)
    symptom = str((rule_candidates[0] if rule_candidates else {}).get("symptom_family") or "unknown")
    matrix_row = canonical_error_type(symptom)
    forced_row = canonical_error_type(forced_error_type)
    if forced_error_type:
        matrix_row = forced_row
        symptom = _symptom_for_error_type(matrix_row, evidence_text)
    inferred_status, inferred_domain = _infer_failure_disposition(
        frame,
        failed_phase,
        failed_tool,
        matrix_row,
    )
    classification_status = forced_classification_status or inferred_status
    failure_domain = forced_failure_domain or inferred_domain
    if classification_status != "classified":
        matrix_row = "unknown"
        symptom = "unknown"
    entity = _infer_entity(frame, symptom)
    version_relation = extract_version_relation(evidence_text, failed_tool)
    failure_mechanism = (
        "version_shortfall"
        if version_relation
        else "configuration_mismatch"
        if has_tool_configuration_failure(evidence_text)
        else "unknown"
    )
    cell = lookup_cell(failed_phase, symptom)
    direct_actions = list(repair_actions_for(failed_phase, symptom))
    conditional_actions = list(conditional_repair_actions_for(failed_phase, symptom))
    fallback_actions = list(fallback_repair_actions_for(failed_phase, symptom))
    matrix_actions = direct_actions
    conditional_directions = tuple(
        item.get("fix_direction", "")
        for item in conditional_actions
        if item.get("fix_direction")
    )
    fallback_directions = tuple(_unique(
        str(item.get("fix_direction") or "")
        for item in fallback_actions
        if item.get("fix_direction")
    ))
    direction_order = _ordered_route_directions(
        tuple(cell.fix_directions),
        tuple(conditional_directions),
    )
    # Fallback alternatives are kept in their own pool.  They become
    # executable only after the selected direct/conditional action is
    # exhausted against the live Build Plan.
    decisions = _candidate_decisions(direction_order, tags)
    checks = _next_checks(symptom, entity, build_plan or {})
    repair_scope = {
        "matrix_key": f"{failed_phase}/{matrix_row}",
        "canonical_error_type": matrix_row,
        "matrix_actions": matrix_actions,
        "selected_matrix_action": matrix_actions[0] if matrix_actions else None,
        "matrix_empty": not bool(matrix_actions),
        "direct_matrix_actions": direct_actions,
        "conditional_actions": conditional_actions,
        "fallback_actions": fallback_actions,
        "fallback_directions": list(fallback_directions),
        "target_decisions": decisions,
        # Conditional revisions become visible to card matching only after
        # their provenance gate opens. Fallback revisions stay in the separate
        # fallback_actions pool until Build Plan alternatives are exhausted.
        "allowed_revision_types": list(_unique(
            list(cell.allowed_revisions)
            + [
                revision
                for action in conditional_actions
                for revision in action.get("revision_types") or []
            ]
        )),
        "fallback_revision_types": list(_unique(
            revision
            for action in fallback_actions
            for revision in action.get("revision_types") or []
        )),
    }
    route = DiagnosisRoute(
        failed_phase=failed_phase,
        failed_tool=failed_tool,
        invocation_role=invocation_role,
        symptom_family=symptom,
        canonical_error_type=matrix_row,
        entity=entity,
        candidate_decisions=decisions,
        next_checks=checks,
        evidence=evidence,
        repair_scope=repair_scope,
        classification_source=classification_source,
        classification_confidence=(
            classification_confidence
            or str((rule_candidates[0] if rule_candidates else {}).get("confidence") or "low")
        ),
        classification_status=classification_status,
        failure_domain=failure_domain,
        rule_candidates=rule_candidates,
        failed_segment=failed_segment,
        failure_mechanism=failure_mechanism,
        version_relation=version_relation,
    )
    route.failure_identity = build_failure_identity(route, frame, failed_command, build_plan).to_dict()
    return route


def fuse_diagnosis(
    failure_frame: dict[str, Any],
    failed_command: Optional[dict[str, Any]],
    build_plan: Optional[dict[str, Any]],
    card_matches: Iterable[dict[str, Any]] = (),
    llm_result: Optional[dict[str, Any]] = None,
) -> DiagnosisRoute:
    """Fuse deterministic classification with card category recall.

    Cards expand the candidate set; they never select the final category or
    repair direction. The LLM is used only when the evidence-supported set is
    ambiguous, and its answer must remain inside that set.
    """
    base = route_failure(failure_frame, failed_command, build_plan)
    cards = list(card_matches or [])
    card_candidates, recalled_card_types = _aggregate_card_candidates(cards)
    supported_card_types = {
        item for item in recalled_card_types
        if _classification_evidence_supported(
            item,
            failure_frame,
            base.failed_phase,
            base.failed_tool,
        )
    }
    rule_candidates = base.rule_candidates
    top_rule_types = _top_candidate_types(rule_candidates)
    recalled_rule_types = {
        str(item.get("canonical_error_type") or "")
        for item in rule_candidates
        if item.get("canonical_error_type")
    }
    causal_evidence = has_causal_diagnostic(_evidence_text(failure_frame))
    allowed_types = set(recalled_rule_types) | set(supported_card_types)
    if not allowed_types and causal_evidence:
        allowed_types = set(CANONICAL_ERROR_TYPES)

    rule_type = base.canonical_error_type
    chosen_type = rule_type
    chosen_phase = base.failed_phase
    source = "rule"
    confidence = base.classification_confidence
    chosen_status = base.classification_status
    chosen_domain = base.failure_domain
    needs_llm = False
    arbitration = dict(llm_result or {})
    strong_rule = bool(
        base.classification_status == "classified"
        and rule_type != "unknown"
        and len(top_rule_types) == 1
        and base.classification_confidence in {"high", "medium"}
    )

    if base.classification_status == "out_of_matrix":
        chosen_type = "unknown"
        source = "rule_disposition"
        confidence = base.classification_confidence
    elif llm_result is not None:
        llm_status = str(llm_result.get("classification_status") or "insufficient_evidence")
        llm_domain = str(llm_result.get("failure_domain") or "unknown")
        llm_type = canonical_error_type(str(llm_result.get("canonical_error_type") or ""))
        llm_confidence = str(llm_result.get("confidence") or "low")
        if llm_status == "out_of_matrix":
            chosen_type = "unknown"
            chosen_status = "out_of_matrix"
            chosen_domain = llm_domain if llm_domain in {"source", "infrastructure"} else "unknown"
            source = "llm_arbitration"
            confidence = llm_confidence
        elif (
            llm_status == "classified"
            and llm_type in allowed_types
            and llm_confidence in {"high", "medium"}
        ):
            chosen_type = llm_type
            chosen_status = "classified"
            chosen_domain = "matrix"
            source = "llm_arbitration"
            confidence = llm_confidence
        else:
            chosen_type = "unknown"
            chosen_status = "insufficient_evidence"
            chosen_domain = "unknown"
            source = "unresolved_after_llm"
            confidence = "low"
    elif strong_rule:
        source = "rule+card_recall" if rule_type in supported_card_types else "rule"
        confidence = "high" if rule_type in supported_card_types else base.classification_confidence
    elif allowed_types:
        chosen_type = "unknown"
        chosen_status = "insufficient_evidence"
        chosen_domain = "unknown"
        source = "ambiguous_candidates"
        confidence = "low"
        needs_llm = True
    else:
        chosen_type = "unknown"
        chosen_status = "insufficient_evidence"
        chosen_domain = "unknown"
        source = "non_causal_or_unsupported"
        confidence = "low"

    if chosen_phase == "unknown":
        card_phases = {
            str(phase)
            for card in cards
            for phase in card.get("matrix_stages") or []
            if str(phase) in VALID_PHASES and str(phase) != "unknown"
        }
        if len(card_phases) == 1:
            chosen_phase = next(iter(card_phases))

    route = route_failure(
        failure_frame,
        failed_command,
        build_plan,
        forced_error_type=chosen_type,
        forced_phase=chosen_phase,
        classification_source=source,
        classification_confidence=confidence,
        forced_classification_status=chosen_status,
        forced_failure_domain=chosen_domain,
    )
    route.rule_candidates = rule_candidates
    route.card_candidates = card_candidates
    route.rule_recalled_error_types = sorted(recalled_rule_types)
    route.card_recalled_error_types = sorted(supported_card_types)
    route.llm_allowed_error_types = sorted(allowed_types) if needs_llm or llm_result is not None else []
    route.card_category_support = (
        chosen_type in supported_card_types
        if cards and chosen_status == "classified" and chosen_type != "unknown"
        else None
    )
    route.llm_arbitration = arbitration
    route.needs_llm_arbitration = needs_llm
    route.failure_identity = build_failure_identity(route, failure_frame, failed_command, build_plan).to_dict()
    return route


def build_failure_identity(
    route: DiagnosisRoute,
    failure_frame: dict[str, Any],
    failed_command: Optional[dict[str, Any]] = None,
    build_plan: Optional[dict[str, Any]] = None,
) -> FailureIdentity:
    """Build a semantic identity that survives paths, line numbers, and log noise."""
    raw_command = str(
        (failed_command or failure_frame.get("primary_command") or failure_frame.get("failed_command") or {}).get("command")
        or ""
    )
    failed_segment = route.failed_segment or _select_failed_segment(raw_command, _evidence_text(failure_frame))
    step_text = _stable_failed_step_signature(
        failed_segment,
        route.failed_tool,
        route.invocation_role,
    )
    failed_step_hash = hashlib.sha1(step_text.encode("utf-8")).hexdigest()[:12] if step_text else ""
    entity_kind = str((route.entity or {}).get("kind") or "").strip().lower()
    entity_name = _normalize_entity_name(str((route.entity or {}).get("name") or ""))
    primary_error = _primary_failure_text(failure_frame) or _evidence_text(failure_frame)
    normalized_error = _normalize_error_signature(primary_error)
    semantic = {
        "phase": route.failed_phase or "unknown",
        "error_type": route.canonical_error_type or "unknown",
        "failed_tool": route.failed_tool or "unknown",
        "invocation_role": route.invocation_role or "unknown",
        "entity_kind": entity_kind,
        "entity_name": entity_name,
        "failed_step_hash": failed_step_hash,
        "normalized_error_signature": normalized_error if not entity_name else "",
    }
    identity_key = hashlib.sha1(
        json.dumps(semantic, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    plan = build_plan or failure_frame.get("build_plan_context") or {}
    selected = plan.get("selected") or {}
    primary_command = failure_frame.get("primary_command") or {}
    context = {
        "sysuse": str(failure_frame.get("sysuse") or selected.get("sysuse") or ""),
        "container_generation": str(failure_frame.get("container_generation") or ""),
        "entry": str(selected.get("build_entry_id") or failure_frame.get("build_entry_id") or ""),
        "target": str(selected.get("target_id") or failure_frame.get("target_id") or ""),
        "workdir": str(primary_command.get("workdir") or ""),
    }
    context_key = hashlib.sha1(
        json.dumps(context, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    occurrence_signature = str(failure_frame.get("failure_signature") or normalized_error or "")
    return FailureIdentity(
        identity_key=identity_key,
        context_key=context_key,
        occurrence_signature=occurrence_signature,
        **semantic,
    )


def _normalize_error_signature(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"(?:[a-z]:)?/(?:[^\s:'\"]+/)+", "<path>/", value)
    value = re.sub(r"(?<=:|\s)\d+(?::\d+)?(?=:|\s|$)", "<n>", value)
    value = re.sub(r"\b0x[0-9a-f]+\b", "<addr>", value)
    value = re.sub(r"\b[0-9a-f]{12,40}\b", "<hash>", value)
    value = re.sub(r"\s+", " ", value).strip()
    return hashlib.sha1(value[:2000].encode("utf-8")).hexdigest()[:16] if value else ""


def _stable_failed_step_signature(command: str, failed_tool: str, invocation_role: str) -> str:
    try:
        tokens = shlex.split(str(command or ""))
    except ValueError:
        tokens = str(command or "").split()
    while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens.pop(0)
    executable = re.sub(r"[^a-z0-9+_.-]", "", (tokens[0].rsplit("/", 1)[-1] if tokens else "").lower())
    salient: list[str] = []
    if executable in {"make", "gmake", "ninja"}:
        skip_next = False
        for token in tokens[1:]:
            if skip_next:
                skip_next = False
                continue
            if token in {"-C", "-f", "-j", "--jobs"}:
                skip_next = True
                continue
            if token.startswith("-") or "=" in token:
                continue
            if re.fullmatch(r"[A-Za-z0-9_.+-]+", token):
                salient.append(token.lower())
    elif executable == "cmake" and "--target" in tokens:
        index = tokens.index("--target")
        if index + 1 < len(tokens):
            salient.append(tokens[index + 1].lower())
    else:
        salient.extend(
            token.rsplit("/", 1)[-1].lower()
            for token in tokens[1:]
            if re.search(r"\.(?:c|cc|cpp|cxx|m|mm)$", token, re.I)
        )
    return "|".join(_unique((failed_tool or "unknown", executable, invocation_role or "unknown", *salient)))


def _shell_segments(command: str) -> list[str]:
    text = str(command or "").strip()
    if not text:
        return []
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return [item.strip() for item in re.split(r"&&|\|\||[;|]", text) if item.strip()]
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= {";", "&", "|"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [shlex.join(tokens) for tokens in segments if tokens]


def _select_failed_segment(command: str, evidence: str) -> str:
    segments = _shell_segments(command)
    if not segments:
        return str(command or "").strip()
    evidence_tool = _evidence_tool(evidence)
    if evidence_tool:
        for segment in reversed(segments):
            tool = _command_tool(segment)
            if tool == evidence_tool:
                return segment
            if tool in {"make", "ninja", "cmake-build", "meson-compile", "build-runner"} and evidence_tool in {
                "compiler", "ld", "make", "ninja"
            }:
                return segment
    for segment in reversed(segments):
        if _command_tool(segment):
            return segment
    return segments[-1]


def _aggregate_card_candidates(
    card_matches: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    details: list[dict[str, Any]] = []
    recalled_types: set[str] = set()
    for card in card_matches:
        candidates = [
            canonical_error_type(item)
            for item in card.get("canonical_error_type_candidates") or []
        ]
        candidates = [item for item in _unique(candidates) if item != "unknown"]
        if not candidates:
            continue
        recalled_types.update(candidates)
        matched_patterns = [str(item) for item in card.get("matched_patterns") or []]
        details.append({
            "card_id": str(card.get("id") or ""),
            "problem_class": str(card.get("problem_class") or ""),
            "canonical_error_type_candidates": candidates,
            "matched_patterns": matched_patterns[:4],
            "confidence": str(card.get("confidence") or "medium"),
            "type_specificity": "exact" if len(candidates) == 1 else "broad",
            "summary": str(card.get("summary") or "")[:400],
        })
    return details, recalled_types


def _top_candidate_types(candidates: Iterable[dict[str, Any]]) -> set[str]:
    values = list(candidates)
    if not values:
        return set()
    top_score = max(int(item.get("score") or 0) for item in values)
    return {
        str(item.get("canonical_error_type") or "")
        for item in values
        if int(item.get("score") or 0) == top_score and item.get("canonical_error_type")
    }


def _infer_phase(
    command: str,
    frame: dict[str, Any],
    build_plan: dict[str, Any],
    *,
    failed_tool: str = "unknown",
) -> str:
    command_l = command.lower()
    selected_stage = str((frame.get("primary_command") or {}).get("stage") or "")
    primary_signals = _combined_signals(frame)
    primary_text = _evidence_text(frame)
    if _is_setup_install_command(command_l, failed_tool):
        # Package-manager wrappers can build native extensions. Preserve the
        # actual compiler/linker failure instead of routing by wrapper name.
        if primary_signals.get("missing_libraries") or _linker_signal(primary_text):
            return "link"
        if _has_compile_evidence(primary_text, primary_signals, failed_tool):
            return "compile"
        return "setup"
    if failed_tool in {"dependency-fetch", "toolchain-manager"}:
        return "setup"
    if failed_tool in {"autoreconf", "autoconf", "automake", "aclocal", "autoheader", "libtoolize", "m4", "bison", "flex", "protoc", "moc", "uic"} or (
        failed_tool in {"", "unknown"}
        and re.search(r"autoreconf|autogen\.sh|bootstrap(?:\.sh)?|buildconf|aclocal|autoheader|libtoolize", command_l)
    ):
        return "bootstrap"
    if failed_tool in {"configure", "cmake", "meson"} or re.search(
        r"\./configure|\bcmake\s+(?:-s|-b|\.|\.\.|--configure)|\bmeson\s+(?!compile\b)|\bqmake\b",
        command_l,
    ):
        return "configure"
    if primary_signals.get("missing_libraries") or _linker_signal(primary_text):
        return "link"
    if failed_tool in {"make", "ninja", "cmake-build", "meson-compile", "build-runner"}:
        return "compile"
    if _is_compile_command(command_l) or re.search(r"\s(?:-c|--compile)\b", command_l):
        return "compile"
    if _has_compile_evidence(primary_text, primary_signals, failed_tool):
        return "compile"
    if selected_stage in {"setup", "bootstrap", "configure", "compile", "link"}:
        return selected_stage
    if re.search(r"\b(?:make|ninja|cmake\s+--build|meson\s+compile|gcc|g\+\+|clang|clang\+\+)\b", command_l):
        return "compile"
    if primary_signals.get("missing_tools"):
        return "setup"
    if failed_tool == "unknown" and has_tool_configuration_failure(primary_text):
        return "setup"
    return "unknown"


def _is_setup_install_command(command: str, failed_tool: str) -> bool:
    if failed_tool in {"apt", "dpkg"}:
        return True
    return bool(re.search(
        r"\b(?:apt(?:-get)?\s+install|apt-cache|dpkg\b|"
        r"pip(?:3)?\s+install|npm\s+(?:install|ci)|"
        r"cargo\s+install|gem\s+install)\b",
        command,
    ))


def _is_compile_command(command: str) -> bool:
    return bool(re.search(r"\b(?:cargo\s+(?:build|check|test|run)|npm\s+run\s+(?:build|compile))\b", command))


def _has_compile_evidence(primary_text: str, primary_signals: dict[str, Any], failed_tool: str) -> bool:
    return bool(
        failed_tool == "compiler"
        or _evidence_tool(primary_text) == "compiler"
        or has_compiler_policy(primary_text)
        or primary_signals.get("missing_headers")
    )


def _infer_tool(command: str, evidence: str) -> str:
    command_tool = _command_tool(command)
    evidence_tool = _evidence_tool(evidence)
    if evidence_tool and re.search(r"&&|\|\||;", str(command or "")) and command_tool not in {"apt", "dpkg"}:
        return evidence_tool
    # make/ninja/cmake --build are wrappers: their child tool can be identified
    # from the actual diagnostic. A direct compiler/configure invocation is
    # authoritative and must not be reclassified by filenames in the log.
    if command_tool in {"make", "ninja", "cmake-build", "meson-compile"}:
        return evidence_tool or command_tool
    return command_tool or evidence_tool or "unknown"


_GENERATOR_TOOLS = {
    "autoreconf", "autoconf", "automake", "aclocal", "autoheader",
    "libtoolize", "m4", "bison", "flex", "protoc", "moc", "uic",
}


def _command_tool(command: str) -> str:
    """Identify the executable invoked by the failed command, not its log."""
    text = str(command or "").strip()
    if not text:
        return ""
    executables = []
    for segment in re.split(r"&&|\|\||[;|]", text):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        while tokens and (tokens[0] == "cd" or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0])):
            tokens = tokens[2:] if tokens[0] == "cd" and len(tokens) > 1 else tokens[1:]
        if tokens and tokens[0] in {"env", "sudo", "command", "exec"}:
            tokens = tokens[1:]
        if tokens:
            executable = tokens[0].rsplit("/", 1)[-1].lower()
            executables.append(executable)
    for executable in executables:
        if executable in {"apt", "apt-get"}:
            return "apt"
        if executable in {"gcc", "g++", "cc", "c++", "clang", "clang++"}:
            return "compiler"
        if executable in {"ld", "collect2"}:
            return "ld"
        if executable in {"make", "ninja"}:
            return executable
        if executable in {"scons", "bazel", "rake"}:
            return "build-runner"
        if executable in {"waf", "waf-light"}:
            return "configure" if re.search(r"\bconfigure\b", text, re.I) else "build-runner"
        if executable == "qmake":
            return "configure"
        if executable in {"wget", "curl", "git", "svn"}:
            return "dependency-fetch"
        if executable == "rustup":
            return "toolchain-manager"
        if executable == "cmake":
            return "cmake-build" if "--build" in text else "cmake"
        if executable == "meson":
            return "meson-compile" if "compile" in text else "meson"
        if executable in _GENERATOR_TOOLS:
            return executable
        if executable in {"dpkg", "pip", "pip3", "npm", "cargo", "gem"}:
            return "pip" if executable == "pip3" else executable
        if executable == "configure" or executable == "configure.ac":
            return "configure"
        if executable in {"pkg-config", "pkgconf"}:
            return "pkg-config"
    return ""


def _evidence_tool(evidence: str) -> str:
    text = str(evidence or "")
    patterns = (
        ("ld", r"(?:^|\n)\s*(?:/[^\s:]+/)?ld:|collect2:|undefined reference|cannot find -l"),
        ("autoreconf", r"(?:^|\n)\s*autoreconf(?:\W|$)|autoreconf:|autom4te:"),
        ("aclocal", r"(?:^|\n)\s*aclocal(?:-[0-9.]+)?(?:\W|$)|aclocal:"),
        ("autoheader", r"(?:^|\n)\s*autoheader(?:\W|$)|autoheader:"),
        ("automake", r"(?:^|\n)\s*automake(?:-[0-9.]+)?(?:\W|$)|automake:"),
        ("libtoolize", r"(?:^|\n)\s*libtoolize(?:\W|$)|libtoolize:"),
        ("autoconf", r"(?:^|\n)\s*autoconf(?:-[0-9.]+)?(?:\W|$)|autoconf:"),
        ("m4", r"(?:^|\n)\s*m4:|gnu m4"),
        ("bison", r"(?:^|\n)\s*bison(?:\W|$)|bison:"),
        ("flex", r"(?:^|\n)\s*flex(?:\W|$)|flex:"),
        ("protoc", r"(?:^|\n)\s*protoc(?:\W|$)|protoc:"),
        ("moc", r"(?:^|\n)\s*moc(?:-qt[0-9]+)?(?:\W|$)|moc:"),
        ("uic", r"(?:^|\n)\s*uic(?:-qt[0-9]+)?(?:\W|$)|uic:"),
        ("cmake", r"cmake error|(?:^|\n)\s*cmake(?:\W|$)"),
        ("meson", r"meson error|(?:^|\n)\s*meson(?:\W|$)"),
        ("pkg-config", r"pkgconf:|pkg-config:"),
        ("configure", r"(?:^|\n)\s*configure:"),
        ("compiler", r"(?:^|\n)\s*(?:gcc|g\+\+|clang|clang\+\+|cc|c\+\+):|fatal error:"),
        ("make", r"(?:^|\n)\s*make(?:\[\d+\])?:\s*\*\*\*"),
        ("ninja", r"(?:^|\n)\s*ninja:\s*build stopped"),
    )
    for tool, pattern in patterns:
        if re.search(pattern, text, re.I):
            return tool
    return ""


def _invocation_role(phase: str, tool: str) -> str:
    if phase == "setup":
        return "dependency_setup"
    if phase == "bootstrap":
        return "generate_build_inputs"
    if phase == "configure":
        return "configure_project"
    if phase == "compile":
        return "compile_sources"
    if phase == "link":
        return "link_artifact"
    return f"invoke_{tool}" if tool and tool != "unknown" else "unknown"


def _source_compile_diagnostic(text: str, phase: str) -> bool:
    if phase != "compile":
        return False
    source_location = r"(?:^|\n)(?:[^\n:]+/)?[^\n:]+\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx):\d+(?::\d+)?:\s*(?:fatal\s+)?error:"
    source_error = (
        r"stray\s+[^\n]+in program|missing terminating\s+['\"]\s+character|"
        r"unterminated\s+(?:comment|string|character constant)|"
        r"invalid suffix\s+[^\n]+on (?:integer|floating) constant|"
        r"exponent has no digits"
    )
    return bool(re.search(source_location, text, re.I) and re.search(source_error, text, re.I))


def _infer_failure_disposition(
    frame: dict[str, Any],
    failed_phase: str,
    failed_tool: str,
    matrix_row: str,
) -> tuple[str, str]:
    if matrix_row != "unknown":
        return "classified", "matrix"
    evidence_text = _evidence_text(frame)
    if _source_compile_diagnostic(evidence_text, failed_phase):
        return "out_of_matrix", "source"
    if re.search(INFRASTRUCTURE_FAILURE_PATTERN, evidence_text, re.I):
        return "out_of_matrix", "infrastructure"
    return "insufficient_evidence", "unknown"


def _classification_evidence_supported(
    error_type: str,
    frame: dict[str, Any],
    failed_phase: str,
    failed_tool: str,
) -> bool:
    """Apply the same phase/tool/evidence boundary to every classifier source."""
    # Callers may pass either a canonical T-code or a legacy row name; both
    # normalize to the FDM v5 T-code used by the branches below.
    error_type = canonical_error_type(error_type)
    evidence_text = _evidence_text(frame)
    signals = _combined_signals(frame)
    tags = set(frame.get("primary_diagnose_tags") or []) | set(frame.get("diagnose_tags") or [])
    if error_type == "T1a":
        return failed_phase == "setup" and failed_tool in {"apt", "dpkg"} and bool(re.search(
            PACKAGE_MANAGER_FAILURE_PATTERN,
            evidence_text,
            re.I,
        ))
    if error_type == "T1c":
        # Legacy rows dependency_discovery_failure, build_input_parsing_failure,
        # and missing_generated_artifacts all map to T1c; the gate accepts
        # evidence supporting any of the three mechanisms.
        if bool(
            signals.get("missing_tools")
            or signals.get("missing_pkg_configs")
            or signals.get("missing_cmake_packages")
            or has_missing_tool(evidence_text)
            or re.search(
                r"dependency\s+[\"'][^\"']+[\"']\s+(?:was\s+)?not found|"
                r"could not find (?:a )?package configuration|not found on system or in local libs|"
                r"(?:program|executable)\s+[\"'][^\"']+[\"']\s+(?:was\s+)?not found|"
                r"(?:subproject|wrap file)[^\n]*(?:not found|missing)|"
                r"Neither a subproject directory nor a [A-Za-z0-9_.+\-]+\.wrap file was found|"
                r"development package[^\n]*(?:not installed|not found|missing)|"
                r"unable to find development files for\s+[A-Za-z0-9_.+\-]+|"
                r"checking for\s+['\"]?[A-Za-z0-9_.+\-]+['\"]?(?:\s*[<>=][^:\n]*)?\s*:\s*not found|"
                r"could not find prerequisite package\s+['\"][A-Za-z0-9_.+\-]+['\"]|"
                r"\b[A-Za-z0-9_.+\-]+\s+could not be found automatically|"
                r"[A-Za-z0-9_.+\-]+\s+not found on system or in local libs|"
                r"configure:\s*error:[^\n]*(?:not found|is required|missing required)|"
                r"(?:perl|python|ruby) module\s+[A-Za-z0-9_.+\-]+\s+not found|"
                r"No\s+[A-Za-z0-9_.+\-]+(?:\s+or\s+[A-Za-z0-9_.+\-]+)+\s+present|"
                r"please install\s+[^\n]+",
                evidence_text,
                re.I,
            )
        ):
            return True
        if failed_phase in {"bootstrap", "configure", "compile"} and bool(re.search(
            r"(?:generated|generation)[^\n]*(?:missing|not found|required)|"
            r"(?:file|macro)[^\n]*(?:does not exist|not found)|"
            r"(?:must|please)\s+(?:first\s+)?run[^\n]*(?:bootstrap|autogen|generate)|"
            r"required (?:generated )?(?:file|artifact)[^\n]*(?:missing|not found)|"
            r"no rule to make target[^\n]+needed by|"
            r"(?:\./|/work/)[^\s:]*configure:\s*No such file|"
            r"(?:configure_file|protoc|bison|flex|moc|uic)[^\n]*(?:output|generated)[^\n]*(?:missing|not found)",
            evidence_text,
            re.I,
        )):
            return True
        if _source_compile_diagnostic(evidence_text, failed_phase):
            return False
        return failed_phase in {"bootstrap", "configure"} and failed_tool in {
            "autoreconf", "autoconf", "automake", "aclocal", "autoheader",
            "libtoolize", "m4", "cmake", "meson", "configure",
        } and bool(re.search(
            r"parse error|syntax error|invalid build definition|first statement must be a call to project|"
            r"meson\.build.*error|cmakelists\.txt.*error|possibly undefined macro|"
            r"unknown (?:cmake|meson) command|cannot find source file|"
            r"no cmake_minimum_required command is present",
            evidence_text,
            re.I,
        ))
    if error_type == "T1b":
        return failed_phase in {"configure", "compile"} and bool(
            signals.get("missing_headers")
            or re.search(MISSING_HEADER_PATTERN, evidence_text, re.I)
        )
    if error_type == "T3a":
        # Legacy rows library_resolution_failure and unresolved_link_symbols
        # both map to T3a; accept evidence for either mechanism.
        return bool(
            signals.get("missing_libraries")
            or re.search(r"cannot find\s+-l|library not found", evidence_text, re.I)
            or _has_linker_signal(evidence_text)
            or _has_unresolved_symbol(evidence_text)
        )
    if error_type == "T4":
        return is_wrong_entry_or_target(evidence_text)
    if error_type == "T6":
        compiler_context = failed_tool == "compiler" or failed_phase in {"compile", "link"} or _evidence_tool(evidence_text) == "compiler"
        return compiler_context and has_compiler_policy(evidence_text)
    if error_type == "T8":
        tool_configuration = bool(re.search(
            r"\b[A-Z][A-Z0-9_]+\s+is set to an invalid directory",
            evidence_text,
            re.I,
        )) or (
            failed_tool in {
                "configure", "cmake", "meson", "pip", "toolchain-manager",
                "autoreconf", "autoconf", "automake", "aclocal", "autoheader",
                "libtoolize", "m4", "bison", "flex", "protoc", "moc", "uic",
            }
            and bool(re.search(TOOL_CONFIGURATION_PATTERN, evidence_text, re.I))
        )
        return tool_configuration or bool(re.search(VERSION_REQUIREMENT_PATTERN, evidence_text, re.I)) or bool(re.search(
            rf"{API_COMPATIBILITY_PATTERN}|"
            r"implicit declaration of function|unknown type name|"
            r"toolchain[^\n]*does not support target",
            evidence_text,
            re.I,
        ))
    return False


def _infer_rule_candidates(
    frame: dict[str, Any],
    tags: set[str],
    failed_phase: str,
    failed_tool: str,
) -> list[dict[str, Any]]:
    evidence_text = _evidence_text(frame)
    signals = _combined_signals(frame)
    candidates: dict[str, dict[str, Any]] = {}

    def add(symptom: str, score: int, reason: str, pattern: str = "") -> None:
        row = canonical_error_type(symptom)
        if row == "unknown" or not _classification_evidence_supported(
            row,
            frame,
            failed_phase,
            failed_tool,
        ):
            return
        evidence_refs = _matching_evidence_refs(frame, pattern) if pattern else _signal_evidence_refs(frame)
        record = {
            "source": "rule",
            "symptom_family": symptom,
            "canonical_error_type": row,
            "score": score,
            "confidence": "high" if score >= 4 else "medium" if score >= 3 else "low",
            "reason": reason,
            "evidence_refs": evidence_refs,
        }
        current = candidates.get(row)
        if current is None or score > int(current.get("score") or 0):
            candidates[row] = record

    if {"runtime_package_conflict", "package_not_found"} & tags or re.search(
        PACKAGE_MANAGER_FAILURE_PATTERN,
        evidence_text,
        re.I,
    ):
        add(
            "apt_install_failure",
            4,
            "package manager reported an unavailable or conflicting provider",
            PACKAGE_MANAGER_FAILURE_PATTERN,
        )
    if is_wrong_entry_or_target(evidence_text):
        add("wrong_entrypoint", 4, "the selected entrypoint or target does not contain the expected build input")
    if failed_phase == "bootstrap" and re.search(
        r"(?:file|macro)[^\n]*(?:does not exist|not found)|"
        r"aclocal[^\n]*configure\.ac[^\n]*(?:missing|not found)",
        evidence_text,
        re.I,
    ):
        add(
            "generated_file_missing",
            5,
            "bootstrap failed because a required generated macro/input is absent",
            r"(?:file|macro)[^\n]*(?:does not exist|not found)|aclocal",
        )
    if signals.get("missing_tools") or has_missing_tool(evidence_text):
        add(
            "build_tool_not_found",
            4,
            "the active build step invokes a tool whose provider is unavailable",
            MISSING_TOOL_PATTERN,
        )
    if signals.get("missing_pkg_configs"):
        add("pkg_config_failure", 4, "dependency metadata or development package is missing")
    if signals.get("missing_cmake_packages"):
        add("cmake_package_failure", 4, "CMake package discovery failed")
    if signals.get("missing_headers") or re.search(MISSING_HEADER_PATTERN, evidence_text, re.I):
        add(
            "header_not_found",
            4,
            "compiler or configure check reported a missing header",
            MISSING_HEADER_PATTERN,
        )
    if signals.get("missing_libraries"):
        add("library_not_found", 4, "linker reported a missing library")
    if _has_linker_signal(evidence_text) or _has_unresolved_symbol(evidence_text):
        add("unresolved_symbol", 4, "the diagnostic is emitted by the real linker", r"undefined|multiple definition|cannot find\s+-l|collect2|ld returned")

    if re.search(VERSION_REQUIREMENT_PATTERN, evidence_text, re.I) or re.search(API_COMPATIBILITY_PATTERN, evidence_text, re.I) or "version requirement" in signals.get("toolchain_errors", []):
        add("api_or_toolchain_mismatch", 4, "an observed tool or dependency version violates a declared requirement", rf"{VERSION_REQUIREMENT_PATTERN}|{API_COMPATIBILITY_PATTERN}")

    dependency_pattern = (
        r"dependency\s+[\"'][^\"']+[\"']\s+(?:was\s+)?not found|"
        r"development package installed|could not find\s+[^\n]+\s+include director|"
        r"unable to find development files for\s+[A-Za-z0-9_.+\-]+|"
        r"checking for\s+['\"]?[A-Za-z0-9_.+\-]+['\"]?(?:\s*[<>=][^:\n]*)?\s*:\s*not found|"
        r"could not find prerequisite package\s+['\"][A-Za-z0-9_.+\-]+['\"]|"
        r"\b[A-Za-z0-9_.+\-]+\s+could not be found automatically|"
        r"[A-Za-z0-9_.+\-]+\s+not found on system or in local libs|"
        r"cannot find\s+[^\n]+\s+development librar|"
        r"not found on system or in local libs|"
        r"(?:subproject|wrap file)[^\n]*(?:not found|missing)|"
        r"Neither a subproject directory nor a [A-Za-z0-9_.+\-]+\.wrap file was found|"
        r"(?:program|executable)\s+[\"'][^\"']+[\"']\s+(?:was\s+)?not found|"
        r"configure:\s*error:[^\n]*(?:not found|is required|missing required)|"
        r"(?:perl|python|ruby) module\s+[A-Za-z0-9_.+\-]+\s+not found|"
        r"No\s+[A-Za-z0-9_.+\-]+(?:\s+or\s+[A-Za-z0-9_.+\-]+)+\s+present|"
        r"please install\s+[^\n]+"
    )
    if re.search(dependency_pattern, evidence_text, re.I):
        add("pkg_config_failure", 3, "dependency discovery failed with an explicit missing provider", dependency_pattern)

    generation_prerequisite_pattern = (
        r"(?:must|please)\s+(?:first\s+)?run\s+[^\n]*(?:bootstrap|autogen|generate)|"
        r"(?:generated|generation)[^\n]*(?:file|artifact|output)[^\n]*(?:missing|not found|required)|"
        r"(?:file|macro)[^\n]*(?:does not exist|not found)|"
        r"no rule to make target[^\n]+needed by|"
        r"(?:\./|/work/)[^\s:]*configure:\s*No such file"
    )
    if re.search(generation_prerequisite_pattern, evidence_text, re.I):
        add("generated_file_missing", 4, "the build explicitly requires a generator or prerequisite setup step", generation_prerequisite_pattern)

    parsing_pattern = (
        r"first statement must be a call to project|no backend specified|"
        r"build input.*(?:parse|syntax) error|invalid build definition|"
        r"unknown (?:cmake|meson) command|cannot find source file|"
        r"no cmake_minimum_required command is present"
    )
    if re.search(parsing_pattern, evidence_text, re.I):
        add("source_path_missing", 3, "the selected build definition cannot be parsed as a valid project", parsing_pattern)
    if failed_phase in {"bootstrap", "configure"} and re.search(
        r"syntax error near unexpected token|possibly undefined macro|parse error|invalid build definition",
        evidence_text,
        re.I,
    ):
        add("source_path_missing", 4, "the active build tool cannot parse the selected build definition")

    if re.search(TOOL_CONFIGURATION_PATTERN, evidence_text, re.I):
        add("api_or_toolchain_mismatch", 4, "the active environment or tool does not support the requested interface", TOOL_CONFIGURATION_PATTERN)
    if re.search(r"toolchain[^\n]*does not support target", evidence_text, re.I):
        add("api_or_toolchain_mismatch", 4, "the active toolchain does not expose the requested target interface", r"toolchain[^\n]*does not support target")

    ci_wrapper_pattern = (
        r"(?:travis|github actions|ci (?:job|environment))[^\n]*(?:must be set|unbound variable)|"
        r"\.(?:travis|github)/[^\n]*(?:unbound variable|no such file)"
    )
    if re.search(ci_wrapper_pattern, evidence_text, re.I):
        add("wrong_entrypoint", 4, "the selected wrapper requires CI-only state unavailable in this build plan", ci_wrapper_pattern)

    if signals.get("missing_paths") and not signals.get("missing_tools"):
        symptom = "generated_file_missing" if failed_phase == "bootstrap" else "source_path_missing"
        add(symptom, 4, "the failed command references a missing build input")
    if signals.get("bootstrap_errors") and not signals.get("missing_tools"):
        symptom = (
            "generated_file_missing"
            if re.search(r"no such file|not found|missing|required file", evidence_text, re.I)
            else "source_path_missing"
        )
        add(symptom, 3, "bootstrap or generator diagnostics identify missing or invalid build input")
    if failed_phase == "bootstrap" and re.search(
        r"syntax error|undefined macro|possibly undefined macro|invalid directive|"
        r"unrecognized option|cannot open|m4 failed|parse error|unexpected token",
        evidence_text,
        re.I,
    ):
        add("source_path_missing", 3, "bootstrap input parsing failed")

    if has_compiler_policy(evidence_text):
        add("compiler_policy_error", 4, "compiler or linker policy rejected the current build configuration")
    if "build_layer_compat_needed" in tags and not candidates:
        add("api_or_toolchain_mismatch", 2, "generic build-layer compatibility evidence was observed")

    return sorted(
        candidates.values(),
        key=lambda item: (-int(item.get("score") or 0), str(item.get("canonical_error_type") or "")),
    )


def _infer_symptom(
    frame: dict[str, Any],
    tags: set[str],
    failed_phase: str,
    failed_tool: str = "unknown",
) -> str:
    candidates = _infer_rule_candidates(frame, tags, failed_phase, failed_tool)
    return str((candidates[0] if candidates else {}).get("symptom_family") or "unknown")


def _symptom_for_error_type(error_type: str, evidence_text: str = "") -> str:
    error_type = canonical_error_type(error_type)
    evidence_lower = evidence_text.lower()
    if error_type == "T1c":
        # Dependency-discovery, build-input, and generation failures share
        # T1c; pick the most specific symptom family the evidence supports.
        if has_missing_tool(evidence_text):
            return "build_tool_not_found"
        if re.search(
            r"(?:generated|generation)[^\n]*(?:missing|not found|required)|"
            r"(?:must|please)\s+(?:first\s+)?run[^\n]*(?:bootstrap|autogen|generate)",
            evidence_text,
            re.I,
        ):
            return "generated_file_missing"
        if "cmake" in evidence_lower:
            return "cmake_package_failure"
        if "pkg-config" in evidence_lower or "pkg_config" in evidence_lower:
            return "pkg_config_failure"
        return "source_path_missing"
    if error_type == "T3a":
        return "unresolved_symbol" if "undefined reference" in evidence_lower else "library_not_found"
    return {
        "T1a": "apt_install_failure",
        "T1b": "header_not_found",
        "T4": "wrong_entrypoint",
        "T6": "compiler_policy_error",
        "T8": "api_or_toolchain_mismatch",
    }.get(error_type, "unknown")


_AUTOCONF_CONTEXT_PATTERN = re.compile(
    r"\b(?:autoreconf|autoconf|autoheader|automake|aclocal|libtoolize|configure\.ac|configure\.in|m4)\b",
    re.I,
)
_AUTOCONF_MACRO_PREFIX_PATTERN = re.compile(r"^(?:AX|AC|AM|LT|PKG)_[A-Za-z0-9_]+$")
_AUTOCONF_MACRO_PATTERNS = (
    r"(?:possibly\s+)?undefined\s+macro:\s*[`'\"]?([A-Za-z][A-Za-z0-9_]{2,})",
    r"(?:macro|aclocal)[^\n]{0,100}[`'\"]([A-Za-z][A-Za-z0-9_]{2,})[`'\"][^\n]{0,100}(?:not found|missing|undefined)",
    r"error:\s+macro\s+[`'\"]?([A-Za-z][A-Za-z0-9_]{2,})[`'\"]?\s+(?:not found|undefined)",
)


def _extract_autoconf_macro_entity(text: str) -> dict[str, str]:
    if not text or not _AUTOCONF_CONTEXT_PATTERN.search(text):
        return {}
    for pattern in _AUTOCONF_MACRO_PATTERNS:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        macro = _normalize_entity_name(match.group(1))
        if macro and _AUTOCONF_MACRO_PREFIX_PATTERN.match(macro):
            return {"kind": "build_macro", "name": macro}
    return {}


def _infer_entity(frame: dict[str, Any], symptom: str) -> dict[str, str]:
    signals = _combined_signals(frame)
    text = _evidence_text(frame)
    if symptom in {"generated_file_missing", "source_path_missing", "pkg_config_failure"}:
        macro_entity = _extract_autoconf_macro_entity(text)
        if macro_entity:
            return macro_entity
    mapping = {
        "header_not_found": ("header", signals.get("missing_headers")),
        "library_not_found": ("library", signals.get("missing_libraries")),
        "pkg_config_failure": ("pkg_config_module", signals.get("missing_pkg_configs")),
        "cmake_package_failure": ("cmake_package", signals.get("missing_cmake_packages")),
        "build_tool_not_found": ("build_tool", signals.get("missing_tools")),
        "source_path_missing": ("path", signals.get("missing_paths")),
    }
    if symptom in mapping:
        kind, values = mapping[symptom]
        name = _normalize_entity_name(str((values or [""])[0]))
        return {"kind": kind, "name": name} if name else {}
    if symptom == "unresolved_symbol":
        symbol = _extract_unresolved_symbol(_evidence_text(frame))
        if symbol:
            return {"kind": "function_symbol", "name": symbol}
    generic_patterns = (
        ("build_tool", r"Program\s+[\"']([^\"']+)[\"']\s+not found"),
        ("build_tool", r"Cannot continue without\s+([A-Za-z0-9_.+\-]+)"),
        ("build_tool", r"(?:command|executable)\s+[\"']([^\"']+)[\"']\s+(?:was\s+)?not found"),
        ("cmake_package", r"could not find a Qt installation"),
        ("header", r"([A-Za-z0-9_.+\-/]+)\s+headers? (?:were |was )?not found"),
    )
    for kind, pattern in generic_patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        name = _normalize_entity_name(match.group(1) if match.groups() else "qt")
        if name:
            return {"kind": kind, "name": name}
    return {}


_INVALID_ENTITY_NAMES = {
    "", "source", "command", "exec", "test", "true", "false", "none",
    "cxx", "cc", "cpp", "cflags", "cxxflags", "ldflags",
}


def _normalize_entity_name(value: str) -> str:
    name = str(value or "").strip().strip("`'\"")
    if not name or name.lower() in _INVALID_ENTITY_NAMES:
        return ""
    if name.startswith(("-", "./", "../", "/")) or "/" in name and not re.search(
        r"\.(?:h|hh|hpp|hxx|pc|cmake|m4)$", name, re.I
    ):
        return ""
    if re.search(r"[<>{}$()]", name):
        return ""
    return name


def _candidate_decisions(matrix_directions: tuple[str, ...], tags: Iterable[str] = ()) -> list[str]:
    decisions = list(matrix_directions)
    normalized = [canonical_fix_direction(item) for item in decisions]
    if "D2" in normalized:
        decisions.insert(normalized.index("D2"), "link_command")
    expanded = []
    deferred_canonical: list[str] = []
    for decision in decisions:
        canonical = canonical_fix_direction(decision)
        # Keep the legacy link label first.  The canonical linker direction is
        # retained later in the list without changing Matrix action order.
        if canonical == "D2":
            expanded.append("link_command")
            deferred_canonical.extend(["linker_configuration", "D2"])
            continue
        if canonical == "D4":
            # D4 merges the legacy dependency_discovery, build_entry, and
            # build_target directions; keep all three labels visible to
            # downstream consumers that match on them directly.
            expanded.extend(["dependency_discovery", "build_entry", "build_target", "D4"])
            continue
        compatible = list(compatibility_directions(decision))
        expanded.extend(compatible)
    expanded.extend(deferred_canonical)
    return _unique(expanded)


def _ordered_route_directions(
    direct: tuple[str, ...],
    conditional: tuple[str, ...],
) -> tuple[str, ...]:
    """Keep compatibility decision aliases while preserving direct action rank."""
    if any(canonical_fix_direction(item) == "D2" for item in direct):
        return tuple(_unique(
            ("D2",)
            + conditional
            + tuple(item for item in direct if canonical_fix_direction(item) != "D2")
        ))
    return tuple(_unique(conditional + direct))


def _next_checks(symptom: str, entity: dict[str, str], build_plan: dict[str, Any]) -> list[str]:
    name = entity.get("name", "")
    checks = {
        "apt_install_failure": [
            "inspect failed apt package list for mutually exclusive providers",
            "verify one provider per family in the current Debian suite",
        ],
        "build_tool_not_found": [
            f"resolve a verified system or source provider for build tool {name}",
            "check whether the project supplies or bootstraps the tool before installing a package",
        ],
        "pkg_config_failure": [
            f"verify package/provider for pkg-config module {name}",
            "check PKG_CONFIG_PATH and source dependency metadata exports",
        ],
        "cmake_package_failure": [
            f"verify package/provider for CMake package {name}",
            "check CMAKE_PREFIX_PATH and source dependency install prefix",
        ],
        "header_not_found": [
            f"check whether {name} exists under /work",
            "if present, inspect actual include paths",
            "if absent, resolve dependency provider or source dependency",
        ],
        "library_not_found": [
            f"check whether lib{name}.so or lib{name}.a exists",
            "inspect actual link -L paths and provider package/source dependency",
        ],
        "source_path_missing": [
            f"check whether {name} is generated, vendored, or submodule-managed",
            "prefer executable source_dependency or generation step over apt guesses",
        ],
        "wrong_entrypoint": [
            "inspect build_plan entrypoint candidates and wrappers",
            "try high-confidence documented/script entry before installing dependencies",
        ],
        "generated_file_missing": [
            "inspect bootstrap/autogen/generation commands in build plan",
            "check generated files before editing source",
        ],
        "compiler_policy_error": [
            "identify exact compiler/linker policy signal",
            "try build-system flag or tool shim before package churn",
        ],
        "api_or_toolchain_mismatch": [
            "identify dependency header/provider version used by actual command",
            "compare current suite/provider against target commit era",
        ],
        "unresolved_symbol": [
            f"identify owner of unresolved symbol {name}",
            "inspect actual link line for missing library, link order, or omitted project object",
            "if dependency-owned, verify provider version and ABI against target commit era",
        ],
    }.get(symptom, ["read latest full build log and identify controlled build-plan decision"])
    if build_plan.get("entrypoints"):
        checks.append("avoid repeating entrypoints marked failed in build_plan")
    return checks


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _evidence_text(frame: dict[str, Any]) -> str:
    return "\n".join(_evidence_lines_from_frame(frame))


def _evidence_lines_from_frame(frame: dict[str, Any]) -> list[str]:
    bundle = [
        str(item.get("text") or "").strip()
        for item in frame.get("evidence_bundle") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    primary = _primary_failure_text(frame)
    supporting = [str(item).strip() for item in frame.get("supporting_evidence") or [] if str(item).strip()]
    evidence = [str(item).strip() for item in frame.get("evidence") or [] if str(item).strip()]
    return _unique(bundle + ([primary] if primary else []) + supporting + evidence)[:12]


def _combined_signals(frame: dict[str, Any]) -> dict[str, list[str]]:
    primary = frame.get("primary_signals") or {}
    keys = (
        "missing_headers",
        "missing_libraries",
        "missing_pkg_configs",
        "missing_cmake_packages",
        "missing_paths",
        "missing_tools",
        "toolchain_errors",
        "bootstrap_errors",
    )
    return {
        key: _unique(
            [str(item) for item in primary.get(key) or []]
            + [str(item) for item in frame.get(key) or []]
        )
        for key in keys
    }


def _matching_evidence_refs(frame: dict[str, Any], pattern: str) -> list[str]:
    if not pattern:
        return _signal_evidence_refs(frame)
    refs = []
    for item in frame.get("evidence_bundle") or []:
        if not isinstance(item, dict):
            continue
        if re.search(pattern, str(item.get("text") or ""), re.I):
            refs.append(str(item.get("id") or ""))
    return [item for item in refs if item][:4] or _signal_evidence_refs(frame)


def _signal_evidence_refs(frame: dict[str, Any]) -> list[str]:
    refs = [
        str(item.get("id") or "")
        for item in frame.get("evidence_bundle") or []
        if isinstance(item, dict) and item.get("id")
    ]
    return refs[:2]


def _primary_failure_text(frame: dict[str, Any]) -> str:
    """Read the stable diagnostic fragment field, with legacy fallback."""
    text = str(frame.get("primary_failure_text") or "").strip()
    if text:
        return text
    return str((frame.get("primary_failure") or {}).get("text") or "").strip()


def _has_unresolved_symbol(text: str) -> bool:
    return bool(re.search(r"undefined reference to|unresolved external|undefined symbol", text, re.I))


def _has_linker_signal(text: str) -> bool:
    return bool(re.search(r"undefined reference|multiple definition|cannot find -l|collect2:|ld returned", text, re.I))


def _linker_signal(line: str) -> bool:
    return bool(re.search(r"undefined reference|multiple definition|cannot find -l|collect2:|ld returned", str(line or ""), re.I))


def _extract_unresolved_symbol(text: str) -> str:
    patterns = [
        r"undefined reference to\s+[`'‘\"]([^`'’\"\n]+)",
        r"undefined symbol:\s*([A-Za-z0-9_:@~.$<>(), +*\-/]+)",
        r"unresolved external symbol\s+([A-Za-z0-9_:@~.$<>(), +*\-/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).strip()
    return ""
