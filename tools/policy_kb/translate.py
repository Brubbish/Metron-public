"""Translate one selected policy-card action into a typed RepairDecision.

Card actions are natural-language hints (e.g. "CFLAGS=\"-include grp.h\"").
To execute them we need actual shell commands adapted to the project's build
system (autotools / cmake / make / meson). This module makes ONE lightweight
LLM call to perform that translation — the LLM does NOT diagnose (the card
already did that), it only adapts the action to the project context.

Output is a raw JSON string compatible with `repair_plan.parse_repair_decision`.
On failure, returns a cannot_solve JSON so the caller falls back to the
regular Phase 3 LLM path.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from llm.ask_llm import async_ask

logger = logging.getLogger(__name__)


def _format_actions(actions: list) -> str:
    lines = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        t = a.get("type") or a.get("card_action_type") or "?"
        h = a.get("hint", "")
        lines.append(f"  - [{t}] {h}")
    return "\n".join(lines) if lines else "  (no actions)"


def _format_non_solutions(non_solutions: list) -> str:
    if not non_solutions:
        return "  (none)"
    return "\n".join(f"  - {n}" for n in non_solutions)


def build_translation_prompt(card: Dict[str, Any], project_context: Dict[str, Any]) -> tuple[str, str]:
    """Build prompts for one controller-selected card action."""
    system_prompt = (
        "You are CardTranslator. Convert only the controller-selected policy-card "
        "hint into one typed operation. Treat the failure classification and selected "
        "action as immutable. Follow only the supplied operation schema and candidate "
        "IDs; return cannot_solve when they do not support the hint. Do not diagnose, "
        "select another action, generate shell commands, or invent values.\n\n"
        "Output only raw RepairDecision JSON."
    )

    failure = project_context.get("failure")
    if not isinstance(failure, dict):
        failure = {
            "failure_identity": project_context.get("failure_identity") or "",
            "failed_command": project_context.get("failed_command") or "",
        }
    context = {
        "failure": failure,
        "selected_action": project_context.get("selected_action") or {},
        "supporting_card": {
            "id": card.get("id") or "",
            "summary": card.get("summary") or "",
            "hints": project_context.get("selected_card_actions") or [],
            "non_solutions": card.get("non_solutions") or [],
        },
        "candidate_catalog": project_context.get("candidate_catalog") or {},
    }
    if project_context.get("current_execution"):
        context["current_execution"] = project_context["current_execution"]
    user_prompt = (
        "Concretize the selected card hint from this compact context:\n"
        f"```json\n{json.dumps(context, ensure_ascii=False, indent=2)}\n```\n\n"
        "Return exactly one raw JSON object:\n"
        '{"status":"repair|cannot_solve","operation":{"kind":"<supplied kind>",'
        '"<schema fields>":"<supplied values>"},"evidence_refs":["<supplied id>"],'
        '"expected_effect":"<observable change>","reason":"<short summary>"}'
    )
    return system_prompt, user_prompt


async def translate_card_to_fix_plan(
    card: Dict[str, Any],
    project_context: Dict[str, Any],
    model: Optional[str] = None,
    caller: str = "card_translation",
) -> str:
    """Translate a card to a typed RepairDecision JSON string.

    Returns the raw LLM output (a JSON string). The caller should pass it to
    `agent_build_main.parse_fix_plan()` which tolerates malformed output and
    falls back to cannot_solve.
    """
    system_prompt, user_prompt = build_translation_prompt(card, project_context)
    try:
        response = await async_ask(
            system_prompt, user_prompt,
            max_retries=2,
            retry_wait=2,
            model=model,
            caller=caller,
        )
    except Exception as exc:
        logger.warning("card translation LLM call failed for %s: %s", card.get("id"), exc)
        return '{"status":"cannot_solve","reason":"card translation LLM call failed"}'

    if not response or response.strip().startswith("Error:"):
        logger.warning("card translation LLM returned error for %s: %s", card.get("id"), response)
        return '{"status":"cannot_solve","reason":"card translation LLM error"}'

    return response


def parse_card_translation(raw_output: str, card_id: str) -> Optional[Dict[str, Any]]:
    """Lightweight parser for card translation output.

    Returns parsed dict on success, None on parse failure. Caller can use
    this directly or just feed raw_output to parse_fix_plan.
    """
    text = (raw_output or "").strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Ensure hypothesis field is set for dedup
            reason = str(parsed.get("reason") or "")
            if "hypothesis:" not in reason:
                parsed["reason"] = f"hypothesis: card_{card_id}; detail: {reason or 'translated card action'}"
            return parsed
    except json.JSONDecodeError:
        return None
    return None


# ---------------------------------------------------------------------------
# Batch translation: translate ALL cards in ONE LLM call.
# ---------------------------------------------------------------------------

def build_batch_translation_prompt(
    cards: list,
    project_context: Dict[str, Any],
) -> tuple[str, str]:
    """Build (system, user) prompts for batch card translation.

    The LLM sees every card at once and emits a JSON array of FixPlan dicts,
    one per card, in the SAME order as the input cards (which are already
    sorted by cost ascending). Each emitted plan carries `card_id` so the
    caller can trace it back to the source card.
    """
    card_blocks = []
    for idx, card in enumerate(cards):
        card_blocks.append("\n".join([
            f"### Card #{idx} (id={card.get('id', '?')})",
            f"- Summary: {card.get('summary', '')}",
            f"- Recommended actions:",
            _format_actions(card.get("actions", [])),
            f"- Known non-solutions (do NOT output these):",
            _format_non_solutions(card.get("non_solutions", [])),
        ]))
    cards_section = ("\n\n".join(card_blocks)) if card_blocks else "(no cards)"

    system_prompt = (
        "You are CardBatchTranslator. Translate ALL of the provided policy "
        "cards' action hints into concrete build commands for THIS project. "
        "Do NOT diagnose — the cards already did that. Your only job is to "
        "adapt each card's actions to the project's actual build system.\n\n"
        "Output ONLY a raw JSON ARRAY with exactly ONE FixPlan object per "
        "input card, in the SAME ORDER as the input cards. Each object MUST "
        "include a `card_id` field echoing the input card id. No prose, no "
        "markdown fences, no explanations outside the array."
    )

    user_prompt = "\n\n".join([
        "## Project Context",
        f"- Build system: {project_context.get('build_system', 'unknown')}",
        f"- Build workdir: {project_context.get('build_workdir', '/work')}",
        f"- Current Debian suite: {project_context.get('sysuse', 'unknown')}",
        f"- Already-known build_options: {project_context.get('build_options', [])}",
        f"- Most recent failed command: {project_context.get('failed_command', '')}",

        "## Input Cards (already diagnosed, sorted by cost ascending)",
        cards_section,

        "## Output Format",
        "Return ONLY a JSON array. Exactly one element per input card, in the",
        "same order as the cards above. Each element has this shape:",
        '[{"card_id":"<input card id>","status":"fix","reason":"hypothesis: card_<id>; detail: <one sentence>","commands":["<one or more shell commands>"],"sysuse":null,"source_dependency":null}]',
        "For a card you cannot translate, emit an element with status "
        '"cannot_solve" (still with its card_id) instead of skipping it.',

        "## Translation Rules",
        "1. Preserve the input order and card_id fields.",
        "2. Each object must concretize only the matrix action represented by that card; do not rank or select alternatives.",
        "3. Adapt commands to the recorded build system, verify packages, preserve the commit, and avoid source edits or invented targets.",
        "4. Return cannot_solve for a card whose selected action lacks evidence or a safe operation.",
    ])
    return system_prompt, user_prompt


async def translate_cards_to_fix_plans(
    cards: list,
    project_context: Dict[str, Any],
    model: Optional[str] = None,
    caller: str = "card_translation_batch",
) -> list:
    """Translate multiple cards in ONE LLM call, returning ordered FixPlans.

    Each card's actions are translated to concrete commands. The LLM sees all
    cards at once and emits one JSON array of FixPlan objects, in the same
    order as the input cards (which the caller should already sort by cost
    ascending). Each returned dict carries a `card_id` field for traceability.

    On LLM failure, returns an empty list — the caller can then fall back to
    single-card translation (`translate_card_to_fix_plan`).
    """
    if not cards:
        return []

    system_prompt, user_prompt = build_batch_translation_prompt(cards, project_context)
    try:
        response = await async_ask(
            system_prompt, user_prompt,
            max_retries=2,
            retry_wait=2,
            model=model,
            caller=caller,
        )
    except Exception as exc:
        logger.warning(
            "batch card translation LLM call failed: %s", exc,
        )
        return []

    if not response or response.strip().startswith("Error:"):
        logger.warning("batch card translation LLM returned error: %s", response)
        return []

    parsed = parse_batch_translation(response)
    return parsed


def parse_batch_translation(raw_output: str) -> list:
    """Parse the LLM's array output into a list of FixPlan dicts.

    Tolerates: markdown fences, leading/trailing prose. Returns list of dicts;
    each dict has at least 'card_id' and 'status' fields. On parse failure
    returns an empty list.
    """
    text = (raw_output or "").strip()

    # Strip markdown fences if present (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:]).strip()

    # Locate the outermost JSON array when there is surrounding prose.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    result = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "card_id" not in item or "status" not in item:
            continue
        result.append(item)
    return result
