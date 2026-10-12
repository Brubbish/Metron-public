import json
from pathlib import Path

import pytest

from core.enhanced_dep_extractor.conflict_filter import filter_conflicting_candidates

# The conflicts index is part of the optional Build_RAG_Info dataset
# (rebuildable from public Debian data and not shipped with the repo).
_CONFLICTS_INDEX = Path("Build_RAG_Info/conflicts_index.json")


FIXTURES = [
    "_agent_batch_gemma4_collects_23_050/worktrees/00000_01org__tpm2.0-tools__c5d72beaab1c/c5d72beaab1cbbbe68271f4bc4b6670d69985157/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00015_DanBloomberg__leptonica__3c18c43b6a3f/3c18c43b6a3f753f0dfff99610d46ad46b8bfac4/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00022_FFmpeg__FFmpeg__9cb4eb772839/9cb4eb772839c5e1de2855d126bf74ff16d13382/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00020_Exim__exim__31323b30449a/31323b30449a59713ed7a99f86e356cb1b9d0fb7/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00025_FRRouting__frr__d5f17cd51e6e/d5f17cd51e6e5fcef4c8b8509a3538d655dadf36/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00021_FFmpeg__FFmpeg__afc9c683ed9d/afc9c683ed9db01edb357bc8c19edad4282b3a97/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00009_BelledonneCommunications__belle-sip__d3f0651531e4/d3f0651531e45e91c2e60f3a16a8b612802e5d2d/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00046_GNOME__nautilus__1630f53481f4/1630f53481f445ada0a455e9979236d31a8d3bb0/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00023_FFmpeg__FFmpeg__7ba100d3e6e8/7ba100d3e6e8b1e5d5342feb960a7f081d6e15af/ccscanner_result.json",
    "_agent_batch_gemma4_collects_23_050/worktrees/00027_FreeRDP__FreeRDP__ed53cd148f43/ed53cd148f43cbab905eaa0f5308c2bf3c48cc37/ccscanner_result.json",
]


def _collect_depnames(ccscanner_result: dict) -> list[str]:
    names: set[str] = set()
    for extractor in ccscanner_result.get("extractors", []):
        for dep in extractor.get("deps", []):
            depname = dep.get("depname")
            if isinstance(depname, str) and depname and not depname.startswith("#"):
                names.add(depname)
    return sorted(names)


@pytest.mark.skipif(
    not _CONFLICTS_INDEX.exists(),
    reason="optional Build_RAG_Info/conflicts_index.json dataset is not present",
)
def test_conflict_strategy_smoke_on_10_real_projects():
    conflicts = json.loads(_CONFLICTS_INDEX.read_text())["conflicts"]["bookworm"]

    # Synthetic resolved set to exercise well-known hard conflicts.
    resolved = {
        "libcurl4-gnutls-dev",
        "bird",
    }

    checked = 0
    total_hits = 0
    for fixture in FIXTURES:
        path = Path(fixture)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        candidates = _collect_depnames(data)
        kept, dropped = filter_conflicting_candidates(candidates, resolved, conflicts)
        assert len(kept) + len(dropped) == len(candidates)
        for item in dropped:
            assert item["candidate"] in candidates
            assert item["conflicts_with"] in resolved
        total_hits += len(dropped)
        checked += 1

    assert checked >= 10
    assert total_hits >= 0
