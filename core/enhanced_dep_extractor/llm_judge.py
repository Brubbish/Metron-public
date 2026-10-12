"""LLM judge: post-prediction KEEP/DROP filter using structured evidence.

After the rule+backfill pipeline produces candidate packages, this module sends
a single batched LLM call per project to judge each candidate as KEEP (real
build-dep) or DROP (false positive).

The LLM sees structured evidence per candidate (not raw source code):
  - which bucket/strategy produced it
  - the matched_text from the rule match
  - artifact kind (library/header/cmake_package/executable/pkgconfig)
  - provider token + cardinality
  - stem occurrences in project source

No few-shot examples, no package-name lists, no tool-name allowlists in the prompt.
The LLM uses its own code understanding to interpret the evidence.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .confidence_filter import _pkg_stem, _load_cardinality, _load_pkg_meta
from .metadata import DEFAULT_DEBIAN_VERSION


JUDGE_SYSTEM_PROMPT = """\
You are a Debian packaging expert. For each candidate package, decide whether it
is a REAL build dependency (KEEP) or a FALSE POSITIVE (DROP) based on the
structured evidence provided.

Decision criteria (apply structurally — do NOT use memorized package lists):

1. KEEP if the evidence shows the project explicitly links against or requires
   this library/tool at build time:
   - matched_text contains find_package(), find_library(), pkg_check_modules(),
     target_link_libraries(), dependency(), AC_CHECK_LIB, or #include directive
   - artifact_kind is cmake_package, pkgconfig, or header
   - stem appears in source within a dependency-declaration context

2. DROP if:
   - evidence_source is "backfill" with no other supporting evidence
   - provider_token is a short common word that appears in source as a variable
     name, type name, or English word (e.g. "file" in "FILE* f", "make" in
     "Makefile") rather than a library reference
   - stem appears 0 times in project source (no textual evidence at all)

3. If unclear or ambiguous, KEEP (prefer recall over precision).

Respond ONLY as a JSON array:
[{"package": "<name>", "decision": "KEEP"|"DROP", "reason": "<one sentence>"}]"""


def _collect_evidence(
    per_file: dict,
    project_path: str,
    *,
    version: str = DEFAULT_DEBIAN_VERSION,
    max_candidates: int = 30,
) -> list[dict]:
    """Collect structured evidence for each predicted package.

    Returns a list of dicts, one per unique package, with evidence fields.
    Caps at max_candidates to control prompt size.

    Packages from `debian/control` and per-file rule matches with explicit
    dependency-declaration evidence (find_package/find_library/etc) are
    EXCLUDED from judging — they already have strong provenance and the judge
    would only risk losing true positives.
    """
    from .confidence_filter import _build_project_blob, _stem_in_blob

    card = _load_cardinality(version)
    meta = _load_pkg_meta(version)
    pkg2tok = _build_pkg2tok_reverse(version)
    blob = _build_project_blob(per_file, project_path=project_path)

    # Skip packages that already have strong evidence from:
    # 1. debian/control — project's own Build-Depends declaration
    # 2. per-file buckets where software_name contains explicit dep declarations
    _STRONG_SW_RE = re.compile(
        r'find_package|find_library|pkg_check_modules|target_link_libraries'
        r'|AC_CHECK_LIB|AC_CHECK_HEADER|dependency\s*\(|#include',
        re.IGNORECASE,
    )
    # Buckets that are authoritative (skip judging)
    _AUTHORITATIVE_BUCKETS = frozenset({'debian/control'})

    seen: set[str] = set()
    # Collect all pkgs that have strong evidence (skip them)
    strong_pkgs: set[str] = set()
    for bucket, deps in per_file.items():
        if bucket in _AUTHORITATIVE_BUCKETS or (isinstance(deps, list) and bucket in per_file and not bucket.startswith('__')):
            if not isinstance(deps, list):
                continue
            for d in deps:
                if not isinstance(d, dict):
                    continue
                sw = d.get('software_name') or ''
                if _STRONG_SW_RE.search(sw):
                    for pkg in d.get('suite_packages', d.get('packages', [])):
                        strong_pkgs.add(pkg)
        # debian/control is always authoritative
        if bucket in _AUTHORITATIVE_BUCKETS:
            if isinstance(deps, list):
                for d in deps:
                    if isinstance(d, dict):
                        strong_pkgs.update(d.get('suite_packages', d.get('packages', [])))

    candidates: list[dict] = []
    for bucket, deps in per_file.items():
        if not isinstance(deps, list):
            continue
        for d in deps:
            if not isinstance(d, dict):
                continue
            sw = d.get('software_name') or ''
            sources = d.get('sources') or []
            pkgs = d.get('suite_packages') or d.get('packages', [])
            for pkg in pkgs:
                if not isinstance(pkg, str) or pkg in seen:
                    continue
                seen.add(pkg)
                # Skip strong-evidence pkgs — no need to judge them
                if pkg in strong_pkgs:
                    continue
                stem = _pkg_stem(pkg)
                toks = pkg2tok.get(pkg, [])
                stem_ctx = ''
                if len(stem) >= 4:
                    ctx = _find_stem_context(project_path, stem, max_hits=1)
                    if ctx:
                        stem_ctx = ctx[0][:120]
                candidates.append({
                    'pkg': pkg,
                    'bucket': bucket[:40],
                    'software_name': sw[:80],
                    'sources': sources[:3],
                    'library_token': toks[0] if toks else '',
                    'token_cardinality': card.get(pkg, 0),
                    'stem': stem,
                    'stem_in_source': _stem_in_blob(stem, blob),
                    'stem_context': stem_ctx,
                    'essential': meta.get(pkg, {}).get('essential', False),
                })
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _build_pkg2tok_reverse(version: str = DEFAULT_DEBIAN_VERSION) -> dict[str, list[str]]:
    """Reverse-map {pkg: [tokens]} from provider_index.libraries."""
    import json
    from .metadata import provider_index_path
    from collections import defaultdict
    path = provider_index_path(version)
    out: dict[str, list[str]] = defaultdict(list)
    try:
        with open(path) as f:
            idx = json.load(f)
        for tok, pkgs in idx.get('libraries', {}).items():
            if not pkgs:
                continue
            for p in pkgs:
                out[p].append(tok)
    except Exception:
        pass
    return out


def _find_stem_context(project_path: str, stem: str, *, max_hits: int = 1) -> list[str]:
    """Find lines in project source containing stem as word-boundary match."""
    if len(stem) < 3:
        return []
    pattern = re.compile(rf'\b{re.escape(stem)}\b', re.IGNORECASE)
    hits: list[str] = []
    _exts = {'.c', '.h', '.cpp', '.cc', '.hpp', '.cxx', '.cmake', '.am', '.ac',
             '.pro', '.py', '.build', '.mk', '.m4'}
    _names = {'cmakelists.txt', 'meson.build', 'makefile', 'makefile.am', 'configure.ac'}
    _skip = frozenset({'.git', '__pycache__', 'node_modules', 'debian', 'build',
                       'builddir', 'third_party', '3rdparty', 'vendor', 'external',
                       'subprojects', 'out'})
    try:
        for dirpath, dirnames, filenames in os.walk(project_path):
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d.lower() not in _skip]
            for fn in filenames:
                fn_l = fn.lower()
                ext = os.path.splitext(fn_l)[1]
                if fn_l not in _names and ext not in _exts and not fn_l.startswith('readme'):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > 100_000:
                        continue
                    with open(p, errors='replace') as f:
                        for line in f:
                            if pattern.search(line):
                                hits.append(' '.join(line.split())[:140])
                                if len(hits) >= max_hits:
                                    return hits
                except Exception:
                    pass
    except Exception:
        pass
    return hits


def _build_judge_prompt(project_name: str, candidates: list[dict]) -> str:
    """Build the user-message prompt with structured evidence."""
    lines = [f'Project: {project_name}', '', 'Candidates to judge:']
    for i, c in enumerate(candidates, 1):
        src_ctx = f" [context: {c['stem_context']}]" if c.get('stem_context') else ''
        lines.append(
            f"{i}. package: {c['pkg']}\n"
            f"   evidence_source: {c['bucket']}\n"
            f"   matched_text: {c['software_name']!r}\n"
            f"   sources: {c.get('sources', [])}\n"
            f"   library_token: {c.get('library_token', '')!r} (cardinality={c.get('token_cardinality', 0)})\n"
            f"   stem '{c.get('stem', '')}' in project source: "
            f"{'yes' if c.get('stem_in_source') else 'no'}{src_ctx}"
        )
    return '\n'.join(lines)


def _parse_judge_response(text: str) -> dict[str, str]:
    """Parse JSON array response → {pkg: 'KEEP'|'DROP'}."""
    text = re.sub(r'^```\w*\n|\n```$', '', text.strip(), flags=re.MULTILINE)
    out: dict[str, str] = {}
    try:
        m = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if m:
            arr = json.loads(m.group(0))
            for item in arr:
                if isinstance(item, dict):
                    pkg = item.get('package') or item.get('pkg_name') or item.get('pkg') or ''
                    dec = item.get('decision') or item.get('verdict') or ''
                    if pkg and dec:
                        out[pkg] = dec.upper()
    except Exception:
        pass
    if out:
        return out
    # Fallback regex
    for line in text.splitlines():
        m = re.search(
            r'["\']?(?:package|pkg_name|pkg)["\']?\s*[:=]\s*["\']?([a-z0-9][a-z0-9.+-_]*)["\']?.*?(KEEP|DROP)',
            line, re.IGNORECASE,
        )
        if m:
            out[m.group(1)] = m.group(2).upper()
    return out


def apply_llm_judge(
    per_file: dict,
    project_path: str,
    client,
    model: str,
    *,
    max_retries: int = 2,
    timeout: float | None = 60.0,
    version: str = DEFAULT_DEBIAN_VERSION,
    max_candidates: int = 30,
) -> list[dict]:
    """Apply LLM judge to filter false-positive predictions.

    Sends one batched LLM call per project. Modifies per_file in-place by
    removing DROP'd packages from all buckets.

    Returns a log of decisions made.
    """
    project_name = os.path.basename(os.path.normpath(project_path))
    candidates = _collect_evidence(
        per_file, project_path, version=version, max_candidates=max_candidates,
    )
    if not candidates:
        return []

    prompt = _build_judge_prompt(project_name, candidates)

    import time
    response_text = ''
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0.0,
                max_tokens=3000,
                timeout=timeout,
            )
            response_text = resp.choices[0].message.content
            break
        except Exception:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return []  # LLM failed — keep all candidates (no filtering)

    judgments = _parse_judge_response(response_text)
    if not judgments:
        return []  # parse failed — keep all

    # Apply DROP decisions
    log: list[dict] = []
    for bucket, deps in list(per_file.items()):
        if not isinstance(deps, list):
            continue
        new_deps = []
        for d in deps:
            if not isinstance(d, dict):
                new_deps.append(d)
                continue
            pkgs_field = d.get('suite_packages')
            if pkgs_field is None:
                pkgs_field = d.get('packages', [])
            kept = []
            for pkg in pkgs_field:
                dec = judgments.get(pkg)
                if dec == 'DROP':
                    log.append({'package': pkg, 'bucket': bucket, 'decision': 'DROP'})
                    continue
                kept.append(pkg)
            if kept:
                nd = dict(d)
                if 'suite_packages' in nd:
                    nd['suite_packages'] = kept
                if 'packages' in nd:
                    nd['packages'] = [p for p in nd.get('packages', []) if p in kept]
                new_deps.append(nd)
        if new_deps:
            per_file[bucket] = new_deps
        else:
            per_file.pop(bucket, None)

    return log
