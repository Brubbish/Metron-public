"""
Per-artifact summarizers for build document discovery.

Each discovered artifact gets an independent summary before being passed
to the composition agent. debian/control uses rule parsing (fixed format),
all other artifact types use LLM with typed prompts.
"""

import asyncio
import json
import threading
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import openai

from LLM_Config import get_phase_model_config


# ============================================================
# Usage tracking
# ============================================================

_usage_lock = threading.Lock()
_summarizer_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed": 0.0}


def get_summarizer_usage() -> dict:
    with _usage_lock:
        return dict(_summarizer_usage)


def _track(resp, elapsed: float):
    with _usage_lock:
        _summarizer_usage["calls"] += 1
        _summarizer_usage["elapsed"] += elapsed
        usage = getattr(resp, 'usage', None)
        if usage:
            _summarizer_usage["input_tokens"] += getattr(usage, 'prompt_tokens', 0)
            _summarizer_usage["output_tokens"] += getattr(usage, 'completion_tokens', 0)


# ============================================================
# Configuration
# ============================================================

def _get_summarizer_config() -> dict:
    """Use the shared project model configuration for artifact summaries."""
    return get_phase_model_config("artifact_summarizer")


# ============================================================
# Output structure
# ============================================================

@dataclass
class ArtifactSummary:
    kind: str
    path: str
    apt_packages: list = field(default_factory=list)
    apt_package_alternatives: list = field(default_factory=list)
    source_deps: list = field(default_factory=list)
    setup_commands: list = field(default_factory=list)
    build_commands: list = field(default_factory=list)
    notes: str = ''
    status: str = 'ok'
    error: str = ''

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'path': self.path,
            'apt_packages': self.apt_packages,
            'apt_package_alternatives': self.apt_package_alternatives,
            'source_deps': self.source_deps,
            'setup_commands': self.setup_commands,
            'build_commands': self.build_commands,
            'notes': self.notes,
            'status': self.status,
            'error': self.error,
        }


# ============================================================
# debian/control — rule-based (format is fixed)
# ============================================================

_BUILD_DEPENDS_RE = re.compile(
    r'^Build-Depends(?:-Indep)?:\s*(.+?)(?=^\S|\Z)',
    re.MULTILINE | re.DOTALL,
)

_PKG_NAME_RE = re.compile(r'^([a-z0-9][a-z0-9.+\-]+)')


def _parse_debian_control(filepath: str) -> ArtifactSummary:
    try:
        content = open(filepath, encoding='utf-8', errors='ignore').read()
    except OSError as exc:
        return ArtifactSummary(
            kind='debian_control',
            path=filepath,
            status='error',
            error=f'{type(exc).__name__}: {exc}',
        )

    packages = []
    package_alternatives = []
    for m in _BUILD_DEPENDS_RE.finditer(content):
        deps_text = m.group(1)
        for dep_entry in re.split(r',', deps_text):
            dep_entry = dep_entry.strip()
            alternatives = []
            for raw_alt in dep_entry.split('|'):
                pkg_match = _PKG_NAME_RE.match(raw_alt.strip())
                if not pkg_match:
                    continue
                package = pkg_match.group(1)
                if package in ('debhelper', 'debhelper-compat', 'dh-sequence-sphinxdoc'):
                    continue
                if package not in alternatives:
                    alternatives.append(package)
            if len(alternatives) == 1:
                packages.append(alternatives[0])
            elif alternatives:
                package_alternatives.append(alternatives)

    return ArtifactSummary(
        kind='debian_control',
        path=filepath,
        apt_packages=packages,
        apt_package_alternatives=package_alternatives,
    )


# ============================================================
# LLM-based summarizers
# ============================================================

_PROMPTS = {
    'cicd': """Extract build-related information from this CI/CD configuration file (GitHub Actions / GitLab CI / Travis CI).

Output JSON with these fields:
- "apt_packages": list of Debian/Ubuntu packages installed via apt-get/apt (package names only, no versions)
- "source_deps": list of dependencies that are **git repositories** to clone, build, and install (e.g., libraries not available in apt). For each, provide an object with:
  - "name": library/tool name
  - "url": git clone URL (must be a git repository, e.g., https://github.com/org/repo.git)
  - "build_commands": ordered list of shell commands to build and install it
  - "reason": why it must be built from source (e.g., "version in apt is too old", "not in apt")
- "setup_commands": list of environment setup steps that require executing commands but are NOT git clones (e.g., installing rustup, pip packages, SDKs, or running install scripts). For each, provide an object with:
  - "name": tool/name
  - "commands": ordered list of shell commands to execute
  - "reason": why it is needed
- "build_commands": list of shell commands used to build the main project (configure, cmake, make, etc.)
- "notes": any special build requirements or constraints mentioned

Key distinction: if the URL is a git repository that needs to be cloned and compiled, put it in "source_deps". If it is a script to run (curl|sh), a package to install via pip/npm/cargo, or an SDK installer, put it in "setup_commands".

Only extract what is explicitly present. Do not infer or guess.
Output ONLY valid JSON, no markdown fences, no explanation.""",

    'dockerfile': """Extract build-related information from this Dockerfile.

Output JSON with these fields:
- "apt_packages": list of packages installed via apt-get/apt (package names only)
- "source_deps": list of dependencies that are **git repositories** cloned and built from source inside the container. For each, provide an object with:
  - "name": library/tool name
  - "url": git clone URL (must be a git repository)
  - "build_commands": ordered list of shell commands to build and install it
  - "reason": why it must be built from source
- "setup_commands": list of environment setup steps that require executing commands but are NOT git clones (e.g., installing rustup, pip packages, SDKs, or running install scripts). For each, provide an object with:
  - "name": tool/name
  - "commands": ordered list of shell commands to execute
  - "reason": why it is needed
- "build_commands": list of commands that build the main project (not dependency installation)
- "notes": any special build requirements (e.g., specific base image, multi-stage)

Key distinction: if the URL is a git repository that needs to be cloned and compiled, put it in "source_deps". If it is a script to run (curl|sh), a package to install via pip/npm/cargo, or an SDK installer, put it in "setup_commands".

Only extract what is explicitly present. Do not infer or guess.
Output ONLY valid JSON, no markdown fences, no explanation.""",

    'debian_rules': """Extract build commands from this debian/rules file.

Output JSON with these fields:
- "apt_packages": [] (leave empty, deps come from debian/control)
- "source_deps": [] (leave empty)
- "setup_commands": [] (leave empty)
- "build_commands": list of actual build/configure commands found in override_dh_auto_configure, override_dh_auto_build, or custom targets
- "notes": any special flags, environment variables, or disabled features

Only extract what is explicitly present. Do not infer or guess.
Output ONLY valid JSON, no markdown fences, no explanation.""",

    'root_doc': """Extract build-related information from this project documentation file (README/INSTALL/BUILDING).

Focus ONLY on building on Linux/Debian x64. Ignore Windows, macOS, and other platform instructions.

Output JSON with these fields:
- "apt_packages": list of system packages mentioned for installation via apt/apt-get (package names only)
- "source_deps": list of dependencies that are **git repositories** to clone, build, and install. For each, provide an object with:
  - "name": library/tool name
  - "url": git clone URL (must be a git repository, e.g., https://github.com/org/repo.git)
  - "build_commands": ordered list of shell commands to build and install it
  - "reason": why it must be built from source
- "setup_commands": list of environment setup steps that require executing commands but are NOT git clones (e.g., installing rustup, pip packages, SDKs, or running install scripts). For each, provide an object with:
  - "name": tool/name
  - "commands": ordered list of shell commands to execute
  - "reason": why it is needed
- "build_commands": ordered list of shell commands to build the main project
- "notes": any special requirements, optional features, or important constraints

Key distinction: if the URL is a git repository that needs to be cloned and compiled, put it in "source_deps". If it is a script to run (curl|sh), a package to install via pip/npm/cargo, or an SDK installer, put it in "setup_commands".

Only extract what is explicitly stated. Do not infer or guess package names.
Output ONLY valid JSON, no markdown fences, no explanation.""",
}


def _llm_summarize(filepath: str, kind: str, client: openai.OpenAI, model: str) -> ArtifactSummary:
    prompt_key = kind if kind in _PROMPTS else 'root_doc'
    system_prompt = _PROMPTS[prompt_key]

    try:
        content = open(filepath, encoding='utf-8', errors='ignore').read()
    except OSError as exc:
        return ArtifactSummary(
            kind=kind,
            path=filepath,
            status='error',
            error=f'{type(exc).__name__}: {exc}',
        )

    try:
        import time as _time
        _t0 = _time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': content},
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=60,
        )
        _track(response, _time.perf_counter() - _t0)
        raw = response.choices[0].message.content.strip()
        parsed = _parse_json_response(raw)
    except Exception as e:
        return ArtifactSummary(
            kind=kind,
            path=filepath,
            notes=f'LLM error: {e}',
            status='error',
            error=f'{type(e).__name__}: {e}',
        )

    if parsed is None:
        return ArtifactSummary(
            kind=kind,
            path=filepath,
            notes='invalid_summary',
            status='invalid_summary',
            error='model response was not a valid JSON object',
        )

    return _summary_from_parsed(kind, filepath, parsed)


async def _llm_summarize_async(filepath: str, kind: str, client: openai.AsyncOpenAI, model: str) -> ArtifactSummary:
    prompt_key = kind if kind in _PROMPTS else 'root_doc'
    system_prompt = _PROMPTS[prompt_key]

    try:
        content = open(filepath, encoding='utf-8', errors='ignore').read()
    except OSError as exc:
        return ArtifactSummary(
            kind=kind,
            path=filepath,
            status='error',
            error=f'{type(exc).__name__}: {exc}',
        )

    try:
        import time as _time
        _t0 = _time.perf_counter()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': content},
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=60,
        )
        _track(response, _time.perf_counter() - _t0)
        raw = response.choices[0].message.content.strip()
        parsed = _parse_json_response(raw)
    except Exception as e:
        return ArtifactSummary(
            kind=kind,
            path=filepath,
            notes=f'LLM error: {e}',
            status='error',
            error=f'{type(e).__name__}: {e}',
        )

    if parsed is None:
        return ArtifactSummary(
            kind=kind,
            path=filepath,
            notes='invalid_summary',
            status='invalid_summary',
            error='model response was not a valid JSON object',
        )

    return _summary_from_parsed(kind, filepath, parsed)


def _summary_from_parsed(kind: str, filepath: str, parsed: dict) -> ArtifactSummary:
    """Keep valid fields while surfacing malformed producer output."""
    errors = []

    def list_field(name: str, item_type) -> list:
        value = parsed.get(name, [])
        if not isinstance(value, list):
            errors.append(f'{name} is not a list')
            return []
        valid = [item for item in value if isinstance(item, item_type)]
        if len(valid) != len(value):
            errors.append(f'{name} contains invalid entries')
        return valid

    apt_packages = [item.strip() for item in list_field('apt_packages', str) if item.strip()]
    source_deps = list_field('source_deps', dict)
    setup_commands = list_field('setup_commands', (dict, str))
    build_commands = [item.strip() for item in list_field('build_commands', str) if item.strip()]
    notes = parsed.get('notes', '')
    if not isinstance(notes, str):
        errors.append('notes is not a string')
        notes = ''
    has_payload = bool(apt_packages or source_deps or setup_commands or build_commands or notes)
    return ArtifactSummary(
        kind=kind,
        path=filepath,
        apt_packages=apt_packages,
        source_deps=source_deps,
        setup_commands=setup_commands,
        build_commands=build_commands,
        notes=notes,
        status='partial' if errors and has_payload else 'invalid_summary' if errors else 'ok',
        error='; '.join(errors),
    )


def _parse_json_response(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    # Decode the first complete object. Some providers repeat an otherwise
    # valid JSON object after it; a greedy ``{.*}`` extraction turns that into
    # invalid JSON and silently loses the whole artifact summary.
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


# ============================================================
# Main dispatcher
# ============================================================

_client_cache: Optional[openai.OpenAI] = None
_async_client_cache: Optional[openai.AsyncOpenAI] = None


def _get_client() -> openai.OpenAI:
    global _client_cache
    if _client_cache is None:
        config = _get_summarizer_config()
        _client_cache = openai.OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
    return _client_cache


def _get_async_client() -> openai.AsyncOpenAI:
    global _async_client_cache
    if _async_client_cache is None:
        config = _get_summarizer_config()
        _async_client_cache = openai.AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )
    return _async_client_cache


def summarize_artifact(
    filepath: str,
    kind: str,
    client: Optional[openai.OpenAI] = None,
    model: Optional[str] = None,
) -> ArtifactSummary:
    """Summarize a single build artifact (sync)."""
    if kind == 'debian_control':
        return _parse_debian_control(filepath)

    client = client or _get_client()
    model = model or _get_summarizer_config()["model"]
    return _llm_summarize(filepath, kind, client, model)


async def summarize_artifact_async(
    filepath: str,
    kind: str,
    model: Optional[str] = None,
) -> ArtifactSummary:
    """Summarize a single build artifact (async)."""
    if kind == 'debian_control':
        return _parse_debian_control(filepath)

    client = _get_async_client()
    model = model or _get_summarizer_config()["model"]
    return await _llm_summarize_async(filepath, kind, client, model)


def summarize_all_artifacts(
    project_root: str,
    artifacts: list,
    client: Optional[openai.OpenAI] = None,
    model: Optional[str] = None,
) -> list[ArtifactSummary]:
    """Summarize all discovered artifacts (sync, serial)."""
    results = []
    for artifact in artifacts:
        filepath = os.path.join(project_root, artifact.path)
        summary = summarize_artifact(filepath, artifact.kind, client, model)
        summary.path = artifact.path
        results.append(summary)
    return results


async def summarize_all_artifacts_async(
    project_root: str,
    artifacts: list,
    model: Optional[str] = None,
    max_concurrent: int = 3,
    semaphore: asyncio.Semaphore = None,
) -> list[ArtifactSummary]:
    """Summarize all discovered artifacts (async, concurrency-limited LLM calls)."""
    sem = semaphore or asyncio.Semaphore(max_concurrent)

    async def _limited(filepath, kind, mdl):
        async with sem:
            return await summarize_artifact_async(filepath, kind, mdl)

    tasks = []
    for artifact in artifacts:
        filepath = os.path.join(project_root, artifact.path)
        tasks.append(_limited(filepath, artifact.kind, model))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = []
    for artifact, result in zip(artifacts, results):
        if isinstance(result, BaseException):
            summary = ArtifactSummary(
                kind=artifact.kind,
                path=artifact.path,
                status='error',
                error=f'{type(result).__name__}: {result}',
            )
        else:
            summary = result
            summary.path = artifact.path
        summaries.append(summary)
    return summaries
