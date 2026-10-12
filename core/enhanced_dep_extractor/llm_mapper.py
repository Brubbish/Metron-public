"""Per-file LLM mapping: send rule-matched context to LLM for name→package mapping."""

import json
import re
import sys
import textwrap
import threading
import time
from typing import Any

import openai

from .openrouter_compat import (
    get_openrouter_extra_body,
    get_retry_delay,
    is_retryable_provider_error,
)
from .rules import RuleMatch

CHUNK_SIZE = 5000
SKIP_MARKER = '[skip]'

# Module-level usage tracker for LLM calls in this subprocess
_usage_lock = threading.Lock()
_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed": 0.0}


def get_llm_usage() -> dict:
    with _usage_lock:
        return dict(_usage)


def _track_usage(resp, elapsed: float):
    with _usage_lock:
        _usage["calls"] += 1
        _usage["elapsed"] += elapsed
        usage = getattr(resp, 'usage', None)
        if usage:
            _usage["input_tokens"] += getattr(usage, 'prompt_tokens', 0)
            _usage["output_tokens"] += getattr(usage, 'completion_tokens', 0)
CONTEXT_LINE_RE = re.compile(r'^(\d+): ?(.*)$')

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert in Linux build systems and Debian packaging.

    Input contains raw snippets from configuration files. Infer Debian
    Build-Depends for external dependencies visible in the snippets. Ignore
    local targets, relative paths, generated in-tree artifacts, and bundled
    fallback code unless the snippet also identifies an external tool, package,
    library, or provider.

    Map evidence to build-time providers: headers, libraries, linker flags,
    pkg-config modules, CMake/Meson/qmake packages or imported targets,
    config scripts, macros/modules, executable tools, generators, schemas, and
    stylesheets should resolve to packages that provide the compile-time or
    build-time artifact. Prefer concrete -dev/SDK/tool packages over runtime
    shared-library ABI packages or virtual names.

    Preserve components and variants. For find_package/find_dependency,
    pkg_check_modules(), Meson dependency(), target/imported targets, and QT +=
    lines, map each external component when it has a separate Debian provider.
    If a CMake call has COMPONENTS or OPTIONAL_COMPONENTS, enumerate every
    component instead of returning only the package namespace or the first
    component. Imported targets such as Boost::program_options, Qt5::Svg,
    KF6::KIO, GTest::gtest, and protobuf::libprotobuf are component evidence.
    For Boost, output component packages only for linkable components or
    *_LIBS/*_LDFLAGS/AX_BOOST_* component checks; header-only checks usually map
    to libboost-dev. Do not collapse to only an umbrella package. If an install
    hint or error lists multiple plausible Debian providers, include all
    plausible ones.

    Link flags in real link commands or link variables (LIBS, LDLIBS, LDADD,
    LIBADD, LDFLAGS, etc.) are dependency evidence. Ignore standard toolchain
    libs such as -lm, -lc, -ldl, -lrt, -lpthread, -lstdc++, and -lgcc unless the
    snippet explicitly identifies an external dependency.

    Build tools invoked or checked in the snippets are Build-Depends; output
    the package shipping the executable or helper macro/module. Documentation,
    tests, language packaging metadata, and ecosystem config files count when
    the snippets declare them (Python setup/pyproject/tox/docs, Ruby extconf or
    gemspec, Perl Makefile.PL, OCaml dune/opam, Java Maven/Gradle/SBT/Ant).

    Important special cases: PGXS/USE_PGXS/pg_config --pgxs/PGXN PostgreSQL
    extension evidence implies postgresql-server-dev-all and postgresql-all;
    libpq or pq-fe.h maps to libpq-dev. Sphinx extensions/themes, Qt tools,
    Boost components, GTest/Catch/doctest, DocBook/XML/XSL/manpage tools, gettext
    and intltool are separate providers when Debian ships them and they are
    external/system requirements, not bundled fallback code.

    Additional high-recall rules: pkg-config module names inside variables or
    string lists (deps, pkg_modules, pkg-module-spec, *_PKGS, *_MODULES,
    Requires, Libs) are dependency evidence. Fortran enablement implies
    gfortran; lex/flex scanner rules or .l sources imply flex; yacc/bison
    grammar rules or .y sources imply bison. Doxygen HAVE_DOT=YES, DOT_PATH, or
    checks for dot imply graphviz. Translation/completion/schema/codegen tools
    count when invoked or installed: lrelease/lupdate, bash-completion install
    dirs, g-ir-scanner/g-ir-compiler/gnome.generate_gir, glib-compile-*,
    gdbus-codegen, and protoc plugins. pkg-config --variable queries for
    systemd unit directories are systemd provider evidence.

    Be recall-oriented for plausible external dependencies visible in the
    snippets, including optional build/test/doc deps unless clearly disabled,
    non-Linux-only, or only used for bundled fallback. For compile/link/config
    evidence, output development providers rather than runtime ABI packages
    even if a runtime package name appears in a dependency string. Output only
    Debian package names that very likely exist.

    For software_name, use the original dependency declaration verbatim from
    the snippet (e.g. "find_package(OpenSSL REQUIRED)", "dependency('glib-2.0')",
    "-lssl -lcrypto", "pkg_check_modules(... hunspell)"). Do NOT use the
    project name or a free-form description. Each JSON entry should correspond
    to one dependency declaration in the source.

    Return only JSON, with no explanation:
    [{"software_name":"original code evidence","packages":["pkg1","pkg2"]}]
    If no dependencies are found, return [].""")


TOOLCHAIN_PACKAGES: frozenset[str] = frozenset()  # deprecated; kept for import compatibility


def _is_toolchain_package(pkg: str) -> bool:
    return False

def normalize_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    normalized = {item.strip() for item in values if isinstance(item, str) and item.strip()}
    return sorted(normalized)


def merge_dependencies(entries: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for entry in entries:
        software_name = ' '.join(entry.get('software_name', '').split())
        packages = normalize_string_list(entry.get('packages', []))
        sources = normalize_string_list(entry.get('sources', []))
        if not software_name or not packages:
            continue

        key = software_name.strip().lower()
        if key not in merged:
            merged[key] = {
                'software_name': software_name,
                'packages': set(packages),
                'sources': set(sources),
            }
        else:
            merged[key]['packages'].update(packages)
            merged[key]['sources'].update(sources)

    return [
        {
            'software_name': v['software_name'],
            'packages': sorted(v['packages']),
            **({'sources': sorted(v['sources'])} if v['sources'] else {}),
        }
        for v in sorted(merged.values(), key=lambda x: x['software_name'].lower())
    ]


def parse_json_like_text(text: str) -> Any:
    decoder = json.JSONDecoder()
    stripped = text.strip()

    for match in re.finditer(r'```(?:json)?\s*(.*?)```', stripped, re.IGNORECASE | re.DOTALL):
        block = match.group(1).strip()
        if block:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for idx, ch in enumerate(stripped):
        if ch not in '[{':
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[idx:])
            return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError('No valid JSON found in model output')


def parse_dependency_response(data: Any) -> list[dict]:
    if isinstance(data, list):
        raw_deps = data
    elif isinstance(data, dict) and 'dependencies' in data:
        raw_deps = data['dependencies']
    elif isinstance(data, dict) and 'packages' in data:
        raw_deps = [
            {'software_name': pkg, 'packages': [pkg]}
            for pkg in normalize_string_list(data['packages'])
        ]
    else:
        raise ValueError('LLM response must be a JSON array or include a "dependencies" key')

    if not isinstance(raw_deps, list):
        raise ValueError('"dependencies" must be a list')

    normalized = []
    for item in raw_deps:
        if not isinstance(item, dict):
            continue
        software_name = item.get('software_name') or item.get('dependency_name') or item.get('name')
        packages = normalize_string_list(
            item.get('packages', item.get('packages', item.get('package_names', [])))
        )
        if isinstance(software_name, str):
            normalized.append({'software_name': software_name.strip(), 'packages': packages})
        elif isinstance(software_name, list):
            names = normalize_string_list(software_name)
            if len(names) == 1:
                normalized.append({'software_name': names[0], 'packages': packages})
            elif len(names) == len(packages):
                for name, pkg in zip(names, packages):
                    normalized.append({'software_name': name, 'packages': [pkg]})
            elif names and packages:
                normalized.append({'software_name': ' / '.join(names), 'packages': packages})

    return merge_dependencies(normalized)


def _message_header(
    file_path: str,
    build_systems: set[str],
    project_name: str | None = None,
    repo_name: str | None = None,
    chunk_idx: int | None = None,
    resolved_packages: list[str] | None = None,
    header_anchor_packages: list[str] | None = None,
) -> str:
    parts = [f'File: {file_path}']
    if chunk_idx is not None:
        parts[0] += f' (chunk {chunk_idx + 1})'
    if project_name:
        parts.append(f'Project name: {project_name}')
    if repo_name:
        parts.append(f'Repository directory: {repo_name}')
    if build_systems:
        parts.append(f'Build systems detected: {", ".join(sorted(build_systems))}')
    if resolved_packages:
        parts.append(f'Already resolved (do NOT include): {", ".join(resolved_packages[:40])}')
    if header_anchor_packages:
        parts.append(
            'Header-derived anchor packages (prefer to stay consistent with these): '
            + ', '.join(header_anchor_packages[:40])
        )
    parts.append('')
    return '\n'.join(parts)


def _format_direct_dependency(match: RuleMatch) -> dict | None:
    pkgs = normalize_string_list(match.packages)
    if not pkgs:
        return None
    name = ' '.join((match.extracted_name or match.matched_text or match.rule_type).split())
    return {'software_name': name, 'packages': pkgs}


def _trim_context(context: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ''
    if len(context) <= max_chars:
        return context

    kept = []
    used = 0
    for line in context.splitlines():
        line_len = len(line) + 1
        if kept and used + line_len > max_chars:
            break
        if not kept and line_len > max_chars:
            return line[:max_chars]
        kept.append(line)
        used += line_len
    return '\n'.join(kept)


def _format_match_block(match: RuleMatch, max_chars: int | None = None) -> str:
    parts = [f'[{match.rule_type}] Line {match.line_num}']
    if getattr(match, 'condition', 'unknown') != 'unknown':
        parts[0] += f' condition={match.condition}'
    if match.packages:
        parts[0] += f' direct_packages={",".join(match.packages)}'
    parts[0] += f': {match.matched_text}'
    summary = _summarize_evidence(match)
    if summary:
        parts.append(f'Evidence summary: {summary}')
    elif match.context:
        parts.append('Context (with line numbers):')
        parts.append(match.context)
    block = '\n'.join(parts)

    if max_chars is None or len(block) <= max_chars or not match.context:
        return block

    base = '\n'.join(parts[:2])
    context_budget = max_chars - len(base) - 1
    trimmed_context = _trim_context(match.context, context_budget)
    if trimmed_context:
        return f'{base}\n{trimmed_context}'
    return parts[0][:max_chars]


def _summarize_evidence(match: RuleMatch) -> str:
    text = ' '.join(f'{match.extracted_name} {match.matched_text}'.split())
    if not text:
        return ''
    tokens = []
    for pattern in (
        r'find_package\s*\(([^)]*)\)',
        r'find_dependency\s*\(([^)]*)\)',
        r'pkg_(?:check_modules|search_module)\s*\(([^)]*)\)',
        r'dependency\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'find_program\s*\(([^)]*)\)',
        r'(?:^|[\s="])-l([A-Za-z0-9_+.-]+)',
        r'QT\s*[+*]?=\s*([^;\n]+)',
        r'PKGCONFIG\s*[+*]?=\s*([^;\n]+)',
        r'(?:requires|depends|dependencies)\s*=\s*([^;\n]+)',
    ):
        for item in re.findall(pattern, text, re.IGNORECASE):
            if isinstance(item, tuple):
                item = ' '.join(part for part in item if part)
            item = ' '.join(str(item).split())
            if item and item not in tokens:
                tokens.append(item[:180])
    if not tokens:
        tokens = [text[:220]]
    cond = getattr(match, 'condition', 'unknown')
    prefix = f'condition={cond}; ' if cond != 'unknown' else ''
    return prefix + '; '.join(tokens[:4])


def _context_lines_from_match(match: RuleMatch) -> list[tuple[int, str]]:
    """Parse scanner context into numbered raw source lines."""
    parsed: list[tuple[int, str]] = []
    for line in (match.context or '').splitlines():
        if line == '...':
            continue
        m = CONTEXT_LINE_RE.match(line)
        if not m:
            continue
        parsed.append((int(m.group(1)), m.group(2)))
    return parsed


def _raw_context_blocks(matches: list[RuleMatch], excluded_lines: set[int] | None = None) -> list[str]:
    """Merge nearby match windows and return raw snippets without metadata."""
    excluded_lines = excluded_lines or set()
    by_line: dict[int, str] = {}
    fallbacks: list[str] = []

    for match in matches:
        parsed = _context_lines_from_match(match)
        if parsed:
            for line_no, text in parsed:
                if line_no in excluded_lines:
                    continue
                by_line.setdefault(line_no, text)
        else:
            text = (match.matched_text or match.extracted_name or '').strip()
            if text:
                fallbacks.append(text)

    blocks: list[str] = []
    current: list[str] = []
    prev_line: int | None = None
    for line_no in sorted(by_line):
        if prev_line is not None and line_no != prev_line + 1:
            if current:
                blocks.append('\n'.join(current).rstrip())
            current = []
        current.append(by_line[line_no])
        prev_line = line_no
    if current:
        blocks.append('\n'.join(current).rstrip())

    blocks.extend(fallbacks)
    return [block for block in blocks if block.strip()]


def _split_raw_block(block: str, max_chars: int) -> list[str]:
    """Split a raw snippet on line boundaries to fit the prompt budget."""
    if len(block) <= max_chars:
        return [block]

    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for line in block.splitlines():
        line_len = len(line) + 1
        if current and used + line_len > max_chars:
            pieces.append('\n'.join(current).rstrip())
            current = []
            used = 0
        if line_len > max_chars:
            if current:
                pieces.append('\n'.join(current).rstrip())
                current = []
                used = 0
            pieces.append(line[:max_chars].rstrip())
            continue
        current.append(line)
        used += line_len
    if current:
        pieces.append('\n'.join(current).rstrip())
    return [piece for piece in pieces if piece.strip()]


def _format_raw_block(block: str) -> str:
    return f'{SKIP_MARKER}\n{block.rstrip()}\n{SKIP_MARKER}'


def _build_user_messages(
    file_path: str,
    matches: list[RuleMatch],
    build_systems: set[str],
    project_name: str | None = None,
    repo_name: str | None = None,
    max_chars: int = CHUNK_SIZE,
    resolved_packages: list[str] | None = None,
    header_anchor_packages: list[str] | None = None,
) -> list[str]:
    """Build LLM messages from merged raw context windows up to max_chars."""
    seen_lines = set()
    llm_matches = []
    excluded_context_lines: set[int] = set()
    for m in matches:
        key = (m.rule_type, m.line_num, m.matched_text)
        if key in seen_lines:
            continue
        seen_lines.add(key)
        if file_path.startswith('__direct__/'):
            continue
        if m.rule_type.endswith(':dependency-list:high'):
            for line_no in range(m.line_num, m.line_num + max(1, m.matched_text.count('\n') + 1)):
                excluded_context_lines.add(line_no)
            continue
        # Directly handled generic toolchain/doc matches still contribute to
        # output, but sending them again to the LLM creates cost and FP noise.
        if (
            m.rule_type in {
                'cmake:find_doc_tool', 'cmake:doc_custom_command',
                'cmake:docbook-xsl', 'cmake:docbook-xml',
                'doxygen:have-dot', 'doxygen:dot-path',
                'make:doc-tool-var', 'make:docbook-xsl',
                'make:docbook-xml', 'make:tool-invocation',
                'make:pgxs', 'make:bison', 'make:flex', 'make:texinfo',
                'make:lexer-parser-rule', 'meson:doc_tool',
                'meson:docbook-xsl', 'meson:docbook-xml',
                'qmake:tool',
            }
            and all(_is_toolchain_package(pkg) for pkg in m.packages)
        ):
            continue
        llm_matches.append(m)

    raw_blocks = _raw_context_blocks(llm_matches, excluded_context_lines)
    if not raw_blocks:
        return []

    messages = []
    header = _message_header(
        file_path,
        build_systems,
        project_name,
        repo_name,
        0,
        resolved_packages,
        header_anchor_packages,
    )
    current_parts = [header]
    current_len = len(header)
    has_block = False

    for raw_block in raw_blocks:
        chunk_budget = max(200, max_chars - len(header) - len(SKIP_MARKER) * 2 - 4)
        for piece in _split_raw_block(raw_block, chunk_budget):
            block = _format_raw_block(piece)
            separator = '\n\n' if has_block else ''
            projected_len = current_len + len(separator) + len(block)

            if has_block and projected_len > max_chars:
                messages.append(''.join(current_parts))
                chunk_header = _message_header(
                    file_path,
                    build_systems,
                    project_name,
                    repo_name,
                    len(messages),
                    resolved_packages,
                    header_anchor_packages,
                )
                current_parts = [chunk_header]
                current_len = len(chunk_header)
                has_block = False
                separator = ''

            if separator:
                current_parts.append(separator)
                current_len += len(separator)
            current_parts.append(block)
            current_len += len(block)
            has_block = True

    if has_block:
        messages.append(''.join(current_parts))

    return messages


def map_file_deps(
    client: openai.OpenAI,
    model: str,
    file_path: str,
    matches: list[RuleMatch],
    build_systems: set[str],
    project_name: str | None = None,
    repo_name: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    retry_backoff: float = 2.0,
    timeout: float | None = None,
    resolved_packages: list[str] | None = None,
    header_anchor_packages: list[str] | None = None,
) -> list[dict]:
    """Send LLM calls over size-bounded groups of rule matches."""
    direct_deps = [
        dep for match in matches
        if (dep := _format_direct_dependency(match)) is not None
    ]
    if file_path.startswith('__direct__/'):
        return merge_dependencies(direct_deps)

    user_messages = _build_user_messages(
        file_path, matches, build_systems, project_name, repo_name,
        resolved_packages=resolved_packages,
        header_anchor_packages=header_anchor_packages,
    )

    all_deps = list(direct_deps)
    for chunk_idx, user_msg in enumerate(user_messages):
        deps = _call_llm(client, model, user_msg, file_path, chunk_idx,
                         max_retries, retry_delay, retry_backoff, timeout)
        all_deps.extend(deps)

    return merge_dependencies(all_deps)


def _call_llm(
    client: openai.OpenAI,
    model: str,
    user_msg: str,
    file_path: str,
    chunk_idx: int,
    max_retries: int,
    retry_delay: float,
    retry_backoff: float,
    timeout: float | None,
) -> list[dict]:
    attempts = max_retries + 1
    extra_body = get_openrouter_extra_body(getattr(client, "base_url", None))
    for attempt in range(1, attempts + 1):
        try:
            t0 = time.perf_counter()
            request_kwargs = dict(
                model=model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_msg},
                ],
                temperature=0,
                max_tokens=1024,
                timeout=timeout,
            )
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**request_kwargs)
            _track_usage(resp, time.perf_counter() - t0)
            if not getattr(resp, "choices", None):
                raise RuntimeError("OpenRouter returned no choices in chat completion response")
            message = resp.choices[0].message
            raw = (message.content or "").strip()
            if not raw:
                raise RuntimeError("OpenRouter returned empty response content")
            data = parse_json_like_text(raw)
            return parse_dependency_response(data)
        except Exception as e:
            if attempt >= attempts:
                print(f'  [warn] LLM error on {file_path} chunk {chunk_idx}: {e}', file=sys.stderr)
                return []
            sleep_s = get_retry_delay(attempt, base_delay=retry_delay, backoff=retry_backoff) if is_retryable_provider_error(e) else retry_delay * (retry_backoff ** (attempt - 1))
            print(
                f'  [warn] LLM error on {file_path} chunk {chunk_idx} '
                f'(attempt {attempt}/{attempts}): {e}; retrying in {sleep_s:.1f}s',
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    return []
