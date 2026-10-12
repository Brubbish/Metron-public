"""File-level artifact extraction for build/config/source evidence.

This module extracts generic artifacts only. Debian package resolution remains
centralized in artifacts.packages_from_artifacts().
"""

from __future__ import annotations

import re

from .artifacts import (
    Artifact,
    _boost_autoconf_artifacts,
    _cmake_artifacts,
    _dependency_list_artifacts,
    _feature_artifacts,
    _header_artifacts,
    _is_standard_header,
    _library_artifacts,
    _looks_like_pkgconfig_token,
    _normal_pkgconfig_token,
    _pkgconfig_artifacts,
    _program_artifacts,
    _provider_index,
    _qmake_qt_artifacts,
)


_M4_CALL_RE = re.compile(r'(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\(')
_M4_DEFINITION_MACROS = frozenset({
    'AC_DEFUN',
    'AC_DEFUN_ONCE',
    'AU_DEFUN',
    'm4_define',
    'm4_defun',
    'm4_defun_once',
})
_TOKEN_RE = re.compile(r'-l[A-Za-z0-9_+.-]+|[A-Za-z0-9_+./-]+\.h(?:pp|xx|h)?\b|[A-Za-z0-9_+.-]+')
_SOURCE_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
_QML_IMPORT_RE = re.compile(r'^\s*import\s+([A-Za-z][A-Za-z0-9_.]*)\b', re.MULTILINE)
_RUNTIME_ABI_PACKAGE_RE = re.compile(r'^(?:lib.*|[a-z0-9.+-]*lib[a-z0-9.+-]*)\d(?:[a-z]*\d*)?(?:t64)?$')
_COMMENT_RE = re.compile(r'^\s*(?:#|//|/\*+|\*|dnl\b)(.*)$', re.IGNORECASE)
_ASSEMBLY_COMMENT_RE = re.compile(r'^\s*;(.*)$')
_ASSEMBLY_COMMENT_EXTENSIONS = ('.asm', '.s', '.nasm')
_COMMENT_EVIDENCE_RE = re.compile(
    r'\b(?:require[sd]?|need[sd]?|depend(?:s|encies)?|prereq(?:uisite)?s?|'
    r'build[- ]?(?:requires?|deps?)|link(?:s|ed)? with|uses?)\b',
    re.IGNORECASE,
)
_SCRIPT_SOURCE_GENERATOR_TOOLS = frozenset({
    'bison',
    'flatc',
    'flex',
    'gdbus-codegen',
    'glib-compile-resources',
    'glib-compile-schemas',
    'gperf',
    'lrelease',
    'lupdate',
    'moc',
    'protoc',
    'ragel',
    'rcc',
    'swig',
    'uic',
    'valac',
    'vapigen',
    'wayland-scanner',
    'xxd',
})
_SHELL_COMMAND_RE = re.compile(r'^\s*(?:time\s+|exec\s+|command\s+)*([A-Za-z0-9_+.-]+)\b')
_SHELL_SOURCE_RE = re.compile(r'^\s*(?:source|\.)\s+([A-Za-z0-9_+.-]+)\b')
_GENERATED_BY_TOOL_RE = re.compile(
    r'\bgenerated\s+by(?:\s+a\s+\w+\s+version\s+of)?\s+([A-Za-z0-9_+.-]+)\b',
    re.IGNORECASE,
)
_MAKE_RECIPE_COMMAND_RE = re.compile(r'^\s*\t\s*[@+-]?\s*([^\s;&|]+)', re.MULTILINE)
_TOOL_ASSIGNMENT_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_+-]*)\s*[:?+]?=\s*([^\n#]+)',
    re.MULTILINE,
)
_SHELL_DEFAULT_TOOL_ASSIGNMENT_RE = re.compile(
    r'\$\{\s*([A-Za-z_][A-Za-z0-9_+-]*)\s*:?=\s*([A-Za-z0-9_+./-]+)',
)
_SHELL_ASSIGNMENT_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\n#]+))',
    re.MULTILINE,
)
_COMMAND_LOOKUP_RE = re.compile(
    r'\b(?:command\s+-v|which|type\s+-P)\s+([A-Za-z0-9_+./-]+)\b',
)
_BUILD_FILE_EXTRA_TOOLS = frozenset({
    'antiword',
    'asciidoc',
    'asciidoctor',
    'help2man',
    'makeinfo',
    'pandoc',
})
_SCRIPT_IGNORED_TOOLS = _BUILD_FILE_EXTRA_TOOLS
_PKGCONFIG_WRAPPER_CALL_RE = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]*(?:pkg|pkgconfig|pkg_config|pkg_configure)[A-Za-z0-9_]*)\s*(?:\((.*?)\)|([^\n;#]+))',
    re.IGNORECASE | re.DOTALL,
)
# Shell builtins and stopwords previously maintained as blocklists; the downstream
# _provider_allowed suffix filter (-dev/-tools/-bin/-utils) now handles the noise
# that these sets were designed to catch. Packages like coreutils, grep, sed that
# result from shell-builtin tokens are rejected because they don't end in -dev.
_SHELL_BUILTINS: frozenset[str] = frozenset()
_STOPWORDS: frozenset[str] = frozenset()
_TOOL_VAR_NAME_RE = re.compile(
    r'(?:(?:^|_)(?:TOOL|TOOLS|PROG|PROGRAM|EXECUTABLE|CMD|COMMAND|COMPILER|GEN|'
    r'GENERATOR|CC|CXX|CPP|FC|F77|F90|OBJC|OBJCXX|LD|LEX|YACC|AS)(?:_|$)|'
    r'(?:TOOL|TOOLS|PROG|PROGRAM|EXECUTABLE|CMD|COMMAND|COMPILER|GENERATOR)$)',
    re.IGNORECASE,
)
_COMPILER_VAR_NAME_RE = re.compile(
    r'^(?:CC|CXX|CPP|FC|F77|F90|OBJC|OBJCXX)$|(?:^|_)(?:COMPILER)(?:_|$)',
    re.IGNORECASE,
)
_DEFAULT_COMPILER_COMMAND_RE = re.compile(
    r'^(?:cc|c\+\+|cpp|gcc|g\+\+|clang|clang\+\+)(?:-\d+(?:\.\d+)*)?$|'
    r'^[A-Za-z0-9_+.-]+-(?:gcc|g\+\+|clang|clang\+\+)(?:-\d+(?:\.\d+)*)?$',
    re.IGNORECASE,
)

# (stopwords removed — downstream _provider_allowed suffix filter handles noise)


def extract_file_artifacts(
    file_path: str,
    content: str,
    build_system: str,
    build_variables: dict[str, str] | None = None,
) -> list[Artifact]:
    """Extract dependency artifacts from an entire build/source file."""
    artifacts: list[Artifact] = []
    if build_system == 'source':
        artifacts.extend(_source_include_artifacts(content))
        if file_path.lower().endswith('.qml'):
            artifacts.extend(_qml_import_artifacts(content))
    elif build_system == 'script':
        artifacts.extend(_script_dependency_artifacts(content))
        artifacts.extend(_generic_pkgconfig_wrapper_artifacts(content))
        artifacts.extend(_probe_assignment_artifacts(content))
    elif build_system == 'package_list':
        artifacts.extend(_package_list_artifacts(content))
    else:
        artifacts.extend(_build_file_artifacts(file_path, content, build_system, build_variables))
        if build_system in {'autoconf', 'automake', 'configure_script', 'bootstrap'} or file_path.endswith('.m4'):
            artifacts.extend(_m4_argument_artifacts(content))
    artifacts.extend(_comment_artifacts(file_path, content))
    return _dedupe_artifacts(artifacts)


def _build_file_artifacts(
    file_path: str,
    content: str,
    build_system: str,
    build_variables: dict[str, str] | None,
) -> list[Artifact]:
    artifacts: list[Artifact] = []
    artifacts.extend(_cmake_artifacts(content, build_variables))
    artifacts.extend(_pkgconfig_artifacts(content))
    artifacts.extend(_dependency_list_artifacts(content))
    artifacts.extend(_program_artifacts(content))
    artifacts.extend(_qmake_qt_artifacts(content))
    artifacts.extend(_boost_autoconf_artifacts(content))
    artifacts.extend(_generic_pkgconfig_wrapper_artifacts(content))
    artifacts.extend(_probe_assignment_artifacts(content))
    if build_system in {'makefile', 'automake', 'configure_script', 'bootstrap'}:
        artifacts.extend(_generic_command_position_artifacts(content))
    artifacts.extend(_autoconf_libltdl_artifacts(content))
    artifacts.extend(_header_artifacts(content))
    artifacts.extend(_library_artifacts(content))
    artifacts.extend(_feature_artifacts(content))
    return artifacts


def _m4_argument_artifacts(content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for name, body in _iter_m4_calls(content):
        if name in _M4_DEFINITION_MACROS:
            continue
        for arg in _split_m4_args(body):
            artifacts.extend(_token_artifacts(arg))
    return artifacts


def _generic_pkgconfig_wrapper_artifacts(content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for match in _PKGCONFIG_WRAPPER_CALL_RE.finditer(content):
        body = match.group(2) if match.group(2) is not None else match.group(3)
        if not body:
            continue
        for token in _words(body):
            token = _clean_pkgconfig_arg(token)
            if _looks_like_pkgconfig_token(token):
                artifacts.append(Artifact('pkgconfig', token))
    return artifacts


def _clean_pkgconfig_arg(token: str) -> str:
    token = token.strip().strip('"\'`()[]{}<>:,;')
    token = re.sub(r'^(?:--?[A-Za-z0-9_+-]+=?)+', '', token)
    return token


def _words(text: str) -> list[str]:
    return re.findall(r'[A-Za-z0-9_+.-]+', text)


def _probe_assignment_artifacts(content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for match in _SHELL_ASSIGNMENT_RE.finditer(content):
        name = match.group(1).lower()
        value = next((group for group in match.groups()[1:] if group is not None), '')
        if not value:
            continue
        if re.search(r'(?:inc|include|includes|header|headers)', name):
            artifacts.extend(_source_include_artifacts(value))
            artifacts.extend(
                artifact for artifact in _token_artifacts(value)
                if artifact.kind == 'header'
            )
        if re.search(r'(?:lib|libs|ldflags)', name):
            artifacts.extend(_token_artifacts(value))
        if re.search(r'(?:pkg|cflags)', name):
            artifacts.extend(_generic_pkgconfig_wrapper_artifacts(value))
            artifacts.extend(
                artifact for artifact in _token_artifacts(value)
                if artifact.kind == 'pkgconfig'
            )
        if re.search(r'(?:prog|program|tool|cmd|command|compiler)', name):
            artifacts.extend(_command_artifact(
                value.split()[0] if value.split() else '',
                allow_provider=True,
            ))
        if _COMPILER_VAR_NAME_RE.search(name):
            artifacts.extend(_command_artifact(
                value.split()[0] if value.split() else '',
                allow_provider=True,
                hints=('compiler',),
            ))
        artifacts.extend(_command_lookup_artifacts(value))
    for name, value in _SHELL_DEFAULT_TOOL_ASSIGNMENT_RE.findall(content):
        if not _TOOL_VAR_NAME_RE.search(name):
            continue
        artifacts.extend(_command_artifact(value.split()[0] if value.split() else '', allow_provider=True))
    artifacts.extend(_command_lookup_artifacts(content))
    return artifacts


def _command_lookup_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for token in _COMMAND_LOOKUP_RE.findall(text):
        artifacts.extend(_command_artifact(token, allow_provider=True))
    return artifacts


def _autoconf_libltdl_artifacts(content: str) -> list[Artifact]:
    if not re.search(r'\b(?:AC_LIBLTDL[A-Z0-9_]*|AC_LTDL[A-Z0-9_]*|LT_LIB_DLLOAD|LTDL_INIT)\b', content, re.IGNORECASE):
        return []
    return [Artifact('library', 'ltdl'), Artifact('header', 'ltdl.h')]


def _generic_command_position_artifacts(content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    tool_vars = _tool_variable_commands(content)
    for raw in _MAKE_RECIPE_COMMAND_RE.findall(content):
        artifacts.extend(_build_command_artifact(_resolve_command_token(raw, tool_vars)))
    for block, _was_continued in _continued_shell_lines(content):
        stripped = block.strip()
        if not stripped or stripped.startswith('#'):
            continue
        for segment in re.split(r'\s*(?:&&|\|\||;|\|)\s*', stripped):
            match = _SHELL_COMMAND_RE.match(segment)
            if match:
                artifacts.extend(_build_command_artifact(_resolve_command_token(match.group(1), tool_vars)))
    return artifacts


def _tool_variable_commands(content: str) -> dict[str, str]:
    commands: dict[str, str] = {}
    for name, value in _TOOL_ASSIGNMENT_RE.findall(content):
        if not _TOOL_VAR_NAME_RE.search(name):
            continue
        words = value.strip().split()
        if not words:
            continue
        command = _strip_command_path(words[0])
        if _command_artifact(command, allow_provider=True):
            commands[name] = command
    for name, value in _SHELL_DEFAULT_TOOL_ASSIGNMENT_RE.findall(content):
        if not _TOOL_VAR_NAME_RE.search(name):
            continue
        command = _strip_command_path(value.split()[0] if value.split() else '')
        if _command_artifact(command, allow_provider=True):
            commands[name] = command
    return commands


def _resolve_command_token(token: str, tool_vars: dict[str, str]) -> str:
    token = token.strip().strip('"\'')
    if token.startswith('$(') and token.endswith(')'):
        return tool_vars.get(token[2:-1], '')
    if token.startswith('${') and token.endswith('}'):
        return tool_vars.get(token[2:-1], '')
    token = token.strip().strip('`(){}')
    return _strip_command_path(token)


def _strip_command_path(token: str) -> str:
    token = token.strip().strip('"\'`(){}')
    if not token:
        return ''
    return token.rsplit('/', 1)[-1].lower()


def _command_artifact(
    command: str,
    allow_provider: bool = False,
    hints: tuple[str, ...] = (),
) -> list[Artifact]:
    command = _strip_command_path(command)
    if not command or command in _SHELL_BUILTINS or command.isupper():
        return []
    if 'compiler' in hints and _DEFAULT_COMPILER_COMMAND_RE.match(command):
        return []
    index = _provider_index()
    if (
        command in _SCRIPT_SOURCE_GENERATOR_TOOLS
        or (allow_provider and command in index.executables)
        or (allow_provider and command in index.config_scripts)
        or command.endswith(('-config', '_config'))
    ):
        return [Artifact('executable', command, hints)]
    return []


def _build_command_artifact(command: str, hints: tuple[str, ...] = ()) -> list[Artifact]:
    command = _strip_command_path(command)
    artifacts = _command_artifact(command, allow_provider=True, hints=hints)
    if artifacts:
        return artifacts
    if command in _BUILD_FILE_EXTRA_TOOLS:
        return [Artifact('executable', command, hints)]
    return []


def _iter_m4_calls(content: str):
    for match in _M4_CALL_RE.finditer(content):
        parsed = _parse_call_body(content, match.end() - 1)
        if parsed is not None:
            yield match.group(1), parsed


def _parse_call_body(content: str, open_paren: int) -> str | None:
    paren_depth = 0
    bracket_depth = 0
    start = open_paren + 1
    i = open_paren
    while i < len(content):
        ch = content[i]
        if ch == '[':
            bracket_depth += 1
        elif ch == ']' and bracket_depth:
            bracket_depth -= 1
        elif bracket_depth == 0:
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    return content[start:i]
        i += 1
    return None


def _split_m4_args(body: str) -> list[str]:
    args: list[str] = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    for idx, ch in enumerate(body):
        if ch == '[':
            bracket_depth += 1
        elif ch == ']' and bracket_depth:
            bracket_depth -= 1
        elif bracket_depth == 0:
            if ch == '(':
                paren_depth += 1
            elif ch == ')' and paren_depth:
                paren_depth -= 1
            elif ch == ',' and paren_depth == 0:
                args.append(_unquote_m4(body[start:idx]))
                start = idx + 1
    args.append(_unquote_m4(body[start:]))
    return [arg for arg in args if arg.strip()]


def _unquote_m4(arg: str) -> str:
    arg = arg.strip()
    while arg.startswith('[') and arg.endswith(']') and _balanced_brackets(arg[1:-1]):
        arg = arg[1:-1].strip()
    return arg


def _balanced_brackets(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == '[':
            depth += 1
        elif ch == ']':
            if depth == 0:
                return False
            depth -= 1
    return depth == 0


def _has_structural_dep_marker(token: str) -> bool:
    """Check if token has a structural pattern indicating a real dependency name.

    Pure-alpha words without separators (and, check, at, since, just, buffer,
    filter, global, screen, dialog) are almost certainly natural language, not
    dependency names. Real dependency tokens have at least one of:
    - lib prefix
    - -l linker flag prefix
    - digit in name (zlib1g, libssl3)
    - .h/.hpp/.pc suffix (header/pkgconfig file)
    - -dev/-tools/-bin suffix (Debian package naming)
    - hyphen-separated multi-part name with segments >= 2 chars (protobuf-c)
    """
    t = token.lower().strip()
    if t.startswith('-l') and len(t) > 2:
        return True
    if t.startswith('lib'):
        return True
    if re.search(r'\d', t):
        return True
    if t.endswith(('.h', '.hpp', '.hxx', '.hh', '.pc')):
        return True
    if t.endswith(('-dev', '-tools', '-tool', '-bin', '-utils', '-headers', '-compiler')):
        return True
    if '-' in t:
        parts = t.split('-')
        if len(parts) >= 2 and all(len(p) >= 2 for p in parts):
            return True
    if '.' in t and not t.endswith('.'):
        return True
    return False


def _comment_artifacts(file_path: str, content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    use_assembly_comments = file_path.lower().endswith(_ASSEMBLY_COMMENT_EXTENSIONS)
    for line in content.splitlines():
        match = _COMMENT_RE.match(line)
        if not match and use_assembly_comments:
            match = _ASSEMBLY_COMMENT_RE.match(line)
        if not match:
            continue
        comment = match.group(1)
        if not _COMMENT_EVIDENCE_RE.search(comment):
            if use_assembly_comments:
                artifacts.extend(_script_tool_invocation_artifacts(comment.strip()))
            continue
        # Only extract artifacts from tokens with structural dependency markers.
        # Pure-alpha English words in comments (and, check, at, since, buffer,
        # filter, etc.) match Debian package names via provider_index but are
        # almost never real build-deps.
        for token in _TOKEN_RE.findall(comment):
            token = token.strip().strip('"\'`()[]{}<>:,;')
            if not token:
                continue
            if not _has_structural_dep_marker(token):
                continue
            artifacts.extend(_artifact_from_token(token))
    return artifacts


def _source_include_artifacts(content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for header in _SOURCE_INCLUDE_RE.findall(content):
        artifacts.extend(_artifact_from_token(header))
    artifacts.extend(_generated_by_tool_artifacts(content))
    return artifacts


def _generated_by_tool_artifacts(content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for tool in _GENERATED_BY_TOOL_RE.findall(content):
        artifacts.extend(_command_artifact(tool, allow_provider=True))
    return artifacts


def _qml_import_artifacts(content: str) -> list[Artifact]:
    return [Artifact('qml_import', module) for module in _QML_IMPORT_RE.findall(content)]


def _script_dependency_artifacts(content: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    lines = content.splitlines()
    continued_install_lines = _continued_install_line_numbers(lines)
    for line_number, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if re.search(r'\b(?:apt(?:-get)?|dnf|yum|apk|pacman)\s+.*\binstall\b', lower):
            artifacts.extend(_token_artifacts(stripped))
        elif re.match(r'(?:packages|deps|dependencies|build_deps)\s*=', lower):
            artifacts.extend(_token_artifacts(stripped))
        elif line_number not in continued_install_lines:
            artifacts.extend(_script_tool_invocation_artifacts(stripped))
            artifacts.extend(_sourced_script_artifacts(stripped))
    for block, was_continued in _continued_shell_lines(content):
        if not was_continued:
            continue
        if not re.search(r'\b(?:apt(?:-get)?|dnf|yum|apk|pacman)\s+.*\binstall\b', block.lower()):
            continue
        artifacts.extend(
            artifact for artifact in _token_artifacts(block)
            if artifact.kind == 'package' and artifact.name.endswith('-headers')
        )
    return artifacts


def _script_tool_invocation_artifacts(line: str) -> list[Artifact]:
    if not line or line.startswith('#'):
        return []
    artifacts: list[Artifact] = []
    for segment in re.split(r'\s*(?:&&|\|\||;)\s*', line):
        match = _SHELL_COMMAND_RE.match(segment)
        if not match:
            continue
        command = match.group(1).lower()
        if command in _SCRIPT_IGNORED_TOOLS:
            continue
        artifacts.extend(_command_artifact(command, allow_provider=True))
    return artifacts


def _sourced_script_artifacts(line: str) -> list[Artifact]:
    match = _SHELL_SOURCE_RE.match(line)
    if not match:
        return []
    token = match.group(1).lower()
    if token in _SHELL_BUILTINS or token.endswith(('.sh', '.bash', '.zsh')):
        return []
    if '/' in token or token.startswith('-'):
        return []
    return [Artifact('executable', token)]


def _continued_install_line_numbers(lines: list[str]) -> set[int]:
    install_lines: set[int] = set()
    start = 0
    block_has_install = False
    in_continuation = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not in_continuation:
            start = index
            block_has_install = bool(
                re.search(r'\b(?:apt(?:-get)?|dnf|yum|apk|pacman)\s+.*\binstall\b', stripped.lower())
            )
        in_continuation = stripped.endswith('\\')
        if block_has_install:
            install_lines.update(range(start, index + 1))
        if not in_continuation:
            block_has_install = False
    return install_lines


def _package_list_artifacts(content: str) -> list[Artifact]:
    return [
        artifact for artifact in _token_artifacts(content)
        if artifact.kind == 'package' and _RUNTIME_ABI_PACKAGE_RE.match(artifact.name)
    ]


def _continued_shell_lines(content: str):
    buffered = ''
    was_continued = False
    for line in content.splitlines():
        stripped = line.strip()
        buffered = f'{buffered} {stripped}'.strip() if buffered else stripped
        if stripped.endswith('\\'):
            buffered = buffered[:-1].rstrip()
            was_continued = True
            continue
        yield buffered, was_continued
        buffered = ''
        was_continued = False
    if buffered:
        yield buffered, was_continued


def _token_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for token in _TOKEN_RE.findall(text):
        token = token.strip().strip('"\'`()[]{}<>:,;')
        if not token:
            continue
        artifacts.extend(_artifact_from_token(token))
    return artifacts


def _artifact_from_token(token: str) -> list[Artifact]:
    if '$' in token or token.startswith('@') or token.endswith('@'):
        return []
    lower = token.lower()
    normalized = _normal_pkgconfig_token(lower)
    if normalized in _STOPWORDS or re.fullmatch(r'\d+(?:\.\d+)*', normalized):
        return []
    index = _provider_index()
    if lower.startswith('-l') and len(lower) > 2:
        return [Artifact('library', lower[2:])]
    if lower.endswith(('.h', '.hpp', '.hxx', '.hh')) and not _is_standard_header(lower):
        return [Artifact('header', lower)]
    artifacts: list[Artifact] = []
    if normalized in index.pkgconfig:
        artifacts.append(Artifact('pkgconfig', normalized))
    if normalized in index.libraries:
        artifacts.append(Artifact('library', normalized))
    if normalized in index.cmake:
        artifacts.append(Artifact('cmake_package', token))
    if normalized in index.executables or normalized in index.config_scripts:
        artifacts.append(Artifact('executable', normalized))
    if normalized.endswith('-config') or normalized.endswith('_config'):
        artifacts.append(Artifact('executable', normalized))
    if _looks_like_package_name(normalized):
        artifacts.append(Artifact('package', normalized))
    return artifacts


def _looks_like_package_name(token: str) -> bool:
    if not re.fullmatch(r'[a-z0-9][a-z0-9.+-]+', token):
        return False
    return (
        token.endswith(('-dev', '-tools', '-tool', '-bin', '-utils', '-headers', '-compiler'))
        or '-dev-' in token
        or re.match(r'^lib[a-z0-9.+-]*\d+(?:t64)?$', token)
        or _RUNTIME_ABI_PACKAGE_RE.match(token)
        or token.startswith(('architecture-is-', 'dh-sequence-'))
    )


def _dedupe_artifacts(artifacts: list[Artifact]) -> list[Artifact]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    deduped: list[Artifact] = []
    for artifact in artifacts:
        key = (artifact.kind, artifact.name.lower(), tuple(c.lower() for c in artifact.components))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(artifact)
    return deduped
