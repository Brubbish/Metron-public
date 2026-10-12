"""File walker, rule matcher, context collector, and debian/control parser."""

import ast
import configparser
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None

from .dependency_lists import find_dependency_lists
from .artifact_extractors import extract_file_artifacts
from .artifacts import Artifact, extract_artifacts
from .sysuse_select import VersionSignals, extract_signals_from_file, should_extract_signal_file
from .rules import (
    RuleMatch, CONTEXT_LINES, classify_file, get_patterns, NOISE_LIBS,
)


SKIP_DIR_NAMES = frozenset({
    '.circleci', '.git', '.github', '.gitlab', '.pc', '__pycache__',
    'ci', 'debian', 'node_modules', 'script', 'scripts',
    # Phase-1 source-dep clones live under this name; never scan them as part
    # of the main repo — they are scanned by source_dep_resolver individually.
    '_source_deps',
})

THIRD_PARTY_PREFIXES = (
    '3rdparty', '3rdparties', 'thirdparty', 'thirdparties',
    'vendor', 'vendors', 'external', 'externals',
    'dependency', 'dependencies', 'deps',
    'subprojects', 'bundled',
)

_DEDUP_RULE_TYPES = frozenset({
    'ac:AX_macro',
    'ac:AC_CHECK_LIB',
    'configure:-l',
    'make:-l',
})

_REAL_ONE_LETTER_LIBS = frozenset({'z'})
_BAD_L_FLAG_NAMES = frozenset({
    'NAME',
    'dir', 'ib', 'ibdir', 'ibexecdir', 'ink', 'ocaledir', 'ocalstatedir',
    'dld', 'svld', 'ltdl',
})

# Python module paths / config keys that should never become package names.
# Catches bugs like `python3-setuptools.command.sdist` (dotted module path) being
# returned as a package by _debianize_python_req's default `python3-{norm}` rule.
_SUSPICIOUS_PKG_RE = re.compile(
    r'^(?:'
    r'python3-[a-z][a-z0-9-]*\.[a-z][a-z0-9.-]*'  # python3-foo.bar (dotted module path)
    r'|python3-(?:cpplint|lint|ninja|cmake|this-cli|cython\.build|cythonize|'
    r'setuptools\.command|setuptools\.build|testenv\.deps|tox\.envlist|'
    r'distutils\.core|extension|install-requires|libraries|build-meta|'
    r'afflib|xz|zlib|zstd|beautysh|clang|itk|pip|skbuild|all|all-dev|dev|wheel|setuptools|pytest)'
    r'|ruby-(?:bundler|dev|minitest|mkmf|rake(?:-compiler|/testtask))'
    r'|node-bats'
    r')$',
    re.IGNORECASE,
)



_M4_DEPENDENCY_FILE_RE = re.compile(
    r'^(?:lib[A-Za-z0-9_+.-]+|zlib|openssl|systemd|keyutils|fuse|gcrypt|gnutls|'
    r'nettle|blas|lapack|sqlite3|readline|ncurses|curl|uuid|xml2|xslt|gtk-doc|'
    r'docbook|asciidoc|asciidoctor)\.m4$',
    re.IGNORECASE,
)

_PYTHON_INTERESTING_EXTRAS = frozenset({
    'test', 'tests', 'testing', 'check', 'checks', 'dev', 'devel',
    'build', 'doc', 'docs', 'documentation', 'sphinx',
})

_PYTHON_IMPORTS = frozenset({
    'setuptools', 'setuptools_scm', 'distutils', 'wheel', 'pip', 'pytest', 'sphinx',
    'sphinx_rtd_theme', 'breathe', 'cython', 'numpy', 'scipy', 'poetry', 'flit', 'flit_core', 'hatchling', 'mesonpy',
    'meson_python', 'build', 'skbuild', 'scikit_build_core', 'pybind11',
    'pkgconfig', 'nbsphinx', 'numpydoc', 'docutils',
})

_PYPROJECT_BACKEND_HINTS = {
    'setuptools': 'setuptools',
    'setuptools_scm': 'setuptools_scm',
    'flit_core': 'flit_core',
    'hatchling': 'hatchling',
    'poetry': 'poetry',
    'mesonpy': 'mesonpy',
    'meson_python': 'meson_python',
    'scikit_build_core': 'scikit_build_core',
}


_MACRO_LIBRARY_FILES = frozenset({
    'aclocal.m4', 'libtool.m4', 'ltoptions.m4', 'ltsugar.m4',
    'ltversion.m4', 'lt~obsolete.m4', 'lt_obsolete.m4',
    'boost.m4',
})
_GENERIC_AUTOCONF_MACRO_FILES = frozenset({
    'aclocal.m4', 'libtool.m4', 'ltoptions.m4', 'ltsugar.m4',
    'ltversion.m4', 'lt~obsolete.m4', 'lt_obsolete.m4',
})
_CMAKE_HELPER_MODULE_FILES = frozenset({
    'findpackagehandlestandardargs.cmake',
    'findpackagemessage.cmake',
})

_STANDARD_HEADERS = frozenset({
    'alloca.h', 'arpa/inet.h', 'assert.h', 'ctype.h', 'dirent.h', 'dlfcn.h',
    'errno.h', 'fcntl.h', 'float.h', 'getopt.h', 'grp.h', 'inttypes.h',
    'langinfo.h', 'libgen.h', 'limits.h', 'locale.h', 'math.h', 'memory.h',
    'netdb.h', 'netinet/in.h', 'poll.h', 'pthread.h', 'pwd.h', 'signal.h',
    'stdarg.h', 'stdbool.h', 'stddef.h', 'stdint.h', 'stdio.h', 'stdlib.h',
    'string.h', 'strings.h', 'sys/ioctl.h', 'sys/mman.h', 'sys/param.h',
    'sys/select.h', 'sys/socket.h', 'sys/stat.h', 'sys/time.h', 'sys/types.h',
    'sys/uio.h', 'sys/un.h', 'sys/wait.h', 'termios.h', 'time.h', 'unistd.h',
    'utime.h', 'wchar.h', 'wctype.h',
})

_SOURCE_ARTIFACT_EXTS = frozenset({
    '.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx',
    '.py', '.rb', '.rs', '.go', '.java', '.js', '.ts', '.qml',
    '.x', '.y', '.l', '.g', '.proto', '.asm', '.s',
})
_MAX_SOURCE_ARTIFACT_BYTES = 2_000_000
_ARTIFACT_SNIFF_BYTES = 64 * 1024
_EXCLUDED_ARTIFACT_DIRS = frozenset({
    '.github', '.gitlab', '.circleci', '.pc', 'ci', 'debian', 'script', 'scripts',
})
_EXCLUDED_ARTIFACT_BASENAME_RE = re.compile(
    r'^(?:readme(?:\..*)?|dockerfile(?:\..*)?|containerfile(?:\..*)?|'
    r'\.gitlab-ci\.ya?ml)$',
    re.IGNORECASE,
)
_CONFIGURE_ARTIFACT_BASENAMES = frozenset({
    'config', 'configure.project',
})
_SCRIPT_ARTIFACT_BASENAMES = frozenset()
_CONTENT_SNIFF_BASENAME_RE = re.compile(
    r'^(?:config|configure(?:[._-].*)?|makefile(?:[._-].*)?|[^/]+\.mk)$',
    re.IGNORECASE,
)
_CONTENT_SNIFF_PATH_RE = re.compile(
    r'(?:^|/)(?:w_conf|build|m4|config|configure|scripts?)(?:/|$)',
    re.IGNORECASE,
)
_AUTOCONF_CONTENT_RE = re.compile(r'\b(?:AC_|AM_|AX_|m4_)[A-Za-z0-9_]*\s*\(', re.IGNORECASE)
_CONFIGURE_CONTENT_RE = re.compile(
    r'\b(?:pkg-config|pkgconf|--cflags|--libs|ngx_feature_|ngx_module_|'
    r'w_pkgconfig_check|command -v|AC_CHECK_PROGS?)\b',
    re.IGNORECASE,
)
_SOURCE_PROBE_CONTENT_RE = re.compile(r'^\s*#\s*include\b|(?:^|[\s="\'])-l[A-Za-z0-9_+.-]+\b', re.MULTILINE)
_SCRIPT_CONTENT_RE = re.compile(
    r'^\s*[A-Za-z_][A-Za-z0-9_]*(?:TOOL|TOOLS|PROG|PROGRAM|EXECUTABLE|CMD|COMMAND|COMPILER|GENERATOR)\s*[:?+]?=',
    re.IGNORECASE | re.MULTILINE,
)

_DIRECT_RULE_TYPES = frozenset({
    'ac:known_macro', 'ac:libtool', 'ac:docbook-xsl', 'ac:docbook-xml',
    'ac:m4_dependency_file', 'cmake:find_doc_tool', 'cmake:doc_custom_command',
    'cmake:docbook-xsl', 'cmake:docbook-xml', 'cmake:qt_auto_tool',
    'doxygen:have-dot', 'doxygen:dot-path', 'make:doc-tool-var',
    'make:docbook-xsl', 'make:docbook-xml', 'make:tool-invocation',
    'make:bison', 'make:flex', 'make:lexer-parser-rule', 'make:texinfo',
    'meson:doc_tool', 'meson:docbook-xsl', 'meson:docbook-xml',
    'meson:gnome_generate_gir', 'meson:gnome_generate_vapi', 'meson:gnome_gtkdoc',
    'meson:option_introspection', 'meson:option_vapi', 'meson:option_gtkdoc',
    'qmake:tool', 'qmake:TRANSLATIONS', 'docbook:doctype', 'docbook:xsl', 'docbook:xml',
})

_TOOL_RULE_TYPES = frozenset({
    'ac:AC_CHECK_PROG', 'ac:AC_PATH_PROG',
    'cmake:find_doc_tool', 'cmake:doc_custom_command',
    'doxygen:have-dot', 'doxygen:dot-path',
    'make:doc-tool-var', 'make:tool-invocation',
    'make:bison', 'make:flex', 'make:lexer-parser-rule', 'make:texinfo',
    'meson:doc_tool', 'qmake:tool',
})

_DIRECT_RULE_TYPES = _DIRECT_RULE_TYPES | _TOOL_RULE_TYPES

_TOOL_TOKENS = frozenset({
    'a2x',
    'asciidoc',
    'asciidoctor',
    'bison',
    'bpftool',
    'cargo',
    'docbook2man',
    'gdbus-codegen',
    'glib-compile-resources',
    'glib-compile-schemas',
    'genhtml',
    'gtkdoc-fixxref',
    'gtkdoc-mkdb',
    'gtkdoc-scan',
    'gtkdocize',
    'lcov',
    'makeinfo',
    'nasm',
    'rst2html',
    'rst2man',
    'scdoc',
    'sphinx-build',
    'swig',
    'txt2man',
    'wayland-scanner',
    'xxd',
    'yasm',
})

_TOOL_TOKEN_RE = re.compile(
    r'(?<![A-Za-z0-9_+.-])('
    + '|'.join(re.escape(name) for name in sorted(_TOOL_TOKENS, key=len, reverse=True))
    + r')(?![A-Za-z0-9_+.-])',
    re.IGNORECASE,
)



def _normalize_dir(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def _should_skip_dir(name: str, skip_third_party: bool = True) -> bool:
    if name.startswith('.'):
        return True
    low = name.lower()
    if low in SKIP_DIR_NAMES:
        return True
    if low.startswith(('build-', 'build_', 'cmake-build', 'cmake_build')):
        return True
    if low in ('build', '_build', 'builddir', 'out', 'output'):
        return True
    if not skip_third_party:
        return False
    norm = _normalize_dir(name)
    return any(norm == p or norm.startswith(p) for p in THIRD_PARTY_PREFIXES)


def _should_descend_artifact_probe_dir(root: str, dirpath: str, dirname: str) -> bool | None:
    rel = os.path.relpath(os.path.join(dirpath, dirname), root)
    rel_l = rel.replace(os.sep, '/').lower()
    if rel_l == 'build' or rel_l in {'build/with', 'build/has'}:
        return True
    if rel_l.startswith('build/'):
        return False
    return None


@dataclass
class ProjectScan:
    build_systems: set = field(default_factory=set)
    file_matches: dict = field(default_factory=dict)
    file_artifacts: dict = field(default_factory=dict)
    direct_packages: set = field(default_factory=set)
    build_driver_packages: set = field(default_factory=set)
    build_variables: dict = field(default_factory=dict)
    version_signals: VersionSignals = field(default_factory=VersionSignals)


_CMAKE_SET_RE = re.compile(
    r'^\s*set\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+(.*?)\)',
    re.MULTILINE | re.DOTALL,
)
_CMAKE_VAR_COMPARE_RE = re.compile(
    r'\b(?:if|elseif)\s*\(\s*"?\$\{([A-Za-z_][A-Za-z0-9_]*)\}"?\s+'
    r'(?:STREQUAL|EQUAL|VERSION_EQUAL)\s+"?([A-Za-z0-9_.+-]+)"?',
    re.IGNORECASE,
)
_CMAKE_INTERPOLATED_NAME_VAR_RE = re.compile(
    r'[A-Za-z0-9_.+-]\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$\{([A-Za-z_][A-Za-z0-9_]*)\}[A-Za-z0-9_.+-]'
)
_MAKE_VAR_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:?]?=\s*(.+)$',
    re.MULTILINE,
)
_QMAKE_VAR_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$',
    re.MULTILINE,
)
_MESON_STRING_VAR_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\'"])(.*?)\2\s*(?:#.*)?$',
    re.MULTILINE,
)
_AUTOCONF_VAR_RE = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\'"]?)([A-Za-z0-9][A-Za-z0-9.+:~_-]*)\2\s*(?:$|dnl\b|#)',
    re.MULTILINE,
)
_AUTOCONF_FULL_CALL_MACROS = {
    'PKG_CHECK_MODULES': 'ac:PKG_CHECK_MODULES',
    'PKG_CHECK_MODULES_STATIC': 'ac:PKG_CHECK_MODULES',
    'PKG_CHECK_EXISTS': 'ac:PKG_CHECK_EXISTS',
    'PKG_CHECK_VAR': 'ac:PKG_CHECK_VAR',
}


def _collect_build_variables(content: str, build_system: str) -> dict[str, str]:
    """Extract simple variable assignments from build file content."""
    variables: dict[str, str] = {}

    def remember(var: str, value: str) -> None:
        existing = variables.get(var)
        if existing is None:
            variables[var] = value
        elif isinstance(existing, list):
            if value not in existing:
                existing.append(value)
        elif existing != value:
            variables[var] = [existing, value]

    if build_system == 'cmake':
        name_interpolated_vars = {
            var
            for m in _CMAKE_INTERPOLATED_NAME_VAR_RE.finditer(content)
            for var in (m.group(1), m.group(2))
            if var
        }
        for m in _CMAKE_SET_RE.finditer(content):
            var, value = m.group(1), m.group(2).strip()
            value = re.split(r'\b(?:CACHE|PARENT_SCOPE)\b', value, 1)[0].strip().strip('"\'')
            if '$' not in value and '@' not in value:
                remember(var, value)
        for m in _CMAKE_VAR_COMPARE_RE.finditer(content):
            var, value = m.group(1), m.group(2).strip()
            if var in name_interpolated_vars and value:
                remember(var, value)
    elif build_system in {'makefile', 'automake'}:
        for m in _MAKE_VAR_RE.finditer(content):
            var, value = m.group(1), m.group(2).strip()
            if '$' not in value and '@' not in value and value:
                remember(var, value)
    elif build_system == 'autoconf':
        for m in _AUTOCONF_VAR_RE.finditer(content):
            var, value = m.group(1), m.group(3).strip()
            if value:
                remember(var, value)
    elif build_system == 'qmake':
        for m in _QMAKE_VAR_RE.finditer(content):
            var, value = m.group(1), m.group(2).strip()
            if '$' not in value and value:
                remember(var, value)
    elif build_system == 'meson':
        for m in _MESON_STRING_VAR_RE.finditer(content):
            var, value = m.group(1), m.group(3).strip()
            if '$' not in value and '@' not in value and value:
                remember(var, value)
    return variables


def _find_balanced_call(content: str, open_paren: int) -> tuple[int, str] | None:
    """Return the end offset and full macro call starting at open_paren's macro."""
    start = open_paren
    while start > 0 and re.match(r'[A-Za-z0-9_]', content[start - 1]):
        start -= 1
    depth = 0
    quote = ''
    escaped = False
    for index in range(open_paren, len(content)):
        char = content[index]
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = ''
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return index + 1, content[start:index + 1]
    return None


def _autoconf_full_call_matches(filepath: str, content: str) -> list[tuple[int, str, str, str]]:
    """Extract complete pkg-config Autoconf macro calls, including multiline bodies."""
    matches: list[tuple[int, str, str, str]] = []
    for macro, rule_type in _AUTOCONF_FULL_CALL_MACROS.items():
        for m in re.finditer(rf'\b{macro}\s*\(', content, re.IGNORECASE):
            open_paren = content.find('(', m.start())
            balanced = _find_balanced_call(content, open_paren)
            if not balanced:
                continue
            _end, call_text = balanced
            line_num = content[:m.start()].count('\n')
            extracted = _autoconf_pkgconfig_extracted_name(call_text)
            matches.append((line_num, rule_type, call_text, extracted))
    return matches


def _autoconf_pkgconfig_extracted_name(call_text: str) -> str:
    body_match = re.search(r'\((.*)\)\s*$', call_text, re.DOTALL)
    if not body_match:
        return ''
    body = body_match.group(1)
    parts = body.split(',', 2)
    if len(parts) < 2:
        return ''
    deps = parts[1]
    deps = deps.replace('\\\n', ' ')
    deps = deps.replace('[', ' ').replace(']', ' ')
    deps = re.sub(r'\s+', ' ', deps).strip()
    return deps[:500]


def _extract_context(lines: list[str], match_lines: set[int], context_n: int = CONTEXT_LINES) -> str:
    """Build a context string covering ±context_n lines around all match_lines."""
    include = set()
    for lno in match_lines:
        for i in range(max(0, lno - context_n), min(len(lines), lno + context_n + 1)):
            include.add(i)
    parts = []
    prev = -2
    for i in sorted(include):
        if i != prev + 1:
            parts.append('...')
        parts.append(f'{i+1}: {lines[i]}')
        prev = i
    return '\n'.join(parts)


def _context_window_for_rule(rule_type: str, default: int) -> int:
    """Use rule-aware snippets where nearby lines carry dependency context."""
    window_12 = {
        'ac:AC_CHECK_LIB', 'ac:PKG_CHECK_MODULES', 'ac:pkg-config',
        'cmake:find_dependency', 'cmake:find_dependency_call',
        'cmake:find_package', 'cmake:find_package_call',
        'cmake:pkg_check_modules', 'cmake:pkg_get_variable', 'cmake:dependency-list',
    }
    window_10 = {
        'ac:AC_CHECK_HEADER', 'ac:AC_PATH_PROG', 'ac:AC_SEARCH_LIBS',
        'ac:AX_macro', 'ac:generic_dependency_macro', 'ac:PKG_CHECK_EXISTS',
        'ac:PKG_CHECK_VAR', 'ac:dependency-list', 'ac:error_missing',
        'ac:known_macro', 'cmake:FetchContent', 'cmake:ExternalProject',
        'cmake:FetchContent_MakeAvailable', 'cmake:CPMAddPackage',
        'cmake:find_package_handle_standard_args', 'cmake:include_find_module',
        'cmake:find_program', 'cmake:find_program_call',
        'cmake:msg_missing', 'cmake:target_link_libraries',
        'make:-l', 'make:config-script', 'make:pkg-config', 'make:dependency-list',
        'meson:dependency', 'meson:dependency-list', 'meson:target_command',
        'qmake:LIBS', 'configure:dependency-list', 'bootstrap:dependency-list', 'vcpkg:dependencies',
    }
    window_8 = {
        'cmake:check_include_file', 'cmake:check_library_exists',
        'cmake:doc_custom_command', 'cmake:docbook-xml',
        'cmake:docbook-xsl', 'cmake:find_doc_tool',
        'cmake:find_library', 'cmake:find_path',
        'doxygen:config', 'doxygen:dot-path', 'doxygen:have-dot',
        'make:doc-tool-var', 'make:docbook-xml', 'make:docbook-xsl',
        'make:link-libs-var', 'make:tool-invocation',
        'meson:doc_tool', 'meson:docbook-xml', 'meson:docbook-xsl',
        'meson:find_library', 'meson:has_header', 'meson:has_function',
        'meson:find_program', 'meson:find_program_call',
        'qmake:INCLUDEPATH', 'qmake:PKGCONFIG', 'qmake:QT', 'qmake:tool',
    }
    base_rule_type = rule_type.rsplit(':', 1)[0] if ':dependency-list:' in rule_type else rule_type
    if base_rule_type in window_12:
        return max(default, 12)
    if base_rule_type in window_10:
        return max(default, 10)
    if base_rule_type in window_8:
        return max(default, 8)
    # Most rule matches are self-contained; keep their snippets small so wider
    # rules above do not push prompt cost too high.
    return min(default, 5)


def _l_flags(text: str) -> list[str]:
    return re.findall(r'(?:^|[\s="])-l([A-Za-z0-9_+.-]+)', text)


def _only_noise_l_flags(text: str) -> bool:
    libs = _l_flags(text)
    return bool(libs) and all(lib in NOISE_LIBS for lib in libs)


def _normalize_tool_token(name: str) -> str:
    token = name.strip().strip('"\'`[](){}<>').lower()
    token = os.path.basename(token)
    token = re.sub(r'(?:-\d+(?:\.\d+)*)$', '', token)
    return token


def _tool_artifacts_from_text(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for match in _TOOL_TOKEN_RE.finditer(text):
        token = _normalize_tool_token(match.group(1))
        if token in {'lrelease', 'lupdate'} and re.search(r'(?i)\b(?:Qt6|qt6|/qt6/|QT_VERSION_MAJOR\\s+6)\b', text):
            artifacts.append(Artifact('executable', token, ('qt6',)))
            continue
        if token in _TOOL_TOKENS:
            artifacts.append(Artifact('executable', token))
    return _dedupe_artifacts(artifacts)


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


def _package_tokens_from_text(text: str) -> set[str]:
    return set(re.findall(r'\b[a-z0-9][a-z0-9.+-]*(?:-dev|-tools?|-[a-z0-9][a-z0-9.+-]*)\b', text.lower()))


def _has_unexplained_evidence(match: RuleMatch) -> bool:
    text = f'{match.matched_text}\n{match.extracted_name}'
    stripped = text.strip()
    if not stripped:
        return False
    if match.rule_type in {
        'ac:docbook-xsl', 'ac:docbook-xml',
        'cmake:docbook-xsl', 'cmake:docbook-xml',
        'make:docbook-xsl', 'make:docbook-xml',
        'meson:docbook-xsl', 'meson:docbook-xml',
        'docbook:doctype', 'docbook:xsl', 'docbook:xml',
        'doxygen:config',
        'ac:m4_dependency_file',
    }:
        return False
    if match.rule_type == 'make:pgxs':
        return bool(re.search(r'(?i)\b(?:libpq|pq-fe\.h|-lpq|PQ[A-Za-z0-9_]*)\b', stripped))
    if match.rule_type in _TOOL_RULE_TYPES:
        residue = _TOOL_TOKEN_RE.sub(' ', stripped)
        residue = re.sub(r'\$\([^)]*\)|\$\{[^}]*\}|@[A-Za-z0-9_]+@', ' ', residue)
        residue = re.sub(
            r'(?i)\b(?:find_program|find_path|find_file|add_custom_(?:command|target)|execute_process|'
            r'program|names?|command|output|input|target|comment|required|quiet|path|paths|'
            r'if|endif|set|true|false|yes|no|have_dot|dot_path|help2man|xsltproc|xmlto|'
            r'docbook2x|docbook2xman|docbooktoman|docbook|asciidoctor|asciidoc|a2x|'
            r'sphinxbuild|doxygen|txt2man|scdoc|bison|flex|yacc|lex|missing|'
            r'found|required|install|enable|disable|with|without)\b',
            ' ',
            residue,
        )
        residue = re.sub(r'[-_/.:=+,"\'\\[\]{}()<>0-9]+', ' ', residue)
        words = [
            word for word in re.findall(r'[A-Za-z][A-Za-z0-9_+.-]*', residue)
            if len(word) > 2 and word.upper() != word
        ]
        if match.packages:
            return False
        return bool(words)
    return not bool(match.packages)


def _condition_label(lines: list[str], line_num: int, build_system: str) -> str:
    if build_system != 'cmake':
        return 'unknown'

    stack: list[str] = []
    for idx, line in enumerate(lines[:line_num + 1]):
        stripped = line.strip()
        if re.match(r'(?i)^if\s*\(', stripped):
            stack.append(stripped)
        elif re.match(r'(?i)^else\s*(?:\(|$)', stripped) and stack:
            top = stack.pop()
            stack.append(f'NOT({top})')
        elif re.match(r'(?i)^endif\s*(?:\(|$)', stripped) and stack:
            stack.pop()
        if idx == line_num:
            break

    condition = ' '.join(stack)
    if not condition:
        return 'unknown'
    if re.search(r'(?i)\b(?:WIN32|MSVC|MINGW|APPLE|DARWIN|CYGWIN)\b', condition):
        if not re.search(r'(?i)\bNOT\s*(?:\(|\s+)(?:WIN32|MSVC|MINGW|APPLE|DARWIN|CYGWIN)\b', condition):
            return 'non-linux'
    if re.search(r'(?i)\b(?:UNIX|LINUX|CMAKE_SYSTEM_NAME\s+STREQUAL\s+["\']?Linux)\b', condition):
        return 'linux'
    return 'unknown'


def _normalize_artifact_name(name: str) -> str:
    name = re.sub(r'(\$\{[^}]+\}|@[A-Za-z0-9_]+@)', '', name)
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def _pkgconfig_stem(filepath: str) -> str:
    name = os.path.basename(filepath).lower()
    for suffix in ('.pc.in.in', '.pc.cmake', '.pc.in', '.pc'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return _normalize_artifact_name(name)


def _only_project_libs(filepath: str, text: str) -> bool:
    libs = [_normalize_artifact_name(lib) for lib in _l_flags(text)]
    libs = [lib for lib in libs if lib]
    stem = _pkgconfig_stem(filepath)
    if not libs or not stem:
        return False
    return all(stem in lib or lib in stem for lib in libs)


def _bad_l_flag_name(name: str, build_system: str) -> bool:
    if name in NOISE_LIBS:
        return True
    if build_system != 'configure_script':
        return False
    if not name:
        return True
    if name in _BAD_L_FLAG_NAMES:
        return True
    if '...' in name or name.endswith('.'):
        return True
    if len(name) == 1 and name not in _REAL_ONE_LETTER_LIBS:
        return True
    if name.isupper():
        return True
    return False


def _bad_configure_error(text: str) -> bool:
    lower = text.lower()
    return (
        'missing argument to $ac_option' in lower
        or 'install-sh found' in lower
        or 'install.sh found' in lower
    )


def _is_macro_library_file(filepath: str) -> bool:
    return os.path.basename(filepath).lower() in _MACRO_LIBRARY_FILES


def _is_generic_autoconf_macro_file(filepath: str) -> bool:
    return os.path.basename(filepath).lower() in _GENERIC_AUTOCONF_MACRO_FILES


def _is_cmake_helper_module_file(filepath: str) -> bool:
    basename = os.path.basename(filepath).lower()
    return basename in _CMAKE_HELPER_MODULE_FILES or any(
        basename == f'{name}.in' for name in _CMAKE_HELPER_MODULE_FILES
    )


def _normalize_header_name(name: str) -> str:
    name = name.strip().strip("[](){}<>\"'")
    name = re.sub(r'^(?:include/|/usr/include/)', '', name)
    return name.lower()


def _is_standard_header(name: str) -> bool:
    return _normalize_header_name(name) in _STANDARD_HEADERS


def _is_source_artifact_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in _SOURCE_ARTIFACT_EXTS


def _is_excluded_artifact_path(rel_path: str) -> bool:
    rel_l = rel_path.replace(os.sep, '/').lower()
    parts = [part for part in rel_l.split('/') if part]
    if any(part in _EXCLUDED_ARTIFACT_DIRS for part in parts[:-1]):
        return True
    basename = parts[-1] if parts else os.path.basename(rel_l)
    return bool(_EXCLUDED_ARTIFACT_BASENAME_RE.match(basename))


def _is_build_script_artifact_file(rel_path: str) -> bool:
    return False


def _is_package_list_artifact_file(rel_path: str) -> bool:
    return os.path.basename(rel_path.lower()) == 'required_packages'


def _is_non_probe_build_artifact_path(rel_path: str) -> bool:
    rel_l = rel_path.replace(os.sep, '/').lower()
    if not rel_l.startswith('build/'):
        return False
    return not bool(re.match(r'^build/(?:with|has)/[^/]+\.(?:c|cc|cpp|cxx)$', rel_l))


def _artifact_build_system_for_path(rel_path: str, filename: str) -> str | None:
    if _is_excluded_artifact_path(rel_path):
        return None
    rel_l = rel_path.replace(os.sep, '/').lower()
    basename = os.path.basename(rel_l)
    ext = os.path.splitext(basename)[1]
    if basename in _CONFIGURE_ARTIFACT_BASENAMES:
        return 'configure_script'
    if basename in _SCRIPT_ARTIFACT_BASENAMES:
        return 'script'
    if rel_l.startswith('w_conf/'):
        return 'configure_script'
    if re.search(r'(?:^|/)build/(?:with|has)/[^/]+\.(?:c|cc|cpp|cxx)$', rel_l):
        return 'source'
    if ext in {'.x', '.y', '.l', '.g', '.proto', '.asm', '.s'}:
        return 'source'
    if ext == '.m4':
        return 'autoconf'
    return None


def _should_sniff_artifact_file(rel_path: str, filename: str) -> bool:
    if _is_excluded_artifact_path(rel_path):
        return False
    rel_l = rel_path.replace(os.sep, '/').lower()
    basename = os.path.basename(rel_l)
    if _CONTENT_SNIFF_BASENAME_RE.match(basename):
        return True
    return bool(_CONTENT_SNIFF_PATH_RE.search(rel_l)) and not os.path.splitext(basename)[1]


def _artifact_build_system_for_content(content: str) -> str | None:
    if _AUTOCONF_CONTENT_RE.search(content):
        return 'autoconf'
    if _CONFIGURE_CONTENT_RE.search(content):
        return 'configure_script'
    if _SOURCE_PROBE_CONTENT_RE.search(content):
        return 'source'
    if _SCRIPT_CONTENT_RE.search(content):
        return 'script'
    return None


def _read_artifact_sample(filepath: str) -> str:
    try:
        with open(filepath, errors='replace') as f:
            return f.read(_ARTIFACT_SNIFF_BYTES)
    except Exception:
        return ''


def _artifact_only_build_system(rel_path: str, filename: str, filepath: str) -> str | None:
    if _is_excluded_artifact_path(rel_path):
        return None
    if _is_non_probe_build_artifact_path(rel_path):
        return None
    if _is_package_list_artifact_file(rel_path):
        return 'package_list'
    if _is_build_script_artifact_file(rel_path):
        return 'script'
    path_build_system = _artifact_build_system_for_path(rel_path, filename)
    if path_build_system is not None:
        return path_build_system
    if _is_source_artifact_file(filename):
        return 'source'
    if not _should_sniff_artifact_file(rel_path, filename):
        return None
    if os.path.getsize(filepath) > _MAX_SOURCE_ARTIFACT_BYTES:
        return None
    return _artifact_build_system_for_content(_read_artifact_sample(filepath))


def _in_m4_macro_body(lines: list[str], line_num: int) -> bool:
    idx = max(0, line_num - 1)
    defun_idx = None
    for i in range(idx, max(-1, idx - 80), -1):
        if re.search(r'\bAC_DEFUN(?:_ONCE)?\s*\(', lines[i]):
            defun_idx = i
            break
    if defun_idx is None:
        return False
    if defun_idx == idx:
        return True
    bracket_depth = 0
    for line in lines[defun_idx:idx]:
        bracket_depth += line.count('[')
        bracket_depth -= line.count(']')
    return bracket_depth > 0


def _safe_literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _flatten_python_values(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_python_values(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_python_values(item))
        return result
    return []


def _is_interesting_extra(name: str) -> bool:
    parts = [p for p in re.split(r'[^a-z0-9]+', name.lower()) if p]
    return any(p in _PYTHON_INTERESTING_EXTRAS for p in parts)


def _find_line_for_text(lines: list[str], needle: str) -> int:
    if not needle:
        return 1
    lower = needle.lower()
    for idx, line in enumerate(lines, 1):
        if lower in line.lower():
            return idx
    return 1


def _python_match(filepath: str, lines: list[str], line_num: int, rule_type: str, text: str) -> RuleMatch:
    text = ' '.join(str(text).split())
    return RuleMatch(
        file_path=filepath,
        line_num=max(1, line_num),
        rule_type=rule_type,
        matched_text=text[:1000],
        context='',
        extracted_name=text,
        packages=[],
    )


def _add_python_match(
    matches: list[RuleMatch],
    seen: set[tuple[str, str]],
    filepath: str,
    lines: list[str],
    line_num: int,
    rule_type: str,
    text: str,
):
    text = ' '.join(str(text).split())
    if not text:
        return
    key = (rule_type, text.lower())
    if key in seen:
        return
    seen.add(key)
    matches.append(_python_match(filepath, lines, line_num, rule_type, text))


def _parse_pyproject(filepath: str, content: str, lines: list[str]) -> list[RuleMatch]:
    if tomllib is None:
        return []
    try:
        data = tomllib.loads(content)
    except Exception:
        return []

    matches = []
    seen = set()
    build_system = data.get('build-system') if isinstance(data, dict) else None
    if isinstance(build_system, dict):
        requires = _flatten_python_values(build_system.get('requires'))
        if requires:
            text = 'build-system.requires = ' + ', '.join(requires)
            _add_python_match(
                matches, seen, filepath, lines,
                _find_line_for_text(lines, 'requires'), 'python:pyproject-build-requires', text,
            )
        backend = build_system.get('build-backend')
        if isinstance(backend, str) and backend.strip():
            backend_text = backend.strip()
            for prefix, hint in _PYPROJECT_BACKEND_HINTS.items():
                if backend_text.startswith(prefix):
                    backend_text = f'{backend_text} ({hint})'
                    break
            _add_python_match(
                matches, seen, filepath, lines,
                _find_line_for_text(lines, 'build-backend'), 'python:pyproject-build-backend', backend_text,
            )

    project = data.get('project') if isinstance(data, dict) else None
    if isinstance(project, dict):
        deps = _flatten_python_values(project.get('dependencies'))
        if deps:
            _add_python_match(
                matches, seen, filepath, lines,
                _find_line_for_text(lines, 'dependencies'), 'python:pyproject-dependencies',
                'project.dependencies = ' + ', '.join(deps),
            )
        optional = project.get('optional-dependencies')
        if isinstance(optional, dict):
            for extra, deps_value in optional.items():
                if not _is_interesting_extra(str(extra)):
                    continue
                deps = _flatten_python_values(deps_value)
                if deps:
                    _add_python_match(
                        matches, seen, filepath, lines,
                        _find_line_for_text(lines, str(extra)), 'python:pyproject-optional-dependencies',
                        f'project.optional-dependencies.{extra} = ' + ', '.join(deps),
                    )

    tool = data.get('tool') if isinstance(data, dict) else None
    if isinstance(tool, dict):
        for key in tool:
            if str(key).lower().startswith('pytest'):
                _add_python_match(
                    matches, seen, filepath, lines,
                    _find_line_for_text(lines, str(key)), 'python:pyproject-tool',
                    f'tool.{key}',
                )
    return matches


def _parse_setup_cfg_or_tox(filepath: str, content: str, lines: list[str]) -> list[RuleMatch]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        parser.read_string(content)
    except configparser.Error:
        return []

    matches = []
    seen = set()
    if parser.has_section('options'):
        for opt in ('setup_requires', 'install_requires', 'tests_require'):
            if parser.has_option('options', opt):
                value = parser.get('options', opt)
                _add_python_match(
                    matches, seen, filepath, lines,
                    _find_line_for_text(lines, opt), f'python:setupcfg-{opt}',
                    f'{opt} = {value}',
                )

    if parser.has_section('options.extras_require'):
        for extra, value in parser.items('options.extras_require'):
            if _is_interesting_extra(extra):
                _add_python_match(
                    matches, seen, filepath, lines,
                    _find_line_for_text(lines, extra), 'python:setupcfg-extras-require',
                    f'options.extras_require.{extra} = {value}',
                )

    if parser.has_section('tox') and parser.has_option('tox', 'envlist'):
        value = parser.get('tox', 'envlist')
        _add_python_match(
            matches, seen, filepath, lines,
            _find_line_for_text(lines, 'envlist'), 'python:tox-envlist',
            f'tox.envlist = {value}',
        )

    for section in parser.sections():
        section_l = section.lower()
        if section_l in ('pytest', 'tool:pytest') or section_l.startswith('tool:pytest'):
            _add_python_match(
                matches, seen, filepath, lines,
                _find_line_for_text(lines, section), 'python:setupcfg-tool',
                section,
            )
        if section_l.startswith('testenv') and parser.has_option(section, 'deps'):
            value = parser.get(section, 'deps')
            _add_python_match(
                matches, seen, filepath, lines,
                _find_line_for_text(lines, section), 'python:tox-deps',
                f'{section}.deps = {value}',
            )
    return matches


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f'{prefix}.{node.attr}' if prefix else node.attr
    return ''


def _string_list_from_ast(node: ast.AST) -> list[str]:
    value = _safe_literal(node)
    return [item for item in _flatten_python_values(value) if isinstance(item, str)]


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _target_name(node.value)
        return f'{prefix}.{node.attr}' if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _target_name(node.value)
    return ''


def _is_setup_link_var(name: str) -> bool:
    return name.split('.')[-1] in {
        'libraries', 'extra_link_args', 'extra_compile_args',
        'link_args', 'compile_args',
    }


def _parse_setup_py(filepath: str, content: str, lines: list[str]) -> list[RuleMatch]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    matches = []
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                if name.lower() in _PYTHON_IMPORTS or name.split('.')[0].lower() in _PYTHON_IMPORTS:
                    _add_python_match(
                        matches, seen, filepath, lines, node.lineno,
                        'python:setup-py-import', name,
                    )

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            for target in targets:
                name = _target_name(target)
                if _is_setup_link_var(name):
                    values = _string_list_from_ast(value_node)
                    if values:
                        _add_python_match(
                            matches, seen, filepath, lines, getattr(value_node, 'lineno', node.lineno),
                            'python:setup-py-link-vars', f'{name} = ' + ', '.join(values),
                        )

        if isinstance(node, ast.AugAssign):
            name = _target_name(node.target)
            if _is_setup_link_var(name):
                values = _string_list_from_ast(node.value)
                if values:
                    _add_python_match(
                        matches, seen, filepath, lines, getattr(node.value, 'lineno', node.lineno),
                        'python:setup-py-link-vars', f'{name} += ' + ', '.join(values),
                    )

        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name.endswith(('append', 'extend')):
                receiver = call_name.rsplit('.', 1)[0]
                if _is_setup_link_var(receiver) and node.args:
                    values = []
                    for arg in node.args:
                        values.extend(_string_list_from_ast(arg))
                    if values:
                        _add_python_match(
                            matches, seen, filepath, lines, node.lineno,
                            'python:setup-py-link-vars', f'{receiver} append/extend ' + ', '.join(values),
                        )

        if not isinstance(node, ast.Call):
            continue
        call = _call_name(node.func)
        if call.endswith('setup'):
            for kw in node.keywords:
                if kw.arg in ('setup_requires', 'install_requires', 'tests_require'):
                    values = _flatten_python_values(_safe_literal(kw.value))
                    if values:
                        _add_python_match(
                            matches, seen, filepath, lines, getattr(kw.value, 'lineno', node.lineno),
                            f'python:setup-py-{kw.arg}', f'{kw.arg} = ' + ', '.join(values),
                        )
                elif kw.arg == 'extras_require':
                    value = _safe_literal(kw.value)
                    if isinstance(value, dict):
                        for extra, deps_value in value.items():
                            if _is_interesting_extra(str(extra)):
                                deps = _flatten_python_values(deps_value)
                                if deps:
                                    _add_python_match(
                                        matches, seen, filepath, lines, getattr(kw.value, 'lineno', node.lineno),
                                        'python:setup-py-extras-require',
                                        f'extras_require.{extra} = ' + ', '.join(deps),
                                    )
        elif call.endswith('Extension'):
            for kw in node.keywords:
                if kw.arg in ('libraries', 'extra_link_args', 'extra_compile_args'):
                    libs = _flatten_python_values(_safe_literal(kw.value))
                    if libs:
                        _add_python_match(
                            matches, seen, filepath, lines, getattr(kw.value, 'lineno', node.lineno),
                            'python:setup-py-extension-libraries',
                            f'Extension {kw.arg} = ' + ', '.join(libs),
                        )
        elif call.endswith('cythonize'):
            _add_python_match(
                matches, seen, filepath, lines, node.lineno,
                'python:setup-py-cythonize', call,
            )
        elif call in ('numpy.get_include', 'pybind11.get_include') or call.endswith('pkgconfig.parse'):
            _add_python_match(
                matches, seen, filepath, lines, node.lineno,
                'python:setup-py-build-helper', call,
            )
    return matches


def _parse_python_config(filepath: str, content: str, lines: list[str]) -> list[RuleMatch]:
    basename = os.path.basename(filepath).lower()
    if basename == 'pyproject.toml':
        return _parse_pyproject(filepath, content, lines)
    if basename in ('setup.cfg', 'tox.ini', 'pytest.ini'):
        return _parse_setup_cfg_or_tox(filepath, content, lines)
    if basename == 'setup.py':
        return _parse_setup_py(filepath, content, lines)
    if basename == 'conf.py':
        return _parse_python_import_config(filepath, content, lines, 'python:sphinx-conf')
    if basename.startswith('requirements') and basename.endswith('.txt'):
        return _parse_requirements_txt(filepath, content, lines)
    return []


def _parse_requirements_txt(filepath: str, content: str, lines: list[str]) -> list[RuleMatch]:
    matches = []
    seen = set()
    deps = []
    first_line = 1
    for idx, line in enumerate(lines, 1):
        item = line.strip()
        if not item or item.startswith('#') or item.startswith(('-', 'http://', 'https://')):
            continue
        item = item.split('#', 1)[0].strip()
        if item:
            if not deps:
                first_line = idx
            deps.append(item)
    if deps:
        _add_python_match(
            matches, seen, filepath, lines, first_line,
            'python:requirements', 'requirements = ' + ', '.join(deps),
        )
    return matches


def _parse_python_import_config(
    filepath: str,
    content: str,
    lines: list[str],
    rule_type: str,
) -> list[RuleMatch]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    matches = []
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                if name.lower() in _PYTHON_IMPORTS or name.split('.')[0].lower() in _PYTHON_IMPORTS:
                    _add_python_match(matches, seen, filepath, lines, node.lineno, rule_type, name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    'extensions', 'needs_sphinx', 'html_theme',
                }:
                    values = _flatten_python_values(_safe_literal(node.value))
                    if values:
                        _add_python_match(
                            matches, seen, filepath, lines, node.lineno,
                            rule_type, f'{target.id} = ' + ', '.join(values),
                        )
    return matches


def _parse_cargo_toml(filepath: str, content: str, lines: list[str]) -> list[RuleMatch]:
    if tomllib is None:
        return []
    try:
        data = tomllib.loads(content)
    except Exception:
        return []

    matches = []
    seen = set()
    for section in ('build-dependencies', 'dev-dependencies', 'dependencies'):
        deps = data.get(section)
        if isinstance(deps, dict) and deps:
            text = f'{section} = ' + ', '.join(str(name) for name in sorted(deps))
            _add_python_match(
                matches, seen, filepath, lines,
                _find_line_for_text(lines, f'[{section}]'), f'cargo:{section}', text,
            )

    package = data.get('package')
    if isinstance(package, dict) and package.get('name'):
        _add_python_match(
            matches, seen, filepath, lines,
            _find_line_for_text(lines, 'name'), 'cargo:package-name',
            f'package.name = {package.get("name")}',
        )
    if isinstance(package, dict) and package.get('build'):
        _add_python_match(
            matches, seen, filepath, lines,
            _find_line_for_text(lines, 'build'), 'cargo:build-script',
            f'build = {package.get("build")}',
        )
    return matches


def _parse_opam_file(filepath: str, content: str, lines: list[str]) -> list[RuleMatch]:
    matches = []
    seen = set()
    match = re.search(r'(?ms)^\s*depends:\s*\[(.*?)\]', content)
    if match:
        deps = re.findall(r'"([^"]+)"', match.group(1))
        if deps:
            _add_python_match(
                matches, seen, filepath, lines,
                _find_line_for_text(lines, 'depends:'), 'ocaml:opam-depends',
                'depends = ' + ', '.join(deps),
            )
    return matches



def _dependency_list_matches(filepath: str, content: str, build_system: str) -> list[RuleMatch]:
    matches: list[RuleMatch] = []
    for statement in find_dependency_lists(content, build_system):
        matches.append(RuleMatch(
            filepath,
            statement.line_num,
            f'{statement.rule_type}:{statement.confidence}',
            statement.raw[:1000],
            '',
            statement.var_name,
        ))
    return matches

def _structured_matches(filepath: str, content: str, lines: list[str], build_system: str) -> list[RuleMatch]:
    basename = os.path.basename(filepath).lower()
    if build_system == 'python':
        return _parse_python_config(filepath, content, lines)
    if build_system == 'cargo' and basename == 'cargo.toml':
        return _parse_cargo_toml(filepath, content, lines)
    if build_system == 'ocaml' and basename.endswith('.opam'):
        return _parse_opam_file(filepath, content, lines)
    return []


def _debianize_python_req(req: str) -> str | None:
    name = req.strip().strip('"\'')
    if name in {'requirements', '=', 'build-system.requires', 'project.dependencies'}:
        return None
    if name.startswith(('python:', 'tool.', 'options.', 'extras_require.')):
        return None
    name = re.split(r'\s*(?:[<>=!~]=?|;|\[)', name, 1)[0].strip()
    if not name:
        return None
    norm = name.lower().replace('_', '-')
    if norm.startswith('sphinx-ext-'):
        norm = 'sphinx'
    if norm in {
        'requirements', 'build-system.requires', 'project.dependencies',
        'optional-dependencies', 'dependencies', 'requires', 'extensions',
        'html-theme', 'needs-sphinx',
    }:
        return None
    # Only entries whose Debian package name diverges from the default `python3-{norm}`
    # rule. All other names are auto-generated and validated downstream by valid_pkgs.
    special_rules = {
        'cython': 'cython3',
        'furo': 'python3-docs-theme',
        'meson-python': 'python3-mesonpy',
        'python-docs-theme': 'python3-docs-theme',
        'sphinxcontrib-images': 'python3-sphinxcontrib.images',
        'sphinxcontrib-jquery': 'python3-sphinxcontrib.jquery',
        'sphinxcontrib-moderncmakedomain': 'python3-sphinxcontrib.moderncmakedomain',
        'sphinxcontrib-qthelp': 'python3-sphinxcontrib.qthelp',
    }
    if norm in special_rules:
        return special_rules[norm]
    if re.fullmatch(r'[a-z0-9][a-z0-9.+-]*', norm):
        candidate = f'python3-{norm}'
        if _SUSPICIOUS_PKG_RE.match(candidate):
            return None
        return candidate
    return None


def _debianize_go_module(module: str) -> str | None:
    module = module.strip().strip('"\'')
    module = module.split()[0] if module.split() else ''
    if not module or module.startswith(('module', 'require', 'replace', '//')):
        return None
    if module.endswith('/v2') or re.search(r'/v\d+$', module):
        module = module.rsplit('/', 1)[0]
    parts = module.lower().split('/')
    if len(parts) < 3:
        return None
    host = parts[0]
    aliases = {
        'github.com/godbus/dbus': 'golang-dbus-dev',
        'github.com/stretchr/testify': 'golang-github-stretchr-testify-dev',
        'github.com/urfave/cli': 'golang-github-urfave-cli-dev',
        'github.com/urfave/cli/v2': 'golang-github-urfave-cli-v2-dev',
    }
    if module.lower() in aliases:
        return aliases[module.lower()]
    if host == 'github.com':
        return 'golang-github-' + '-'.join(parts[1:]).replace('_', '-') + '-dev'
    if host == 'golang.org' and parts[1] == 'x':
        return 'golang-golang-x-' + '-'.join(parts[2:]).replace('_', '-') + '-dev'
    return None


def _debian_token(name: str) -> str:
    token = name.strip().strip('"\'').lower()
    token = re.split(r'\s*(?:[<>=!~]=?|;|\[)', token, 1)[0].strip()
    token = token.replace('_', '-').replace('.', '-')
    token = re.sub(r'[^a-z0-9+@/-]+', '-', token).strip('-')
    return token


def _npm_debian_stem(name: str) -> str:
    token = _debian_token(name)
    if token.startswith('@'):
        token = token[1:].replace('/', '-')
    else:
        token = token.replace('/', '-')
    return token


def _debianize_node_packages(name: str) -> set[str]:
    stem = _npm_debian_stem(name)
    if not stem or stem.startswith('$'):
        return set()
    if stem in {'dependencies', 'devdependencies', 'optionaldependencies', 'peerdependencies'}:
        return set()
    return {stem} if stem.startswith('node-') else {f'node-{stem}'}


def _debianize_ruby_gem_packages(name: str) -> set[str]:
    stem = _debian_token(name)
    if not stem:
        return set()
    return {stem} if stem.startswith('ruby-') else {f'ruby-{stem}'}


def _debianize_cargo_crate_packages(name: str) -> set[str]:
    stem = _debian_token(name).replace('/', '-')
    if not stem:
        return set()
    candidates = {f'librust-{stem}-dev'}
    if stem.endswith('-sys'):
        native = stem[:-len('-sys')]
        if native:
            candidates.add(f'lib{native}-dev')
    return candidates


def _python_build_backend_package(text: str, file_path: str) -> str | None:
    basename = os.path.basename(file_path)
    lower = text.lower()
    if basename == 'pyproject.toml' and (
        'build-system.requires' in lower
        or 'build-backend' in lower
        or 'setuptools.build_meta' in lower
        or 'hatchling' in lower
        or 'flit_core' in lower
        or 'poetry.core' in lower
        or 'mesonpy' in lower
        or 'scikit_build_core' in lower
    ):
        return 'pybuild-plugin-pyproject'
    return None


def _python_extension_packages(text: str, file_path: str) -> set[str]:
    basename = os.path.basename(file_path)
    lower = text.lower()
    pkgs: set[str] = set()
    has_extension = bool(re.search(r'(?i)\b(?:Extension|cythonize|build_ext|Python\.h|python3?-config)\b', text))
    if has_extension or basename == 'setup.py':
        pkgs.add('python3-all-dev')
    if re.search(r'(?i)\b(?:numpy|get_include|numpy/arrayobject\.h|np\.get_include)\b', text):
        pkgs.add('python3-numpy')
    if 'cython' in lower or 'cythonize' in lower:
        pkgs.add('cython3')
    return pkgs


def _direct_packages_from_text(
    build_system: str,
    rule_type: str,
    matched_text: str,
    extracted_name: str,
    file_path: str = '',
) -> set[str]:
    """Map only high-confidence generic evidence directly to Debian packages.

    Ecosystem-specific provider inference is intentionally left to the LLM so a
    multi-provider match (for example Qt/XCB/Sphinx components) is not collapsed
    to a partial direct mapping.
    """
    text = f'{matched_text}\n{extracted_name}'
    extracted = extracted_name.strip().lower()
    pkgs = set()

    if (
        build_system == 'cmake'
        and rule_type in ('cmake:find_package_call', 'cmake:find_package')
        and re.search(r'\bECM\b', text)
    ):
        pkgs.add('extra-cmake-modules')

    # Covered by artifact/header paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)\b(systemd|libsystemd|journald)\b', text):
    #     pkgs.add('systemd-dev')

    # Covered by artifact/header paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)(?:-lz\b|\bzlib\b|zlib\.h)', text) or (
    #     extracted == 'z'
    #     and rule_type in {
    #         'ac:AC_CHECK_LIB',
    #         'cmake:check_library_exists',
    #         'configure:-l',
    #         'make:-l',
    #         'python:setup-py-extension-libraries',
    #         'ruby:have_library',
    #     }
    # ):
    #     pkgs.add('zlib1g-dev')

    # Covered by artifact/header paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)(?:-l(?:ssl|crypto)\b|\bopenssl\b|openssl/[A-Za-z0-9_./+-]+\.h)', text):
    #     pkgs.add('libssl-dev')

    # Covered by artifact/header paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)(?:-lsqlite3\b|\bsqlite3\b|sqlite3\.h)', text):
    #     pkgs.add('libsqlite3-dev')

    # Covered by build/language artifact paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)\b(Python3|python3|python)\b', text):
    #     pkgs.add('python3')

    # Covered by tool artifact paths; keep disabled while measuring direct_rule overlap.
    # if rule_type.startswith('doxygen:'):
    #     pkgs.add('doxygen')

    # Covered by tool artifact paths; keep disabled while measuring direct_rule overlap.
    # if (
    #     rule_type in {'doxygen:have-dot', 'doxygen:dot-path'}
    #     or re.search(r'(?i)\bgraphviz\b', text)
    #     or re.search(r'(?i)(?:^|[^A-Za-z0-9_+.-])dot(?:[^A-Za-z0-9_+.-]|$)', text)
    # ):
    #     pkgs.add('graphviz')

    # Covered by XML artifact paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)(?:docbook\.(?:dtd|xsd)|docbook-xml)', text):
    #     pkgs.add('docbook-xml')

    # Covered by XML artifact paths; keep disabled while measuring direct_rule overlap.
    # if rule_type in {'docbook:doctype', 'docbook:xml'}:
    #     pkgs.add('docbook-xml')

    if re.search(r'(?i)(?:/usr/share/xml/docbook/stylesheet|docbook\.xsl|xsltproc)', text):
        pkgs.add('docbook-xsl')
        # Covered by executable artifact paths; keep disabled while measuring direct_rule overlap.
        # pkgs.add('xsltproc')

    if rule_type == 'docbook:xsl':
        pkgs.add('docbook-xsl')
        # Covered by executable artifact paths; keep disabled while measuring direct_rule overlap.
        # pkgs.add('xsltproc')

    if re.search(r'(?i)\bpytest\b', text):
        pkgs.add('python3-pytest')

    if (
        build_system == 'python'
        and (re.search(r'(?i)\bsetuptools\b', text) or os.path.basename(file_path) in {'setup.py', 'setup.cfg'})
    ):
        pkgs.add('python3-setuptools')

    if re.search(r'(?i)\bpython3-all\b', text) or (
        rule_type == 'python:tox-envlist'
        and re.search(r'(?i)\bpy3(?:[0-9]{1,2})?\b.*\bpy3(?:[0-9]{1,2})?\b', text)
    ):
        pkgs.add('python3-all')
        pkgs.add('python3-all-dev')

    # Covered by header/tool artifact paths; keep disabled while measuring direct_rule overlap.
    # if (
    #     re.search(r'(?i)\bPython\.h\b', text)
    #     or re.search(r'(?i)\bpython3-config\b[^\n]*(?:--includes|--ldflags)', text)
    #     or re.search(r'(?i)\bPython3?\b[^\n]*(?:Development(?:\.(?:Module|Embed))?|Embed)\b', text)
    # ):
    #     pkgs.add('libpython3-dev')
    #     pkgs.add('python3-dev')

    # Covered by library/header artifact paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)\b(?:jackd2|libjack-jackd2-dev)\b', text):
    #     pkgs.add('libjack-jackd2-dev')
    # if re.search(r'(?i)(?:\bjack\b|\blibjack\b|-ljack\b|libjack-dev)', text):
    #     pkgs.add('libjack-dev')

    if (
        rule_type == 'make:pgxs'
        or re.search(r'(?i)\b(?:PGXS|USE_PGXS|pg_config\s+--pgxs)\b', text)
    ):
        pkgs.add('postgresql-server-dev-all')
        pkgs.add('postgresql-all')

    # Covered by tool artifact paths; keep disabled while measuring direct_rule overlap.
    # if re.search(r'(?i)\b(?:BISON|bison|YACC|yacc|AC_PROG_YACC)\b', text):
    #     pkgs.add('bison')
    if re.search(r'(?i)\b(?:FLEX|flex|LEX|lex|AC_PROG_LEX|AM_PROG_LEX)\b', text):
        pkgs.add('flex')
    if (
        rule_type in {
            'cmake:enable_fortran', 'cmake:project_fortran', 'cmake:fortran-compiler',
            'ac:fortran', 'make:fortran-var', 'make:fortran-rule',
            'meson:fortran-language',
        }
        or re.search(r'(?i)\b(?:AC_PROG_FC|AC_PROG_F77|AC_FC_[A-Z0-9_]+|Fortran|CMAKE_Fortran_COMPILER|gfortran|add_languages\s*\([^)]*[\'"]fortran[\'"])\b', text)
    ):
        pkgs.add('gfortran')

    if re.search(r'(?i)\b(?:IT_PROG_INTLTOOL|intltoolize|intltool-(?:extract|update|merge))\b', text):
        pkgs.add('intltool')

    if re.search(r'(?i)\b(?:gtkdocize|gtkdoc-(?:scan|mkdb|fixxref)|GTK_DOC_CHECK|gnome\.gtkdoc)\b', text) or rule_type in {'meson:gnome_gtkdoc', 'meson:option_gtkdoc'}:
        pkgs.add('gtk-doc-tools')

    if re.search(r'(?i)\b(?:g-ir-scanner|g-ir-compiler|GOBJECT_INTROSPECTION_CHECK|gobject-introspection-1\.0|gnome\.generate_gir)\b', text) or rule_type in {'meson:gnome_generate_gir', 'meson:option_introspection'}:
        pkgs.add('gobject-introspection')
        pkgs.add('libgirepository1.0-dev')

    if re.search(r'(?i)\b(?:VAPIGEN_CHECK|VALAC|VAPIGEN|valac|vapigen|gnome\.generate_vapi)\b', text) or rule_type in {'meson:gnome_generate_vapi', 'meson:option_vapi'}:
        pkgs.add('valac')

    if re.search(r'(?i)\b(?:AM_GNU_GETTEXT|find_package\s*\(\s*Gettext|Gettext|msgfmt|xgettext|msgmerge)\b', text):
        pkgs.add('gettext')

    if re.search(r'(?i)\b(?:Qt5LinguistTools|Qt5_LRELEASE_EXECUTABLE|Qt5_LUPDATE_EXECUTABLE)\b', text):
        pkgs.add('qttools5-dev')
        pkgs.add('qttools5-dev-tools')
    elif re.search(r'(?i)\b(?:Qt6LinguistTools|Qt6_LRELEASE_EXECUTABLE|Qt6_LUPDATE_EXECUTABLE)\b', text):
        pkgs.add('qt6-tools-dev')
        pkgs.add('qt6-tools-dev-tools')
    elif re.search(r'(?i)\b(?:LinguistTools|lrelease|lupdate)\b', text) or rule_type == 'qmake:TRANSLATIONS':
        if re.search(r'(?i)\b(?:QT_VERSION_MAJOR|Qt5|qt5|/qt5/)\b', text):
            pkgs.add('qttools5-dev')
            pkgs.add('qttools5-dev-tools')
        elif re.search(r'(?i)\b(?:Qt6|qt6|/qt6/)\b', text):
            pkgs.add('qt6-tools-dev')
            pkgs.add('qt6-tools-dev-tools')
        else:
            pkgs.add('qttools5-dev-tools')

    if build_system == 'qmake':
        if re.search(r'(?i)\bQT\s*[+*]?=.*\bopengl\b', text):
            pkgs.add('libqt5opengl5-dev')
            pkgs.add('qt5-qmake')
        if re.search(r'(?i)\bQT\s*[+*]?=', text):
            pkgs.add('qt5-qmake')
        if re.search(r'(?i)\bQT\s*[+*]?=.*\bdeclarative\b', text):
            pkgs.add('qtdeclarative5-dev')

    if re.search(r'(?i)\b(?:docbook2man|docbook-utils)\b', text):
        pkgs.add('docbook-utils')

    if re.search(r'(?i)\bdocbook2x-man\b', text):
        pkgs.add('docbook2x')

    if re.search(r'(?i)\bdocbook-to-man\b', text):
        pkgs.add('docbook-to-man')

    if build_system == 'ocaml' and re.search(r'(?i)\bdune(?:-configurator)?\b', text):
        pkgs.add('ocaml-dune')
        if re.search(r'(?i)\bdune-configurator\b', text):
            pkgs.add('libdune-ocaml-dev')

    if build_system == 'python':
        backend_pkg = _python_build_backend_package(text, file_path)
        if backend_pkg:
            pkgs.add(backend_pkg)
        pkgs.update(_python_extension_packages(text, file_path))

    return pkgs


def _direct_packages_from_match(build_system: str, match: RuleMatch) -> set[str]:
    pkgs = _direct_packages_from_text(
        build_system,
        match.rule_type,
        match.matched_text,
        match.extracted_name,
        match.file_path,
    )
    if build_system == 'python':
        for part in re.split(r'[,;\s]+', match.matched_text):
            pkg = _debianize_python_req(part)
            if pkg:
                pkgs.add(pkg)
        pkgs.update(_python_extension_packages(match.matched_text, match.file_path))
    elif build_system == 'go' and match.rule_type.startswith('go:require'):
        for token in re.findall(r'(?:github\.com|golang\.org)/[A-Za-z0-9_./-]+', match.matched_text):
            pkg = _debianize_go_module(token)
            if pkg:
                pkgs.add(pkg)
    elif build_system == 'cargo' and match.rule_type.startswith(('cargo:dependencies', 'cargo:build-dependencies', 'cargo:dev-dependencies', 'cargo:package-name')):
        cargo_body = match.matched_text.split('=', 1)[1] if '=' in match.matched_text else match.matched_text
        for token in re.findall(r'[A-Za-z0-9_.+-]+', cargo_body):
            if token in {'build', 'build-dependencies', 'dependencies', 'dev-dependencies', 'package', 'name', 'package-name'}:
                continue
            pkgs.update(_debianize_cargo_crate_packages(token))
    elif build_system == 'node':
        for token in re.findall(r'"(@?[^"@/][^"]*?|@[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"\s*:', match.matched_text):
            pkgs.update(_debianize_node_packages(token))
        if match.rule_type == 'node:require':
            pkgs.update(_debianize_node_packages(match.extracted_name))
        if re.search(r'\bnode-gyp\b', match.matched_text, re.IGNORECASE):
            pkgs.add('node-gyp')
    elif build_system == 'ruby':
        if match.rule_type in {'ruby:gemspec-dependency', 'ruby:require'}:
            pkgs.update(_debianize_ruby_gem_packages(match.extracted_name))
        if re.search(r'\bRake::(?:TestTask|ExtensionTask)\b|\brake\b', match.matched_text, re.IGNORECASE):
            pkgs.add('rake')
        if re.search(r'\bmkmf\b|extconf\.rb|create_makefile|ExtensionTask', match.matched_text, re.IGNORECASE):
            pkgs.add('ruby-dev')
            pkgs.add('gem2deb')
    return pkgs


def _attach_match_artifacts(
    matches: list[RuleMatch],
    build_variables: dict[str, str] | None = None,
) -> None:
    for match in matches:
        artifacts = list(getattr(match, 'artifacts', []) or [])
        artifacts.extend(extract_artifacts([match], build_variables=build_variables))
        artifacts.extend(_tool_artifacts_from_text(f'{match.matched_text}\n{match.extracted_name}'))
        match.artifacts = _dedupe_artifacts(artifacts)


def _is_direct_only_match(build_system: str, match: RuleMatch) -> bool:
    if _is_generic_autoconf_macro_file(match.file_path):
        match.packages = []
        return True
    if _is_macro_library_file(match.file_path):
        return True
    if match.rule_type in _DIRECT_RULE_TYPES and match.packages:
        return not _has_unexplained_evidence(match)
    if match.rule_type.startswith('doxygen:') and match.packages:
        return not _has_unexplained_evidence(match)
    if build_system == 'python' and match.packages:
        if 'extension-libraries' in match.rule_type or 'link' in match.rule_type:
            return False
        return not _has_unexplained_evidence(match)
    if build_system == 'ocaml' and match.packages:
        return not _has_unexplained_evidence(match)
    if match.rule_type in {'qmake:QT', 'qmake:PKGCONFIG', 'qmake:LIBS'}:
        return False
    if match.packages and _package_tokens_from_text(match.matched_text) <= set(match.packages):
        return True
    return False


# Secondary build tools that are only emitted when their text appears in project source.
# Primary build-system front-ends (cmake, meson, autoconf, ...) are emitted unconditionally
# whenever the corresponding build file is detected; secondary tools below require explicit
# textual evidence to avoid spraying toolchain packages into every project's prediction.
_BUILD_DRIVER_EVIDENCE: dict[str, re.Pattern] = {
    'ninja-build': re.compile(r'\b(?:ninja|generate_ninja|GenerationRules)\b', re.IGNORECASE),
    'automake':    re.compile(r'\b(?:automake|AM_INIT_AUTOMAKE|AC_CONFIG_MACRO_DIRS|AM_PROG_LEX)\b'),
    'libtool':     re.compile(r'\b(?:libtoolize|LT_INIT|AC_PROG_LIBTOOL|LT_PREREQ)\b'),
    'autoconf-archive': re.compile(r'\b(?:AX_(?:CHECK|INIT|PROG|require)|aclocal)\b'),
    'autotools-dev': re.compile(r'\b(?:autoreconf|dh[-_]autoreconf)\b'),
    'pkgconf':     re.compile(r'\bpkg-config\b|\bpkgconf\b', re.IGNORECASE),
    'pkg-config':  re.compile(r'\bpkg-config\b|\bpkgconf\b', re.IGNORECASE),
    'python3':     re.compile(r'\b(?:python3|find_package\s*\(\s*Python3|python_requires)\b', re.IGNORECASE),
    'python3-setuptools': re.compile(r'\b(?:setuptools|setup\.py|setup\.cfg|pyproject\.toml)\b', re.IGNORECASE),
    'python3-pytest': re.compile(r'\b(?:pytest|tox)\b', re.IGNORECASE),
    'python3-all': re.compile(r'\bpython3-all\b', re.IGNORECASE),
    'python3-all-dev': re.compile(r'\bpython3-all-dev\b', re.IGNORECASE),
    'python3-dev': re.compile(r'\bpython3-dev\b', re.IGNORECASE),
    'doxygen':     re.compile(r'\b(?:doxygen|find_package\s*\(\s*Doxygen|Doxyfile)\b', re.IGNORECASE),
}

# Primary build-system front-end packages — emitted on build-file detection alone.
_BUILD_DRIVER_PRIMARY: dict[str, str] = {
    'cmake':    'cmake',
    'meson':    'meson',
    'cargo':    'cargo',
    'autoconf': 'autoconf',
    'automake': 'automake',
    'scons':    'scons',
    'doxygen':  'doxygen',
    'go':       'golang-any',
}


def _auto_packages_from_build_file(
    build_system: str,
    filename: str | None = None,
    root_level: bool = False,
    file_content: str | None = None,
) -> set[str]:
    """Map discovered build-system files to their required driver packages.

    Emits the primary build-system front-end unconditionally; secondary toolchain
    packages (ninja-build, automake, libtool, autoconf-archive, autotools-dev, pkgconf,
    python3-*, doxygen) are emitted only when textual evidence is present in
    `file_content` — this prevents spraying the full autotools chain into every project
    that happens to contain a single Makefile.am.
    """
    pkgs: set[str] = set()
    primary = _BUILD_DRIVER_PRIMARY.get(build_system)
    if primary:
        pkgs.add(primary)

    # Cargo is special — it always pulls rustc and libstd-rust-dev because the cargo
    # toolchain cannot function without them.
    if build_system == 'cargo':
        pkgs.update({'rustc', 'libstd-rust-dev'})

    # Meson projects almost always need ninja (its default backend), so emit ninja
    # unconditionally for meson. CMake projects may use Make or Ninja; we still emit
    # ninja unconditionally because it's the Debian-packaged default.
    if build_system in ('cmake', 'meson'):
        pkgs.add('ninja-build')
        if build_system == 'meson':
            pkgs.add('pkgconf')

    # qmake → Qt resolved via component path (not here)
    if build_system == 'qmake':
        return set()

    # Pure ecosystems: emit nothing unconditionally beyond what's above. Specific
    # python/postgresql/ruby/etc tools are gated on evidence below.
    if build_system in {'postgresql', 'ruby', 'ocaml', 'java', 'dlang', 'node'}:
        return pkgs

    # Python: emit python3 unconditionally (it's the language runtime), plus
    # setup-tools when a setup file is detected.
    if build_system == 'python':
        pkgs.add('python3')
        if filename in {'setup.py', 'setup.cfg', 'pyproject.toml'}:
            pkgs.add('python3-setuptools')
        if filename == 'pytest.ini':
            pkgs.add('python3-pytest')

    # Secondary toolchain: gated on textual evidence
    if file_content:
        for pkg, pat in _BUILD_DRIVER_EVIDENCE.items():
            if pkg not in pkgs and pat.search(file_content):
                pkgs.add(pkg)

    return pkgs


def _strip_comments_for_regex(content: str, build_system: str) -> str:
    """Strip per-line comments before regex matching.

    Per build_system, strips:
    - cmake/meson/makefile/automake/qmake/python: '#' to end of line
    - autoconf/configure_script/bootstrap: 'dnl' and '#' to end of line

    Conservative: only strips when the comment marker appears outside of a quoted
    string context (heuristic: marker must be at start of line or preceded by whitespace).
    """
    if build_system in ('cmake', 'meson', 'makefile', 'automake', 'qmake',
                        'python', 'ruby', 'node'):
        # Strip '# comment' (but not inside strings — heuristic: require whitespace or BOL before #)
        return re.sub(r'(?m)(^|\s)#.*$', r'\1', content)
    if build_system in ('autoconf', 'configure_script', 'bootstrap'):
        # dnl is m4 comment; # is shell comment
        return re.sub(r'(?m)(^|\s)(?:dnl\b|#).*$', r'\1', content)
    return content


def _apply_rules(filepath: str, content: str, lines: list[str], build_system: str) -> list[RuleMatch]:
    """Apply all rules for a build system and collect matches with context."""
    if build_system == 'cmake' and _is_cmake_helper_module_file(filepath):
        return []

    patterns = get_patterns(build_system)
    if not patterns:
        return []

    # Strip comments before regex matching to avoid picking up disabled/commented-out
    # dependencies (e.g. `# find_package(OldLib)` or `dnl PKG_CHECK_MODULES(...)`).
    content = _strip_comments_for_regex(content, build_system)

    raw_matches = []
    seen_dedup = set()

    structured_matches = _structured_matches(filepath, content, lines, build_system)
    dependency_list_matches = _dependency_list_matches(filepath, content, build_system)
    for match in structured_matches + dependency_list_matches:
        raw_matches.append((match.line_num - 1, match.rule_type, match.matched_text, match.extracted_name))
    if build_system == 'autoconf':
        for match in _autoconf_full_call_matches(filepath, content):
            line_num, _rule_type, _matched_text, _extracted_name = match
            if filepath.endswith('.m4') and _in_m4_macro_body(lines, line_num + 1):
                continue
            raw_matches.append(match)

    for regex, rule_type in patterns:
        if build_system == 'autoconf' and rule_type in _AUTOCONF_FULL_CALL_MACROS.values():
            continue
        if structured_matches and rule_type.split(':', 1)[0] == build_system:
            continue
        for m in regex.finditer(content):
            line_num = content[:m.start()].count('\n')

            extracted = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            extracted = extracted.strip()
            if rule_type in {'ac:AC_CHECK_HEADER', 'cmake:check_include_file'} and _is_standard_header(extracted):
                continue
            if (
                rule_type in {
                    'ac:AC_CHECK_LIB',
                    'ac:AC_CHECK_PROG',
                    'ac:generic_dependency_macro',
                    'ac:pkg-config',
                }
                and filepath.endswith('.m4')
                and _in_m4_macro_body(lines, line_num + 1)
            ):
                continue
            if rule_type == 'ac:fortran' and filepath.endswith('.m4'):
                continue
            if _is_macro_library_file(filepath) and re.search(r'\$\d+|\$\{?\w+\}?|@', extracted):
                continue
            if rule_type in {'ac:BOOST_macro', 'ac:BOOST_libs_var'} and filepath.endswith('.m4'):
                continue
            if rule_type == 'configure:error-missing':
                continue
            if rule_type.endswith(':-l') and _bad_l_flag_name(extracted, build_system):
                continue
            if (
                rule_type.endswith(':link-command')
                or rule_type.endswith(':link-libs-var')
                or rule_type in ('pkgconfig:libs', 'pkgconfig:libs-private')
            ) and _only_noise_l_flags(m.group(0)):
                continue
            if rule_type == 'pkgconfig:libs' and _only_project_libs(filepath, m.group(0)):
                continue
            if rule_type in _DEDUP_RULE_TYPES:
                key = (rule_type, extracted.lower())
                if key in seen_dedup:
                    continue
                seen_dedup.add(key)

            raw_matches.append((line_num, rule_type, m.group(0), extracted))

    if build_system == 'autoconf' and _M4_DEPENDENCY_FILE_RE.match(os.path.basename(filepath)):
        raw_matches.insert(0, (0, 'ac:m4_dependency_file', os.path.basename(filepath), os.path.basename(filepath)))
    if build_system == 'cmake':
        find_module = re.match(r'Find([A-Za-z0-9_+.-]+)\.cmake(?:\.in|\.in\.in)?$', os.path.basename(filepath), re.IGNORECASE)
        if find_module:
            raw_matches.insert(0, (0, 'cmake:find_module_file', os.path.basename(filepath), find_module.group(1)))

    if not raw_matches:
        return []

    results = []
    for line_num, rule_type, matched_text, extracted_name in raw_matches:
        max_matched_text = 1000 if (
            rule_type.endswith('_call')
            or rule_type.startswith('python:')
            or ':dependency-list' in rule_type
            or rule_type in _AUTOCONF_FULL_CALL_MACROS.values()
        ) else 200
        rule_match = RuleMatch(
            file_path=filepath,
            line_num=line_num + 1,
            rule_type=rule_type,
            matched_text=matched_text.strip()[:max_matched_text],
            context='',
            extracted_name=extracted_name,
            condition=_condition_label(lines, line_num, build_system),
        )
        if rule_match.condition == 'non-linux':
            continue
        rule_match.packages = sorted(_direct_packages_from_match(build_system, rule_match))
        results.append(rule_match)

    return results


def _parse_debian_control(ctrl_path: str) -> set[str]:
    """Extract Build-Depends package names from debian/control."""
    try:
        content = open(ctrl_path, errors='replace').read()
    except Exception:
        return set()

    pkgs = set()
    m = re.search(r'Build-Depends:\s*(.*?)(?:\n[A-Z]|\Z)', content, re.DOTALL)
    if m:
        bd = m.group(1)
        for entry in re.split(r',\s*', bd):
            entry = entry.strip()
            pkg_m = re.match(r'([a-z0-9][a-z0-9.+\-]+)', entry)
            if pkg_m:
                pkgs.add(pkg_m.group(1))
    return pkgs


# debian/rules contains configure/build commands and extra dependency hints.
# We extract:
#   - -l<lib> linker flags → candidate -dev packages
#   - --with-<X> / --enable-<X> / --disable-<X> → build flags (informational)
#   - pkg-config / python module deps mentioned in override_dh_* blocks
_DEBIAN_RULES_LIB_RE = re.compile(r'(?<![a-zA-Z-])-l(\w+)')
_DEBIAN_RULES_PKG_RE = re.compile(r'(?:pkg-config|pkgconf)\s+--(?:exists|cflags|libs)\s+([\w.-]+)')


def _parse_debian_rules(rules_path: str) -> tuple[set[str], list[str]]:
    """Extract extra package hints and build commands from debian/rules.

    Returns (extra_packages, build_commands).
    extra_packages: set of candidate package names derived from -l flags and pkg-config calls.
    build_commands: list of configure/build commands found in override_dh_* blocks.
    """
    try:
        content = open(rules_path, errors='replace').read()
    except Exception:
        return set(), []

    extra_packages: set[str] = set()
    system_libs = frozenset({
        'stdc++', 'gcc_s', 'c', 'm', 'pthread', 'dl', 'rt', 'iconv',
        'crypt', 'resolv', 'nsl', 'anl',
    })

    for m in _DEBIAN_RULES_LIB_RE.finditer(content):
        lib = m.group(1)
        if lib in system_libs:
            continue
        extra_packages.add(f'lib{lib}-dev')

    for m in _DEBIAN_RULES_PKG_RE.finditer(content):
        pkg = m.group(1)
        extra_packages.add(pkg)

    return extra_packages, []


def scan_project(
    root: str | Path,
    context_lines: int = CONTEXT_LINES,
    scan_third_party: bool = False,
    scan_debian_rules: bool = False,
) -> ProjectScan:
    """Walk project tree, find build files, apply rules, collect matches.

    Args:
        root: Project root path.
        context_lines: Context lines around rule matches.
        scan_third_party: Include thirdparty/vendor directories.
        scan_debian_rules: If True, also parse debian/rules for extra -l<lib>
            and pkg-config hints that debian/control doesn't cover.
    """
    root = str(root)
    result = ProjectScan()
    root_configure_sources = {
        name for name in ('configure.ac', 'configure.in')
        if os.path.isfile(os.path.join(root, name))
    }

    ctrl_path = os.path.join(root, 'debian', 'control')
    if os.path.isfile(ctrl_path):
        result.direct_packages = _parse_debian_control(ctrl_path)

    # Optional: parse debian/rules for extra linker/pkg-config hints that
    # debian/control doesn't cover (e.g. -lwavpack in override_dh_auto_configure).
    if scan_debian_rules:
        rules_path = os.path.join(root, 'debian', 'rules')
        if os.path.isfile(rules_path):
            extra_pkgs, _ = _parse_debian_rules(rules_path)
            result.direct_packages.update(extra_pkgs)

    for dirpath, dirnames, filenames in os.walk(root):
        kept_dirnames = []
        for dirname in dirnames:
            probe_descend = _should_descend_artifact_probe_dir(root, dirpath, dirname)
            if probe_descend is True:
                kept_dirnames.append(dirname)
            elif probe_descend is None and not _should_skip_dir(dirname, skip_third_party=not scan_third_party):
                kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

        if os.path.basename(dirpath) == 'debian':
            ctrl = os.path.join(dirpath, 'control')
            if os.path.isfile(ctrl) and dirpath != os.path.join(root, 'debian'):
                result.direct_packages.update(_parse_debian_control(ctrl))
        for ctrl_name in filenames:
            if ctrl_name == 'control.in':
                ctrl = os.path.join(dirpath, ctrl_name)
                if os.path.isfile(ctrl):
                    result.direct_packages.update(_parse_debian_control(ctrl))

        for fname in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fname), root)
            filepath = os.path.join(dirpath, fname)
            if (
                fname == 'configure'
                and root_configure_sources
                and os.path.abspath(dirpath) == os.path.abspath(root)
            ):
                continue
            bs = classify_file(fname, rel_path=rel)
            artifact_build_system = bs
            artifact_only = False
            if bs is None:
                artifact_build_system = _artifact_only_build_system(rel, fname, filepath)
                artifact_only = artifact_build_system is not None
            if bs is None and not artifact_only:
                continue

            root_level = os.path.abspath(dirpath) == os.path.abspath(root)
            # Read file content early so we can pass it to _auto_packages_from_build_file
            # for evidence-gated toolchain emission. Skip large artifact-only files.
            raw = ''
            if not (artifact_only and os.path.getsize(filepath) > _MAX_SOURCE_ARTIFACT_BYTES):
                try:
                    raw = open(filepath, errors='replace').read()
                except Exception:
                    raw = ''
            if bs is not None:
                result.build_systems.add(bs)
                result.build_driver_packages.update(_auto_packages_from_build_file(
                    bs, fname, root_level,
                    file_content=raw if not artifact_only else None,
                ))
            if artifact_only and os.path.getsize(filepath) > _MAX_SOURCE_ARTIFACT_BYTES:
                continue

            if not raw.strip():
                continue

            if should_extract_signal_file(rel):
                extract_signals_from_file(rel, raw, result.version_signals)

            if bs in {'cmake', 'makefile', 'automake', 'qmake', 'meson', 'autoconf'}:
                result.build_variables.update(_collect_build_variables(raw, bs))

            file_artifacts = extract_file_artifacts(
                rel,
                raw,
                artifact_build_system,
                build_variables=result.build_variables,
            )
            if file_artifacts:
                result.file_artifacts[rel] = file_artifacts

            if artifact_only:
                continue

            file_lines = raw.splitlines(keepends=True)
            matches = _apply_rules(rel, raw, [l.rstrip('\n') for l in file_lines], bs)

            if matches:
                direct_only = []
                llm_matches = []
                for m_obj in matches:
                    if _is_direct_only_match(bs, m_obj):
                        direct_only.append(m_obj)
                        continue
                    m_obj.context = _extract_context(
                        [l.rstrip('\n') for l in file_lines],
                        {m_obj.line_num - 1},
                        _context_window_for_rule(m_obj.rule_type, context_lines),
                    )
                    llm_matches.append(m_obj)
                _attach_match_artifacts(matches, result.build_variables)
                if direct_only:
                    direct_key = f'__direct__/{rel}'
                    result.file_matches[direct_key] = direct_only
                if llm_matches:
                    result.file_matches[rel] = llm_matches

    return result
