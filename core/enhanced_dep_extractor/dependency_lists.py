"""Shared parser for dependency-like variable/list statements.

The scanner uses these statements as raw LLM evidence. Artifact backfill uses
only the value tokens, so variable names and symbolic substitutions do not turn
into provider lookups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .provider_normalization import NOISE_LIBS  # noqa: F401 (re-exported)


@dataclass(frozen=True)
class DependencyListStatement:
    line_num: int
    rule_type: str
    var_name: str
    value: str
    raw: str
    confidence: str


VAR_NAME_RE = re.compile(
    r'(?:^|[_-])(?:deps?|dependencies|requires?|modules?|packages?|pkgs?|pkgconfig|pkg_config|pkg-module|libs?|libraries|components?)(?:$|[_-])',
    re.IGNORECASE,
)
CMAKE_CALL_RE = re.compile(r'\b(?:set|list)\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
MESON_ASSIGN_RE = re.compile(
    r'^\s*([A-Za-z0-9_.+-]+)\s*=\s*(?:\\\n\s*)?(\[(?:.|\n)*?\]|[^\n#]+)',
    re.IGNORECASE | re.MULTILINE,
)
ASSIGN_RE = re.compile(r'^\s*([A-Za-z0-9_.+-]+)\s*([:+?]?=)\s*(.+)$', re.IGNORECASE | re.MULTILINE)
TOKEN_RE = re.compile(r'(?<![A-Za-z0-9_+./-])-l[A-Za-z0-9_+.-]+|[A-Za-z0-9_+.-]+(?:\.pc)?')
VAR_REF_RE = re.compile(r'(?:\$\([^)]*\)|\$\{[^}]*\}|@[A-Za-z0-9_]+@)')
CMAKE_WORD_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\$\{[^}]+\}|\$\([^)]*\)|@[A-Za-z0-9_]+@|[A-Za-z0-9_+./:-]+')

STOPWORDS = frozenset({
    'append', 'set', 'true', 'false', 'yes', 'no', 'on', 'off', 'required',
    'quiet', 'private', 'public', 'interface', 'version', 'debug', 'optimized',
    'general', 'and', 'or', 'not', 'if', 'else', 'endif', 'for', 'foreach',
    'endforeach', 'with', 'without', 'found', 'missing', 'module', 'modules',
    'package', 'packages', 'lib', 'libs', 'dependency', 'dependencies', 'deps',
})
# NOISE_LIBS imported from provider_normalization at top of module.
LIB_VAR_NAMES = frozenset({'libs', 'ldlibs', 'loadlibs', 'ldadd', 'libadd', 'ldflags'})
INTERNAL_VAR_PREFIXES = ('am__', 'cpack_', 'cmake_', 'package_')
INTERNAL_VAR_PARTS = ('_obj', '_objs', '_src', '_srcs', 'search_path', 'soname', 'abi')
INTERNAL_VAR_NAMES = frozenset({
    'aclocal_amflags', 'configure_dependencies', 'dist_pkgconfig_data',
    'module', 'modules',
    'pkg_config', 'pkgconfig', 'pkg_name', 'pkgconfig_data',
})
SPECIAL_PKGCONFIG_NAMES = frozenset({'openssl', 'zlib', 'alsa', 'krb5', 'systemd', 'udev', 'libcurl'})
LINKER_OPTIONS = frozenset({'-l', '-shared'})


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        cut = len(line)
        for marker in ('#', ' dnl '):
            idx = line.find(marker)
            if idx >= 0:
                cut = min(cut, idx)
        out.append(line[:cut])
    return '\n'.join(out)


def clean_token(token: str) -> str:
    return token.strip().strip('"\'`()[]{}<>:,;')


def normalize_var_name(name: str) -> str:
    return clean_token(name).lower().replace('.', '_').replace('-', '_')


def is_dependency_var(name: str) -> bool:
    normalized = normalize_var_name(name)
    if not normalized:
        return False
    if normalized in INTERNAL_VAR_NAMES or normalized.startswith(INTERNAL_VAR_PREFIXES):
        return False
    if any(part in normalized for part in INTERNAL_VAR_PARTS):
        return False
    if normalized in LIB_VAR_NAMES:
        return True
    return bool(VAR_NAME_RE.search(normalized.replace('_', '-')))


def is_symbolic_token(token: str) -> bool:
    token = clean_token(token)
    if not token:
        return True
    if VAR_REF_RE.fullmatch(token):
        return True
    if '$' in token or token.startswith('@') or token.endswith('@'):
        return True
    if token in LINKER_OPTIONS:
        return True
    if token.startswith(('/', '-W')):
        return True
    if token.startswith('Qt') and '::' in token:
        return False
    if token.startswith('-') and not token.startswith('-l'):
        return True
    if '/' in token and not token.endswith('.pc'):
        return True
    if re.fullmatch(r'[A-Z][A-Z0-9_]*(?:_(?:LIBS|CFLAGS|LDADD|LDFLAGS|DEPS|MODULES|PKGS))?', token):
        return True
    return False


def concrete_token(token: str) -> str | None:
    token = clean_token(token)
    if not token or is_symbolic_token(token):
        return None
    low = token.lower()
    normalized = low[:-3] if low.endswith('.pc') else low
    normalized = normalized.replace('_', '-')
    if normalized in STOPWORDS or normalized in NOISE_LIBS:
        return None
    if token.startswith('-l') and len(token) > 2:
        lib = token[2:].lower()
        return token if lib and lib not in NOISE_LIBS else None
    if re.fullmatch(r'\d+(?:\.\d+)*', normalized):
        return None
    if normalized.endswith(('.a', '.so', '.o', '.d', '.lo', '.la', '.cc', '.cpp', '.c', '.h', '.hpp', '.m4')):
        return None
    if normalized.endswith(('-dir', '-path', '-prefix', '-suffix', '-name', '-names', '-version', '-actions')):
        return None
    if normalized.startswith(('am-', 'lt-', 'sys-')):
        return None
    if low.endswith('.pc') or normalized in SPECIAL_PKGCONFIG_NAMES:
        return normalized
    if (
        re.fullmatch(r'[a-z0-9_+.-]+', low)
        and re.search(r'[a-z]', normalized)
        and ('-' in low or re.search(r'\d', low))
    ):
        return normalized
    return None


def concrete_tokens(value: str) -> list[str]:
    value = strip_comments(value)
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(value):
        concrete = concrete_token(token)
        if concrete and concrete not in seen:
            seen.add(concrete)
            result.append(concrete)
    return result


def has_concrete_artifact(value: str) -> bool:
    return bool(concrete_tokens(value))


HIGH_CONF_VAR_NAMES = frozenset({
    'pkg_modules', 'pkg_module', 'pkg_module_spec', 'pkg_modules_spec',
    'pkgconfig', 'pkg_config', 'pkgconfig_modules', 'pkg_config_modules',
})
HIGH_CONF_VAR_SUFFIXES = ('_pkg_modules', '_pkgconfig', '_pkg_config', '_pc_requires')


def dependency_confidence(var_name: str, value: str) -> str:
    normalized = normalize_var_name(var_name)
    tokens = concrete_tokens(value)
    if not tokens:
        return 'low'
    if all(token.startswith('-l') for token in tokens):
        return 'high'
    if normalized in HIGH_CONF_VAR_NAMES or normalized.endswith(HIGH_CONF_VAR_SUFFIXES):
        return 'high'
    if re.search(r'\bpkg-?config\b|\$\(?PKG_CONFIG\)?|\$\{?PKG_CONFIG\}?', value, re.IGNORECASE):
        return 'high'
    return 'medium'

def split_cmake_words(body: str) -> list[str]:
    return [clean_token(word) for word in CMAKE_WORD_RE.findall(body)]


def find_dependency_lists(content: str, build_system: str) -> list[DependencyListStatement]:
    statements: list[DependencyListStatement] = []
    if build_system == 'cmake':
        for match in CMAKE_CALL_RE.finditer(content):
            body = match.group(1)
            words = split_cmake_words(body)
            if not words:
                continue
            command = content[match.start():match.start() + 16].split('(', 1)[0].strip().lower()
            if command == 'list' and words[0].lower() == 'append' and len(words) > 1:
                var_name = words[1]
                value = ' '.join(words[2:])
            else:
                var_name = words[0]
                value = ' '.join(words[1:])
            if not is_dependency_var(var_name):
                continue
            confidence = dependency_confidence(var_name, value)
            if confidence == 'low':
                continue
            statements.append(DependencyListStatement(
                line_num=content[:match.start()].count('\n') + 1,
                rule_type='cmake:dependency-list',
                var_name=var_name,
                value=value,
                raw=match.group(0).strip(),
                confidence=confidence,
            ))
        return statements

    if build_system == 'meson':
        for match in MESON_ASSIGN_RE.finditer(content):
            var_name, value = match.group(1), match.group(2)
            if not is_dependency_var(var_name):
                continue
            confidence = dependency_confidence(var_name, value)
            if confidence == 'low':
                continue
            statements.append(DependencyListStatement(
                line_num=content[:match.start()].count('\n') + 1,
                rule_type='meson:dependency-list',
                var_name=var_name,
                value=value,
                raw=match.group(0).strip(),
                confidence=confidence,
            ))
        return statements

    if build_system not in {'makefile', 'automake', 'autoconf', 'configure_script', 'bootstrap'}:
        return statements

    prefix = 'make' if build_system in {'makefile', 'automake'} else 'ac'
    if build_system in {'configure_script', 'bootstrap'}:
        prefix = build_system.split('_', 1)[0]

    for match in ASSIGN_RE.finditer(content):
        var_name, value = match.group(1), match.group(3)
        if not is_dependency_var(var_name):
            continue
        confidence = dependency_confidence(var_name, value)
        if confidence == 'low':
            continue
        statements.append(DependencyListStatement(
            line_num=content[:match.start()].count('\n') + 1,
            rule_type=f'{prefix}:dependency-list',
            var_name=var_name,
            value=value,
            raw=match.group(0).strip(),
            confidence=confidence,
        ))
    return statements
