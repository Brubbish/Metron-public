"""Artifact-level evidence extraction and conservative package backfill."""

from __future__ import annotations

import re
import json
import os
from functools import lru_cache
from dataclasses import dataclass, field

from .dependency_lists import clean_token, concrete_tokens, find_dependency_lists, is_symbolic_token
from .provider_normalization import (
    BOOST_LINK_COMPONENTS,
    BOOST_NON_COMPONENT_MACROS,
    EXECUTABLE_NORMALIZATION,
    NOISE_LIBS,
    NON_DEV_PROVIDER_ALLOWLIST,
    boost_component_package,
    boost_library_package,
)
from . import component_templates
from .rules import RuleMatch
from .metadata import (
    DEFAULT_DEBIAN_VERSION,
    DISTRO_MAPPING_PATH,
    normalize_debian_version,
    pkginfo_paths,
    pkgpath_path,
    provider_index_path,
)
from .default_headers import header_path_suffixes, load_default_header_filter


@dataclass(frozen=True)
class Artifact:
    kind: str
    name: str
    components: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ProviderIndex:
    pkgconfig: dict[str, set[str]] = field(default_factory=dict)
    cmake: dict[str, set[str]] = field(default_factory=dict)
    headers_exact: dict[str, set[str]] = field(default_factory=dict)
    headers_suffix: dict[str, set[str]] = field(default_factory=dict)
    libraries: dict[str, set[str]] = field(default_factory=dict)
    executables: dict[str, set[str]] = field(default_factory=dict)
    config_scripts: dict[str, set[str]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    provides: dict[str, set[str]] = field(default_factory=dict)


DEBIAN_VERSION = normalize_debian_version(
    os.environ.get('DEBIAN_VERSION') or os.environ.get('DEBIAN_DISTRO') or DEFAULT_DEBIAN_VERSION
)
PROVIDER_INDEX_PATH = (
    os.environ.get('DEBIAN_PROVIDER_INDEX_PATH')
    or os.environ.get('TRIXIE_PROVIDER_INDEX_PATH')
    or str(provider_index_path(DEBIAN_VERSION))
)
PKGPATH_PATH = (
    os.environ.get('DEBIAN_PKGPATH_PATH')
    or os.environ.get('TRIXIE_PKGPATH_PATH')
    or str(pkgpath_path(DEBIAN_VERSION))
)
PACKAGES_PATH = (
    os.environ.get('DEBIAN_PACKAGES_PATH')
    or os.environ.get('TRIXIE_PACKAGES_PATH')
    or ''
)
_PROVIDER_INDEX_CACHE: ProviderIndex | None = None
_DISTRO_MAPPING_CACHE: dict | None = None


def _distro_mapping() -> dict:
    global _DISTRO_MAPPING_CACHE
    if _DISTRO_MAPPING_CACHE is None:
        if DISTRO_MAPPING_PATH.exists():
            with open(DISTRO_MAPPING_PATH) as f:
                _DISTRO_MAPPING_CACHE = json.load(f)
        else:
            _DISTRO_MAPPING_CACHE = {}
    return _DISTRO_MAPPING_CACHE


def resolve_from_distro_mapping(kind: str, name: str, distro: str | None = None) -> list[str]:
    """Query the cross-distro mapping for packages matching (kind, name, distro)."""
    distro = distro or DEBIAN_VERSION
    mapping = _distro_mapping()
    entry = mapping.get(kind, {}).get(name.lower(), {})
    return entry.get(distro, [])


_SAFE_NON_DEV_EXECUTABLE_PACKAGES = frozenset({
    'chrpath',
    'graphviz',
    'help2man',
    'xauth',
    'xvfb',
})
_HEADER_ONLY_PACKAGE_CANDIDATES = frozenset({
    'doctest-dev',
    'libsimde-dev',
    'rapidjson-dev',
})
_HEADER_ONLY_STEMS = {
    'doctest': 'doctest-dev',
    'simde': 'libsimde-dev',
    'rapidjson': 'rapidjson-dev',
    'tclap': 'libtclap-dev',
}
_BAD_PROVIDER_RE = re.compile(
    r'(?:^|-)doc(?:-|$)|(?:^|-)dbg(?:-|$)|(?:^|-)debug(?:-|$)|'
    r'(?:^|-)examples?(?:-|$)|(?:^|-)tests?(?:-|$)|(?:^|-)data(?:-|$)'
)
_RUNTIME_ABI_PACKAGE_RE = re.compile(r'^(?:lib.*|[a-z0-9.+-]*lib[a-z0-9.+-]*)\d(?:[a-z]*\d*)?(?:t64)?$')
_CMAKE_CALL_RE = re.compile(
    r'\b((?:qt_(?:internal_)?find_package|find_package)|find_dependency)\s*\((.*?)\)',
    re.IGNORECASE | re.DOTALL,
)
_CMAKE_INCLUDE_FIND_RE = re.compile(r'\binclude\s*\(\s*Find([A-Za-z0-9_]+)', re.IGNORECASE)
_CMAKE_FPHSA_RE = re.compile(r'\bfind_package_handle_standard_args\s*\(\s*([A-Za-z0-9_.+-]+)', re.IGNORECASE)
_CMAKE_FETCH_RE = re.compile(
    r'\b(?:CPMFindPackage)\s*\((.*?)\)',
    re.IGNORECASE | re.DOTALL,
)
_CMAKE_FETCH_DECLARE_RE = re.compile(r'\bFetchContent_Declare\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_CMAKE_SET_CALL_RE = re.compile(r'\bset\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_CMAKE_DEP_LIST_VAR_RE = re.compile(
    r'(?:^|_)(?:DEPS?|DEPENDENCIES|LIBS?|LIBRARIES|PACKAGES|MODULES)(?:_|$)',
    re.IGNORECASE,
)
_PKG_CHECK_RE = re.compile(r'\bpkg_(?:check_modules|search_module)(?:_static)?\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_CMAKE_PKG_GET_VARIABLE_RE = re.compile(r'\bpkg_get_variable\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_MESON_DEP_RE = re.compile(r'\bdependency\s*\(\s*[\'"]([^\'"]+)[\'"]', re.IGNORECASE)
_MESON_DEP_LIST_RE = re.compile(r'\bdependency\s*\(\s*\[(.*?)\]', re.IGNORECASE | re.DOTALL)
_MESON_DEP_CALL_RE = re.compile(r'\bdependency\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_MESON_FIND_PROGRAM_RE = re.compile(r'\bfind_program\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_CMAKE_FIND_PROGRAM_RE = re.compile(r'\bfind_program\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_CMAKE_TARGET_LINK_RE = re.compile(r'\btarget_link_libraries\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_IMPORTED_TARGET_RE = re.compile(r'\b([A-Za-z0-9_+.-]+)::([A-Za-z0-9_+.-]+)\b')
_QMAKE_QT_RE = re.compile(r'^\s*QT\s*[+*]?=\s*(.+)$', re.IGNORECASE | re.MULTILINE)
_QMAKE_PKGCONFIG_RE = re.compile(r'^\s*PKGCONFIG\s*[+*]?=\s*(.+)$', re.IGNORECASE | re.MULTILINE)
_QMAKE_TRANSLATIONS_RE = re.compile(r'^\s*TRANSLATIONS\s*[+*]?=\s*\S.*$', re.IGNORECASE | re.MULTILINE)
_LFLAG_RE = re.compile(r'(?:^|[\s="\'])-l([A-Za-z0-9_+.-]+)')
_PY_EXT_LIBS_RE = re.compile(r'(?:libraries|extra_link_args|link_args)\s*(?:=|\+=|append/extend)\s*([^;\n]+)', re.IGNORECASE)
_BUILD_LIBS_LIST_RE = re.compile(
    r"(?:\[['\"]LIBS['\"]\]|\bLIBS\b)\s*=\s*\[([^\]]+)\]",
    re.IGNORECASE | re.DOTALL,
)
_BUILD_LIBS_APPEND_RE = re.compile(
    r"(?:\[['\"]LIBS['\"]\]\s*\.\s*(?:append|extend)|\bLIBS\s*\.\s*(?:append|extend))\s*\((.*?)\)",
    re.IGNORECASE | re.DOTALL,
)
_BUILD_ACTION_RE = re.compile(r'\b(?:Command|Action)\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_QUOTED_TEXT_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_MAKE_PKG_CONFIG_RE = re.compile(r'(?:pkg-config|pkgconf|\$\(?PKG_CONFIG\)?|\$\{?PKG_CONFIG\}?)\b[^\n;&|`]*', re.IGNORECASE)
_TOOL_VAR_RE = re.compile(
    r'^\s*([A-Za-z0-9_+-]*(?:TOOL|TOOLS|PROG|PROGRAM|EXECUTABLE|XMLLINT|XMLCATALOG|'
    r'VALAC|VAPIGEN|INTLTOOL|GTKDOC|G_IR)[A-Za-z0-9_+-]*)\s*[:?+]?=\s*([^\n#]+)',
    re.IGNORECASE | re.MULTILINE,
)
_COMMAND_EXEC_RE = re.compile(r'\b(?:command|COMMAND)\s*[:=]\s*\[?\s*[\'"]?([A-Za-z0-9_+.-]+)')
_RUBY_GEM_RE = re.compile(r'\b(?:add_)?(?:runtime_|development_)?dependency\s*\(?\s*[\'"]([^\'"]+)[\'"]', re.IGNORECASE)
_PERL_MODULE_RE = re.compile(r'[\'"]([A-Z][A-Za-z0-9_:]+)[\'"]\s*=>')
_OCAML_CONF_RE = re.compile(r'\bconf-([A-Za-z0-9_.+-]+)\b', re.IGNORECASE)
_BOOST_MACRO_RE = re.compile(r'\b(?:AX_)?BOOST_([A-Z0-9_]+)\b')
_BOOST_LIB_VAR_RE = re.compile(r'\bBOOST_([A-Z0-9_]+)_(?:LIBS|LDFLAGS|LDPATH)\b')
_BOOST_TARGET_RE = re.compile(r'\bBoost::([A-Za-z0-9_]+)\b')
_AC_CHECK_LIB_RE = re.compile(r'\bAC_CHECK_LIBS?\s*\(\s*\[?([A-Za-z0-9_+.-]+)', re.IGNORECASE)
_AC_SEARCH_LIBS_RE = re.compile(
    r'\bAC_SEARCH_LIBS?\s*\(\s*(?:\[[^\]]*\]|[^,)]*)\s*,\s*(?:\[([^\]]+)\]|([^),]+))',
    re.IGNORECASE,
)
_AC_CHECK_LIBS_RE = re.compile(
    r'\bAC_CHECK_LIBS?\s*\(\s*(?:\[([^\]]+)\]|([^),]+))',
    re.IGNORECASE,
)
_CMAKE_CHECK_LIBRARY_RE = re.compile(r'\bcheck_library_exists\s*\(\s*([A-Za-z0-9_+.-]+)', re.IGNORECASE)
_FIND_LIBRARY_RE = re.compile(
    r'\b(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?find_library\s*\(\s*(?:\w+\s+(?:NAMES?\s+)?)?[\'"]?([A-Za-z0-9_+.-]+)',
    re.IGNORECASE,
)
_HEADER_CALL_RE = re.compile(
    r'\b(?:AC_CHECK_HEADERS?|check_include_file(?:_cxx|s)?|find_(?:path|file)|have_header|has_(?:header|header_symbol))\s*\((.*?)\)',
    re.IGNORECASE | re.DOTALL,
)
_HEADER_TOKEN_RE = re.compile(r'[A-Za-z0-9_./+-]+\.h(?:pp|xx)?\b', re.IGNORECASE)
_FEATURE_PATTERNS = (
    (re.compile(r'\b(?:gnome\.generate_gir\s*\(|GOBJECT_INTROSPECTION_CHECK)\b', re.IGNORECASE), 'gir_generation'),
    (re.compile(r'\b(?:gnome\.gtkdoc\s*\(|GTK_DOC_CHECK)\b', re.IGNORECASE), 'gtk_doc_generation'),
    (re.compile(r'\b(?:VAPIGEN_CHECK|find_package\s*\(\s*Vala\b|VALAC\b|VAPIGEN\b)', re.IGNORECASE), 'vala_generation'),
    (re.compile(r'\b(?:IT_PROG_INTLTOOL|intltoolize|intltool-(?:extract|merge|update)|find_package\s*\(\s*Intltool\b)', re.IGNORECASE), 'intltool_i18n'),
    (re.compile(r'\b(?:AM_GNU_GETTEXT|GETTEXT_PACKAGE|msgfmt|xgettext|msgmerge|find_package\s*\(\s*Gettext\b)\b', re.IGNORECASE), 'gettext_i18n'),
    (re.compile(r'\b(?:XMLLINT|XMLCATALOG|xmllint|xmlcatalog)\b', re.IGNORECASE), 'xml_validation'),
    (re.compile(r'\bfind_package\s*\(\s*(?:CoverageReport|Intltool|Vala)\b', re.IGNORECASE), 'cmake_extra_modules'),
)
_GENERIC_AUTOCONF_DEP_MACRO_RE = re.compile(
    r'\b(?:AX_(?:LIB|CHECK|PATH|WITH)_([A-Z0-9_]+)|AM_PATH_([A-Z0-9_]+))\b',
    re.IGNORECASE,
)


_CMAKE_KEYWORDS = {
    'required', 'quiet', 'components', 'optional_components', 'module', 'config',
    'no_module', 'exact', 'version', 'global', 'names',
}
_CMAKE_COMPONENT_BOUNDARIES = {
    'components', 'optional_components',
}
_CMAKE_COMPONENT_STOPWORDS = {
    'required', 'quiet', 'module', 'config', 'no_module', 'exact', 'version',
    'global', 'names', 'hints', 'paths', 'path_suffixes', 'registry_view',
    'no_default_path', 'no_package_root_path', 'no_cmake_path',
    'no_cmake_environment_path', 'no_system_environment_path',
    'no_cmake_system_path', 'no_cmake_install_prefix',
}
_PKGCONFIG_KEYWORDS = {
    'required', 'quiet', 'imported_target', 'global', 'no_cmake_path',
    'no_cmake_environment_path',
}
_FIND_PROGRAM_KEYWORDS = {
    'names', 'hints', 'paths', 'path_suffixes', 'doc', 'required', 'quiet',
    'no_default_path', 'cache', 'advanced',
}
_LINK_LIBRARY_KEYWORDS = {
    'debug', 'optimized', 'general', 'public', 'private', 'interface',
    'link_public', 'link_private', 'link_interface_libraries',
}
_HEADER_SUFFIX_HARD_FANOUT_LIMIT = 128
_HEADER_SUFFIX_MAX_DEPTH = 3
_GENERIC_BASENAME_HEADERS = frozenset({
    'common.h', 'config.h', 'debug.h', 'error.h', 'file.h', 'hash.h',
    'list.h', 'log.h', 'object.h', 'parser.h', 'random.h', 'string.h',
    'thread.h', 'timer.h', 'types.h', 'util.h', 'utils.h', 'version.h',
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


def configure_metadata(
    debian_version: str | None = None,
    provider_index: str | None = None,
    pkgpath: str | None = None,
    packages: str | None = None,
) -> None:
    """Switch provider metadata used by artifact backfill."""
    global DEBIAN_VERSION, PROVIDER_INDEX_PATH, PKGPATH_PATH, PACKAGES_PATH, _PROVIDER_INDEX_CACHE, _DISTRO_MAPPING_CACHE
    DEBIAN_VERSION = normalize_debian_version(debian_version or DEBIAN_VERSION)
    PROVIDER_INDEX_PATH = provider_index or str(provider_index_path(DEBIAN_VERSION))
    PKGPATH_PATH = pkgpath or str(pkgpath_path(DEBIAN_VERSION))
    PACKAGES_PATH = packages or ''
    _PROVIDER_INDEX_CACHE = None
    _DISTRO_MAPPING_CACHE = None
    _provider_has_matching_pkgconfig.cache_clear()
    _provider_has_executable_related_public_header.cache_clear()
    _is_sole_executable_provider.cache_clear()


def _provider_allowed(pkg: str, valid_packages: set[str] | None = None) -> bool:
    pkg_l = pkg.lower()
    if valid_packages is not None and pkg not in valid_packages:
        return False
    if pkg_l in NON_DEV_PROVIDER_ALLOWLIST:
        return True
    if _BAD_PROVIDER_RE.search(pkg_l):
        return False
    return (
        pkg_l.endswith('-dev')
        or '-dev-' in pkg_l
        or pkg_l.endswith('-tools')
        or pkg_l.endswith('-tool')
        or pkg_l.endswith('-bin')
        or pkg_l.endswith('-utils')
    )


def _provider_index_allowed(pkg: str, valid_packages: set[str] | None = None) -> bool:
    pkg_l = pkg.lower()
    if pkg_l in {
        'dietlibc-dev',
        'libc6-dev',
        'libc6-dev-i386',
        'musl-dev',
        'ruby3.3-dev',
    }:
        return False
    if re.fullmatch(r'libboost-[a-z0-9.+-]+[0-9.]+-dev', pkg_l):
        return False
    if pkg_l in _HEADER_ONLY_PACKAGE_CANDIDATES:
        return True
    return _provider_allowed(pkg, valid_packages)


def _path_provider_allowed(pkg: str, section: str, valid_packages: set[str] | None = None) -> bool:
    pkg_l = pkg.lower()
    if pkg_l in {
        'dietlibc-dev',
        'libc6-dev',
        'libc6-dev-i386',
        'musl-dev',
        'ruby3.3-dev',
    }:
        return False
    if _provider_index_allowed(pkg, valid_packages):
        return True
    if valid_packages is not None and pkg not in valid_packages:
        return False
    if _BAD_PROVIDER_RE.search(pkg_l):
        return False
    return section in {'executables', 'config_scripts', 'pkgconfig', 'cmake', 'headers_exact', 'headers_suffix'}


def _stable_packages(packages: set[str], valid_packages: set[str]) -> set[str]:
    return {
        pkg for pkg in packages
        if _provider_allowed(pkg, valid_packages) or _provider_index_allowed(pkg, valid_packages)
    }


def _small_same_source_dev_siblings(seed_packages: set[str], valid_packages: set[str]) -> set[str]:
    siblings = {
        pkg for pkg in _same_source_packages(seed_packages, valid_packages)
        if pkg.endswith('-dev') or '-dev-' in pkg
    }
    return siblings if 1 < len(siblings) <= 3 else set()


def _header_only_provider_from_token(token: str, valid_packages: set[str]) -> str | None:
    normalized = token.lower().strip().strip('/').replace('_', '-')
    stem = normalized.split('/', 1)[0]
    pkg = _HEADER_ONLY_STEMS.get(stem)
    return pkg if pkg in valid_packages else None


def _candidate_score(pkg: str) -> tuple[int, int, str]:
    """Prefer canonical build providers over runtime or overly specific packages."""
    pkg_l = pkg.lower()
    if pkg_l in _HEADER_ONLY_PACKAGE_CANDIDATES:
        score = 60
    elif pkg_l.endswith('-dev'):
        score = 60
    elif '-dev-' in pkg_l:
        score = 45
    elif pkg_l.endswith(('-tools', '-tool', '-bin')):
        score = 35
    elif pkg_l in NON_DEV_PROVIDER_ALLOWLIST:
        score = 30
    else:
        score = 0

    if re.search(r'(?:^|-)lib[a-z0-9.+-]*\d+(?:t64)?(?:-|$)', pkg_l) and not pkg_l.endswith('-dev'):
        score -= 40
    if re.fullmatch(r'libboost-[a-z0-9.+-]+[0-9.]+-dev', pkg_l):
        score -= 30
    if pkg_l.endswith(('-doc', '-data')) or _BAD_PROVIDER_RE.search(pkg_l):
        score -= 100
    return (score, -len(pkg_l), pkg_l)


def _artifact_provider_score(pkg: str, artifact_name: str, section: str) -> tuple[int, int, str]:
    score, length_score, pkg_l = _candidate_score(pkg)
    artifact_l = artifact_name.lower().replace('_', '-')
    if section in {'pkgconfig', 'headers_exact', 'headers_suffix'} and artifact_l == pkg_l:
        score = max(score, 55)
    if section == 'headers_exact' and _provider_has_matching_pkgconfig(pkg_l):
        score = max(score, 55)
    return (score, length_score, pkg_l)


def _naming_affinity(pkg: str, artifact_name: str) -> int:
    """Score how well a package name matches Debian naming convention for an artifact.

    For a library 'FOO', canonical is lib<foo><soversion>-dev.
    Higher score = better match. 0 = no affinity signal.
    """
    if not artifact_name:
        return 0
    pkg_l = pkg.lower()
    art = artifact_name.lower().replace('_', '-')
    # Skip affinity for very short names — too ambiguous for substring matching
    if len(art) <= 2:
        if re.match(rf'^lib{re.escape(art)}\d*-dev$', pkg_l):
            return 20
        return 0
    # lib<art>-dev or lib<art><digits>-dev (exact canonical form)
    if re.match(rf'^lib{re.escape(art)}\d*-dev$', pkg_l):
        return 20
    # <art><abi>-dev forms such as zlib1g-dev
    if re.match(rf'^{re.escape(art)}[a-z0-9.+-]*-dev$', pkg_l):
        return 15
    # lib<art> appears at package start (e.g. libedit-dev for 'edit')
    if pkg_l.startswith(f'lib{art}'):
        return 10
    # art appears as a word boundary in package name
    if re.search(rf'(?:^|[-.]){re.escape(art)}(?:[-.]|$)', pkg_l):
        return 5
    return 0


def _select_provider_candidates(
    candidates: set[str],
    valid_packages: set[str],
    artifact_name: str = '',
    section: str = '',
) -> set[str]:
    """Return only high-confidence providers; skip ambiguous weak ties.

    When multiple candidates tie on score, use naming convention affinity
    to the artifact_name as tiebreaker. If still ambiguous (>2), return empty.
    """
    allowed = {
        pkg for pkg in candidates
        if _path_provider_allowed(pkg, section, valid_packages)
    }
    if not allowed:
        return set()
    if section == 'pkgconfig' and len(allowed) == 1:
        return allowed
    scored = sorted((_artifact_provider_score(pkg, artifact_name, section), pkg) for pkg in allowed)
    best_score = scored[-1][0][0]
    if best_score < 30:
        return set()
    best = {pkg for score, pkg in scored if score[0] == best_score}
    if len(best) <= 1:
        return best
    # Tiebreak using naming convention affinity
    if artifact_name:
        affinity = {pkg: _naming_affinity(pkg, artifact_name) for pkg in best}
        max_affinity = max(affinity.values())
        if max_affinity > 0:
            best = {pkg for pkg, a in affinity.items() if a == max_affinity}
    return best if len(best) <= 2 else set()



def _same_source_packages(seed_packages: set[str], valid_packages: set[str]) -> set[str]:
    index = _provider_index()
    if not index.sources:
        return set()
    seed_sources = {
        source for pkg in seed_packages
        if (source := index.sources.get(pkg))
    }
    if not seed_sources:
        return set()
    return {
        pkg for pkg, source in index.sources.items()
        if source in seed_sources and pkg in valid_packages
    }


def _same_source_named_packages(
    seed_packages: set[str],
    valid_packages: set[str],
    names: set[str],
) -> set[str]:
    return {pkg for pkg in _same_source_packages(seed_packages, valid_packages) if pkg in names}


def _same_source_stable_siblings(seed_packages: set[str], valid_packages: set[str]) -> set[str]:
    return _stable_packages(_same_source_packages(seed_packages, valid_packages), valid_packages)


def _packages_from_index_family(
    seeds: set[str],
    valid_packages: set[str],
    preferred: set[str] | None = None,
) -> set[str]:
    packages = set(seeds)
    if preferred:
        packages.update(_same_source_named_packages(seeds, valid_packages, preferred))
    return _stable_packages(packages, valid_packages)


def _executable_provider_allowed(
    pkg: str,
    executable: str,
    valid_packages: set[str],
    hints: tuple[str, ...] = (),
) -> bool:
    pkg_l = pkg.lower()
    exe_l = executable.lower()
    if valid_packages is not None and pkg not in valid_packages:
        return False
    if pkg_l in NON_DEV_PROVIDER_ALLOWLIST:
        return True
    if pkg_l in _SAFE_NON_DEV_EXECUTABLE_PACKAGES:
        return True
    if re.search(r'(?:^|-|\+)compiler(?:-|$)', pkg_l):
        return True
    if _is_protocol_generator_tool_package(pkg_l):
        return True
    if _is_program_suite_package(pkg_l, exe_l):
        return True
    if _provider_has_executable_related_public_header(pkg_l, exe_l):
        return True
    if 'compiler' in hints and _is_sole_executable_provider(pkg_l, exe_l):
        return True
    if _BAD_PROVIDER_RE.search(pkg_l):
        return False
    if pkg_l in {
        'coreutils', 'dash', 'sed', 'grep', 'findutils',
        'libc-bin', 'perl-base', 'python3-minimal',
    }:
        return False
    if re.fullmatch(r'lib.*\d(?:[a-z]*\d*)?(?:t64)?', pkg_l):
        return False
    if pkg_l == executable.lower():
        return True
    return _provider_allowed(pkg, valid_packages)


def _executable_candidate_score(pkg: str, executable: str, hints: tuple[str, ...] = ()) -> tuple[int, int, str]:
    pkg_l = pkg.lower()
    exe_l = executable.lower()
    score = _candidate_score(pkg)[0]
    if pkg_l.endswith('-utils'):
        score = max(score, 34)
    if re.search(r'(?:^|-|\+)compiler(?:-|$)', pkg_l):
        score = max(score, 35)
    if _is_protocol_generator_tool_package(pkg_l):
        score = max(score, 35)
    if _is_program_suite_package(pkg_l, exe_l):
        score = max(score, 34)
    if _provider_has_executable_related_public_header(pkg_l, exe_l):
        score = max(score, 35)
    if 'compiler' in hints and _is_sole_executable_provider(pkg_l, exe_l):
        score = max(score, 35)
    if pkg_l in _SAFE_NON_DEV_EXECUTABLE_PACKAGES:
        score = max(score, 55 if pkg_l == executable.lower() else 35)
    if pkg_l == exe_l:
        score = max(score, 55)
    elif exe_l.startswith(pkg_l):
        score = max(score, 45)
    if 'qt5' in hints:
        score += 20 if ('qttools5' in pkg_l or 'qt5' in pkg_l) else -20
    if 'qt6' in hints:
        score += 20 if pkg_l.startswith('qt6') else -20
    return (score, -len(pkg_l), pkg_l)


def _select_executable_provider_candidates(
    executable: str,
    candidates: set[str],
    valid_packages: set[str],
    hints: tuple[str, ...] = (),
) -> set[str]:
    allowed = {
        pkg for pkg in candidates
        if _executable_provider_allowed(pkg, executable, valid_packages, hints)
    }
    if not allowed:
        return set()
    scored = sorted((_executable_candidate_score(pkg, executable, hints), pkg) for pkg in allowed)
    best_score = scored[-1][0][0]
    if best_score < 30:
        return set()
    best = {pkg for score, pkg in scored if score[0] == best_score}
    return best if len(best) <= 2 else set()


def _is_protocol_generator_tool_package(pkg_l: str) -> bool:
    """Allow exact executable providers packaged as protocol generator toolsets."""
    return pkg_l.endswith('-proto')


def _is_program_suite_package(pkg_l: str, exe_l: str) -> bool:
    """Allow exact executable providers from narrowly named command suites."""
    if not pkg_l.endswith('-progs'):
        return False
    stem = pkg_l[:-len('-progs')]
    return bool(stem) and (exe_l == stem or exe_l.startswith(f'{stem}-') or exe_l.startswith(f'{stem}.'))


@lru_cache(maxsize=8192)
def _is_sole_executable_provider(pkg_l: str, exe_l: str) -> bool:
    """True when an executable maps to exactly one provider package."""
    providers = {pkg.lower() for pkg in _provider_index().executables.get(exe_l, set())}
    return providers == {pkg_l}


@lru_cache(maxsize=8192)
def _provider_has_matching_pkgconfig(pkg_l: str) -> bool:
    """True when a provider exposes a pkg-config module with its package name."""
    index = _provider_index()
    return pkg_l in index.pkgconfig and pkg_l in index.pkgconfig.get(pkg_l, set())


@lru_cache(maxsize=8192)
def _provider_has_executable_related_public_header(pkg_l: str, exe_l: str) -> bool:
    """True when an executable provider also ships matching public headers."""
    index = _provider_index()
    prefix = f'{pkg_l}/'
    for header, providers in index.headers_exact.items():
        if not header.startswith(prefix) or pkg_l not in providers:
            continue
        basename = header.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        if basename == exe_l or basename.startswith(f'{exe_l}_') or basename.startswith(f'{exe_l}-'):
            return True
    return False


def _add_provider(index: dict[str, set[str]], key: str, pkg: str) -> None:
    key = key.lower().strip()
    if not key:
        return
    index.setdefault(key, set()).add(pkg)


def _library_name_from_path(path: str) -> str | None:
    basename = os.path.basename(path)
    match = re.match(r'lib(.+)\.so(?:$|\.)', basename)
    if not match:
        return None
    return match.group(1).lower()


def _cmake_names_from_path(path: str) -> list[str]:
    basename = os.path.basename(path)
    lower = basename.lower()
    if lower.startswith('find') and lower.endswith('.cmake'):
        return []
    names: list[str] = []
    for suffix in ('config.cmake', '-config.cmake', 'targets.cmake', '-targets.cmake'):
        if lower.endswith(suffix):
            stem = basename[:-len(suffix)].strip('-_')
            if stem:
                names.append(stem)
            break
    return names


def _executable_names_from_path(path: str) -> list[str]:
    if path.startswith(('usr/bin/', 'bin/', 'usr/sbin/', 'sbin/')) or re.search(r'^usr/lib(?:/[^/]+)?/qt[56]/bin/', path):
        basename = os.path.basename(path)
    else:
        return []
    names = [basename]
    match = re.match(r'^[a-z0-9_+-]+-linux-gnu(?:eabi|abihf)?-(.+)$', basename)
    if match:
        names.append(match.group(1))
    return list(dict.fromkeys(name for name in names if name))


def _add_path_provider(index: ProviderIndex, path: str, pkg: str) -> None:
    lower = path.lower().lstrip('./')
    for exe in _executable_names_from_path(lower):
        if not _BAD_PROVIDER_RE.search(pkg.lower()):
            _add_provider(index.executables, exe, pkg)
            if exe.endswith(('-config', '_config')):
                _add_provider(index.config_scripts, exe, pkg)
    if '/pkgconfig/' in lower and lower.endswith('.pc'):
        _add_provider(index.pkgconfig, os.path.basename(lower)[:-3], pkg)
    if not _provider_allowed(pkg):
        return
    if '/cmake/' in lower and lower.endswith('.cmake'):
        for name in _cmake_names_from_path(lower):
            _add_provider(index.cmake, name, pkg)
    if lower.startswith('usr/include/') and lower.endswith(('.h', '.hpp', '.hxx', '.hh')):
        _add_provider(index.headers_exact, lower[len('usr/include/'):], pkg)
    lib = _library_name_from_path(lower)
    if lib:
        _add_provider(index.libraries, lib, pkg)

def _coerce_provider_map(raw: object) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if not isinstance(raw, dict):
        return result
    for key, values in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(values, str):
            packages = {values}
        elif isinstance(values, list):
            packages = {value for value in values if isinstance(value, str)}
        else:
            continue
        if packages:
            result[key.lower()] = packages
    return result


def _load_provider_index_json(path: str) -> ProviderIndex:
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return ProviderIndex()

    index = ProviderIndex()
    if not isinstance(raw, dict):
        return index
    index.pkgconfig = _coerce_provider_map(raw.get('pkgconfig'))
    index.cmake = _coerce_provider_map(raw.get('cmake'))
    index.headers_exact = _coerce_provider_map(raw.get('headers_exact'))
    index.libraries = _coerce_provider_map(raw.get('libraries'))
    index.executables = _coerce_provider_map(raw.get('executables'))
    index.config_scripts = _coerce_provider_map(raw.get('config_scripts'))
    index.provides = _coerce_provider_map(raw.get('provides'))
    index.headers_suffix = _coerce_provider_map(raw.get('headers_suffix'))
    if not index.headers_suffix:
        _build_header_suffix_index(index)
    _add_development_artifact_providers_from_path_map(index, _serialized_path_map(raw))
    _add_development_artifact_providers_from_pkgpath(index)
    sources = raw.get('sources')
    if isinstance(sources, dict):
        index.sources = {str(pkg): str(src) for pkg, src in sources.items() if isinstance(pkg, str)}

    # Overlay supplemental header-only index for packages missing from pkgpath.
    _overlay_header_only_supplement(index, path)

    return index


def _overlay_header_only_supplement(index: ProviderIndex, base_path: str) -> None:
    """Load {version}_header_only_supplement.json and merge into the provider index.

    This covers Architecture: all header-only libraries (nlohmann-json3-dev,
    libsimde-dev, rapidjson-dev, etc.) that have 0 entries in pkgpath because
    the pkgpath builder couldn't enumerate their files. The supplement file name
    is derived from the provider_index file name to stay version-aware.
    """
    import os
    # Derive version from base_path: .../trixie_provider_index_amd64.json → trixie
    base_name = os.path.basename(base_path)
    version = base_name.split('_')[0] if '_' in base_name else 'trixie'
    supp_path = os.path.join(os.path.dirname(base_path), f'{version}_header_only_supplement.json')
    try:
        if not os.path.exists(supp_path):
            return
        with open(supp_path) as f:
            supp = json.load(f)
        for header, pkgs in _coerce_provider_map(supp.get('headers_exact')).items():
            if header not in index.headers_exact:
                index.headers_exact[header] = pkgs
        for name, pkgs in _coerce_provider_map(supp.get('cmake')).items():
            if name not in index.cmake:
                index.cmake[name] = pkgs
    except Exception:
        pass


def _serialized_path_map(raw: dict) -> object:
    for key in ('pkgpath', 'path_to_pkg', 'paths', 'files'):
        paths = raw.get(key)
        if isinstance(paths, dict):
            return paths
    return None


def _add_development_artifact_providers_from_pkgpath(index: ProviderIndex) -> None:
    try:
        with open(PKGPATH_PATH) as f:
            path_to_pkg = json.load(f)
    except Exception:
        return
    _add_development_artifact_providers_from_path_map(index, path_to_pkg)


def _add_development_artifact_providers_from_path_map(index: ProviderIndex, path_to_pkg: object) -> None:
    if not isinstance(path_to_pkg, dict):
        return
    for path, packages in path_to_pkg.items():
        if not isinstance(path, str):
            continue
        pkg_list = [packages] if isinstance(packages, str) else packages
        if not isinstance(pkg_list, list):
            continue
        lower = path.lower().lstrip('./')
        for pkg in pkg_list:
            if not isinstance(pkg, str) or _BAD_PROVIDER_RE.search(pkg.lower()):
                continue
            for exe in _executable_names_from_path(lower):
                _add_provider(index.executables, exe, pkg)
                if exe.endswith(('-config', '_config')):
                    _add_provider(index.config_scripts, exe, pkg)
            if '/pkgconfig/' in lower and lower.endswith('.pc'):
                _add_provider(index.pkgconfig, os.path.basename(lower)[:-3], pkg)
            if lower.startswith('usr/include/') and lower.endswith(('.h', '.hpp', '.hxx', '.hh')):
                _add_provider(index.headers_exact, lower[len('usr/include/'):], pkg)


def _parse_provides(value: str) -> list[str]:
    provides = []
    for part in value.split(','):
        name = re.split(r'\s*[(:]', part.strip(), 1)[0].strip()
        if name:
            provides.append(name)
    return provides


def _iter_package_blocks(packages_path: str | None = None):
    paths = [packages_path] if packages_path else [str(path) for path in pkginfo_paths(DEBIAN_VERSION)]
    for path in paths:
        if not path:
            continue
        try:
            with open(path, errors='replace') as f:
                content = f.read()
        except Exception:
            continue
        for block in content.split('\n\n'):
            if block.strip():
                yield block


def _add_package_metadata(index: ProviderIndex, packages_path: str | None = None) -> None:
    for block in _iter_package_blocks(packages_path or PACKAGES_PATH):
        pkg_match = re.search(r'^Package:\s*(\S+)', block, re.MULTILINE)
        if not pkg_match:
            continue
        pkg = pkg_match.group(1)
        source_match = re.search(r'^Source:\s*(\S+)', block, re.MULTILINE)
        source = source_match.group(1).split('(')[0].strip() if source_match else pkg
        index.sources[pkg] = source
        provides_match = re.search(
            r'^Provides:\s*(.*?)(?:\n[A-Z][A-Za-z0-9-]*:|\Z)',
            block,
            re.MULTILINE | re.DOTALL,
        )
        if provides_match:
            provides_text = ' '.join(line.strip() for line in provides_match.group(1).splitlines())
            for virtual in _parse_provides(provides_text):
                _add_provider(index.provides, virtual, pkg)


def _build_provider_index(pkgpath_path: str | None = None) -> ProviderIndex:
    index = ProviderIndex()
    pkgpath_path = pkgpath_path or PKGPATH_PATH
    try:
        with open(pkgpath_path) as f:
            path_to_pkg = json.load(f)
    except Exception:
        return index

    for path, pkg in path_to_pkg.items():
        if isinstance(path, str) and isinstance(pkg, str):
            _add_path_provider(index, path, pkg)
        elif isinstance(path, str) and isinstance(pkg, list):
            for item in pkg:
                if isinstance(item, str):
                    _add_path_provider(index, path, item)
    _add_package_metadata(index)
    _build_header_suffix_index(index)
    return index


def _build_header_suffix_index(index: ProviderIndex) -> None:
    """Build multi-level suffix lookup for include tokens.

    A provider path like x86_64-linux-gnu/acme-1.0/include/acme/foo.h
    should be findable from source includes such as acme/foo.h.
    """
    for full_path, pkgs in index.headers_exact.items():
        for suffix in _header_suffixes(full_path, max_depth=_HEADER_SUFFIX_MAX_DEPTH):
            index.headers_suffix.setdefault(suffix, set()).update(pkgs)


def _provider_index() -> ProviderIndex:
    global _PROVIDER_INDEX_CACHE
    if _PROVIDER_INDEX_CACHE is None:
        if PROVIDER_INDEX_PATH and os.path.exists(PROVIDER_INDEX_PATH):
            _PROVIDER_INDEX_CACHE = _load_provider_index_json(PROVIDER_INDEX_PATH)
        else:
            _PROVIDER_INDEX_CACHE = _build_provider_index()
    return _PROVIDER_INDEX_CACHE


_VAR_REF_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$\(([A-Za-z_][A-Za-z0-9_]*)\)')


def _build_variable_values(variables: dict[str, str] | None, name: str) -> list[str]:
    if not variables or name not in variables:
        return []
    value = variables[name]
    values: list[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)):
                values.extend(str(nested) for nested in item if str(nested))
            elif str(item):
                values.append(str(item))
        return values
    return [str(value)] if str(value) else []


def _resolve_variables(token: str, variables: dict[str, str] | None) -> str:
    """Replace ${VAR} and $(VAR) references with known values."""
    if not variables or '$' not in token:
        return token

    def replacer(m):
        var = m.group(1) or m.group(2)
        values = _build_variable_values(variables, var)
        return values[-1] if values else m.group(0)
    return _VAR_REF_RE.sub(replacer, token)


def _resolve_variable_variants(token: str, variables: dict[str, str] | None, max_variants: int = 16) -> list[str]:
    """Return all simple static expansions for ${VAR}/$(VAR) references."""
    if not variables or '$' not in token:
        return [token]

    variants = [token]
    for match in list(_VAR_REF_RE.finditer(token)):
        var = match.group(1) or match.group(2)
        values = _build_variable_values(variables, var)
        if not values:
            continue
        next_variants: list[str] = []
        for variant in variants:
            for value in values:
                next_variants.append(variant.replace(match.group(0), value))
                if len(next_variants) >= max_variants:
                    break
            if len(next_variants) >= max_variants:
                break
        variants = next_variants or variants
    return list(dict.fromkeys(variants))


def _words(text: str) -> list[str]:
    return re.findall(r'[A-Za-z0-9_+.-]+', text)


def _clean_token(token: str) -> str:
    return token.strip().strip('"\'`()[]{}<>:,')


def _is_variable(token: str) -> bool:
    return '$' in token or token.startswith('@') or token.endswith('@')


def _context_text(matches: list[RuleMatch]) -> str:
    parts: list[str] = []
    seen_contexts: set[str] = set()
    for match in matches:
        if match.context and match.context not in seen_contexts:
            parts.append(match.context)
            seen_contexts.add(match.context)
        if match.matched_text:
            parts.append(match.matched_text)
    return '\n'.join(part for part in parts if part)


def _detect_qt_version_hint(text: str) -> str | None:
    """Detect Qt major version from context: set(QT_MAJOR_VERSION 6), find_package(QT NAMES Qt6 ...)."""
    if re.search(r'set\s*\(\s*QT_MAJOR_VERSION\s+6\b', text, re.IGNORECASE):
        return 'Qt6'
    if re.search(r'set\s*\(\s*QT_MAJOR_VERSION\s+5\b', text, re.IGNORECASE):
        return 'Qt5'
    if re.search(r'find_package\s*\(\s*QT\s+NAMES\s+Qt6\b', text, re.IGNORECASE):
        return 'Qt6'
    if re.search(r'find_package\s*\(\s*QT\s+NAMES\s+Qt5\b', text, re.IGNORECASE):
        return 'Qt5'
    if re.search(r'find_package\s*\(\s*Qt6\b', text, re.IGNORECASE):
        return 'Qt6'
    if re.search(r'find_package\s*\(\s*Qt5\b', text, re.IGNORECASE):
        return 'Qt5'
    if re.search(r'\bQt6::', text):
        return 'Qt6'
    if re.search(r'\bQt5::', text):
        return 'Qt5'
    return None


def _resolve_cmake_token(token: str, build_variables: dict[str, str] | None) -> list[str]:
    values: list[str] = []
    for resolved in _resolve_variable_variants(token, build_variables):
        values.extend(_clean_token(value) for value in _words(resolved))
    return [value for value in values if value and not _is_variable(value)]


def _cmake_call_parts(body: str, build_variables: dict[str, str] | None) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []
    for resolved in _resolve_variable_variants(body, build_variables):
        tokens = [_clean_token(t) for t in _words(resolved)]
        tokens = [token for token in tokens if token and not _is_variable(token)]
        if tokens:
            calls.append((tokens[0], tokens[1:]))
    return calls


def _cmake_components(args: list[str], build_variables: dict[str, str] | None) -> list[str]:
    """Extract explicit CMake COMPONENTS values without treating options as components."""
    components: list[str] = []
    in_components = False
    saw_component_boundary = False
    for idx, token in enumerate(args):
        values = _resolve_cmake_token(token, build_variables)
        if not values:
            continue
        low_values = [value.lower() for value in values]
        if any(low in _CMAKE_COMPONENT_BOUNDARIES for low in low_values):
            in_components = True
            saw_component_boundary = True
            continue
        if in_components and any(low in _CMAKE_COMPONENT_STOPWORDS for low in low_values):
            in_components = False
            continue
        if not in_components:
            continue

        prev = args[idx - 1].lower() if idx else ''
        if prev in {'version'}:
            continue
        for value in values:
            low = value.lower()
            if low in _CMAKE_COMPONENT_STOPWORDS or low in _CMAKE_COMPONENT_BOUNDARIES:
                continue
            if re.match(r'^\d+(?:\.\d+)*$', value):
                continue
            components.append(value)
    if components or saw_component_boundary:
        return list(dict.fromkeys(components))

    if any(token.lower() in {'names', 'hints', 'paths', 'path_suffixes'} for token in args):
        return []

    saw_option = False
    for token in args:
        values = _resolve_cmake_token(token, build_variables)
        for value in values:
            low = value.lower()
            if low in _CMAKE_COMPONENT_STOPWORDS:
                saw_option = True
                continue
            if not saw_option:
                continue
            if re.match(r'^\d+(?:\.\d+)*$', value):
                continue
            components.append(value)
    return list(dict.fromkeys(components))


def _pkgconfig_aliases(text: str) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for body in _PKG_CHECK_RE.findall(text):
        tokens = [_clean_token(t) for t in _words(body)]
        if len(tokens) < 2:
            continue
        variable = tokens[0].lower()
        modules = tuple(
            token for token in tokens[1:]
            if _looks_like_pkgconfig_token(token)
        )
        if variable and modules:
            aliases[variable] = modules
    return aliases


def _cmake_imported_target_artifacts(
    text: str,
    pkgconfig_aliases: dict[str, tuple[str, ...]] | None = None,
    build_variables: dict[str, str] | None = None,
    qt_hint: str | None = None,
) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for resolved in _resolve_variable_variants(text, build_variables):
        for namespace, component in _IMPORTED_TARGET_RE.findall(resolved):
            if not namespace or not component:
                continue
            if namespace.lower() == 'pkgconfig' and pkgconfig_aliases:
                for module in pkgconfig_aliases.get(component.lower(), ()):
                    artifacts.append(Artifact('pkgconfig', module))
                continue
            if qt_hint and namespace.lower() == 'qt':
                namespace = qt_hint
            artifacts.append(Artifact('cmake_package', namespace, (component,)))
    return artifacts


def _cmake_link_library_artifacts(body: str, build_variables: dict[str, str] | None) -> list[Artifact]:
    artifacts: list[Artifact] = []
    text = _IMPORTED_TARGET_RE.sub(' ', body)
    for resolved in _resolve_variable_variants(text, build_variables):
        tokens = [_clean_token(t) for t in _words(resolved)]
        if len(tokens) < 2:
            continue
        for token in tokens[1:]:
            low = token.lower()
            if low in _LINK_LIBRARY_KEYWORDS or _is_variable(token):
                continue
            if re.match(r'^\d+(?:\.\d+)*$', token) or token.isupper():
                continue
            if '/' in token or token.endswith(('.a', '.so', '.dylib', '.lib')):
                continue
            lib = token[2:] if token.startswith('-l') else token
            if lib and lib.lower() not in NOISE_LIBS:
                artifacts.append(Artifact('library', lib))
    return artifacts


def _cmake_external_name(body: str) -> str | None:
    tokens = [_clean_token(token) for token in _words(body)]
    tokens = [token for token in tokens if token and not _is_variable(token)]
    if not tokens:
        return None
    if tokens[0].lower() == 'name' and len(tokens) > 1:
        return tokens[1]
    return tokens[0]


def _fetchcontent_declared_names(text: str) -> set[str]:
    names: set[str] = set()
    for body in _CMAKE_FETCH_DECLARE_RE.findall(text):
        if not re.search(r'\b(?:FIND_PACKAGE_ARGS|OVERRIDE_FIND_PACKAGE)\b', body, re.IGNORECASE):
            continue
        name = _cmake_external_name(body)
        if name:
            names.add(name)
    return names


def _cmake_find_module_artifact(filepath: str, text: str) -> Artifact | None:
    basename = os.path.basename(filepath)
    match = re.match(r'Find([A-Za-z0-9_+.-]+)\.cmake(?:\.in|\.in\.in)?$', basename, re.IGNORECASE)
    if match:
        return Artifact('cmake_package', match.group(1), ())
    for module in re.findall(r'\b(?:Find|Use)([A-Za-z0-9_+.-]+)\.cmake\b', text, re.IGNORECASE):
        return Artifact('cmake_package', module, ())
    return None


def _cmake_provider_has_package(name: str) -> bool:
    normalized = name.lower().replace('-', '').replace('_', '')
    raw_name = name.lower()
    index = _provider_index()
    return normalized in index.cmake or raw_name in index.cmake


def _cmake_set_dependency_artifacts(text: str, build_variables: dict[str, str] | None) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for body in _CMAKE_SET_CALL_RE.findall(text):
        for variable, values in _cmake_call_parts(body, build_variables):
            if not _CMAKE_DEP_LIST_VAR_RE.search(variable):
                continue
            for value in values:
                if _is_variable(value) or value.isupper() or '/' in value:
                    continue
                if re.match(r'^\d+(?:\.\d+)*$', value):
                    continue
                if _cmake_provider_has_package(value):
                    artifacts.append(Artifact('cmake_package', value, ()))
    return artifacts


def _cmake_artifacts(text: str, build_variables: dict[str, str] | None = None) -> list[Artifact]:
    artifacts: list[Artifact] = []
    qt_hint = _detect_qt_version_hint(text)
    pkgconfig_aliases = _pkgconfig_aliases(text)
    for _func, body in _CMAKE_CALL_RE.findall(text):
        for package, args in _cmake_call_parts(body, build_variables):
            if not package:
                continue
            # Resolve Qt${QT_MAJOR_VERSION} or bare QT/Qt -> Qt5/Qt6 based on context
            if qt_hint and (re.match(r'^Qt\$', package) or package.upper() == 'QT'):
                package = qt_hint
            components = _cmake_components(args, build_variables)
            artifacts.append(Artifact('cmake_package', package, tuple(dict.fromkeys(components))))
            artifacts.extend(_cmake_imported_target_artifacts(body, pkgconfig_aliases, build_variables, qt_hint))
            artifacts.extend(_protobuf_compiler_artifacts(package, text))
    for name in _CMAKE_INCLUDE_FIND_RE.findall(text):
        if name and not _is_variable(name):
            artifacts.append(Artifact('cmake_package', name, ()))
    for name in _CMAKE_FPHSA_RE.findall(text):
        if name and not _is_variable(name):
            artifacts.append(Artifact('cmake_package', name, ()))
    for body in _CMAKE_FETCH_RE.findall(text):
        name = _cmake_external_name(body)
        if name:
            artifacts.append(Artifact('cmake_package', name, ()))
    for name in _fetchcontent_declared_names(text):
        artifacts.append(Artifact('cmake_package', name, ()))
    for body in _CMAKE_TARGET_LINK_RE.findall(text):
        artifacts.extend(_cmake_imported_target_artifacts(body, pkgconfig_aliases, build_variables, qt_hint))
        artifacts.extend(_cmake_link_library_artifacts(body, build_variables))
    artifacts.extend(_cmake_set_dependency_artifacts(text, build_variables))
    return artifacts


def _protobuf_compiler_artifacts(package: str, text: str) -> list[Artifact]:
    package_l = package.lower().replace('-', '').replace('_', '')
    if package_l not in {'protobuf', 'protobufc'}:
        return []
    if not re.search(
        r'\b(?:protobuf_generate(?:_cpp)?|protobuf::protoc|PROTOBUF_PROTOC_EXECUTABLE|[A-Za-z0-9_./+-]+\.proto)\b',
        text,
        re.IGNORECASE,
    ):
        return []
    return [Artifact('executable', 'protoc')]


def _provider_index_cmake_packages(package: str, components: tuple[str, ...], valid_packages: set[str]) -> set[str]:
    pkgs: set[str] = set()
    name = package.lower().replace('-', '').replace('_', '')
    raw_name = package.lower()
    debian_name = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', package).lower()
    debian_name = debian_name.replace('_', '-')
    component_names = {c.lower().replace('-', '').replace('_', '') for c in components}
    index = _provider_index()

    pkgs.update(resolve_from_distro_mapping('cmake', name))
    pkgs.update(resolve_from_distro_mapping('cmake', raw_name))
    pkgs.update(_select_provider_candidates(index.cmake.get(name, set()), valid_packages, artifact_name=raw_name, section='cmake'))
    pkgs.update(_select_provider_candidates(index.cmake.get(raw_name, set()), valid_packages, artifact_name=raw_name, section='cmake'))
    header_only = _header_only_provider_from_token(package, valid_packages)
    if header_only:
        pkgs.add(header_only)
    if name in {'qt5', 'qt6'} or name.startswith(('qt5', 'qt6')):
        version = 'Qt6' if name.startswith('qt6') else 'Qt5'
        if name in {'qt5', 'qt6'}:
            pkgs.update(_qt_component_packages(version, components, valid_packages))
        else:
            component = name[3:]
            pkgs.update(_qt_component_packages(version, (component,), valid_packages))
            pkgs.update(_qt_component_packages(version, components, valid_packages))
            if component == 'linguisttools':
                if version == 'Qt5':
                    pkgs.update({'qttools5-dev', 'qttools5-dev-tools'})
                else:
                    pkgs.update({'qt6-tools-dev', 'qt6-tools-dev-tools'})
            if component == 'svg':
                pkgs.update(_qt_component_packages(version, ('svg',), valid_packages))
            if component == 'private':
                pkgs.add('qt6-base-private-dev' if version == 'Qt6' else 'qtbase5-private-dev')

    if name == 'qt':
        qt6_hint = any(c.lower().replace('_', '').replace('-', '') in {'6', 'qt6'} for c in components)
        qt5_hint = any(c.lower().replace('_', '').replace('-', '') in {'5', 'qt5'} for c in components)
        version = 'Qt6' if qt6_hint and not qt5_hint else 'Qt5'
        pkgs.update(_qt_component_packages(version, components, valid_packages))

    if component_names & {'linguisttools'}:
        if name.startswith('qt6') or package == 'Qt6':
            pkgs.update({'qt6-tools-dev', 'qt6-tools-dev-tools'})
        elif name.startswith('qt5') or package == 'Qt5':
            pkgs.update({'qttools5-dev', 'qttools5-dev-tools'})

    unversioned_debian_name = re.sub(r'\d+$', '', debian_name)
    for candidate in (
        f'{debian_name}-cmake',
        f'lib{debian_name}-dev',
        f'lib{unversioned_debian_name}-dev',
    ):
        if candidate in valid_packages:
            pkgs.add(candidate)

    # Fallback: lib<name>-dev (same pattern as _library_packages)
    if not pkgs:
        fallback = f'lib{raw_name}-dev'
        if fallback in valid_packages:
            pkgs.add(fallback)

    raw_bases = list(dict.fromkeys((raw_name, re.sub(r'\d+$', '', raw_name))))
    compact_bases = list(dict.fromkeys((name, re.sub(r'\d+$', '', name))))
    for component in components:
        comp = component.lower()
        compact_comp = comp.replace('-', '').replace('_', '')
        for raw_base, compact_base in zip(raw_bases, compact_bases):
            for token in dict.fromkeys((
                f'{raw_base}{comp}',
                f'{raw_base}-{comp}',
                f'{raw_base}_{comp}',
                f'{compact_base}{compact_comp}',
            )):
                if token in index.pkgconfig:
                    pkgs.update(_pkgconfig_packages(token, valid_packages))

    return _stable_packages(pkgs, valid_packages)




def _normal_pkgconfig_token(token: str) -> str:
    token = token.strip().strip('"\'`()[]{}<>:,').lower()
    token = token[:-3] if token.endswith('.pc') else token
    return token.replace('_', '-')


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


def _looks_like_pkgconfig_token(token: str) -> bool:
    low = token.lower()
    normalized = _normal_pkgconfig_token(token).strip('.')
    if low in _PKGCONFIG_KEYWORDS or _is_variable(token):
        return False
    if normalized in {'pc'}:
        return False
    if normalized.startswith('ac-') or normalized.startswith('_pkg') or normalized.startswith('pkg-') or token.isupper():
        return False
    if normalized.endswith(('-cflags', '-libs', '-cppflags')):
        return False
    if normalized in {
        'pkg-config', 'pkgconf', 'pkg_config', 'cflags', 'libs', 'modversion', 'variable',
        'if', 'else', 'endif', 'then', 'fi', 'test', 'yes', 'no', 'true', 'false',
        'set', 'get', 'is', 'are', 'with', 'without', 'for', 'the', 'and', 'or',
        'to', 'from', 'in', 'of', 'not', 'be', 'found', 'required', 'install',
        'ac_subst', 'ac_define', 'ac_msg_error', 'as_if', 'dnl', 'null', 'dev',
        'pkg_check_modules', 'pkg_check_exists', 'pkg_check_var', 'modules',
        'action-if-found', 'action-if-not-found', 'built-in', 'arch-independent',
        'apache-cppflags', 'modules', 'curl', 'define', 'cflags', 'libs', 'fi', 'dnl',
        'action-if-found', 'action-if-not-found',
    }:
        return False
    if low.startswith('-') or re.match(r'^\d+(?:\.\d+)*$', token):
        return False
    if normalized in _provider_index().pkgconfig:
        return True
    return (
        normalized.endswith('.pc')
        or (('-' in normalized or re.search(r'\d', normalized)) and re.fullmatch(r'[a-z0-9_+.-]+', normalized))
        or normalized in {'openssl', 'alsa', 'zlib', 'krb5', 'fcitx', 'systemd', 'udev', 'libcurl'}
    )


def _explicit_package_artifact_token(token: str) -> bool:
    token_l = token.lower()
    return token_l.endswith((
        '-dev', '-tools', '-tool', '-bin', '-utils', '-headers', '-compiler',
    )) or '-dev-' in token_l


def _pkgconfig_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for body in _PKG_CHECK_RE.findall(text):
        tokens = [_clean_token(t) for t in _words(body)]
        if len(tokens) < 2:
            continue
        for token in tokens[1:]:
            if not _looks_like_pkgconfig_token(token):
                continue
            artifacts.append(Artifact('pkgconfig', token))
    for match in re.finditer(r'\bPKG_CHECK_VAR\s*\((.*?)\)', text, re.IGNORECASE | re.DOTALL):
        tokens = [_clean_token(t) for t in _words(match.group(1))]
        for token in tokens[1:2]:
            if _looks_like_pkgconfig_token(token):
                artifacts.append(Artifact('pkgconfig', token))
    for body in _CMAKE_PKG_GET_VARIABLE_RE.findall(text):
        tokens = [_clean_token(t) for t in _words(body)]
        # pkg_get_variable(out_var module variable)
        for token in tokens[1:2]:
            if _looks_like_pkgconfig_token(token):
                artifacts.append(Artifact('pkgconfig', token))
    for token in _MESON_DEP_RE.findall(text):
        if _looks_like_pkgconfig_token(token):
            artifacts.append(Artifact('pkgconfig', token))
    artifacts.extend(_meson_dependency_component_artifacts(text))
    for body in _MESON_DEP_LIST_RE.findall(text):
        for token in _words(body):
            if _looks_like_pkgconfig_token(token):
                artifacts.append(Artifact('pkgconfig', token))
    for line in _QMAKE_PKGCONFIG_RE.findall(text):
        for token in _words(line):
            if _looks_like_pkgconfig_token(token):
                artifacts.append(Artifact('pkgconfig', token))
    for line in _MAKE_PKG_CONFIG_RE.findall(text):
        if re.search(r'(?i)\b(?:should|available|module|ACTION-IF|Autoconf|macro|checking)\b', line):
            continue
        for token in _words(line):
            low = token.lower()
            if not _looks_like_pkgconfig_token(token):
                continue
            artifacts.append(Artifact('pkgconfig', token))
    return artifacts


def _meson_dependency_component_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for body in _MESON_DEP_CALL_RE.findall(text):
        package_match = re.match(r'\s*[\'"]([^\'"]+)[\'"]', body)
        if not package_match:
            continue
        package = package_match.group(1)
        components: list[str] = []
        for modules_body in re.findall(r'\bmodules\s*:\s*\[(.*?)\]', body, re.IGNORECASE | re.DOTALL):
            components.extend(_clean_token(token) for token in _words(modules_body))
        scalar_module = re.search(r'\bmodules\s*:\s*[\'"]([^\'"]+)[\'"]', body, re.IGNORECASE)
        if scalar_module:
            components.append(_clean_token(scalar_module.group(1)))
        components = [
            component for component in dict.fromkeys(components)
            if component and not _is_variable(component) and not component.isupper()
        ]
        if components:
            artifacts.append(Artifact('cmake_package', package, tuple(components)))
    return artifacts



def _dependency_list_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for build_system in ('cmake', 'meson', 'makefile', 'autoconf', 'configure_script', 'bootstrap'):
        for statement in find_dependency_lists(text, build_system):
            tokens = list(dict.fromkeys(
                concrete_tokens(statement.value) + _provider_backed_dependency_tokens(statement.value)
            ))
            for token in tokens:
                if token.startswith('-l') and len(token) > 2:
                    artifacts.append(Artifact('library', token[2:]))
                elif _looks_like_package_name(token):
                    artifacts.append(Artifact('package', token))
                    if (
                        not _explicit_package_artifact_token(token)
                        and _normal_pkgconfig_token(token) in _provider_index().pkgconfig
                    ):
                        artifacts.append(Artifact('pkgconfig', token))
                elif _looks_like_pkgconfig_token(token):
                    artifacts.append(Artifact('pkgconfig', token))
    return artifacts


def _provider_backed_dependency_tokens(value: str) -> list[str]:
    index = _provider_index()
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r'(?<![A-Za-z0-9_+./-])-l[A-Za-z0-9_+.-]+|[A-Za-z0-9_+.-]+(?:\.pc)?', value):
        token = clean_token(raw)
        if not token or is_symbolic_token(token):
            continue
        if token.startswith('-l') and len(token) > 2:
            normalized = token[2:].lower()
        else:
            normalized = _normal_pkgconfig_token(token.lower())
        if normalized in seen:
            continue
        if (
            normalized in index.pkgconfig
            or normalized in index.libraries
            or normalized in index.cmake
            or normalized in index.executables
            or normalized in index.config_scripts
        ):
            seen.add(normalized)
            tokens.append(normalized)
    return tokens


def _normalize_header(header: str) -> str:
    return header.strip().strip('"\'`()[]{}<>:,').lower()


def _header_suffixes(
    header: str,
    max_depth: int | None = None,
    shortest_first: bool = False,
) -> list[str]:
    normalized = _normalize_header(header).lstrip('/')
    if normalized.startswith('usr/include/'):
        normalized = normalized[len('usr/include/'):]
    parts = [part for part in normalized.split('/') if part]
    suffixes = [
        '/'.join(parts[idx:])
        for idx in range(len(parts))
        if max_depth is None or len(parts) - idx <= max_depth
    ]
    if shortest_first:
        suffixes.reverse()
    return suffixes


def _header_stem(header: str) -> str:
    normalized = _normalize_header(header).lstrip('/')
    if normalized.startswith('usr/include/'):
        normalized = normalized[len('usr/include/'):]
    return normalized.split('/')[0].split('.')[0] if '/' in normalized else normalized.rsplit('.', 1)[0]


def _header_affinity_names(header: str) -> tuple[str, ...]:
    normalized = _normalize_header(header).lstrip('/')
    if normalized.startswith('usr/include/'):
        normalized = normalized[len('usr/include/'):]
    basename = normalized.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    return tuple(dict.fromkeys((_header_stem(normalized), basename)))


def _is_standard_header(header: str) -> bool:
    return _normalize_header(header) in _STANDARD_HEADERS


def _header_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for body in _HEADER_CALL_RE.findall(text):
        for header in _HEADER_TOKEN_RE.findall(body):
            header = _normalize_header(header)
            if header and not _is_standard_header(header):
                artifacts.append(Artifact('header', header))
    return artifacts


def _program_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for regex in (_CMAKE_FIND_PROGRAM_RE, _MESON_FIND_PROGRAM_RE):
        for body in regex.findall(text):
            tokens = [_clean_token(t) for t in _words(body)]
            if not tokens:
                continue
            candidates = tokens[1:] if len(tokens) > 1 else tokens
            for token in candidates:
                low = token.lower()
                if low in _FIND_PROGRAM_KEYWORDS or _is_variable(token):
                    continue
                if token.isupper():
                    continue
                artifacts.append(Artifact('executable', token))
    for _var, value in _TOOL_VAR_RE.findall(text):
        for token in _words(value):
            if not _is_variable(token) and not token.isupper():
                artifacts.append(Artifact('executable', token))
    for token in _COMMAND_EXEC_RE.findall(text):
        if not _is_variable(token) and not token.isupper():
            artifacts.append(Artifact('executable', token))
    index = _provider_index()
    for body in _BUILD_ACTION_RE.findall(text):
        for quoted in _QUOTED_TEXT_RE.findall(body):
            words = _words(quoted)
            if not words:
                continue
            token = os.path.basename(words[0]).lower()
            if token in index.executables or token in index.config_scripts:
                artifacts.append(Artifact('executable', token))
    return artifacts


def _qmake_qt_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for line in _QMAKE_QT_RE.findall(text):
        components = [token for token in _words(line) if not _is_variable(token)]
        if components:
            artifacts.append(Artifact('qt', 'Qt5', tuple(dict.fromkeys(components))))
    if _QMAKE_TRANSLATIONS_RE.search(text):
        artifacts.append(Artifact('feature', 'qt_translation', ('qt5',)))
    return artifacts


def _boost_autoconf_artifacts(text: str) -> list[Artifact]:
    if re.search(r'^\s*(?:m4_define|AC_DEFUN(?:_ONCE)?|BOOST_DEFUN)\s*\(', text, re.MULTILINE):
        return []

    components: list[str] = []
    for target_component in _BOOST_TARGET_RE.findall(text):
        target_component = target_component.lower()
        if target_component in BOOST_LINK_COMPONENTS:
            components.append(target_component)
    for line in text.splitlines():
        linked = {name.lower() for name in _BOOST_LIB_VAR_RE.findall(line)}
        for name in linked:
            if name in BOOST_LINK_COMPONENTS:
                components.append(name)

        stripped = line.strip()
        for match in _BOOST_MACRO_RE.finditer(stripped):
            name = match.group(1)
            name_u = name.upper()
            name_l = name.lower()
            if name_u in BOOST_NON_COMPONENT_MACROS:
                continue
            if match.group(0).upper().startswith('AX_BOOST_') and name_l in BOOST_LINK_COMPONENTS:
                components.append(name_l)

    if components:
        return [Artifact('cmake_package', 'Boost', tuple(dict.fromkeys(components)))]
    if re.search(r'\b(?:AX_)?BOOST_REQUIRE\b|\bfind_package\s*\(\s*Boost\b', text, re.IGNORECASE):
        return [Artifact('cmake_package', 'Boost', ('headers',))]
    return []


def _generic_autoconf_macro_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for lib_name, path_name in _GENERIC_AUTOCONF_DEP_MACRO_RE.findall(text):
        name = (lib_name or path_name).lower().strip('_')
        if not name or name in {'pthreads', 'pthread'}:
            continue
        normalized = _normal_pkgconfig_token(name)
        normalized = re.sub(r'-(\d+(?:-\d+)+)$', lambda m: '-' + m.group(1).replace('-', '.'), normalized)
        if lib_name:
            artifacts.append(Artifact('library', normalized))
            artifacts.append(Artifact('pkgconfig', f'lib{normalized}'))
        else:
            artifacts.append(Artifact('pkgconfig', normalized))
    return artifacts


def _autoconf_libltdl_artifacts(text: str) -> list[Artifact]:
    if not re.search(r'\b(?:AC_LIBLTDL[A-Z0-9_]*|AC_LTDL[A-Z0-9_]*|LT_LIB_DLLOAD|LTDL_INIT)\b', text, re.IGNORECASE):
        return []
    return [Artifact('library', 'ltdl'), Artifact('header', 'ltdl.h')]


def _library_artifacts(text: str) -> list[Artifact]:
    artifacts = [
        Artifact('library', lib)
        for lib in _LFLAG_RE.findall(text)
        if lib.lower() not in NOISE_LIBS
    ]
    for regex in (_AC_CHECK_LIB_RE, _CMAKE_CHECK_LIBRARY_RE, _FIND_LIBRARY_RE):
        for lib in regex.findall(text):
            if lib.lower() not in NOISE_LIBS:
                artifacts.append(Artifact('library', lib))
    for bracketed, bare in _AC_CHECK_LIBS_RE.findall(text):
        for lib in _words(bracketed or bare):
            lib_l = lib.lower()
            if lib_l and lib_l not in NOISE_LIBS and lib_l not in {'none', 'required'}:
                artifacts.append(Artifact('library', lib))
    for bracketed, bare in _AC_SEARCH_LIBS_RE.findall(text):
        for lib in _words(bracketed or bare):
            lib_l = lib.lower()
            if lib_l and lib_l not in NOISE_LIBS and lib_l not in {'none', 'required'}:
                artifacts.append(Artifact('library', lib))
    for libs_text in _PY_EXT_LIBS_RE.findall(text):
        for token in _words(libs_text):
            token = token[2:] if token.startswith('-l') else token
            if token and token.lower() not in NOISE_LIBS:
                artifacts.append(Artifact('library', token))
    for libs_text in _BUILD_LIBS_LIST_RE.findall(text):
        for token in _words(libs_text):
            if token and token.lower() not in NOISE_LIBS and not token.isupper():
                artifacts.append(Artifact('library', token))
            elif token and token.lower() not in NOISE_LIBS and token.lower() in _provider_index().libraries:
                artifacts.append(Artifact('library', token))
    for libs_text in _BUILD_LIBS_APPEND_RE.findall(text):
        for token in _words(libs_text):
            if token and token.lower() not in NOISE_LIBS and token.lower() in _provider_index().libraries:
                artifacts.append(Artifact('library', token))
    return artifacts


def _language_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for gem in _RUBY_GEM_RE.findall(text):
        artifacts.append(Artifact('ruby_gem', gem))
    for module in _PERL_MODULE_RE.findall(text):
        artifacts.append(Artifact('perl_module', module))
    for name in _OCAML_CONF_RE.findall(text):
        artifacts.append(Artifact('ocaml_conf', name))
    return artifacts


def _feature_artifacts(text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for regex, feature in _FEATURE_PATTERNS:
        if regex.search(text):
            artifacts.append(Artifact('feature', feature))
    return artifacts


def _expanded_feature_artifacts(artifact: Artifact) -> list[Artifact]:
    name = artifact.name.lower()
    if artifact.kind != 'feature':
        return [artifact]
    if name == 'gir_generation':
        return [
            artifact,
            Artifact('executable', 'g-ir-scanner'),
            Artifact('pkgconfig', 'gobject-introspection-1.0'),
        ]
    if name == 'gtk_doc_generation':
        return [
            artifact,
            Artifact('executable', 'gtkdocize'),
            Artifact('executable', 'gtkdoc-scan'),
            Artifact('executable', 'gtkdoc-mkdb'),
            Artifact('executable', 'gtkdoc-fixxref'),
        ]
    if name == 'qt_translation':
        return [
            artifact,
            Artifact('executable', 'lrelease', artifact.components),
            Artifact('executable', 'lupdate', artifact.components),
        ]
    if name == 'vala_generation':
        return [
            artifact,
            Artifact('executable', 'valac'),
            Artifact('executable', 'vapigen'),
        ]
    if name == 'intltool_i18n':
        return [
            artifact,
            Artifact('executable', 'intltoolize'),
            Artifact('executable', 'intltool-merge'),
            Artifact('executable', 'intltool-update'),
            Artifact('executable', 'intltool-extract'),
        ]
    if name == 'gettext_i18n':
        return [
            artifact,
            Artifact('executable', 'msgfmt'),
            Artifact('executable', 'xgettext'),
            Artifact('executable', 'msgmerge'),
        ]
    if name == 'xml_validation':
        return [
            artifact,
            Artifact('executable', 'xmllint'),
            Artifact('executable', 'xmlcatalog'),
        ]
    if name == 'cmake_extra_modules':
        return [
            artifact,
            Artifact('cmake_package', 'cmake-extras'),
        ]
    return [artifact]


def extract_artifacts(matches: list[RuleMatch], build_variables: dict[str, str] | None = None) -> list[Artifact]:
    """Extract artifact tokens from raw LLM context for post-LLM gap filling."""
    text = _context_text(matches)
    artifacts: list[Artifact] = []
    for match in matches:
        if match.rule_type == 'cmake:find_module_file':
            artifacts.append(Artifact('cmake_package', match.extracted_name, ()))
        elif match.file_path:
            module_artifact = _cmake_find_module_artifact(match.file_path, match.context or match.matched_text)
            if module_artifact:
                artifacts.append(module_artifact)
    artifacts.extend(_cmake_artifacts(text, build_variables))
    artifacts.extend(_pkgconfig_artifacts(text))
    artifacts.extend(_dependency_list_artifacts(text))
    artifacts.extend(_program_artifacts(text))
    artifacts.extend(_qmake_qt_artifacts(text))
    artifacts.extend(_boost_autoconf_artifacts(text))
    artifacts.extend(_generic_autoconf_macro_artifacts(text))
    artifacts.extend(_autoconf_libltdl_artifacts(text))
    artifacts.extend(_header_artifacts(text))
    artifacts.extend(_library_artifacts(text))
    artifacts.extend(_language_artifacts(text))
    artifacts.extend(_feature_artifacts(text))

    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    deduped: list[Artifact] = []
    for artifact in artifacts:
        key = (artifact.kind, artifact.name.lower(), tuple(c.lower() for c in artifact.components))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(artifact)
    return deduped


def _qt_component_packages(version: str, components: tuple[str, ...], valid_packages: set[str]) -> set[str]:
    """Qt component resolver — delegates to component_templates."""
    return component_templates.resolve_qt_components(version, components, valid_packages)


def _kf_component_packages(package: str, components: tuple[str, ...], valid_packages: set[str]) -> set[str]:
    """KDE Frameworks component resolver — delegates to component_templates."""
    return component_templates.resolve_kf_components(package, components, valid_packages)


def _xcb_component_packages(components: tuple[str, ...], valid_packages: set[str]) -> set[str]:
    pkgs: set[str] = set()
    aliases = {'xcb': 'libxcb1-dev', 'xkb': 'libxcb-xkb-dev'}
    for component in components:
        comp = component.lower().replace('_', '-')
        candidate = aliases.get(comp, f'libxcb-{comp}0-dev')
        alt = f'libxcb-{comp}-dev'
        if candidate in valid_packages:
            pkgs.add(candidate)
        elif alt in valid_packages:
            pkgs.add(alt)
    return pkgs


def _library_packages(lib: str, valid_packages: set[str]) -> set[str]:
    lib = lib.lower()
    pkgs: set[str] = set()

    preferred = _preferred_library_package(lib, valid_packages)
    for lookup_name in _library_lookup_names(lib):
        provider_candidates = _provider_index().libraries.get(lookup_name, set())
        if provider_candidates:
            if preferred and preferred in provider_candidates:
                return {preferred}
            provider_pkgs = _select_provider_candidates(
                provider_candidates,
                valid_packages,
                artifact_name=lookup_name,
                section='libraries',
            )
            if provider_pkgs:
                return provider_pkgs
            if preferred:
                return {preferred}

    boost_pkg = boost_library_package(lib, valid_packages)
    if boost_pkg:
        pkgs.add(boost_pkg)
        return _stable_packages(pkgs, valid_packages)

    for lookup_name in _library_lookup_names(lib):
        dm = resolve_from_distro_mapping('library', lookup_name)
        pkgs.update(p for p in dm if p in valid_packages)

    candidate = f'lib{lib}-dev'
    if candidate in valid_packages:
        pkgs.add(candidate)

    return _stable_packages(pkgs, valid_packages)


def _library_lookup_names(lib: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((
        lib,
        re.sub(r'(?<=[a-z])-(?=\d)', '', lib),
    )))


def _preferred_library_package(lib: str, valid_packages: set[str]) -> str | None:
    preferred = {
        'z': 'zlib1g-dev',
        'ssl': 'libssl-dev',
        'crypto': 'libssl-dev',
    }.get(lib)
    if preferred and preferred in valid_packages:
        return preferred

    boost_pkg = boost_library_package(lib, valid_packages)
    if boost_pkg:
        return boost_pkg

    return None


def _library_package(lib: str, valid_packages: set[str]) -> str | None:
    pkgs = _library_packages(lib, valid_packages)
    return sorted(pkgs)[0] if len(pkgs) == 1 else None


def _pkgconfig_packages(name: str, valid_packages: set[str]) -> set[str]:
    token = name.strip().strip('"\'').lower()
    token = re.split(r'\s*(?:[<>=!~]=?|;)', token, 1)[0].strip()
    token = token[:-3] if token.endswith('.pc') else token
    direct_data_package = set()
    if token in valid_packages and re.search(r'-(?:db|proto)$', token):
        direct_data_package.add(token)
    if token.startswith('qt5'):
        component = token[3:]
        return _qt_component_packages('Qt5', (component,), valid_packages)
    if token.startswith('qt6'):
        component = token[3:]
        return _qt_component_packages('Qt6', (component,), valid_packages)

    seed = set(resolve_from_distro_mapping('pkgconfig', token))
    pkgs = _packages_from_index_family(seed & valid_packages, valid_packages)
    provider_pkgs = _select_provider_candidates(
        _provider_index().pkgconfig.get(token, set()),
        valid_packages,
        artifact_name=token,
        section='pkgconfig',
    )
    pkgs.update(provider_pkgs)
    if '-' in token:
        pkgs.update(_small_same_source_dev_siblings(seed & valid_packages, valid_packages))
    return _stable_packages(pkgs - provider_pkgs, valid_packages) | provider_pkgs | direct_data_package


def _qml_import_packages(name: str, valid_packages: set[str]) -> set[str]:
    normalized = name.lower()
    module_map = {
        'qtmultimedia': 'qml-module-qtmultimedia',
        'qtquick': 'qml-module-qtquick2',
        'qtquick.controls': 'qml-module-qtquick-controls',
        'qtquick.controls.2': 'qml-module-qtquick-controls2',
        'qtquick.dialogs': 'qml-module-qtquick-dialogs',
        'qtquick.layouts': 'qml-module-qtquick-layouts',
        'qtquick.window': 'qml-module-qtquick-window2',
        'qtgraphicaleffects': 'qml-module-qtgraphicaleffects',
    }
    pkg = module_map.get(normalized)
    return {pkg} if pkg in valid_packages else set()


def _same_source_sibling_packages(seed_packages: set[str], valid_packages: set[str]) -> set[str]:
    return _same_source_stable_siblings(seed_packages, valid_packages)


def _feature_sibling_packages(feature: str, seed_packages: set[str], valid_packages: set[str]) -> set[str]:
    siblings = _same_source_sibling_packages(seed_packages, valid_packages)
    if feature == 'gir_generation':
        return {
            pkg for pkg in siblings
            if pkg == 'gobject-introspection'
            or pkg in {'libgirepository1.0-dev', 'libgirepository-2.0-dev'}
        }
    if feature == 'gtk_doc_generation':
        return {pkg for pkg in siblings if pkg.startswith('gtk-doc')}
    if feature == 'vala_generation':
        return {pkg for pkg in siblings if pkg in {'valac', 'valac-bin'} or pkg.startswith('libvala-')}
    if feature == 'intltool_i18n':
        return {pkg for pkg in siblings if pkg == 'intltool' or pkg.startswith('intltool')}
    if feature == 'qt_translation':
        return {
            pkg for pkg in siblings
            if (
                pkg in {'qttools5-dev', 'qttools5-dev-tools', 'qt6-tools-dev', 'qt6-tools-dev-tools', 'qt6-l10n-tools', 'qtchooser'}
                or pkg.endswith('-l10n-tools')
            )
        }
    if feature == 'cmake_extra_modules':
        return {pkg for pkg in siblings if pkg.endswith('cmake-modules') or pkg == 'cmake-extras'}
    return set()


def _header_index_suffix_packages(
    header: str,
    index: ProviderIndex,
    valid_packages: set[str],
) -> set[str]:
    affinity_names = _header_affinity_names(header)
    best: set[str] = set()
    for suffix in _header_suffixes(
        header,
        max_depth=_HEADER_SUFFIX_MAX_DEPTH,
        shortest_first=True,
    ):
        candidates = index.headers_suffix.get(suffix, set())
        if not candidates or len(candidates) > _HEADER_SUFFIX_HARD_FANOUT_LIMIT:
            continue
        basename = suffix.rsplit('/', 1)[-1]
        if '/' not in suffix and basename in _GENERIC_BASENAME_HEADERS:
            continue
        for affinity_name in affinity_names:
            resolved = _select_provider_candidates(
                candidates,
                valid_packages,
                artifact_name=affinity_name,
                section='headers_suffix',
            )
            if len(resolved) == 1:
                best = resolved
                if '/' not in suffix and len(candidates) == 1:
                    return resolved
                if '/' in suffix:
                    return resolved
    return best


def _header_packages(header: str, valid_packages: set[str]) -> set[str]:
    normalized = header.lower().strip().lstrip('/')
    if normalized.startswith('usr/include/'):
        normalized = normalized[len('usr/include/'):]
    basename = normalized.rsplit('/', 1)[-1]
    default_suffixes, default_basenames = load_default_header_filter()

    pkgs = set(p for p in resolve_from_distro_mapping('header', normalized) if p in valid_packages)
    if basename not in default_basenames:
        pkgs.update(p for p in resolve_from_distro_mapping('header', basename) if p in valid_packages)
    header_only = _header_only_provider_from_token(normalized, valid_packages)
    if header_only:
        pkgs.add(header_only)
    if re.fullmatch(r'private/q[a-z0-9_]+_p\.h', normalized) and 'qtbase5-private-dev' in valid_packages:
        pkgs.add('qtbase5-private-dev')

    header_stem = _header_stem(normalized)
    index = _provider_index()
    provider_pkgs: set[str] = set()
    seed = _select_provider_candidates(
        index.headers_exact.get(normalized, set()),
        valid_packages,
        artifact_name=header_stem,
        section='headers_exact',
    )
    pkgs.update(seed)
    provider_pkgs.update(seed)
    if not provider_pkgs and _is_standard_header(normalized):
        return set()
    if not provider_pkgs and any(suffix in default_suffixes for suffix in header_path_suffixes(normalized)):
        return set()
    if not seed:
        seed = _header_index_suffix_packages(normalized, index, valid_packages)
        pkgs.update(seed)
        provider_pkgs.update(seed)
    if normalized.startswith('glib-2.0/') or basename == 'glib.h':
        pkgs.update(_same_source_named_packages(seed, valid_packages, {'libglib2.0-dev'}))
    if normalized.startswith('gobject-introspection-') or basename == 'girepository.h':
        pkgs.update(_same_source_named_packages(
            seed, valid_packages, {'libgirepository1.0-dev', 'libgirepository-1.0-dev'}
        ))
    return _stable_packages(pkgs - provider_pkgs, valid_packages) | provider_pkgs


def _executable_packages(
    name: str,
    valid_packages: set[str],
    hints: tuple[str, ...] = (),
) -> set[str]:
    token = name.lower()
    pkgs = set(resolve_from_distro_mapping('executable', token))
    pkgs.update(EXECUTABLE_NORMALIZATION.get(token, set()))
    if token in valid_packages and ('-' in token or re.search(r'\d', token)):
        pkgs.add(token)
    pkgs.update(_select_executable_provider_candidates(
        token,
        _provider_index().executables.get(token, set()),
        valid_packages,
        hints,
    ))
    if 'qt5' in hints:
        pkgs = {pkg for pkg in pkgs if not pkg.startswith('qt6')}
    elif 'qt6' in hints:
        pkgs = {pkg for pkg in pkgs if not pkg.startswith('qttools5')}
    if token.startswith('gtkdoc') and 'gtk-doc-tools' in valid_packages:
        pkgs.add('gtk-doc-tools')
    if token == 'g-ir-scanner':
        pkgs.discard('gobject-introspection-bin')
    return {pkg for pkg in pkgs if _executable_provider_allowed(pkg, token, valid_packages, hints)}


def packages_from_artifacts(artifacts: list[Artifact], valid_packages: set[str]) -> set[str]:
    """Map only high-confidence component/artifact gaps to package names."""
    pkgs: set[str] = set()
    for artifact in artifacts:
        name_l = artifact.name.lower()
        if artifact.kind == 'cmake_package' and name_l == 'boost':
            for component in artifact.components:
                pkg = boost_component_package(component, valid_packages)
                if pkg:
                    pkgs.add(pkg)
        elif artifact.kind == 'cmake_package' and name_l in {'qt5', 'qt6'}:
            pkgs.update(_qt_component_packages(artifact.name, artifact.components, valid_packages))
        elif artifact.kind == 'cmake_package' and name_l in {'xcb'}:
            pkgs.update(_xcb_component_packages(artifact.components, valid_packages))
        elif artifact.kind == 'cmake_package' and (name_l.startswith(('kf5', 'kf6')) or name_l == 'ecm'):
            pkgs.update(_kf_component_packages(artifact.name, artifact.components, valid_packages))
        elif artifact.kind == 'cmake_package':
            pkgs.update(_provider_index_cmake_packages(artifact.name, artifact.components, valid_packages))
        elif artifact.kind == 'qt':
            pkgs.update(_qt_component_packages(artifact.name, artifact.components, valid_packages))
        elif artifact.kind == 'library':
            pkgs.update(_library_packages(artifact.name, valid_packages))
        elif artifact.kind == 'pkgconfig':
            pkgs.update(_pkgconfig_packages(artifact.name, valid_packages))
        elif artifact.kind == 'qml_import':
            pkgs.update(_qml_import_packages(artifact.name, valid_packages))
        elif artifact.kind == 'header':
            pkgs.update(_header_packages(artifact.name, valid_packages))
        elif artifact.kind == 'executable':
            pkgs.update(_executable_packages(artifact.name, valid_packages, artifact.components))
        elif artifact.kind == 'package':
            if artifact.name in valid_packages:
                pkgs.add(artifact.name)
        elif artifact.kind == 'feature':
            feature_seed_pkgs: set[str] = set()
            for expanded in _expanded_feature_artifacts(artifact):
                if expanded.kind == 'executable':
                    feature_seed_pkgs.update(_executable_packages(
                        expanded.name, valid_packages, expanded.components
                    ))
                elif expanded.kind == 'pkgconfig':
                    feature_seed_pkgs.update(_pkgconfig_packages(expanded.name, valid_packages))
                elif expanded.kind == 'cmake_package':
                    feature_seed_pkgs.update(_provider_index_cmake_packages(
                        expanded.name, expanded.components, valid_packages
                    ))
            pkgs.update(feature_seed_pkgs)
            pkgs.update(_feature_sibling_packages(artifact.name.lower(), feature_seed_pkgs, valid_packages))
    return pkgs


def backfill_artifact_packages(
    per_file: dict,
    artifacts_by_file: dict[str, list[Artifact]],
    valid_packages: set[str],
) -> None:
    """Add package gaps found from artifacts after LLM mapping."""
    existing = {
        pkg
        for deps in per_file.values()
        for dep in deps
        for pkg in dep.get('packages', [])
    }
    additions: set[str] = set()
    for artifacts in artifacts_by_file.values():
        additions.update(packages_from_artifacts(artifacts, valid_packages))
    additions -= existing
    if not additions:
        return
    per_file['__artifact_backfill__'] = [
        {'software_name': pkg, 'packages': [pkg]}
        for pkg in sorted(additions)
    ]
