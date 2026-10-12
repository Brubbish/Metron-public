"""Generative component → package resolvers for Qt / Boost / KDE Frameworks.

Replaces hardcoded family tables (e.g. `_qt_component_packages` and Boost functions in
`provider_normalization.py`) with templates that:
  1. Map a component name to one or more module "stems" (component-level grouping).
  2. Generate candidate Debian package names via per-family naming patterns.
  3. Filter candidates by `valid_packages` (the active Debian pkglist).

This automatically supports new components without code changes — they fall into the
generic stem pattern and are validated by `valid_packages`.
"""
from __future__ import annotations

import re


# ───────────────────────── Qt ─────────────────────────

# Component → module stem(s). Components grouped to the same stem share the same package.
_QT_STEMS: dict[str, tuple[str, ...]] = {
    # qtbase modules
    'core': ('base',), 'gui': ('base',), 'widgets': ('base',),
    'network': ('base',), 'dbus': ('base',), 'test': ('base',),
    'printsupport': ('base',), 'xml': ('base',), 'concurrent': ('base',),
    'opengl': ('base',), 'sql': ('base',),
    'private': ('base-private',), 'coreprivate': ('base-private',),
    'guiprivate': ('base-private',), 'widgetsprivate': ('base-private',),
    # Qt Declarative (QML/Quick)
    'qml': ('declarative',), 'quick': ('declarative',),
    'quickcontrols2': ('declarative',), 'quickwidgets': ('declarative',),
    'declarative': ('declarative',), 'qmlmodels': ('declarative',),
    'qmlworkerscript': ('declarative',), 'quickshapes': ('declarative',),
    'quicktest': ('declarative',), 'quick3d': ('quick3d',),
    'qmlprivate': ('declarative-private',),
    'quickprivate': ('declarative-private',),
    'quickcontrols2private': ('declarative-private',),
    # Qt Tools (always emit -dev and -dev-tools)
    'linguisttools': ('tools',), 'linguist': ('tools',),
    'designer': ('tools',), 'help': ('tools',), 'uitools': ('tools',),
    # Standalone Qt add-on modules
    'svg': ('svg',), 'svgwidgets': ('svg',),
    'multimedia': ('multimedia',), 'x11extras': ('x11extras',),
    'webengine': ('webengine',), 'webchannel': ('webchannel',),
    'charts': ('charts',), 'datavisualization': ('datavis3d',),
    'location': ('location',), 'positioning': ('positioning',),
    'serialport': ('serialport',), 'bluetooth': ('connectivity',),
    'connectivity': ('connectivity',), 'networkauth': ('networkauth',),
    'scxml': ('scxml',), 'speech': ('speech',), 'gamepad': ('gamepad',),
    'sensors': ('sensors',), 'serialbus': ('serialbus',),
    'waylandcompositor': ('wayland',), 'waylandclient': ('wayland',),
    'waylandclientprivate': ('wayland-private',),
    'waylandcompositorprivate': ('wayland-private',),
    'xkbcommonsupportprivate': ('wayland-private',),
    'shadertools': ('shadertools',),
    '3d': ('3d',), '3dcore': ('3d',), '3drender': ('3d',),
    'mqtt': ('mqtt',), 'opcua': ('opcua',), 'websockets': ('websockets',),
    'webview': ('webview',), 'lottie': ('lottie',),
    'virtualkeyboard': ('virtualkeyboard',),
}

_QT_COMP_SKIP_RE = re.compile(
    r'^(?:required|components|qt5|qt6|5|6|'
    r'(?:version|minimum|findpackage|addlibrary|set).*)$',
    re.IGNORECASE,
)


def _qt5_patterns(stem: str) -> list[str]:
    if stem == 'base':
        return ['qtbase5-dev']
    if stem == 'base-private':
        return ['qtbase5-private-dev']
    if stem == 'tools':
        return ['qttools5-dev', 'qttools5-dev-tools']
    if stem == 'declarative':
        return ['qtdeclarative5-dev']
    if stem == 'declarative-private':
        return ['qtdeclarative5-private-dev']
    if stem == 'wayland-private':
        return ['qtwayland5-private-dev']
    if stem == 'x11extras':
        return ['libqt5x11extras5-dev']
    if stem == 'opengl':
        return ['libqt5opengl5-dev']
    return [f'qt{stem}5-dev', f'libqt5{stem}5-dev', f'libqt5{stem}5-desktop-dev']


def _qt6_patterns(stem: str) -> list[str]:
    if stem == 'base':
        return ['qt6-base-dev']
    if stem == 'base-private':
        return ['qt6-base-private-dev']
    if stem == 'tools':
        return ['qt6-tools-dev', 'qt6-tools-dev-tools']
    if stem == 'declarative':
        return ['qt6-declarative-dev']
    if stem == 'declarative-private':
        return ['qt6-declarative-private-dev']
    if stem == 'wayland-private':
        return ['qt6-wayland-private-dev']
    return [f'qt6-{stem}-dev']


def _qt5_fallback(comp: str) -> list[str]:
    return [
        f'qt{comp}5-dev',
        f'qt{comp.replace("-", "")}5-dev',
        f'libqt5{comp}5-dev',
        f'libqt5{comp}5-desktop-dev',
    ]


def _qt6_fallback(comp: str) -> list[str]:
    return [f'qt6-{comp}-dev', f'qt6-{comp.replace("-", "")}-dev']


def resolve_qt_components(
    version: str,
    components: tuple[str, ...],
    valid_packages: set[str],
) -> set[str]:
    """Resolve Qt component artifacts to Debian packages.

    Args:
        version: 'Qt5' or 'Qt6' (case-insensitive).
        components: Component names (e.g. ('Core', 'Gui', 'Network')).
        valid_packages: Active Debian pkglist; only packages in this set are returned.

    Returns:
        Set of package names (e.g. {'qtbase5-dev'}).
    """
    is_qt6 = version.lower().startswith('qt6')
    out: set[str] = set()
    for component in components:
        comp = component.lower().replace('_', '').replace('-', '')
        if _QT_COMP_SKIP_RE.match(comp):
            continue
        stems = _QT_STEMS.get(comp)
        if stems:
            patterns = (_qt6_patterns(s) if is_qt6 else _qt5_patterns(s) for s in stems)
            candidates = [c for group in patterns for c in group]
        else:
            candidates = _qt6_fallback(comp) if is_qt6 else _qt5_fallback(comp)
        for cand in candidates:
            if cand in valid_packages:
                out.add(cand)
    return out


# ───────────────────────── KDE Frameworks ─────────────────────────

def resolve_kf_components(
    version: str,
    components: tuple[str, ...],
    valid_packages: set[str],
) -> set[str]:
    """Resolve KDE Frameworks components (KF5/KF6) to Debian packages.

    The `version` argument is the CMake package name (e.g. 'KF5Wayland'); we extract the
    module name from it as an additional component candidate, matching the previous
    hardcoded resolver's behavior.
    """
    name_l = version.lower()
    if name_l in ('ecm', 'extracmakemodules'):
        return {'extra-cmake-modules'} if 'extra-cmake-modules' in valid_packages else set()
    if name_l.startswith('kf6'):
        prefixes = ('libkf6', 'kf6-')
    elif name_l.startswith('kf5'):
        prefixes = ('libkf5',)
    else:
        prefixes = ('libkf6', 'kf6-', 'libkf5')
    # Build component candidate list: module name (extracted from package) first, then
    # explicit components. This lets `find_package(KF5Wayland COMPONENTS Client)` resolve
    # to `libkf5wayland-dev` even though 'Client' alone wouldn't.
    component_candidates: list[str] = list(components)
    for prefix in ('kf5', 'kf6'):
        if name_l.startswith(prefix) and name_l != prefix:
            component_candidates.insert(0, version[len(prefix):])
            break
    out: set[str] = set()
    # De-dup while preserving order
    for component in tuple(dict.fromkeys(component_candidates)) or (version,):
        comp = component.lower().replace('_', '').replace('-', '')
        # Strip leading kf5/kf6 prefix if user included it in the component name
        for prefix in ('kf5', 'kf6'):
            if comp.startswith(prefix):
                comp = comp[len(prefix):]
        for cand in (f'{prefix}{comp}-dev' for prefix in prefixes):
            if cand in valid_packages:
                out.add(cand)
                break
    return out


# ───────────────────────── Boost ─────────────────────────

# Boost component aliases that don't follow the simple `libboost-{comp}-dev` pattern.
_BOOST_COMPONENT_ALIASES: dict[str, str] = {
    'unit-test-framework': 'test',
    'unit_test_framework': 'test',
    'test-exec-monitor': 'test',
    'prg-exec-monitor': 'test',
    'program-options': 'program-options',
    'program_options': 'program-options',
    'date-time': 'date-time',
    'date_time': 'date-time',
    'python': 'python-dev',  # libboost-python-dev is the python-dev variant
    'mpi-python': 'mpi-python',
    'mpi_python': 'mpi-python',
}

# Stems that don't correspond to a `libboost-{stem}-dev` package (umbrella/meta).
_BOOST_NON_COMPONENT_STEMS = frozenset({
    'all', 'doc', 'cmake', 'diagnostic-definitions', 'disable-deprecated',
    'lib', 'libs', 'root', 'version', 'find_header', 'find_lib',
    'find_libs', 'ldflags', 'ldpath', 'cppflags', 'lib_version', 'require',
    'static', 'defun', 'throw_exception',
})

_BOOST_UMBRELLA_STEMS = frozenset({'headers', 'header'})

# Stems that map to a separate "tools" package (e.g. Boost.BJam).
_BOOST_TOOLS_STEMS = frozenset({'tools'})

# Boost macros that are NOT components (e.g. BOOST_LIBS, BOOST_LDFLAGS).
# Mirrors the legacy BOOST_NON_COMPONENT_MACROS list.
BOOST_NON_COMPONENT_MACROS = frozenset({
    'CPPFLAGS', 'LDFLAGS', 'LDPATH', 'LIBS', 'LIB', 'ROOT', 'VERSION',
    'LIB_VERSION', 'REQUIRE', 'STATIC', 'FIND_HEADER', 'FIND_LIB',
    'FIND_LIBS', 'DEFUN',
})

# Known Boost linkable components. Used by artifact extractors to distinguish
# BOOST_<NAME> macros that map to libraries from generic utility macros.
BOOST_LINK_COMPONENTS = frozenset({
    'atomic', 'chrono', 'container', 'context', 'contract', 'coroutine',
    'date_time', 'exception', 'fiber', 'filesystem', 'graph',
    'graph_parallel', 'iostreams', 'json', 'locale', 'log', 'math', 'mpi',
    'mpi_python', 'nowide', 'numpy', 'program_options', 'python', 'random',
    'regex', 'serialization', 'stacktrace', 'system', 'test', 'thread',
    'timer', 'type_erasure', 'url', 'wave', 'wserialization',
})


def _boost_normalize_component(component: str) -> str | None:
    """Normalize a Boost component name to its `libboost-{X}-dev` stem."""
    comp = component.lower()
    if comp in _BOOST_COMPONENT_ALIASES:
        comp = _BOOST_COMPONENT_ALIASES[comp]
    comp = comp.replace('_', '-')
    if comp in _BOOST_NON_COMPONENT_STEMS or comp.upper() in BOOST_NON_COMPONENT_MACROS:
        return None
    return comp


def resolve_boost_component(
    component: str,
    valid_packages: set[str],
) -> set[str]:
    """Resolve a single Boost component to Debian package(s)."""
    comp = _boost_normalize_component(component)
    if comp is None:
        return set()
    if comp in _BOOST_UMBRELLA_STEMS:
        return {'libboost-dev'} if 'libboost-dev' in valid_packages else set()
    if comp in _BOOST_TOOLS_STEMS:
        return {'libboost-tools-dev'} if 'libboost-tools-dev' in valid_packages else set()
    candidates = [f'libboost-{comp}-dev', f'libboost-{comp}']
    return {c for c in candidates if c in valid_packages}


def resolve_boost_library(
    library: str,
    valid_packages: set[str],
) -> set[str]:
    """Resolve a `boost_X[-mt]` library name to its package via component path."""
    lib = library.lower()
    if not lib.startswith('boost_'):
        return set()
    component = lib[len('boost_'):]
    if component.endswith('-mt'):
        component = component[:-len('-mt')]
    # The alias map keys use both `_` and `-` forms; pass through normalize which handles both.
    return resolve_boost_component(component, valid_packages)


def resolve_boost_header(
    header: str,
    valid_packages: set[str],
) -> set[str]:
    """Resolve a Boost header path to the umbrella `libboost-dev` package."""
    stem = header.lower().strip().lstrip('/').split('/', 1)[0]
    if stem == 'boost' and 'libboost-dev' in valid_packages:
        return {'libboost-dev'}
    return set()
