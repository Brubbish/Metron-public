"""Debian provider normalization for stable build artifacts.

Only entries that the provider index (apt metadata) cannot resolve are kept here.
"""

# Packages that were renamed across Debian releases.
# Key: (old_name, new_name, first_distro_with_new_name)
# The resolver should pick old_name for distros before the cutoff, new_name for cutoff and later.
DISTRO_RENAMES: list[tuple[str, str, str]] = [
    ('libgirepository1.0-dev', 'libgirepository-2.0-dev', 'trixie'),
    ('libsystemd-dev', 'systemd-dev', 'bookworm'),
]

CODENAME_ORDER = ['etch', 'lenny', 'squeeze', 'wheezy', 'jessie', 'stretch', 'buster', 'bullseye', 'bookworm', 'trixie']


def resolve_distro_package(candidates: set[str], distro: str) -> set[str]:
    """Filter package candidates based on distro availability for known renames."""
    try:
        distro_idx = CODENAME_ORDER.index(distro)
    except ValueError:
        return candidates

    resolved = set(candidates)
    for old_name, new_name, cutoff_distro in DISTRO_RENAMES:
        try:
            cutoff_idx = CODENAME_ORDER.index(cutoff_distro)
        except ValueError:
            continue
        if old_name in resolved and new_name in resolved:
            if distro_idx >= cutoff_idx:
                resolved.discard(old_name)
            else:
                resolved.discard(new_name)
        elif new_name in resolved and distro_idx < cutoff_idx:
            resolved.discard(new_name)
            resolved.add(old_name)
        elif old_name in resolved and distro_idx >= cutoff_idx:
            resolved.discard(old_name)
            resolved.add(new_name)
    return resolved

NON_DEV_PROVIDER_ALLOWLIST = frozenset({
    'asciidoc',
    'asciidoctor',
    'bash-completion',
    'bpftool',
    'catch',
    'cmake',
    'cmake-extras',
    'docbook-utils',
    'extra-cmake-modules',
    'gfortran',
    'gettext',
    'gobject-introspection',
    'gobject-introspection-bin',
    'googletest',
    'gtk-doc-tools',
    'intltool',
    'libxml2-utils',
    'lcov',
    'pkgconf',
    'python3-docutils',
    'python3-sphinx',
    'qtchooser',
    'qt6-l10n-tools',
    'scdoc',
    'texinfo',
    'txt2man',
    'valac',
    'valac-bin',
    'wayland-protocols',
})

NOISE_LIBS = frozenset({
    'm', 'c', 'dl', 'rt', 'pthread', 'stdc++', 'stdc++fs', 'gcc', 'gcc_s',
    'supc++', 'atomic', 'nsl', 'resolv', 'util',
    'mingw32', 'msvcrt', 'log', 'android',
    'kernel32', 'user32', 'ws2_32', 'advapi32', 'shell32',
    'ole32', 'oleaut32', 'gdi32', 'winmm', 'imm32',
    'version', 'shlwapi', 'psapi', 'iphlpapi', 'setupapi',
    'crypt32', 'wldap32', 'normaliz', 'mswsock', 'rpcrt4',
    'secur32', 'userenv', 'dbghelp',
})

# Boost component tables now live in component_templates.py.
# BOOST_NON_COMPONENT_MACROS and BOOST_LINK_COMPONENTS are re-exported from there
# for callers that still import them from this module
# (e.g. artifact_extractors._boost_autoconf_artifacts, artifacts._boost_target_packages).
from .component_templates import BOOST_NON_COMPONENT_MACROS, BOOST_LINK_COMPONENTS  # noqa: F401


def boost_header_packages(header: str, valid_packages: set[str]) -> set[str]:
    """Boost header resolver — delegates to component_templates."""
    from .component_templates import resolve_boost_header
    return resolve_boost_header(header, valid_packages)


def boost_component_package(component: str, valid_packages: set[str]) -> str | None:
    """Boost component resolver — delegates to component_templates.

    Returns a single package for backward compatibility with existing callers
    (`str | None`); component_templates returns a set, so we pick the first.
    """
    from .component_templates import resolve_boost_component
    matches = resolve_boost_component(component, valid_packages)
    return next(iter(sorted(matches)), None)


def boost_library_package(library: str, valid_packages: set[str]) -> str | None:
    """Boost library name resolver — delegates to component_templates."""
    from .component_templates import resolve_boost_library
    matches = resolve_boost_library(library, valid_packages)
    return next(iter(sorted(matches)), None)

CMAKE_PACKAGE_NORMALIZATION: dict[str, set[str]] = {
    'cmakeextras': {'cmake-extras'},
    'coveragereport': {'cmake-extras'},
    'intltool': {'cmake-extras', 'intltool'},
    'vala': {'cmake-extras', 'valac'},
    'openssl': {'libssl-dev'},
    'zlib': {'zlib1g-dev'},
    'z': {'zlib1g-dev'},
    'gtest': {'libgtest-dev', 'googletest'},
    'googletest': {'libgtest-dev', 'googletest'},
    'gmock': {'libgmock-dev', 'googletest'},
    'googlemock': {'libgmock-dev', 'googletest'},
    'catch': {'libcatch2-dev', 'catch'},
    'catch2': {'libcatch2-dev', 'catch'},
    'gobjectintrospection': {'gobject-introspection', 'libgirepository1.0-dev'},
    'fcitx': {'fcitx-bin', 'fcitx-libs-dev'},
}

PKGCONFIG_NORMALIZATION: dict[str, set[str]] = {
    'bash-completion': {'bash-completion'},
    'fcitx': {'fcitx-bin', 'fcitx-libs-dev'},
    'fcitx-config': {'fcitx-bin', 'fcitx-libs-dev'},
    'fcitx-gclient': {'fcitx-bin', 'fcitx-libs-dev'},
    'gio-2.0': {'libglib2.0-dev'},
    'gio-unix-2.0': {'libglib2.0-dev'},
    'glib-2.0': {'libglib2.0-dev'},
    'gmodule-2.0': {'libglib2.0-dev'},
    'gmodule-export-2.0': {'libglib2.0-dev'},
    'gobject-2.0': {'libglib2.0-dev'},
    'gthread-2.0': {'libglib2.0-dev'},
    'girepository-1.0': {'libgirepository1.0-dev'},
    'girepository-2.0': {'libgirepository-2.0-dev', 'libgirepository1.0-dev'},
    'gobject-introspection-1.0': {'gobject-introspection', 'libgirepository1.0-dev'},
    'hdf5': {'libhdf5-dev'},
    'libhdf5': {'libhdf5-dev'},
    'libsystemd': {'libsystemd-dev', 'systemd-dev'},
    'systemd': {'systemd-dev'},
    'udev': {'libudev-dev'},
}

LIBRARY_NORMALIZATION: dict[str, str] = {
    'crypto': 'libssl-dev',
    'girepository-1.0': 'libgirepository1.0-dev',
    'ssl': 'libssl-dev',
    'z': 'zlib1g-dev',
    'curl': 'libcurl4-gnutls-dev',
}

HEADER_NORMALIZATION: dict[str, set[str]] = {
    # Only headers that the provider_index headers_exact table does NOT cover.
    # All other headers should resolve via provider_index directly.
    'doctest/doctest.h': {'doctest-dev'},
}

EXECUTABLE_NORMALIZATION: dict[str, set[str]] = {
    # Only executables NOT covered by provider_index.executables (which misses most
    # documentation tools installed under /usr/share/ rather than /usr/bin/).
    'a2x': {'asciidoc'},
    'asciidoc': {'asciidoc'},
    'asciidoctor': {'asciidoctor'},
    'bpftool': {'bpftool'},
    'docbook2man': {'docbook-utils'},
    'genhtml': {'lcov'},
    'g-ir-compiler': {'gobject-introspection', 'libgirepository1.0-dev'},
    'g-ir-generate': {'gobject-introspection', 'libgirepository1.0-dev'},
    'g-ir-scanner': {'gobject-introspection', 'libgirepository1.0-dev'},
    'gdbus-codegen': {'libglib2.0-dev-bin'},
    'glib-compile-resources': {'libglib2.0-dev-bin'},
    'glib-compile-schemas': {'libglib2.0-dev-bin'},
    'gtkdoc-fixxref': {'gtk-doc-tools'},
    'gtkdoc-mkdb': {'gtk-doc-tools'},
    'gtkdoc-scan': {'gtk-doc-tools'},
    'gtkdocize': {'gtk-doc-tools'},
    'intltool-extract': {'intltool'},
    'intltool-merge': {'intltool'},
    'intltool-update': {'intltool'},
    'intltoolize': {'intltool'},
    'makeinfo': {'texinfo'},
    'rst2html': {'python3-docutils'},
    'rst2man': {'python3-docutils'},
    'sphinx-build': {'python3-sphinx'},
    'txt2man': {'txt2man'},
    'valac': {'valac'},
    'wayland-scanner': {'wayland-protocols'},
}
