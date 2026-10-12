import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.enhanced_dep_extractor.artifacts import (
    Artifact,
    ProviderIndex,
    _build_header_suffix_index,
    _header_index_suffix_packages,
    extract_artifacts,
    packages_from_artifacts,
    resolve_from_distro_mapping,
)
from core.enhanced_dep_extractor.eval import build_alias_map
from core.enhanced_dep_extractor.header_scan import _is_skip_token
from core.enhanced_dep_extractor.rules import RuleMatch

# Artifact seeding is backed by the optional distro_mapping.json dataset
# (Build_RAG_Info, rebuildable from public Debian data and not shipped).
_requires_distro_dataset = pytest.mark.skipif(
    not resolve_from_distro_mapping('cmake', 'openssl'),
    reason="optional distro_mapping.json dataset is not present",
)


VALID_PACKAGES = {
    'architecture-is-64-bit',
    'architecture-is-little-endian',
    'architecture-properties',
    'libevdev-dev',
    'libluajit-5.1-dev',
    'libssl-dev',
    'libudev-dev',
    'libxrender-dev',
    'libxt-dev',
    'qt6-base-dev',
    'qt6-base-private-dev',
    'qt6-declarative-dev',
    'qt6-declarative-private-dev',
    'qtwayland5-private-dev',
    'zlib1g-dev',
}


def _match(text: str, rule_type: str = 'cmake:find_package_call') -> RuleMatch:
    return RuleMatch(
        file_path='CMakeLists.txt',
        line_num=1,
        rule_type=rule_type,
        matched_text=text,
        context=text,
        extracted_name='',
    )


@_requires_distro_dataset
def test_cmake_build_metadata_extracts_imported_targets_pkgconfig_aliases_and_bare_libraries():
    text = """
    find_package(Qt6 COMPONENTS Core Gui Qml)
    find_package(OpenSSL REQUIRED)
    pkg_check_modules(EVDEV REQUIRED libevdev)
    target_link_libraries(app PRIVATE
        Qt::GuiPrivate Qt::QmlPrivate OpenSSL::SSL ZLIB::ZLIB
        PkgConfig::EVDEV ssl crypto z udev Xrender Xt)
    """

    artifacts = extract_artifacts([_match(text)])
    packages = packages_from_artifacts(artifacts, VALID_PACKAGES)

    assert Artifact('pkgconfig', 'libevdev') in artifacts
    assert Artifact('library', 'ssl') in artifacts
    assert {
        'libevdev-dev',
        'libssl-dev',
        'libudev-dev',
        'libxrender-dev',
        'libxt-dev',
        'qt6-base-private-dev',
        'qt6-declarative-private-dev',
        'zlib1g-dev',
    } <= packages


@_requires_distro_dataset
def test_meson_and_autotools_build_metadata_resolve_common_artifacts():
    text = """
    dependency('luajit-5.1')
    dependency('libudev')
    cc.find_library('luajit-5.1')
    cc.find_library('Xrender')
    PKG_CHECK_MODULES([XT], [xt])
    AC_CHECK_LIB([z], [deflate])
    AC_SEARCH_LIBS([SSL_new], [ssl crypto])
    """

    artifacts = extract_artifacts([_match(text, 'meson:dependency')])
    packages = packages_from_artifacts(artifacts, VALID_PACKAGES)

    assert {
        'libluajit-5.1-dev',
        'libssl-dev',
        'libudev-dev',
        'libxrender-dev',
        'libxt-dev',
        'zlib1g-dev',
    } <= packages


def test_qt_wayland_private_target_maps_to_wayland_private_package():
    packages = packages_from_artifacts(
        [Artifact('cmake_package', 'Qt5', ('WaylandClientPrivate', 'XkbCommonSupportPrivate'))],
        VALID_PACKAGES,
    )

    assert packages == {'qtwayland5-private-dev'}


def test_architecture_helper_renames_are_eval_aliases():
    aliases = build_alias_map()

    assert 'architecture-properties' in aliases['architecture-is-64-bit']
    assert 'architecture-properties' in aliases['architecture-is-little-endian']


def test_header_suffix_index_matches_versioned_provider_paths_by_long_suffix():
    index = ProviderIndex(headers_exact={
        'x86_64-linux-gnu/qt6/qtqml/6.8.2/qtqml/private/qqmlobjectcreator_p.h': {
            'qt6-declarative-private-dev',
        },
        'x86_64-linux-gnu/qt6/qtgui/6.8.2/qtgui/qpa/qplatformwindow_p.h': {
            'qt6-base-private-dev',
        },
        'x11/extensions/xrender.h': {'libxrender-dev'},
        'libevdev-1.0/libevdev/libevdev.h': {'libevdev-dev'},
    })
    _build_header_suffix_index(index)

    assert _header_index_suffix_packages(
        'QtQml/private/qqmlobjectcreator_p.h',
        index,
        VALID_PACKAGES,
    ) == {'qt6-declarative-private-dev'}
    assert _header_index_suffix_packages(
        'QtGui/qpa/qplatformwindow_p.h',
        index,
        VALID_PACKAGES,
    ) == {'qt6-base-private-dev'}
    assert _header_index_suffix_packages(
        'X11/extensions/Xrender.h',
        index,
        VALID_PACKAGES,
    ) == {'libxrender-dev'}
    assert _header_index_suffix_packages(
        'libevdev/libevdev.h',
        index,
        VALID_PACKAGES,
    ) == {'libevdev-dev'}


def test_header_basename_suffix_requires_low_fanout():
    index = ProviderIndex(headers_exact={
        f'include-{idx}/lua.h': {f'liblua{idx}-dev'}
        for idx in range(5)
    })
    _build_header_suffix_index(index)
    valid_packages = {f'liblua{idx}-dev' for idx in range(5)}

    assert _header_index_suffix_packages('lua.h', index, valid_packages) == set()


def test_source_header_scan_allows_non_standard_basename_tokens():
    assert not _is_skip_token('zlib.h')
    assert not _is_skip_token('libudev.h')
    assert _is_skip_token('stdio.h')
