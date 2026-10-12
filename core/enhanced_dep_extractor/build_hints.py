"""Extract build hints: disable flags and bundled library names.

Used by Phase 2/3 to:
- Skip tests/docs/strict mode on initial build attempts
- Avoid installing system packages for bundled/vendored libraries
"""

import os
import re

from .scanner import THIRD_PARTY_PREFIXES

_DISABLE_KEYWORDS = frozenset({
    'test', 'testing', 'tests', 'build_testing', 'build_tests',
    'strict', 'werror', 'wall_error', 'warnings_as_errors',
    'doc', 'docs', 'documentation', 'doxygen', 'man', 'manpages',
    'example', 'examples', 'sample', 'samples', 'demo',
    'bench', 'benchmark', 'benchmarks',
    'coverage', 'gcov', 'lcov',
    'sanitize', 'asan', 'tsan', 'ubsan', 'msan',
    'fuzz', 'fuzzing', 'fuzzer',
    'lint', 'clang_tidy', 'clangtidy', 'cppcheck',
    'install_tests', 'daemon_tests',
    'tools', 'build_tools', 'install_tools',
    'introspection', 'gobject_introspection',
    'bash_completion', 'gtk_doc',
    'completion',
    'glut', 'freeglut', 'gui',
    'systemd',
    'gstreamer', 'cron',
    'plugin', 'plugins',
})

_KEEP_KEYWORDS = frozenset({
    'shared', 'static', 'pic', 'fpic',
    'prefix', 'rpath',
    'debug', 'release', 'relwithdebinfo',
    'openssl', 'ssl', 'tls', 'crypto',
    'python', 'java', 'csharp',
})

_GITMODULES_SECTION_RE = re.compile(
    r'\[submodule\s+"([^"]+)"\](.*?)(?=\[submodule|\Z)',
    re.DOTALL,
)
_GITMODULES_PATH_RE = re.compile(r'^\s*path\s*=\s*(.+)', re.MULTILINE)

_CMAKE_OPTION_RE = re.compile(
    r'(?:option|cmake_dependent_option)\s*\(\s*(\w+)\s',
    re.IGNORECASE,
)

_CMAKE_SET_BOOL_RE = re.compile(
    r'set\s*\(\s*(BUILD_TESTING|ENABLE_TESTING|BUILD_TESTS)\s+(ON|OFF)',
    re.IGNORECASE,
)

_MESON_OPTION_RE = re.compile(
    r"option\s*\(\s*'([\w-]+)'\s*,\s*type\s*:\s*'(boolean|feature|combo|array)'",
)

# Meson array/combo backend-style options: name suggests "backend" or "vendor"
_BACKEND_NAME_KEYWORDS = frozenset({
    'backend', 'vendor', 'toolkit', 'impl',
})

_AC_ARG_ENABLE_RE = re.compile(
    r'AC_ARG_ENABLE\s*\(\s*\[?(\w[\w-]*)\]?',
    re.IGNORECASE,
)

_AC_ARG_WITH_RE = re.compile(
    r'AC_ARG_WITH\s*\(\s*\[?(\w[\w-]*)\]?',
    re.IGNORECASE,
)

_FETCHCONTENT_RE = re.compile(
    r'FetchContent_Declare\s*\(\s*([\w-]+)',
    re.IGNORECASE,
)

_EXTERNALPROJECT_RE = re.compile(
    r'ExternalProject_Add\s*\(\s*([\w-]+)',
    re.IGNORECASE,
)


def _should_disable(name: str) -> bool:
    lower = name.lower()
    tokens = set(re.split(r'[_\-]+', lower))
    if tokens & _KEEP_KEYWORDS:
        return False
    return bool(tokens & _DISABLE_KEYWORDS)


def _extract_cmake_flags(repo_path: str) -> list[dict]:
    flags = []
    files_to_scan = ['CMakeLists.txt']

    # cmake/*.cmake (already covered) + cmake/ subdirs at depth 1
    cmake_dir = os.path.join(repo_path, 'cmake')
    if os.path.isdir(cmake_dir):
        try:
            for f in os.listdir(cmake_dir):
                rel = os.path.join('cmake', f)
                if os.path.isfile(os.path.join(repo_path, rel)) and rel not in files_to_scan:
                    if rel.endswith('.cmake') or f == 'CMakeLists.txt':
                        files_to_scan.append(rel)
        except OSError:
            pass

    # First-level subdirectory CMakeLists.txt (depth-1, like src/CMakeLists.txt)
    try:
        for entry in os.listdir(repo_path):
            sub = os.path.join(repo_path, entry)
            if not os.path.isdir(sub):
                continue
            if entry.startswith('.') or entry in {'build', 'thirdparty', 'third_party',
                                                    'vendor', 'extern', 'external', 'deps'}:
                continue
            sub_cmake = os.path.join(entry, 'CMakeLists.txt')
            if os.path.isfile(os.path.join(repo_path, sub_cmake)):
                files_to_scan.append(sub_cmake)
    except OSError:
        pass

    for fname in files_to_scan:
        path = os.path.join(repo_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            content = open(path, errors='replace').read()
        except OSError:
            continue

        for m in _CMAKE_OPTION_RE.finditer(content):
            name = m.group(1)
            if _should_disable(name):
                block_end = content.find(')', m.start())
                block = content[m.start():block_end + 1] if block_end != -1 else m.group(0)
                flags.append({
                    'flag': f'-D{name}=OFF',
                    'source_file': fname,
                    'line': content[:m.start()].count('\n') + 1,
                    'original': block,
                })

        for m in _CMAKE_SET_BOOL_RE.finditer(content):
            name = m.group(1)
            val = m.group(2).upper()
            if val == 'ON' and _should_disable(name):
                flags.append({
                    'flag': f'-D{name}=OFF',
                    'source_file': fname,
                    'line': content[:m.start()].count('\n') + 1,
                    'original': m.group(0),
                })

    return flags


def _extract_meson_flags(repo_path: str) -> tuple[list[dict], list[dict]]:
    """Return (disable_flags, build_options).

    disable_flags: boolean/feature options that can be turned off.
    build_options: array/combo options that affect build mode (backend/vendor).
    """
    disable_flags: list[dict] = []
    build_options: list[dict] = []
    for fname in ('meson_options.txt', 'meson.options', 'meson_options'):
        path = os.path.join(repo_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            content = open(path, errors='replace').read()
        except OSError:
            continue

        for m in _MESON_OPTION_RE.finditer(content):
            name = m.group(1)
            opt_type = m.group(2)
            # Slice out the full option() block to inspect choices/value
            block_end = content.find(')', m.start())
            block = content[m.start():block_end + 1] if block_end != -1 else m.group(0)

            if opt_type == 'boolean':
                if _should_disable(name):
                    disable_flags.append({
                        'flag': f'-D{name}=false',
                        'source_file': fname,
                        'line': content[:m.start()].count('\n') + 1,
                        'original': block,
                    })
            elif opt_type == 'feature':
                if _should_disable(name):
                    disable_flags.append({
                        'flag': f'-D{name}=disabled',
                        'source_file': fname,
                        'line': content[:m.start()].count('\n') + 1,
                        'original': block,
                    })
            elif opt_type in ('array', 'combo'):
                # Backend/vendor selection — expose as build_options, not disable_flags
                name_lower = name.lower()
                name_tokens = set(re.split(r'[_\-]+', name_lower))
                if name_tokens & _BACKEND_NAME_KEYWORDS:
                    # Find declared choices to help Phase 3 pick a minimal one
                    choices_match = re.search(
                        r"choices\s*:\s*\[([^\]]*)\]",
                        block,
                    )
                    value_match = re.search(
                        r"value\s*:\s*(?:\[([^\]]*)\]|'([^']*)')",
                        block,
                    )
                    build_options.append({
                        'option': name,
                        'kind': opt_type,
                        'choices': (
                            [c.strip().strip("'\"") for c in choices_match.group(1).split(',') if c.strip()]
                            if choices_match else []
                        ),
                        'default': (
                            (value_match.group(1) or value_match.group(2)).strip()
                            if value_match else ''
                        ),
                        'source_file': fname,
                        'line': content[:m.start()].count('\n') + 1,
                    })

    return disable_flags, build_options


def _extract_autotools_flags(repo_path: str) -> list[dict]:
    flags = []
    for fname in ('configure.ac', 'configure.in'):
        path = os.path.join(repo_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            content = open(path, errors='replace').read()
        except OSError:
            continue

        for m in _AC_ARG_ENABLE_RE.finditer(content):
            name = m.group(1)
            if _should_disable(name):
                flags.append({
                    'flag': f'--disable-{name}',
                    'source_file': fname,
                    'line': content[:m.start()].count('\n') + 1,
                    'original': m.group(0),
                })

        for m in _AC_ARG_WITH_RE.finditer(content):
            name = m.group(1)
            if _should_disable(name):
                flags.append({
                    'flag': f'--without-{name}',
                    'source_file': fname,
                    'line': content[:m.start()].count('\n') + 1,
                    'original': m.group(0),
                })

    return flags


_CONFIGURE_HELP_OPTION_RE = re.compile(
    r'^\s+--(disable|enable)-([\w-]+)\s',
    re.MULTILINE,
)

# Cargo.lock git dependency: `source = "git+https://github.com/<owner>/<repo>.git?<query>#<commit>"`
# We extract (url, commit) so the caller can pre-flight `git ls-remote` and flag
# source-rotten refs (the librsvg/cairo-rs failure mode: the fork deleted the
# pinned commit and cargo cannot resolve the lock entry).
_CARGO_GIT_SOURCE_RE = re.compile(
    r'source\s*=\s*"git\+([^?#]+)(?:\?[^#]*)?#([0-9a-f]{40})"',
)


def _extract_cargo_unreachable_refs(repo_path: str) -> list[dict]:
    """Scan Cargo.lock for pinned git deps.

    Returns a list of entries with keys: name, url, commit. The caller (or a
    preflight step) can run `git ls-remote <url>` to detect when the pinned
    commit has been deleted upstream — a common librsvg/cairo-rs-style failure
    where the lock file references a fork commit that no longer exists.

    We do NOT actually run git ls-remote here (build_hints must remain offline
    and fast); we surface the candidates so the agent workflow can pre-flight
    them when network is available.
    """
    path = os.path.join(repo_path, 'Cargo.lock')
    if not os.path.isfile(path):
        return []
    try:
        content = open(path, errors='replace').read()
    except OSError:
        return []

    # Entries in Cargo.lock look like:
    #   [[package]]
    #   name = "cairo-rs"
    #   version = "0.1.1"
    #   source = "git+https://github.com/federicomenaquintero/cairo.git#abcd1234..."
    entries: list[dict] = []
    # Walk package blocks; each starts with `[[package]]` and ends at the next
    # `[[package]]` or EOF.
    for block_match in re.finditer(
        r'\[\[package\]\](.*?)(?=\[\[package\]\]|\[\[|\Z)',
        content,
        re.DOTALL,
    ):
        block = block_match.group(1)
        name_m = re.search(r'^name\s*=\s*"([^"]+)"', block, re.MULTILINE)
        source_m = _CARGO_GIT_SOURCE_RE.search(block)
        if not name_m or not source_m:
            continue
        entries.append({
            'name': name_m.group(1),
            'url': source_m.group(1),
            'commit': source_m.group(2),
            'source_file': 'Cargo.lock',
        })
    return entries

# Makefile: HAVE_GLUT := yes  /  HAVE_X11 = yes  /  HAVE_FREEGLUT ?= yes
# We capture HAVE_<NAME> defined to "yes"/"true" so Phase 3 can override with
# HAVE_<NAME>=no (the convention MuPDF uses for `make HAVE_GLUT=no`).
_MAKEFILE_HAVE_RE = re.compile(
    r'^\s*(HAVE_[A-Z0-9_]+)\s*[:?]?=\s*(yes|true|1)\b',
    re.MULTILINE,
)

# Makefile variables that gate optional components (e.g. WITH_GLUT, BUILD_TOOL)
_MAKEFILE_VAR_KEYWORDS = frozenset({
    'glut', 'freeglut', 'x11', 'opengl', 'egl',
    'tool', 'tools', 'examples', 'samples', 'demo',
    'doc', 'docs', 'tests', 'test',
})


def _extract_makefile_flags(repo_path: str) -> list[dict]:
    """Extract HAVE_*=yes / WITH_*=yes Makefile variables as disable flags."""
    flags: list[dict] = []
    seen_names: set[str] = set()
    for fname in ('Makerules', 'Makefile', 'GNUmakefile', 'makefile'):
        path = os.path.join(repo_path, fname)
        if not os.path.isfile(path):
            continue
        try:
            content = open(path, errors='replace').read()
        except OSError:
            continue

        for m in _MAKEFILE_HAVE_RE.finditer(content):
            name = m.group(1)  # e.g. HAVE_GLUT
            if name in seen_names:
                continue
            seen_names.add(name)
            # Gate: only emit when the variable name actually points at an
            # optional component (glut/x11/opengl/egl/tools/docs/tests/...).
            # Pure HAVE_LIBCURL etc. are real dependencies, not optional features.
            tokens = set(re.split(r'_+', name.lower()))
            if not (tokens & _MAKEFILE_VAR_KEYWORDS):
                continue
            flags.append({
                'flag': f'{name}=no',
                'source_file': fname,
                'line': content[:m.start()].count('\n') + 1,
                'original': m.group(0).strip(),
            })
    return flags


def _extract_configure_script_flags(repo_path: str) -> list[dict]:
    """Extract --disable-* options from custom configure scripts (FFmpeg-style)."""
    path = os.path.join(repo_path, 'configure')
    if not os.path.isfile(path):
        return []
    # Skip if autotools configure.ac exists — those are handled by _extract_autotools_flags
    if os.path.isfile(os.path.join(repo_path, 'configure.ac')) or \
       os.path.isfile(os.path.join(repo_path, 'configure.in')):
        return []
    try:
        content = open(path, errors='replace').read(512 * 1024)
    except OSError:
        return []
    # Only process if it looks like a help-text block (not a generated autoconf script)
    if 'Generated by GNU Autoconf' in content[:2000]:
        return []

    flags = []
    seen = set()
    for m in _CONFIGURE_HELP_OPTION_RE.finditer(content):
        action = m.group(1)  # 'disable' or 'enable'
        name = m.group(2)
        if name in seen:
            continue
        seen.add(name)
        if _should_disable(name):
            flags.append({
                'flag': f'--disable-{name}',
                'source_file': 'configure',
                'line': content[:m.start()].count('\n') + 1,
                'original': m.group(0).strip(),
            })

    return flags


_GENERIC_DIR_NAMES = frozenset({
    'src', 'lib', 'test', 'tests', 'build', 'bin', 'include',
    'doc', 'docs', 'scripts', 'tools', 'cmake', 'config',
    'packagecache', 'packagefiles',
})


def _extract_bundled_from_gitmodules(repo_path: str) -> list[dict]:
    path = os.path.join(repo_path, '.gitmodules')
    if not os.path.isfile(path):
        return []
    try:
        content = open(path, errors='replace').read()
    except OSError:
        return []

    bundled = []
    for m in _GITMODULES_SECTION_RE.finditer(content):
        section_body = m.group(2)
        path_match = _GITMODULES_PATH_RE.search(section_body)
        if not path_match:
            continue
        submodule_path = path_match.group(1).strip()
        name = os.path.basename(submodule_path)
        if name.startswith('.') or name.lower() in _GENERIC_DIR_NAMES:
            continue
        bundled.append({
            'name': name,
            'source': '.gitmodules',
            'path': submodule_path,
        })
    return bundled


def _extract_bundled_from_dirs(repo_path: str) -> list[dict]:
    bundled = []
    try:
        entries = os.listdir(repo_path)
    except OSError:
        return bundled

    for entry in entries:
        if entry.lower() not in THIRD_PARTY_PREFIXES:
            continue
        vendor_dir = os.path.join(repo_path, entry)
        if not os.path.isdir(vendor_dir):
            continue
        try:
            subdirs = os.listdir(vendor_dir)
        except OSError:
            continue
        for sub in subdirs:
            sub_path = os.path.join(vendor_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            if sub.startswith('.') or sub.lower() in _GENERIC_DIR_NAMES:
                continue
            bundled.append({
                'name': sub,
                'source': f'{entry}/',
                'path': f'{entry}/{sub}',
            })
    return bundled


def _extract_bundled_from_cmake(repo_path: str) -> list[dict]:
    bundled = []
    path = os.path.join(repo_path, 'CMakeLists.txt')
    if not os.path.isfile(path):
        return bundled
    try:
        content = open(path, errors='replace').read()
    except OSError:
        return bundled

    for pat in (_FETCHCONTENT_RE, _EXTERNALPROJECT_RE):
        for m in pat.finditer(content):
            name = m.group(1)
            bundled.append({
                'name': name,
                'source': 'CMakeLists.txt',
                'path': '',
            })
    return bundled


def extract_build_hints(repo_path: str) -> dict:
    """Extract disable flags and bundled library names from a project."""
    disable_flags: list[dict] = []
    build_options: list[dict] = []
    disable_flags.extend(_extract_cmake_flags(repo_path))
    meson_disable, meson_options = _extract_meson_flags(repo_path)
    disable_flags.extend(meson_disable)
    build_options.extend(meson_options)
    disable_flags.extend(_extract_autotools_flags(repo_path))
    disable_flags.extend(_extract_configure_script_flags(repo_path))
    disable_flags.extend(_extract_makefile_flags(repo_path))

    # Deduplicate disable flags by flag string, keep deterministic order
    seen_flags: set[str] = set()
    deduped_flags: list[dict] = []
    for flag in disable_flags:
        key = flag['flag']
        if key in seen_flags:
            continue
        seen_flags.add(key)
        deduped_flags.append(flag)

    bundled_libs = []
    gitmodule_libs = _extract_bundled_from_gitmodules(repo_path)
    bundled_libs.extend(gitmodule_libs)
    bundled_libs.extend(_extract_bundled_from_dirs(repo_path))
    bundled_libs.extend(_extract_bundled_from_cmake(repo_path))

    # Deduplicate by name
    seen_names = set()
    deduped = []
    for lib in bundled_libs:
        if lib['name'] not in seen_names:
            seen_names.add(lib['name'])
            deduped.append(lib)

    submodule_names = [lib['name'] for lib in gitmodule_libs]

    result = {
        'disable_flags': deduped_flags,
        'bundled_libraries': deduped,
        'has_submodules': bool(submodule_names),
        'submodule_names': submodule_names,
    }
    if build_options:
        result['build_options'] = build_options
    cargo_refs = _extract_cargo_unreachable_refs(repo_path)
    if cargo_refs:
        result['cargo_git_refs'] = cargo_refs
    return result
