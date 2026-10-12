"""Rule definitions for build system dependency extraction."""

import os
import re
from dataclasses import dataclass, field

from .provider_normalization import NOISE_LIBS  # noqa: F401 (re-exported for callers)


@dataclass
class RuleMatch:
    file_path: str
    line_num: int
    rule_type: str
    matched_text: str
    context: str
    extracted_name: str
    packages: list[str] = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    condition: str = 'unknown'


CONTEXT_LINES = 5

# ──── CMake ────────────────────────────────────────────────────────────────────

CMAKE_PATTERNS = [
    (re.compile(r'\b\w*find_package\s*\(([^)]*)\)', re.IGNORECASE | re.DOTALL), 'cmake:find_package_call'),
    (re.compile(r'\b\w*find_package\s*\(\s*([A-Za-z0-9_:.-]+)', re.IGNORECASE), 'cmake:find_package'),
    (re.compile(r'find_dependency\s*\(([^)]*)\)', re.IGNORECASE | re.DOTALL), 'cmake:find_dependency_call'),
    (re.compile(r'find_dependency\s*\(\s*([A-Za-z0-9_:.-]+)', re.IGNORECASE), 'cmake:find_dependency'),
    (re.compile(r'find_package_handle_standard_args\s*\(\s*([A-Za-z0-9_.+-]+)', re.IGNORECASE), 'cmake:find_package_handle_standard_args'),
    (re.compile(r'pkg_(?:check_modules|search_module)\s*\(\s*[A-Za-z0-9_]+\s+((?:(?:REQUIRED|QUIET|IMPORTED_TARGET|NO_CMAKE_PATH|NO_CMAKE_ENVIRONMENT_PATH)\s+)*)(.*?)\)', re.IGNORECASE | re.DOTALL), 'cmake:pkg_check_modules'),
    (re.compile(r'pkg_get_variable\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'cmake:pkg_get_variable'),
    (re.compile(r'include\s*\(\s*Find([A-Za-z0-9_+.-]+)', re.IGNORECASE), 'cmake:include_find_module'),
    (re.compile(r'find_library\s*\(\s*\w+\s+(?:NAMES?\s+)?([A-Za-z0-9_.-]+)', re.IGNORECASE), 'cmake:find_library'),
    (re.compile(r'find_(?:path|file)\s*\(\s*\w+\s+(?:NAMES?\s+|PATH_SUFFIXES\s+)?([A-Za-z0-9_./\\-]+\.h(?:pp|xx|h)?)', re.IGNORECASE), 'cmake:find_path'),
    (re.compile(r'find_program\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'cmake:find_program_call'),
    (re.compile(r'find_program\s*\(\s*\w+\s+(?:NAMES?\s+)?([A-Za-z0-9_.-]+)', re.IGNORECASE), 'cmake:find_program'),
    (re.compile(r'find_program\s*\([^)]*(?:help2man|xsltproc|xmlto|docbook2x-man|docbook2man|asciidoc|asciidoctor|a2x|sphinx-build|doxygen|dot|rst2man|rst2html|txt2man|scdoc|pod2man|msgfmt|xgettext|msgmerge|lrelease|lupdate|makeinfo|bison|flex)[^)]*\)', re.IGNORECASE | re.DOTALL), 'cmake:find_doc_tool'),
    (re.compile(r'check_library_exists\s*\(\s*([A-Za-z0-9_.-]+)', re.IGNORECASE), 'cmake:check_library_exists'),
    (re.compile(r'(?:check_include_file(?:_cxx|s)?|check_(?:cxx_)?symbol_exists|check_struct_has_member)\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'cmake:check_include_file'),
    (re.compile(r'target_link_libraries\s*\(\s*(\S+)\s+(.*?)\)', re.IGNORECASE | re.DOTALL), 'cmake:target_link_libraries'),
    (re.compile(r'enable_language\s*\(([^)]*Fortran[^)]*)\)', re.IGNORECASE | re.DOTALL), 'cmake:enable_fortran'),
    (re.compile(r'project\s*\(([^)]*(?:LANGUAGES\s+[^)]*)?Fortran[^)]*)\)', re.IGNORECASE | re.DOTALL), 'cmake:project_fortran'),
    (re.compile(r'\bCMAKE_Fortran_COMPILER\b[^\n]*', re.IGNORECASE), 'cmake:fortran-compiler'),
    (re.compile(r'\b(CMAKE_AUTOMOC|CMAKE_AUTOUIC|CMAKE_AUTORCC)\b', re.IGNORECASE), 'cmake:qt_auto_tool'),
    (re.compile(r'\b(qt_add_[A-Za-z0-9_]+|qt5_add_[A-Za-z0-9_]+|qt6_add_[A-Za-z0-9_]+)\s*\(', re.IGNORECASE), 'cmake:qt_helper'),
    (re.compile(r'(?:add_custom_command|add_custom_target|execute_process)\s*\([^)]*(?:help2man|xsltproc|xmlto|docbook2x-man|docbook2man|asciidoc|asciidoctor|a2x|sphinx-build|doxygen|rst2man|rst2html|txt2man|scdoc|pod2man)[^)]*\)', re.IGNORECASE | re.DOTALL), 'cmake:doc_custom_command'),
    (re.compile(r'(?:/usr/share/xml/docbook/stylesheet|docbook\.sourceforge\.net/release/xsl|manpages/docbook\.xsl|docbook\.xsl)', re.IGNORECASE), 'cmake:docbook-xsl'),
    (re.compile(r'(?:docbook\.dtd|docbook\.xsd|docbook-xml)', re.IGNORECASE), 'cmake:docbook-xml'),
    (re.compile(r'FetchContent_Declare\s*\(\s*([A-Za-z0-9_.-]+)[^)]*(?:FIND_PACKAGE_ARGS|OVERRIDE_FIND_PACKAGE)', re.IGNORECASE | re.DOTALL), 'cmake:FetchContent'),
    (re.compile(r'CPMFindPackage\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'cmake:CPMAddPackage'),
    (re.compile(r'ExternalProject_Add\s*\(\s*([A-Za-z0-9_.-]+)', re.IGNORECASE), 'cmake:ExternalProject'),
    # (re.compile(r'enable_language\s*\(\s*([A-Za-z0-9_]+)', re.IGNORECASE), 'cmake:enable_language'),
    (re.compile(r'CPACK_DEBIAN_PACKAGE_DEPENDS\s+"([^"]+)"', re.IGNORECASE), 'cmake:cpack_depends'),
    (re.compile(r'message\s*\(\s*(?:FATAL_ERROR|SEND_ERROR|WARNING|STATUS)\s+"([^"]*(?:not found|missing|(?:is )?required|install)[^"]*)"', re.IGNORECASE), 'cmake:msg_missing'),
]

# ──── Autoconf ─────────────────────────────────────────────────────────────────

AUTOCONF_PATTERNS = [
    (re.compile(r'PKG_CHECK_MODULES(?:_STATIC)?\s*\(\s*\[?([^\],\s]+)\]?\s*,\s*\[?([^\])\s,]+)', re.IGNORECASE), 'ac:PKG_CHECK_MODULES'),
    (re.compile(r'PKG_CHECK_EXISTS\s*\(\s*\[?([^\],\s]+)', re.IGNORECASE), 'ac:PKG_CHECK_EXISTS'),
    (re.compile(r'PKG_CHECK_VAR\s*\(\s*\[?[^\],]+\]?\s*,\s*\[?([^\],\s]+)', re.IGNORECASE), 'ac:PKG_CHECK_VAR'),
    (re.compile(r'(?:pkg-config|pkgconf|\$[({]PKG_CONFIG[)}])\b[^\n;&|`]*', re.IGNORECASE), 'ac:pkg-config'),
    (re.compile(r'^\s*(?:LIBS|LDFLAGS|[A-Za-z0-9_]+_(?:LIBS|LDADD|LIBADD|LDFLAGS))\s*=\s*["\']?[^\n]*\s-l[A-Za-z0-9_+.-]+\b[^\n]*', re.MULTILINE), 'ac:link-libs-var'),
    (re.compile(r'\b(?:AX_(?:LIB|CHECK|PATH|WITH)_[A-Z0-9_]+|AM_PATH_[A-Z0-9_]+)\b', re.IGNORECASE), 'ac:generic_dependency_macro'),
    (re.compile(r'^\s*((?:AX_)?BOOST_[A-Z0-9_]+)\s*(?:\([^)]*\))?\s*(?:$|dnl\b|#)', re.MULTILINE), 'ac:BOOST_macro'),
    (re.compile(r'^\s*(?:LIBS|LDFLAGS)\s*=\s*["\']?[^\n]*\$\{?BOOST_[A-Z0-9_]+_(?:LIBS|LDFLAGS|LDPATH)\}?[^\n]*', re.MULTILINE), 'ac:BOOST_libs_var'),
    (re.compile(r'AC_CHECK_LIBS?\s*\(\s*\[?([^\],\s]+)', re.IGNORECASE), 'ac:AC_CHECK_LIB'),
    (re.compile(r'AC_SEARCH_LIBS\s*\(\s*\[?[^\],]+\]?\s*,\s*\[?([^\])\s,]+)', re.IGNORECASE), 'ac:AC_SEARCH_LIBS'),
    (re.compile(r'AC_CHECK_HEADERS?\s*\(\s*\[?([^\])\s,]+)', re.IGNORECASE), 'ac:AC_CHECK_HEADER'),
    (re.compile(r'AC_(?:CHECK|PATH)_(?:PROG|PROGS|TOOL|TOOLS)\s*\(\s*\[?[^\],]+\]?\s*,\s*\[?([^\])\s,]+)', re.IGNORECASE), 'ac:AC_CHECK_PROG'),
    (re.compile(r'\b(AC_PROG_YACC|AC_PROG_LEX|AM_PROG_LEX)\b', re.IGNORECASE), 'ac:known_macro'),
    (re.compile(r'\b(AC_PROG_FC|AC_PROG_F77|AC_FC_[A-Z0-9_]+|AC_F77_[A-Z0-9_]+)\b', re.IGNORECASE), 'ac:fortran'),
    (re.compile(r'\bAC_LANG(?:_PUSH)?\s*\(\s*\[?Fortran\]?', re.IGNORECASE), 'ac:fortran'),
    (re.compile(r'\bAC_(?:LINK|COMPILE)_IFELSE\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'ac:compile-link-ifelse'),
    # (re.compile(r'AC_CHECK_FUNCS?\s*\(\s*\[?([^\])\s,]+)', re.IGNORECASE), 'ac:AC_CHECK_FUNC'),
    (re.compile(r'AC_PATH_PROGS?\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'ac:AC_PATH_PROG'),
    (re.compile(r'm4_include\s*\(\s*\[?([A-Za-z0-9_./+-]+\.m4)\]?', re.IGNORECASE), 'ac:m4_include'),
    (re.compile(r'^\s*(?!AC_DEFUN\b).*?\b(AX_[A-Z0-9_]+)\b', re.MULTILINE), 'ac:AX_macro'),
    (re.compile(r'(AM_PATH_[A-Z_]+)', re.IGNORECASE), 'ac:AM_PATH'),
    (re.compile(r'(IT_PROG_INTLTOOL|GTK_DOC_CHECK|GOBJECT_INTROSPECTION_CHECK|VAPIGEN_CHECK|YELP_HELP_INIT|AM_GNU_GETTEXT)', re.IGNORECASE), 'ac:known_macro'),
    (re.compile(r'LT_INIT|AC_PROG_LIBTOOL|AM_PROG_LIBTOOL', re.IGNORECASE), 'ac:libtool'),
    (re.compile(r'AC_MSG_ERROR\s*\(\s*\[?([^\]]*(?:not found|missing|required|install)[^\]]*)\]?\)', re.IGNORECASE), 'ac:error_missing'),
    (re.compile(r'(?:/usr/share/xml/docbook/stylesheet|docbook\.sourceforge\.net/release/xsl|manpages/docbook\.xsl|docbook\.xsl)', re.IGNORECASE), 'ac:docbook-xsl'),
    (re.compile(r'(?:docbook\.dtd|docbook\.xsd|docbook-xml)', re.IGNORECASE), 'ac:docbook-xml'),
    # (re.compile(r'AC_MSG_WARN(?:ING)?\s*\(\s*\[?([^\]]*(?:not found|missing|is required)[^\]]*)\]?\)', re.IGNORECASE), 'ac:warn_missing'),
]

# ──── Meson ────────────────────────────────────────────────────────────────────

MESON_PATTERNS = [
    (re.compile(r"""dependency\s*\(\s*['"]([^'"]+)['"]"""), 'meson:dependency'),
    (re.compile(r"""dependency\s*\(\s*\[(.*?)\]""", re.IGNORECASE | re.DOTALL), 'meson:dependency-list'),
    (re.compile(r'\blanguages\s*:\s*\[[^\]]*[\'"]fortran[\'"][^\]]*\]', re.IGNORECASE | re.DOTALL), 'meson:fortran-language'),
    (re.compile(r'\badd_languages\s*\([^\)]*[\'"]fortran[\'"][^\)]*\)', re.IGNORECASE | re.DOTALL), 'meson:fortran-language'),
    (re.compile(r"""\b(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?find_library\s*\(\s*['"]([^'"]+)['"]"""), 'meson:find_library'),
    (re.compile(r"""\b(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?has_(?:header|header_symbol)\s*\(\s*['"]([^'"]+)['"]"""), 'meson:has_header'),
    (re.compile(r"""\b(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?has_function\s*\((.*?)\)""", re.IGNORECASE | re.DOTALL), 'meson:has_function'),
    (re.compile(r"""find_program\s*\((.*?)\)""", re.IGNORECASE | re.DOTALL), 'meson:find_program_call'),
    (re.compile(r"""find_program\s*\(\s*['"]([^'"]+)['"]"""), 'meson:find_program'),
    (re.compile(r"""(?:custom_target|generator|run_target)\s*\((.*?)\)""", re.IGNORECASE | re.DOTALL), 'meson:target_command'),
    (re.compile(r'\b(help2man|asciidoc|asciidoctor|a2x|xmlto|xsltproc|docbook2x-man|docbook2man|sphinx-build|doxygen|rst2man|rst2html|txt2man|scdoc|pod2man|gtkdocize|gtkdoc-scan|gtkdoc-mkdb|gtkdoc-fixxref)\b', re.IGNORECASE), 'meson:doc_tool'),
    (re.compile(r'(?:/usr/share/xml/docbook/stylesheet|docbook\.sourceforge\.net/release/xsl|manpages/docbook\.xsl|docbook\.xsl)', re.IGNORECASE), 'meson:docbook-xsl'),
    (re.compile(r'(?:docbook\.dtd|docbook\.xsd|docbook-xml)', re.IGNORECASE), 'meson:docbook-xml'),
    (re.compile(r"""subproject\s*\(\s*['"]([^'"]+)['"]"""), 'meson:subproject'),
    (re.compile(r"""conf_data\s*\.\s*set\w*\s*\(\s*['"]([^'"]+)['"]""", re.IGNORECASE), 'meson:conf_data'),
    (re.compile(r"""error\s*\(\s*['"]([^'"]*(?:not found|missing|required|install)[^'"]*)['"]""", re.IGNORECASE), 'meson:error_missing'),
    (re.compile(r'\bgnome\s*\.\s*generate_gir\s*\(', re.IGNORECASE), 'meson:gnome_generate_gir'),
    (re.compile(r'\bgnome\s*\.\s*generate_vapi\s*\(', re.IGNORECASE), 'meson:gnome_generate_vapi'),
    (re.compile(r'\bgnome\s*\.\s*gtkdoc\s*\(', re.IGNORECASE), 'meson:gnome_gtkdoc'),
    (re.compile(r"""option\s*\(\s*['"](?:introspection|gir)['"]""", re.IGNORECASE), 'meson:option_introspection'),
    (re.compile(r"""option\s*\(\s*['"](?:vapi|vapigen)['"]""", re.IGNORECASE), 'meson:option_vapi'),
    (re.compile(r"""option\s*\(\s*['"](?:gtk_doc|gtk-doc|gtkdoc|docs)['"]""", re.IGNORECASE), 'meson:option_gtkdoc'),
]

# ──── Makefile ─────────────────────────────────────────────────────────────────

MAKEFILE_PATTERNS = [
    (re.compile(r'(?:^|[\s="])-l([A-Za-z0-9_+.-]+)', re.MULTILINE), 'make:-l'),
    (re.compile(r'^\s*(?:\$\((?:CXX|CC)\)|\w*\+\+|gcc|g\+\+|clang|clang\+\+|ld)\b[^\n]*\s-l[A-Za-z0-9_+.-]+\b[^\n]*', re.MULTILINE), 'make:link-command'),
    (re.compile(r'^\s*(?:LDLIBS|LOADLIBES|LIBS|LDFLAGS|[A-Za-z0-9_]+_(?:LIBS|LDADD|LIBADD|LDFLAGS))\s*[:+?]?=\s*[^\n]*\s-l[A-Za-z0-9_+.-]+\b[^\n]*', re.MULTILINE), 'make:link-libs-var'),
    (re.compile(r'^\s*(?!BOOST_[A-Z0-9_]+_(?:LIBS|LDFLAGS|LDPATH)\s*=)(?:LDLIBS|LOADLIBES|LIBS|LDFLAGS|[A-Za-z0-9_]+_(?:LIBS|LDADD|LIBADD|LDFLAGS))\s*[:+?]?=\s*[^\n]*BOOST_[A-Z0-9_]+_(?:LIBS|LDFLAGS|LDPATH)[^\n]*', re.MULTILINE), 'make:boost-libs-var'),
    (re.compile(r'\b(?:PGXS|PG_CONFIG|pg_config\s+--pgxs)\b[^\n]*', re.IGNORECASE), 'make:pgxs'),
    (re.compile(r'^\s*(?:BISON|BISONFLAGS|YACC|YFLAGS)\b[^\n]*', re.MULTILINE | re.IGNORECASE), 'make:bison'),
    (re.compile(r'^\s*(?:FLEX|FLEXFLAGS|LEX|LFLAGS)\b[^\n]*', re.MULTILINE | re.IGNORECASE), 'make:flex'),
    (re.compile(r'^\s*(?:FC|F77|F90|FORTRAN|MPIF90|GFORTRAN|FFLAGS|FCFLAGS)\s*[:?+]?=\s*[^\n]+', re.MULTILINE | re.IGNORECASE), 'make:fortran-var'),
    (re.compile(r'^\s*[^:\n]+\.(?:o|lo|obj)\s*:\s*[^:\n]+\.(?:f|for|ftn|f77|f90|f95|f03|f08|F|F90|F95|F03|F08)\b[^\n]*', re.MULTILINE), 'make:fortran-rule'),
    (re.compile(r'^\s*(?:FC|F77|F90|FORTRAN|MPIF90|GFORTRAN)\b[^\n]*\s-c\s+[^\n]*\.(?:f|for|ftn|f77|f90|f95|f03|f08|F|F90|F95|F03|F08)\b[^\n]*', re.MULTILINE | re.IGNORECASE), 'make:fortran-rule'),
    (re.compile(r'^\s*[^:\n]+\.(?:c|cc|cpp|cxx|h|hpp)\s*:\s*[^:\n]+\.(?:y|yy|l|ll)\b[^\n]*', re.MULTILINE | re.IGNORECASE), 'make:lexer-parser-rule'),
    (re.compile(r'(?:pkg-config|pkgconf|\$[({]PKG_CONFIG[)}])\b[^\n;&|`]*', re.IGNORECASE), 'make:pkg-config'),
    (re.compile(r'(?:^|[\s`$({])([A-Za-z0-9_+.-]+(?:-config|_config))\b[^\n;&|`]*--(?:cflags|libs|includedir|libdir|ldflags|version|modversion|variable)\b[^\n;&|`]*', re.MULTILINE | re.IGNORECASE), 'make:config-script'),
    (re.compile(r'\$\((?:error|warning)\s+[^)]*(?:not found|missing|required|install)[^)]*\)', re.IGNORECASE), 'make:error-missing'),
    (re.compile(r'(?:apt(?:-get)?\s+install|dnf\s+install|yum\s+install)\s+[^\n]+', re.IGNORECASE), 'make:install-hint'),
    (re.compile(r'(?:please install|install package[s]?)[^\n]*(?:[a-z0-9][a-z0-9.+-]*-dev|[a-z0-9][a-z0-9.+-]*-[a-z0-9][a-z0-9.+-]*)[^\n]*', re.IGNORECASE), 'make:install-hint'),
    (re.compile(r'-I\s*/usr/include/([A-Za-z0-9_+.-]+)', re.MULTILINE), 'make:include-path'),
    (re.compile(r'^\s*(?:HELP2MAN|XSLTPROC|XMLTO|ASCIIDOCTOR|A2X|SPHINXBUILD|DOXYGEN|DOT|TXT2MAN|SCDOC)\s*[:?+]?=\s*[^\n]+', re.MULTILINE | re.IGNORECASE), 'make:doc-tool-var'),
    (re.compile(r'(?:/usr/share/xml/docbook/stylesheet|docbook\.sourceforge\.net/release/xsl|manpages/docbook\.xsl|docbook\.xsl)', re.IGNORECASE), 'make:docbook-xsl'),
    (re.compile(r'(?:docbook\.dtd|docbook\.xsd|docbook-xml)', re.IGNORECASE), 'make:docbook-xml'),
    (re.compile(r'(?:^|\s)(doxygen|help2man|asciidoc|asciidoctor|a2x|xmlto|xsltproc|docbook2x-man|docbook2man|sphinx-build|rst2man|rst2html|txt2man|scdoc|pod2man|gtkdocize|gtkdoc-scan|gtkdoc-mkdb|gtkdoc-fixxref|bpftool|pandoc|ragel|xxd|gperf|wayland-scanner|protoc|flatc|flex|bison|swig|moc|uic|rcc|lrelease|lupdate|gdbus-codegen|glib-compile-resources|glib-compile-schemas|valac|vapigen|makeinfo)(?!\s*=)(?:\s|$)', re.MULTILINE | re.IGNORECASE), 'make:tool-invocation'),
    (re.compile(r'(?:info_TEXINFOS\s*=|TEXINFO_TEX\s*=|AM_MISSING_PROG\s*\(\s*\[?MAKEINFO\]?|^(?!\s*MAKEINFO\s*[:?+]?=\s*@MAKEINFO@\s*$)\s*MAKEINFO\s*[:?+]?=\s*[^\n]+|(?:^|\s)(?-i:makeinfo)(?:\s|$))', re.MULTILINE | re.IGNORECASE), 'make:texinfo'),
]

# ──── qmake ────────────────────────────────────────────────────────────────────

QMAKE_PATTERNS = [
    (re.compile(r'QT\s*[+*]?=\s*(.+)'), 'qmake:QT'),
    (re.compile(r'PKGCONFIG\s*[+*]?=\s*(.+)'), 'qmake:PKGCONFIG'),
    (re.compile(r'LIBS\s*[+*]?=\s*(.+)'), 'qmake:LIBS'),
    (re.compile(r'INCLUDEPATH\s*[+*]?=\s*(.+)'), 'qmake:INCLUDEPATH'),
    (re.compile(r'TRANSLATIONS\s*[+*]?=\s*(.+)'), 'qmake:TRANSLATIONS'),
    (re.compile(r'\b(doxygen|pandoc|lrelease|lupdate|qdoc)\b', re.MULTILINE | re.IGNORECASE), 'qmake:tool'),
]

# ──── pkg-config metadata ─────────────────────────────────────────────────────

PKGCONFIG_PATTERNS = [
    (re.compile(r'^\s*Requires(?:\.private)?\s*:\s*(.+)$', re.MULTILINE), 'pkgconfig:requires'),
    (re.compile(r'^\s*Libs\.private\s*:\s*(?=.*\s-l[A-Za-z0-9_+.-]+\b)(.+)$', re.MULTILINE), 'pkgconfig:libs-private'),
    (re.compile(r'^\s*Libs\s*:\s*(?=.*\s-l[A-Za-z0-9_+.-]+\b)(.+)$', re.MULTILINE), 'pkgconfig:libs'),
]

# ──── Python ───────────────────────────────────────────────────────────────────

PYTHON_PATTERNS = [
    (re.compile(r'(?:import|from)\s+(setuptools|setuptools_scm|distutils|wheel|pip|pytest|sphinx|sphinx_rtd_theme|breathe|cython|numpy|scipy|poetry|flit|hatchling|mesonpy|build)\b', re.IGNORECASE), 'python:import'),
    (re.compile(r'requires\s*=\s*\[([^\]]+)\]', re.IGNORECASE), 'python:requires'),
    (re.compile(r'install_requires\s*=\s*\[([^\]]+)\]', re.IGNORECASE), 'python:install_requires'),
    (re.compile(r'(?:setup_requires|tests_require|extras_require)\s*=\s*\[([^\]]+)\]', re.IGNORECASE), 'python:setup_requires'),
    (re.compile(r'(?:build-system\.requires|project\.dependencies|project\.optional-dependencies\.[A-Za-z0-9_.+-]+)\s*=\s*(.+)', re.IGNORECASE), 'python:structured-requirement'),
    (re.compile(r'(?:Extension|setuptools\.Extension)\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'python:extension-call'),
    (re.compile(r'(?:libraries|extra_link_args|extra_compile_args|include_dirs)\s*=\s*\[([^\]]+)\]', re.IGNORECASE | re.DOTALL), 'python:extension-args'),
    (re.compile(r'^requires\s*=\s*(.+)$', re.MULTILINE | re.IGNORECASE), 'python:cfg_requires'),
    (re.compile(r'^install_requires\s*=\s*(.+)$', re.MULTILINE | re.IGNORECASE), 'python:cfg_install_requires'),
    (re.compile(r'Packages required for building documentation:\s*\n\s*((?:(?!\n\n).)+)', re.IGNORECASE | re.DOTALL), 'python:doc_requires'),
]

# ──── PostgreSQL/PGXS extension metadata ──────────────────────────────────────

POSTGRESQL_PATTERNS = [
    (re.compile(r'\b(?:PGXS|USE_PGXS|PG_CONFIG|pg_config\s+--pgxs)\b[^\n]*', re.IGNORECASE), 'postgresql:pgxs'),
    (re.compile(r'"PostgreSQL"\s*:\s*"?[0-9A-Za-z_.+-]*"?', re.IGNORECASE), 'postgresql:pgxn-prereq'),
    (re.compile(r'^\s*(?:default_version|module_pathname|relocatable|requires|comment)\s*=', re.MULTILINE | re.IGNORECASE), 'postgresql:extension-control'),
]

# ──── Ruby native gems ────────────────────────────────────────────────────────

RUBY_PATTERNS = [
    (re.compile(r"require\s+['\"](mkmf|rake(?:/testtask)?|test/unit|webrick|pkg-config)['\"]", re.IGNORECASE), 'ruby:require'),
    (re.compile(r"\b(?:have_library|find_library)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE), 'ruby:have_library'),
    (re.compile(r"\bhave_header\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE), 'ruby:have_header'),
    (re.compile(r"\bpkg_config\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE), 'ruby:pkg_config'),
    (re.compile(r"\bdir_config\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE), 'ruby:dir_config'),
    (re.compile(r"\bfind_executable\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE), 'ruby:find_executable'),
    (re.compile(r"\bcreate_makefile\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE), 'ruby:create_makefile'),
    (re.compile(r"\b(?:add_)?(?:runtime_|development_)?dependency\s*\(?\s*['\"]([^'\"]+)['\"]", re.IGNORECASE), 'ruby:gemspec-dependency'),
    (re.compile(r"\b(?:spec\.)?(?:extensions|files|test_files|executables)\s*(?:<<|=|\+=)\s*(.+)", re.IGNORECASE), 'ruby:gemspec-metadata'),
    (re.compile(r"\b(?:extensions|test_files|files)\s*(?:<<|=|\+=)\s*(.+)", re.IGNORECASE), 'ruby:gemspec-files'),
    (re.compile(r"\bRake::(?:TestTask|ExtensionTask)\b", re.IGNORECASE), 'ruby:rake-task'),
]

# ──── OCaml/dune/opam ─────────────────────────────────────────────────────────

OCAML_PATTERNS = [
    (re.compile(r'\(depends\s+(.*?)\)', re.IGNORECASE | re.DOTALL), 'ocaml:dune-depends'),
    (re.compile(r'^\s*depends:\s*\[(.*?)\]', re.MULTILINE | re.IGNORECASE | re.DOTALL), 'ocaml:opam-depends'),
    (re.compile(r'\bconf-[A-Za-z0-9_.+-]+\b', re.IGNORECASE), 'ocaml:conf-package'),
    (re.compile(r'\(libraries\s+(.*?)\)', re.IGNORECASE | re.DOTALL), 'ocaml:dune-libraries'),
    (re.compile(r'\(package\s+([A-Za-z0-9_.+-]+)\)', re.IGNORECASE), 'ocaml:dune-package'),
    (re.compile(r'\bdune(?:-configurator)?\b', re.IGNORECASE), 'ocaml:dune-tool'),
    (re.compile(r'Pkg_config\.[A-Za-z0-9_.]+\s+[^"\n]*(?:"([^"]+)"|~(?:package|expr):\s*"([^"]+)")', re.IGNORECASE), 'ocaml:pkg-config'),
    (re.compile(r'libs\s*=\s*\[([^\]]+)\]', re.IGNORECASE | re.DOTALL), 'ocaml:c-library-flags'),
    (re.compile(r'\(foreign_stubs\b|\(c_library_flags\b', re.IGNORECASE), 'ocaml:foreign-stubs'),
]

# ──── Java/JNI/Maven/Gradle/SBT ───────────────────────────────────────────────

JAVA_PATTERNS = [
    (re.compile(r'<artifactId>\s*([^<]+)\s*</artifactId>', re.IGNORECASE), 'java:maven-artifact'),
    (re.compile(r'<groupId>\s*([^<]+)\s*</groupId>', re.IGNORECASE), 'java:maven-group'),
    (re.compile(r'<dependency\b[^>]*(?:org|name|groupId|artifactId)="[^"]+"[^>]*>', re.IGNORECASE), 'java:ivy-dependency'),
    (re.compile(r'<(?:project|javac|javadoc|jar|javah|taskdef|java)\b[^>]*>', re.IGNORECASE), 'java:ant-tool'),
    (re.compile(r'\b(?:implementation|api|compileOnly|testImplementation|testCompile|classpath)\s+["\']([^"\']+)["\']', re.IGNORECASE), 'java:gradle-dependency'),
    (re.compile(r'\blibraryDependencies\s*(?:\+=|:=|\.?)\s*["\']([^"\']+)["\']', re.IGNORECASE), 'java:sbt-dependency'),
    (re.compile(r'\b(?:javac|javadoc|jar|jni|jni\.h|JAVA_HOME|maven|gradle|sbt)\b', re.IGNORECASE), 'java:tool'),
    (re.compile(r'\b(?:junit|org\.junit|junit:junit)\b', re.IGNORECASE), 'java:junit'),
    (re.compile(r'\b(?:sbt|sbt-jni|addSbtPlugin)\b', re.IGNORECASE), 'java:sbt-tool'),
]

# ──── Perl/ExtUtils::MakeMaker ────────────────────────────────────────────────

PERL_PATTERNS = [
    (re.compile(r'\bWriteMakefile\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'perl:WriteMakefile'),
    (re.compile(r'^\s*(?:PREREQ_PM|BUILD_REQUIRES|TEST_REQUIRES|CONFIGURE_REQUIRES)\s*=>\s*\{(.*?)\}', re.MULTILINE | re.IGNORECASE | re.DOTALL), 'perl:requires-hash'),
    (re.compile(r'\b(?:PREREQ_PM|BUILD_REQUIRES|TEST_REQUIRES|CONFIGURE_REQUIRES)\s*=>\s*(.*?)\n\s*(?:[A-Z_]+\s*=>|\))', re.MULTILINE | re.IGNORECASE | re.DOTALL), 'perl:requires-block'),
    (re.compile(r"""['"]([A-Z][A-Za-z0-9_:]+)['"]\s*=>\s*['"]?[0-9._]*['"]?"""), 'perl:module-require'),
    (re.compile(r'\b(?:ExtUtils::MakeMaker|Module::Build|Module::Install|Alien::Base)\b', re.IGNORECASE), 'perl:build-tool'),
]

# ──── D/DUB ───────────────────────────────────────────────────────────────────

DLANG_PATTERNS = [
    (re.compile(r'"name"\s*:\s*"([^"]+)"', re.IGNORECASE), 'dlang:name'),
    (re.compile(r'"(?:dependencies|libs|lflags)"\s*:\s*(\[.*?\]|\{.*?\})', re.IGNORECASE | re.DOTALL), 'dlang:deps'),
    (re.compile(r'\b(?:dub|dmd|ldc2?|gdc)\b', re.IGNORECASE), 'dlang:tool'),
]

# ──── Go modules / Node package metadata ──────────────────────────────────────

GO_PATTERNS = [
    (re.compile(r'^\s*require\s*\((.*?)\)', re.MULTILINE | re.DOTALL), 'go:require-block'),
    (re.compile(r'^\s*require\s+([^\n]+)', re.MULTILINE), 'go:require'),
    (re.compile(r'^\s*module\s+([^\n]+)', re.MULTILINE), 'go:module'),
]

NODE_PATTERNS = [
    (re.compile(r'"(?:dependencies|devDependencies|optionalDependencies)"\s*:\s*\{(.*?)\}', re.IGNORECASE | re.DOTALL), 'node:dependencies'),
    (re.compile(r'\bnode-gyp\b', re.IGNORECASE), 'node:node-gyp'),
    (re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.IGNORECASE), 'node:require'),
    (re.compile(r'"(?:libraries|include_dirs|dependencies)"\s*:\s*\[(.*?)\]', re.IGNORECASE | re.DOTALL), 'node:gyp-list'),
]

# ──── configure script ────────────────────────────────────────────────────────

SCONS_PATTERNS = [
    (re.compile(r"(?:env|Environment)\s*\.\s*ParseConfig\s*\(\s*['\"]([^'\"]+)['\"]"), 'scons:ParseConfig'),
    (re.compile(r"(?:conf|env)\s*\.\s*CheckLib\s*\(\s*['\"]([^'\"]+)['\"]"), 'scons:CheckLib'),
    (re.compile(r"(?:conf|env)\s*\.\s*CheckHeader\s*\(\s*['\"]([^'\"]+)['\"]"), 'scons:CheckHeader'),
    (re.compile(r"(?:conf|env)\s*\.\s*CheckFunc\s*\(\s*['\"]([^'\"]+)['\"]"), 'scons:CheckFunc'),
    (re.compile(r"LIBS\s*=\s*\[([^\]]+)\]"), 'scons:LIBS'),
    (re.compile(r"LIBS\s*\.\s*(?:append|extend)\s*\(\s*['\"]([^'\"]+)['\"]"), 'scons:LIBS_append'),
]

BOOTSTRAP_PATTERNS = [
    (re.compile(r'\b(autoreconf|autopoint|intltoolize|gtkdocize|gnome-autogen\.sh|libtoolize|glib-gettextize)\b', re.IGNORECASE), 'bootstrap:tool'),
    (re.compile(r'(?:pkg-config|pkgconf|\$[({]PKG_CONFIG[)}])\b[^\n;&|`]*', re.IGNORECASE), 'bootstrap:pkg-config'),
    (re.compile(r'(?:apt(?:-get)?\s+install|dnf\s+install|yum\s+install)\s+[^\n]+', re.IGNORECASE), 'bootstrap:install-hint'),
    (re.compile(r'(?:please install|install package[s]?)[^\n]*(?:[a-z0-9][a-z0-9.+-]*-dev|[a-z0-9][a-z0-9.+-]*-[a-z0-9][a-z0-9.+-]*)[^\n]*', re.IGNORECASE), 'bootstrap:install-hint'),
]

CONFIGURE_SCRIPT_PATTERNS = [
    (re.compile(r'(?:^|[\s="])-l([A-Za-z0-9_+.-]+)', re.MULTILINE), 'configure:-l'),
    (re.compile(r'^\s*(?:\$\((?:CXX|CC)\)|\w*\+\+|gcc|g\+\+|clang|clang\+\+|ld)\b[^\n]*\s-l[A-Za-z0-9_+.-]+\b[^\n]*', re.MULTILINE), 'configure:link-command'),
    (re.compile(r'^\s*(?:LDLIBS|LOADLIBES|LIBS|LDFLAGS|[A-Za-z0-9_]+_(?:LIBS|LDADD|LIBADD|LDFLAGS))\s*[:+?]?=\s*[^\n]*\s-l[A-Za-z0-9_+.-]+\b[^\n]*', re.MULTILINE), 'configure:link-libs-var'),
    (re.compile(r'(?:pkg-config|pkgconf|\$[({]PKG_CONFIG[)}])\b[^\n;&|`]*', re.IGNORECASE), 'configure:pkg-config'),
    (re.compile(r'(?:^|[\s`$({])([A-Za-z0-9_+.-]+(?:-config|_config))\b[^\n;&|`]*--(?:cflags|libs|includedir|libdir|ldflags|version|modversion|variable)\b[^\n;&|`]*', re.MULTILINE | re.IGNORECASE), 'configure:config-script'),
    (re.compile(r'(?:apt(?:-get)?\s+install|dnf\s+install|yum\s+install)\s+[^\n]+', re.IGNORECASE), 'configure:install-hint'),
    (re.compile(r'(?:please install|install package[s]?)[^\n]*(?:[a-z0-9][a-z0-9.+-]*-dev|[a-z0-9][a-z0-9.+-]*-[a-z0-9][a-z0-9.+-]*)[^\n]*', re.IGNORECASE), 'configure:install-hint'),
    (re.compile(r'(?:AC_MSG_ERROR|as_fn_error|echo|printf)[^\n]*(?:not found|missing|required|install)[^\n]*', re.IGNORECASE), 'configure:error-missing'),
]

WAF_PATTERNS = [
    (re.compile(r'\bcheck_cfg\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'waf:check_cfg'),
    (re.compile(r'\bcheck_cc\s*\((.*?)\)', re.IGNORECASE | re.DOTALL), 'waf:check_cc'),
    (re.compile(r'\bfind_program\s*\(\s*["\']([^"\']+)["\']', re.IGNORECASE), 'waf:find_program'),
]

XMAKE_PATTERNS = [
    (re.compile(r'\badd_(?:requires|packages|links)\s*\(([^)]*)\)', re.IGNORECASE), 'xmake:add_dep'),
]

PREMAKE_PATTERNS = [
    (re.compile(r'\blinks\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL), 'premake:links'),
]

DOXYGEN_PATTERNS = [
    (re.compile(r'^\s*(?:PROJECT_NAME|OUTPUT_DIRECTORY|INPUT|FILE_PATTERNS|RECURSIVE|GENERATE_(?:HTML|MAN|LATEX|XML))\s*=', re.MULTILINE | re.IGNORECASE), 'doxygen:config'),
    (re.compile(r'^\s*HAVE_DOT\s*=\s*YES\b', re.MULTILINE | re.IGNORECASE), 'doxygen:have-dot'),
    (re.compile(r'^\s*DOT_PATH\s*=.+$', re.MULTILINE | re.IGNORECASE), 'doxygen:dot-path'),
]

CONAN_PATTERNS = [
    (re.compile(r'^\s*([A-Za-z0-9_.+-]+/[^\s#]+)', re.MULTILINE), 'conan:requires'),
    (re.compile(r'\b(?:self\.)?(?:requires|tool_requires|build_requires)\s*\(\s*["\']([^"\']+)["\']', re.IGNORECASE), 'conan:py-requires'),
]

VCPKG_PATTERNS = [
    (re.compile(r'"dependencies"\s*:\s*\[(.*?)\]', re.IGNORECASE | re.DOTALL), 'vcpkg:dependencies'),
]

CARGO_PATTERNS = [
    (re.compile(r'^\s*\[(?:build-)?dependencies(?:\.[^\]]+)?\]\s*$', re.MULTILINE | re.IGNORECASE), 'cargo:dependencies-section'),
    (re.compile(r'^\s*([A-Za-z0-9_.+-]+)\s*=\s*(?:"[^"]+"|\{[^}]+\})', re.MULTILINE), 'cargo:dependency'),
    (re.compile(r'^\s*build\s*=\s*"([^"]+)"', re.MULTILINE | re.IGNORECASE), 'cargo:build-script'),
]

DOCBOOK_PATTERNS = [
    (re.compile(r'<!DOCTYPE\s+[^>]*(?:DocBook|docbookx?\.dtd)[^>]*>', re.IGNORECASE | re.DOTALL), 'docbook:doctype'),
    (re.compile(r'(?:docbook\.sourceforge\.net/release/xsl|/usr/share/xml/docbook/stylesheet|docbook\.xsl)', re.IGNORECASE), 'docbook:xsl'),
    (re.compile(r'(?:docbookx?\.dtd|docbook\.xsd|docbook-xml)', re.IGNORECASE), 'docbook:xml'),
]

# ──── CI config (GitHub Actions, Travis CI, etc.) ────────────────────────────
# Disabled: CI configs are not build config files.

# CI_PATTERNS = [
#     (re.compile(r'(?:apt-get|apt|apt-fast)\s+(?:-\w+\s+)*install\s+(?:-\w+\s+)*(.+)', re.MULTILINE), 'ci:apt-install'),
#     (re.compile(r'(?:dnf|yum)\s+(?:-\w+\s+)*install\s+(?:-\w+\s+)*(.+)', re.MULTILINE), 'ci:rpm-install'),
#     (re.compile(r'(?:pacman)\s+-S(?:yu|y|u)?\s+(?:--\w+\s+)*(.+)', re.MULTILINE), 'ci:pacman-install'),
#     (re.compile(r'(?:brew)\s+install\s+(.+)', re.MULTILINE), 'ci:brew-install'),
#     (re.compile(r'^\s*-\s+(lib\S+-dev|[a-z0-9][\w.+-]*-dev)\s*$', re.MULTILINE), 'ci:yaml-list-dev'),
#     (re.compile(r'^\s*-\s+(cmake|meson|ninja-build|autoconf|automake|libtool|pkg-config|bison|flex|doxygen|gettext|intltool|valac|swig|protobuf-compiler)\s*$', re.MULTILINE), 'ci:yaml-list-tool'),
# ]

# ──── Build system classification ──────────────────────────────────────────────

BUILD_FILE_NAMES = {
    'CMakeLists.txt': 'cmake',
    'configure.ac': 'autoconf',
    'configure.in': 'autoconf',
    'Makefile.am': 'automake',
    'Makefile.in': 'automake',
    'Makefile': 'makefile',
    'makefile': 'makefile',
    'GNUmakefile': 'makefile',
    'Makerules': 'makefile',
    'Makethird': 'makefile',
    'meson.build': 'meson',
    'meson_options.txt': 'meson',
    'meson.options': 'meson',
    'SConstruct': 'scons',
    'SConscript': 'scons',
    'configure': 'configure_script',
    'autogen.sh': 'bootstrap',
    'bootstrap.sh': 'bootstrap',
    'bootstrap': 'bootstrap',
    'wscript': 'waf',
    'wscript_build': 'waf',
    'wscript_configure': 'waf',
    'xmake.lua': 'xmake',
    'premake4.lua': 'premake',
    'premake5.lua': 'premake',
    'conanfile.txt': 'conan',
    'conanfile.py': 'conan',
    'vcpkg.json': 'vcpkg',
    'Cargo.toml': 'cargo',
    'go.mod': 'go',
    'package.json': 'node',
    'binding.gyp': 'node',
    'setup.py': 'python',
    'pyproject.toml': 'python',
    'setup.cfg': 'python',
    'tox.ini': 'python',
    'pytest.ini': 'python',
    'Makefile.PL': 'perl',
    'META.json': 'postgresql',
    'Rakefile': 'ruby',
    'Gemfile': 'ruby',
    'extconf.rb': 'ruby',
    'dune': 'ocaml',
    'dune-project': 'ocaml',
    'dune-workspace': 'ocaml',
    'pom.xml': 'java',
    'build.gradle': 'java',
    'build.gradle.kts': 'java',
    'settings.gradle': 'java',
    'settings.gradle.kts': 'java',
    'build.sbt': 'java',
    'plugins.sbt': 'java',
    'ivy.xml': 'java',
    'build.xml': 'java',
    'dub.json': 'dlang',
    'dub.sdl': 'dlang',
    'doxygen.cfg': 'doxygen',
    'doxygen.conf': 'doxygen',
    # '.travis.yml': 'ci',
    # '.build.yml': 'ci',
    # 'Dockerfile': 'ci',
}

BUILD_FILE_EXTS = {
    '.cmake': 'cmake',
    '.mk': 'makefile',
    '.pro': 'qmake',
    '.pri': 'qmake',
    '.prf': 'qmake',
    '.m4': 'autoconf',
    '.toml': 'python',
    '.cfg': 'python',
    '.gemspec': 'ruby',
    '.opam': 'ocaml',
    '.gradle': 'java',
    '.gradle.kts': 'java',
}

# CI file patterns disabled
# CI_FILE_PATTERNS = [
#     re.compile(r'^\.github/workflows/.*\.ya?ml$'),
#     re.compile(r'^\.circleci/'),
#     re.compile(r'^\.gitlab-ci\.yml$'),
#     re.compile(r'^\.cirrus\.yml$'),
#     re.compile(r'(?:^|/)Dockerfile'),
#     re.compile(r'^.*\.travis\.yml$'),
#     re.compile(r'^\.build\.yml$'),
#     re.compile(r'^azure-pipelines\.yml$'),
# ]
CI_FILE_PATTERNS = []

_EXCLUDED_MAKEFILE_SUFFIXES = frozenset({
    'nt', 'win', 'win32', 'win64', 'windows',
    'mingw', 'mingw32', 'mingw64',
    'msc', 'msvc', 'vc', 'vc7', 'vc8', 'vc9', 'vc10',
    'mac', 'osx', 'darwin', 'macos', 'macosx',
    'cygwin',
    'mak', 'nmake',
    'freebsd', 'openbsd', 'netbsd', 'solaris', 'sunos', 'hpux', 'aix',
    'amiga',
})

_SOURCE_CODE_EXTS = frozenset({
    '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hxx',
    '.py', '.java', '.go', '.rs', '.rb', '.js', '.ts',
})


def _is_excluded_makefile_suffix(suffix: str) -> bool:
    if not suffix:
        return False
    parts = suffix.lower().split('.')
    return any(p in _EXCLUDED_MAKEFILE_SUFFIXES for p in parts)


def classify_file(filename: str, rel_path: str = '') -> str | None:
    """Return build system type for a filename, or None if not a build file."""
    if filename in BUILD_FILE_NAMES:
        return BUILD_FILE_NAMES[filename]
    lower = filename.lower()
    rel_lower = rel_path.replace('\\', '/').lower()
    if lower == 'conf.py' and re.search(r'(^|/)(doc|docs|documentation)(/|$)', rel_lower):
        return 'python'
    if lower.startswith('requirements') and lower.endswith('.txt') and (
        '/' not in rel_lower or re.search(r'(^|/)(doc|docs|documentation|test|tests)(/|$)', rel_lower)
    ):
        return 'python'
    if lower.endswith('.control') and lower != 'control':
        return 'postgresql'
    if lower.startswith('doxyfile'):
        return 'doxygen'
    if lower.endswith(('.xml', '.xml.in')) and re.search(
        r'(^|/)(doc|docs|documentation|manual|manuals|man|manpages)(/|$)',
        rel_lower,
    ):
        return 'docbook'
    if lower.endswith(('.pc', '.pc.in', '.pc.cmake', '.pc.in.in')):
        return 'pkgconfig'
    if lower.endswith(('.cmake.in', '.cmake.in.in')):
        return 'cmake'
    if (
        lower.endswith(('.mk.in', '.mk.in.in', '.make', '.make.in'))
        or lower.startswith('make.')
        or lower in {'config.mak', 'common.mak'}
    ):
        return 'makefile'
    for ext, bs in BUILD_FILE_EXTS.items():
        if filename.endswith(ext):
            return bs
    if lower.startswith('makefile') or lower.startswith('gnumakefile'):
        _, dot_ext = os.path.splitext(lower)
        if dot_ext in _SOURCE_CODE_EXTS:
            return None
        suffix = lower.split('.', 1)[1] if '.' in lower else ''
        if suffix == 'am' or suffix.startswith('am.'):
            return 'automake'
        if suffix == 'in' or suffix.startswith('in.'):
            return 'automake'
        if not _is_excluded_makefile_suffix(suffix):
            return 'makefile'
    if rel_path:
        for pat in CI_FILE_PATTERNS:
            if pat.match(rel_path):
                return 'ci'
    return None


def get_patterns(build_system: str):
    """Return the list of (compiled_regex, rule_type) for a build system."""
    return {
        'cmake': CMAKE_PATTERNS,
        'autoconf': AUTOCONF_PATTERNS,
        'automake': AUTOCONF_PATTERNS + MAKEFILE_PATTERNS,
        'meson': MESON_PATTERNS,
        'makefile': MAKEFILE_PATTERNS,
        'qmake': QMAKE_PATTERNS,
        'pkgconfig': PKGCONFIG_PATTERNS,
        'configure_script': CONFIGURE_SCRIPT_PATTERNS,
        'scons': SCONS_PATTERNS,
        'bootstrap': BOOTSTRAP_PATTERNS,
        'waf': WAF_PATTERNS,
        'xmake': XMAKE_PATTERNS,
        'premake': PREMAKE_PATTERNS,
        'doxygen': DOXYGEN_PATTERNS,
        'conan': CONAN_PATTERNS,
        'vcpkg': VCPKG_PATTERNS,
        'cargo': CARGO_PATTERNS,
        'docbook': DOCBOOK_PATTERNS,
        'python': PYTHON_PATTERNS,
        'postgresql': POSTGRESQL_PATTERNS,
        'ruby': RUBY_PATTERNS,
        'ocaml': OCAML_PATTERNS,
        'java': JAVA_PATTERNS,
        'perl': PERL_PATTERNS,
        'dlang': DLANG_PATTERNS,
        'go': GO_PATTERNS,
        'node': NODE_PATTERNS,
        # 'ci': CI_PATTERNS,
    }.get(build_system, [])


# Noise library names to filter out (Linux toolchain/system libs, not real deps)
# TODO: Generate and review this from Debian Trixie libc/toolchain development
# package linkable artifacts (for example libm.so -> m, libpthread.so ->
# pthread, libstdc++.so -> stdc++) instead of maintaining it by hand.
# NOISE_LIBS now lives in provider_normalization.py and is imported at top of module.
