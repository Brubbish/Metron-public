"""
新的仓库分析流程（增强版）：
1. 使用预先准备好的文档文件（input_json）
2. 使用 enhanced_dep_extractor 提取依赖
3. 从文档中提取 URL 和跳转文件
4. 并行处理所有任务
5. URL 重试机制
6. 超长文档分片和智能过滤
7. 生成项目总结
"""
import os
import json
import asyncio
import re
import signal
import tempfile
import hashlib
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.ask_llm import ask_ds, ask_ds_async
from repo_parse.utils import parse_level2_directory, regex_get_urls, Fetch_URL_Content
from repo_parse.ReadPrompt import URL_SUMMARY_SYS, URL_SUMMARY_USR
from repo_parse.file_matcher import (
    extract_build_related_files,
    filter_by_keyword_content,
    post_process_matched_files
)
from core.build_artifact_discovery import discover_build_artifacts
from core.build_artifact_discovery.summarizers import summarize_all_artifacts_async, get_summarizer_usage
from core.build_family_resolver import resolve_build_families
from core.build_entry_selector import rank_build_entry_candidates

try:
    from auto_build_mcp.runtime.build_detect import detect_build
except ImportError:
    def detect_build(project_path: str, commit: str = "HEAD") -> dict:
        """Fallback detector used only when the external auto-build module is unavailable."""
        build_names = {
            "CMakeLists.txt", "meson.build", "meson.options", "meson_options.txt",
            "configure", "configure.ac", "configure.in", "autogen.sh", "bootstrap",
            "bootstrap.sh", "Makefile", "makefile", "GNUmakefile", "Makefile.am",
            "Makefile.in", "SConstruct", "SConscript", "wscript",
            "wscript_build", "wscript_configure", "xmake.lua",
            ".gn", "BUILD.gn", "configure.py", "Makefile.master",
        }
        build_files = []
        for root, dirs, files in os.walk(project_path):
            rel_root = os.path.relpath(root, project_path)
            depth = 0 if rel_root == "." else len(rel_root.split(os.sep))
            dirs[:] = [item for item in dirs if item not in {".git", "build", "builddir", "node_modules", ".venv"}]
            if depth > 4:
                dirs[:] = []
                continue
            for filename in files:
                if filename not in build_names:
                    continue
                full_path = os.path.join(root, filename)
                try:
                    modified = datetime.fromtimestamp(os.path.getmtime(full_path)).date().isoformat()
                except OSError:
                    modified = ""
                build_files.append({
                    "file": filename,
                    "path": os.path.relpath(full_path, project_path),
                    "modified": modified,
                })
        build_files.sort(key=lambda item: (item["path"].count(os.sep), item["path"]))
        return {"build_files": build_files}

# build_hints 提取在 doc_specify/enhanced_dep_extractor 里（与依赖抽取同源）
_DOC_SPECIFY_ROOT = os.environ.get("METRON_DOC_SPECIFY_ROOT", "")
if _DOC_SPECIFY_ROOT not in sys.path:
    sys.path.insert(0, _DOC_SPECIFY_ROOT)
try:
    from core.enhanced_dep_extractor.build_hints import extract_build_hints as _extract_build_hints
except Exception:
    _extract_build_hints = None

# 模型配置由 ask_llm 模块的 DEFAULT_MODEL 统一管理

# 配置常量
MAX_DOC_CHUNK_SIZE = 15000  # 单个文档片段最大字符数
MAX_TOTAL_DOC_SIZE = 50000  # 总文档内容最大字符数
URL_RETRY_TIMES = 3  # URL 抓取重试次数
URL_RETRY_DELAY = 2  # URL 重试延迟（秒）
MAX_URLS_TO_PROCESS = 5  # 最多处理的 URL 数量
LLM_CONCURRENCY = max(1, int(os.getenv("PHASE1_SUMMARIZER_CONCURRENCY", "1")))
ENHANCED_DEP_EXTRACTOR_TIMEOUT = int(os.getenv("PHASE1_EXTRACTOR_TIMEOUT", "900"))
MAX_DOCUMENT_PATH_LENGTH = 4096

_SUBORDINATE_BUILD_FILES = frozenset({
    "meson.options",
    "meson_options.txt",
    "SConscript",
    "wscript_build",
    "wscript_configure",
})

_MANIFEST_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "__pycache__", "node_modules",
    "build", "builddir", "dist", "target",
})
_COMMIT_ARTIFACT_DIR_RE = re.compile(r"^[0-9a-f]{7,40}$", re.I)
_VISIBLE_HIDDEN_BUILD_DIRS = frozenset({".github", ".circleci"})

_llm_semaphore: Optional[asyncio.Semaphore] = None

def _get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
    return _llm_semaphore

ENHANCED_DEP_EXTRACTOR_PARENT = os.environ.get("METRON_DOC_SPECIFY_ROOT", "")
ENHANCED_DEP_EXTRACTOR_MODULE = "enhanced_dep_extractor.main"
ENHANCED_DEP_SCHEMA_VERSION = 2


def _normalize_debian_version(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("theauto/debian:", "debian:"):
        if text.startswith(prefix):
            return text.split(":", 1)[1]
    return text or "bookworm"

_BUILD_FILE_FAMILY = {
    "CMakeLists.txt": "cmake",
    "meson.build": "meson",
    "meson.options": "meson",
    "meson_options.txt": "meson",
    "configure": "autotools",
    "configure.ac": "autotools",
    "configure.in": "autotools",
    "autogen.sh": "autotools",
    "bootstrap": "autotools",
    "bootstrap.sh": "autotools",
    "Makefile.am": "autotools",
    "Makefile.in": "autotools",
    "Makefile": "make",
    "makefile": "make",
    "GNUmakefile": "make",
    "Makerules": "make",
    "Makethird": "make",
    "SConstruct": "scons",
    "SConscript": "scons",
    "wscript": "waf",
    "wscript_build": "waf",
    "wscript_configure": "waf",
    "xmake.lua": "xmake",
    ".gn": "gn_ninja",
    "BUILD.gn": "gn_ninja",
    "configure.py": "python_configure",
    "Makefile.master": "workspace_make",
}
_MANIFEST_EXACT_NAMES = frozenset({
    *tuple(_BUILD_FILE_FAMILY),
    "Rakefile", "binding.gyp", "build.gradle", "build.gradle.kts",
    "settings.gradle", "settings.gradle.kts", "gradlew", "BUILD.gn",
    "Jamfile", "Jamrules", "config.m4", "mrbgem.rake",
})


@dataclass(frozen=True)
class DocumentSource:
    """A build-document input with its original source bucket."""

    path: str
    source_type: str = "repo_doc"


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_doc_path(file_path: str, compile_path: str) -> str:
    if not file_path:
        return ""
    raw_path = str(file_path)
    if "\x00" in raw_path or len(raw_path) >= MAX_DOCUMENT_PATH_LENGTH:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(compile_path) / path
    try:
        normalized = str(path.resolve(strict=False))
    except Exception:
        normalized = str(path.absolute())
    if "\x00" in normalized or len(normalized) >= MAX_DOCUMENT_PATH_LENGTH:
        return ""
    return normalized


def _safe_is_file(path: str) -> bool:
    if not path or "\x00" in path or len(path) >= MAX_DOCUMENT_PATH_LENGTH:
        return False
    try:
        return Path(path).is_file()
    except (OSError, ValueError):
        return False


def _command_key(command: str) -> str:
    return re.sub(r"\s+", " ", str(command or "").strip())


def _build_family_from_filename(filename: str) -> str:
    return _BUILD_FILE_FAMILY.get(str(filename or "").strip(), "unknown")


def _classify_command_family(command: str) -> str:
    normalized = str(command or "").strip().lower()
    if not normalized:
        return "unknown"
    if normalized.startswith("cmake "):
        return "cmake"
    if normalized.startswith("meson "):
        return "meson"
    if normalized.startswith(("autoreconf", "autogen.sh", "./autogen.sh", "autoconf", "automake", "aclocal", "libtoolize", "./configure", "configure ")):
        return "autotools"
    if normalized.startswith(("make", "gmake")):
        return "make"
    if normalized.startswith("ninja"):
        return "ninja"
    if normalized.startswith("scons"):
        return "scons"
    if normalized.startswith(("waf", "./waf")):
        return "waf"
    if normalized.startswith("xmake"):
        return "xmake"
    if normalized.startswith("bazel"):
        return "bazel"
    return "unknown"


def _command_matches_family(command: str, family: str) -> bool:
    command_family = _classify_command_family(command)
    if family == "cmake":
        return command_family in {"cmake", "ninja", "make"}
    if family == "meson":
        return command_family in {"meson", "ninja"}
    if family == "autotools":
        return command_family in {"autotools", "make"}
    if family == "make":
        return command_family == "make"
    if family == "scons":
        return command_family == "scons"
    if family == "waf":
        return command_family == "waf"
    if family == "xmake":
        return command_family == "xmake"
    if family == "bazel":
        return command_family == "bazel"
    return False


def _merge_source_dependencies(artifact_summaries) -> List[Dict[str, object]]:
    merged: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    for summary in artifact_summaries:
        for raw_dep in summary.source_deps or []:
            dep = dict(raw_dep) if isinstance(raw_dep, dict) else {"name": str(raw_dep)}
            name = str(dep.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            entry = merged.setdefault(
                key,
                {
                    "name": name,
                    "url": "",
                    "build_commands": [],
                    "reason": "",
                    "sources": [],
                },
            )
            if dep.get("url") and not entry["url"]:
                entry["url"] = dep["url"]
            for command in dep.get("build_commands") or []:
                if not command:
                    continue
                normalized = _command_key(command)
                existing = {_command_key(item) for item in entry["build_commands"]}
                if normalized not in existing:
                    entry["build_commands"].append(str(command).strip())
            reason = str(dep.get("reason") or "").strip()
            if reason:
                existing_reasons = [item.strip() for item in str(entry["reason"]).split(";") if item.strip()]
                if reason not in existing_reasons:
                    existing_reasons.append(reason)
                    entry["reason"] = "; ".join(existing_reasons)
            if summary.path not in entry["sources"]:
                entry["sources"].append(summary.path)
    return list(merged.values())


def _merge_build_commands(artifact_summaries, primary_family: str) -> Tuple[List[str], Dict[str, List[str]], List[dict]]:
    primary_commands: List[str] = []
    alternative_groups: "OrderedDict[str, List[str]]" = OrderedDict()
    command_candidates: List[dict] = []
    primary_seen = set()
    alt_seen: Dict[str, Set[str]] = {}

    for summary in artifact_summaries:
        for raw_command in summary.build_commands or []:
            command = str(raw_command or "").strip()
            if not command:
                continue
            key = _command_key(command)
            if primary_family != "unknown" and _command_matches_family(command, primary_family):
                if key not in primary_seen:
                    primary_seen.add(key)
                    primary_commands.append(command)
                    command_candidates.append({
                        "command": command,
                        "source": "doc_explicit",
                        "confidence": "high",
                        "path": summary.path,
                        "family": primary_family,
                    })
                continue
            group = _classify_command_family(command)
            alternative_groups.setdefault(group, [])
            alt_seen.setdefault(group, set())
            if key not in alt_seen[group]:
                alt_seen[group].add(key)
                alternative_groups[group].append(command)
                command_candidates.append({
                    "command": command,
                    "source": "doc_explicit",
                    "confidence": "medium",
                    "path": summary.path,
                    "family": group,
                })
    return primary_commands, dict(alternative_groups), command_candidates


def _scan_root_scripts(compile_path: str) -> List[Dict[str, str]]:
    """Find executable/wrapper scripts in root + first-level dirs.

    These are project-provided build helpers (build.sh, bootstrap.sh,
    get_deps.py, buildconf) that are NOT build-system marker files
    (CMakeLists.txt, configure.ac, etc.) but are critical for correct
    build entrypoint selection.
    """
    _SKIP_EXTS = frozenset({
        '.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.hh', '.inl',
        '.py', '.rs', '.go', '.js', '.ts', '.java', '.kt',
        '.md', '.txt', '.rst', '.adoc', '.org',
        '.json', '.xml', '.yml', '.yaml', '.toml', '.ini', '.cfg',
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
        '.in', '.am', '.ac', '.def', '.rc',
        '.o', '.a', '.so', '.dylib', '.dll', '.exe',
    })
    _SKIP_NAMES = frozenset({
        '.gitignore', '.gitmodules', '.gitattributes', '.dockerignore',
        'LICENSE', 'COPYING', 'COPYRIGHT', 'NOTICE', 'CHANGES', 'CHANGELOG',
        'AUTHORS', 'CONTRIBUTORS', 'VERSION', 'stamp-h1',
        'CMakeLists.txt', 'configure.ac', 'configure.in', 'configure',
        'meson.build', 'meson.options', 'meson_options.txt',
        'Makefile', 'GNUmakefile', 'Makefile.am', 'Makefile.in',
        'config.status', 'libtool', 'libtool.orig',
    })
    scripts: List[Dict[str, str]] = []
    seen = set()

    def _check_file(fpath: str, rel: str) -> None:
        if rel in seen:
            return
        fname = os.path.basename(fpath)
        if fname in _SKIP_NAMES:
            return
        if any(fname.endswith(ext) for ext in _SKIP_EXTS):
            return
        if fname.startswith('.'):
            return
        is_exec = os.access(fpath, os.X_OK)
        shebang = ""
        try:
            with open(fpath, errors='replace') as fh:
                first_line = fh.readline().strip()
        except Exception:
            first_line = ""
        if first_line.startswith('#!'):
            shebang = first_line
        is_script_like = (
            fname.endswith(('.sh', '.pl', '.rb'))
            or ('.' not in fname and (is_exec or shebang))
        )
        if is_exec or shebang or is_script_like:
            seen.add(rel)
            scripts.append({
                "path": rel,
                "executable": is_exec,
                "shebang": shebang[:60],
            })

    # Root directory
    try:
        for name in sorted(os.listdir(compile_path)):
            fp = os.path.join(compile_path, name)
            if os.path.isfile(fp):
                _check_file(fp, name)
    except OSError:
        pass

    # First-level subdirectories (depth=1)
    try:
        for name in sorted(os.listdir(compile_path)):
            dp = os.path.join(compile_path, name)
            if not os.path.isdir(dp) or name.startswith('.'):
                continue
            if name in ('debian', '.github', 'build', '.git', '__pycache__', 'node_modules'):
                continue
            for sub_name in sorted(os.listdir(dp))[:20]:
                fp = os.path.join(dp, sub_name)
                if os.path.isfile(fp):
                    _check_file(fp, f"{name}/{sub_name}")
    except OSError:
        pass

    return scripts


def _collect_manifest_inventory(
    compile_path: str,
    *,
    max_depth: int = 3,
    max_files: int = 320,
) -> dict:
    """Return a bounded root-relative view for zero-entry arbitration.

    This inventory only recalls repository evidence. It deliberately does not
    infer commands or promote a filename to an executable build pipeline.
    """
    root = Path(compile_path).resolve()
    files: List[dict] = []
    directories: List[str] = ["."]

    for current_root, dir_names, file_names in os.walk(root):
        current = Path(current_root)
        try:
            relative_dir = current.relative_to(root)
        except ValueError:
            continue
        depth = 0 if str(relative_dir) == "." else len(relative_dir.parts)
        dir_names[:] = sorted(
            name for name in dir_names
            if (not name.startswith(".") or name in _VISIBLE_HIDDEN_BUILD_DIRS)
            and name not in _MANIFEST_SKIP_DIRS
            and not _COMMIT_ARTIFACT_DIR_RE.fullmatch(name)
        )
        if depth >= max_depth:
            dir_names[:] = []
        for name in dir_names:
            rel_dir = (relative_dir / name).as_posix()
            directories.append(rel_dir)
        def file_priority(name: str) -> tuple[int, str]:
            lower = name.lower()
            manifest = (
                name in _MANIFEST_EXACT_NAMES
                or lower.startswith("makefile")
                or Path(name).suffix.lower() in {".pro", ".gyp"}
            )
            build_doc = lower.startswith(("readme", "install", "build", "contributing"))
            return (0 if manifest else 1 if build_doc else 2, lower)

        for name in sorted(file_names, key=file_priority):
            if len(files) >= max_files:
                break
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            is_manifest = (
                name in _MANIFEST_EXACT_NAMES
                or name.lower().startswith("makefile")
                or path.suffix.lower() in {".pro", ".gyp"}
            )
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            files.append({
                "path": relative,
                "size": int(size),
                "manifest": is_manifest,
            })
        if len(files) >= max_files:
            break

    return {
        "max_depth": max_depth,
        "truncated": len(files) >= max_files,
        "directories": list(dict.fromkeys(directories))[:max_files],
        "files": files,
        "manifest_paths": [item["path"] for item in files if item["manifest"]],
    }


def _script_command_candidates(root_scripts: List[Dict[str, str]]) -> List[dict]:
    candidates: List[dict] = []
    preferred_names = {
        "Config",
        "bootstrap",
        "bootstrap.sh",
        "autogen.sh",
        "build.sh",
        "configure",
        "Makefile.build",
    }
    for item in root_scripts:
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        name = os.path.basename(path)
        if name not in preferred_names and not name.endswith((".build", ".sh")):
            continue
        prefix = "./" if "/" not in path else ""
        candidates.append({
            "command": f"{prefix}{path}",
            "source": "script_detected",
            "confidence": "high" if item.get("executable") else "medium",
            "path": path,
            "family": "script",
        })
    return candidates


def _structural_build_steps(
    build_file: dict,
    available_paths: Optional[Set[str]] = None,
    compile_path: str = "",
) -> List[dict]:
    """Instantiate a conservative command sequence for a known build file."""
    filename = str(build_file.get("file") or "").strip()
    path = str(build_file.get("path") or filename).strip().replace("\\", "/")
    directory = os.path.dirname(path) or "."
    available_paths = {
        str(item or "").strip().replace("\\", "/")
        for item in (available_paths or set())
        if str(item or "").strip()
    }

    def sibling(name: str) -> str:
        return name if directory == "." else f"{directory}/{name}"

    if filename == "CMakeLists.txt":
        return [
            {"command": "cmake -S . -B build", "workdir": directory, "stage": "configure"},
            {"command": "cmake --build build", "workdir": directory, "stage": "compile"},
        ]
    if filename == "meson.build":
        return [
            {"command": "meson setup builddir .", "workdir": directory, "stage": "configure"},
            {"command": "meson compile -C builddir", "workdir": directory, "stage": "compile"},
        ]
    if filename in {"configure.ac", "configure.in"}:
        return [
            {"command": "autoreconf -fi", "workdir": directory, "stage": "bootstrap"},
            {"command": "./configure", "workdir": directory, "stage": "configure"},
            {"command": "make", "workdir": directory, "stage": "compile"},
        ]
    if filename in {"autogen.sh", "bootstrap", "bootstrap.sh"}:
        return [
            {"command": f"./{filename}", "workdir": directory, "stage": "bootstrap"},
            {"command": "make", "workdir": directory, "stage": "compile"},
        ]
    if filename in {"configure"}:
        return [
            {"command": "./configure", "workdir": directory, "stage": "configure"},
            {"command": "make", "workdir": directory, "stage": "compile"},
        ]
    if filename in {"Makefile.am", "Makefile.in"}:
        if sibling("configure") in available_paths:
            return [
                {"command": "./configure", "workdir": directory, "stage": "configure"},
                {"command": "make", "workdir": directory, "stage": "compile"},
            ]
        if any(sibling(item) in available_paths for item in ("configure.ac", "configure.in")):
            return [
                {"command": "autoreconf -fi", "workdir": directory, "stage": "bootstrap"},
                {"command": "./configure", "workdir": directory, "stage": "configure"},
                {"command": "make", "workdir": directory, "stage": "compile"},
            ]
        bootstrap = next(
            (item for item in ("autogen.sh", "bootstrap", "bootstrap.sh") if sibling(item) in available_paths),
            "",
        )
        if bootstrap:
            return [
                {"command": f"./{bootstrap}", "workdir": directory, "stage": "bootstrap"},
                {"command": "make", "workdir": directory, "stage": "compile"},
            ]
        return []
    if filename in {"Makefile", "makefile", "GNUmakefile"}:
        return [{"command": "make", "workdir": directory, "stage": "compile"}]
    if filename == "SConstruct":
        return [{"command": "scons", "workdir": directory, "stage": "compile"}]
    if filename == "wscript":
        waf_command = "./waf" if compile_path and os.path.isfile(os.path.join(compile_path, directory, "waf")) else "waf"
        return [
            {"command": f"{waf_command} configure", "workdir": directory, "stage": "configure"},
            {"command": f"{waf_command} build", "workdir": directory, "stage": "compile"},
        ]
    if filename == "xmake.lua":
        return [{"command": "xmake", "workdir": directory, "stage": "compile"}]
    return []


def _structural_evidence_paths(build_file: dict, available_paths: Set[str]) -> List[str]:
    """Attach same-directory build fragments to their executable parent."""
    filename = str(build_file.get("file") or "").strip()
    path = str(build_file.get("path") or filename).strip().replace("\\", "/")
    directory = os.path.dirname(path) or "."

    def sibling(name: str) -> str:
        return name if directory == "." else f"{directory}/{name}"

    fragments = {
        "meson.build": ("meson.options", "meson_options.txt"),
        "SConstruct": ("SConscript",),
        "wscript": ("wscript_build", "wscript_configure"),
    }.get(filename, ())
    return [path, *(sibling(name) for name in fragments if sibling(name) in available_paths)]


def _build_entry_candidates(
    artifact_summaries,
    root_scripts: List[Dict[str, str]],
    build_files: List[dict],
    compile_path: str = "",
) -> List[dict]:
    """Keep explicit command sequences grouped by their evidence source."""
    candidates: List[dict] = []
    for summary in artifact_summaries:
        sequence = [str(command).strip() for command in (summary.build_commands or []) if str(command).strip()]
        if not sequence:
            continue
        source = {
            "cicd": "cicd",
            "debian_rules": "debian_rules",
            "root_doc": "doc_explicit",
            "dockerfile": "doc_explicit",
        }.get(str(summary.kind or ""), "doc_explicit")
        candidates.append({
            "entry_path": "",
            "evidence_paths": [str(summary.path or "")],
            "source": source,
            "confidence": "high",
            "family": _classify_command_family(sequence[0]),
            "command_sequence": sequence,
        })

    for item in _script_command_candidates(root_scripts):
        candidates.append({
            "entry_path": item.get("path") or "",
            "source": "script_detected",
            "confidence": item.get("confidence") or "medium",
            "family": item.get("family") or "script",
            "command_sequence": [item.get("command") or ""],
        })

    available_paths = {
        str(item.get("path") or item.get("file") or "").strip().replace("\\", "/")
        for item in build_files
        if isinstance(item, dict)
    }
    for build_file in build_files:
        steps = _structural_build_steps(build_file, available_paths, compile_path)
        if not steps:
            continue
        candidates.append({
            "entry_path": build_file.get("path") or build_file.get("file") or "",
            "evidence_paths": _structural_evidence_paths(build_file, available_paths),
            "source": "structural",
            "confidence": "low",
            "family": _build_family_from_filename(build_file.get("file") or ""),
            "command_steps": steps,
            "last_update_at_commit": 0,
        })
    return candidates


def _collect_repo_context(
    compile_path: str,
    artifact_summaries,
    build_detect_result: dict,
    dependencies: List[Dict[str, str]],
    target_commit: str = "HEAD",
) -> dict:
    build_files = list(build_detect_result.get("build_files", []))
    primary_build_file = next(
        (
            item for item in build_files
            if str(item.get("file") or "").strip() not in _SUBORDINATE_BUILD_FILES
        ),
        None,
    )
    build_system = _build_family_from_filename(primary_build_file["file"]) if primary_build_file else "unknown"

    # apt_packages: only materialize the extractor's required closure. Broad
    # scan matches remain in dependency_inventory for evidence-backed repair.
    # artifact_summaries' apt_packages (from doc/CI/Dockerfile scanning) are
    # intentionally NOT merged here — they frequently contain stale
    # cross-distro package names (libpng12-dev, python-dev, etc.) and
    # provider conflicts (pkg-config vs pkgconf) that trigger
    # "held broken packages" failures in Phase 2. enhanced_dep_extractor
    # is the authoritative source for apt packages.
    merged_apt_packages: List[str] = []
    for dep in dependencies:
        if str(dep.get("requiredness") or "conditional") != "required":
            continue
        pkg = str(dep.get("unified_name") or dep.get("name") or "").strip()
        if pkg and pkg not in merged_apt_packages:
            merged_apt_packages.append(pkg)

    source_deps = _merge_source_dependencies(artifact_summaries)
    build_commands, alternative_command_groups, command_candidates = _merge_build_commands(artifact_summaries, build_system)
    root_scripts = _scan_root_scripts(compile_path)
    command_candidates.extend(_script_command_candidates(root_scripts))
    manifest_inventory = _collect_manifest_inventory(compile_path)
    build_entry_candidates = _build_entry_candidates(
        artifact_summaries,
        root_scripts,
        build_files,
        compile_path,
    )
    build_entry_candidates.extend(
        resolve_build_families(
            compile_path,
            manifest_inventory,
            artifact_summaries,
        )
    )
    build_entry_candidates = rank_build_entry_candidates(
        build_entry_candidates,
        repo_path=compile_path,
        target_commit=target_commit,
    )

    merged_setup_commands: List[dict] = []
    for summary in artifact_summaries:
        for sc in (summary.setup_commands or []):
            if not isinstance(sc, dict):
                sc = {"name": str(sc or ""), "commands": [str(sc or "")]}
            item = deepcopy(sc)
            name = str(item.get("name") or "").strip()
            commands = [str(command).strip() for command in item.get("commands") or [] if str(command).strip()]
            if not name or not commands:
                continue
            key = (name.lower(), tuple(commands))
            existing_keys = {
                (
                    str(existing.get("name") or "").lower(),
                    tuple(existing.get("commands") or []),
                )
                for existing in merged_setup_commands
            }
            if key in existing_keys:
                continue
            item["commands"] = commands
            item["source"] = str(summary.path or "")
            item["producer"] = f"artifact_summary:{summary.kind}"
            item["producer_status"] = str(summary.status or "ok")
            merged_setup_commands.append(item)

    notes_by_source = []
    for summary in artifact_summaries:
        note = str(summary.notes or "").strip()
        if note:
            notes_by_source.append({"source": summary.path, "kind": summary.kind, "note": note})

    gitmodules_excerpt = ""
    gitmodules_path = os.path.join(compile_path, ".gitmodules")
    if os.path.isfile(gitmodules_path):
        try:
            with open(gitmodules_path, "r", encoding="utf-8", errors="replace") as gf:
                gitmodules_excerpt = gf.read(4000).strip()
        except Exception:
            gitmodules_excerpt = ""

    return {
        "build_system": build_system,
        "build_file_path": primary_build_file["path"] if primary_build_file else "",
        "build_files": build_files,
        "apt_packages": merged_apt_packages,
        "source_deps": source_deps,
        "build_commands": build_commands,
        "command_candidates": command_candidates,
        "build_entry_candidates": build_entry_candidates,
        "alternative_command_groups": alternative_command_groups,
        "notes_by_source": notes_by_source,
        "has_git_submodules": bool(gitmodules_excerpt),
        "gitmodules_excerpt": gitmodules_excerpt,
        "root_scripts": root_scripts,
        "setup_commands": merged_setup_commands,
        "manifest_inventory": manifest_inventory,
    }


def _dependency_inventory(
    dependency_projects: List[dict],
    build_entries: List[dict],
    artifact_summaries,
) -> dict:
    """Normalize Phase-1 dependency facts without asking ReadRepo to filter them."""
    entries = [item for item in build_entries or [] if isinstance(item, dict)]
    eligible_entry_ids = [
        str(item.get("entry_id") or "")
        for item in entries
        if item.get("entry_id") and item.get("eligible", True)
    ]
    rows: List[dict] = []
    # A package may be required by one entry and merely optional for another;
    # deduplicate only within the same scope/requiredness, never globally.
    seen_packages: Set[tuple[str, tuple[str, ...], str]] = set()
    def summary_value(summary: Any, key: str, default: Any = None) -> Any:
        if isinstance(summary, dict):
            return summary.get(key, default)
        return getattr(summary, key, default)

    for project in dependency_projects or []:
        if not isinstance(project, dict):
            continue
        clues = [item for item in project.get("clues") or [] if isinstance(item, dict)]
        requiredness = str(project.get("requiredness") or "conditional")
        if requiredness not in {"required", "conditional", "unknown"}:
            requiredness = "unknown"
        scope_value = project.get("scope") if isinstance(project.get("scope"), dict) else {}
        scope_paths = {
            str(path).strip().lstrip("./")
            for path in scope_value.get("evidence_paths") or []
            if str(path or "").strip()
        }
        scoped_entry_ids = [
            str(entry.get("entry_id") or "")
            for entry in entries
            if entry.get("entry_id")
            and (
                not scope_paths
                or scope_paths.intersection(
                    str(path).strip().lstrip("./")
                    for path in [entry.get("entry_path"), *(entry.get("evidence_paths") or [])]
                    if str(path or "").strip()
                )
            )
        ]
        providers = []
        for package in project.get("packages") or []:
            if not isinstance(package, dict) or not package.get("name"):
                continue
            package_name = str(package.get("name") or "").strip()
            providers.append({
                "id": f"apt:{package_name.lower()}",
                "kind": "apt",
                "name": package_name,
                "suite": str(package.get("suite") or ""),
                "source_repo": str(package.get("source_repo") or ""),
                "component": str(package.get("component") or ""),
                "channel": str(package.get("channel") or ""),
                "version": str(package.get("version") or ""),
                "availability": str(package.get("availability") or "indexed"),
            })
        entity_clue = next((item for item in clues if item.get("evidence")), {})
        project_id = str(project.get("project_id") or entity_clue.get("evidence") or "dependency").strip()
        entity_kind = str(entity_clue.get("kind") or "dependency")
        dependency_id = "dep:" + hashlib.sha1(project_id.lower().encode("utf-8")).hexdigest()[:12]
        rows.append({
            "dependency_id": dependency_id,
            "entity": {"kind": entity_kind, "name": project_id},
            "requirement": {
                "requiredness": requiredness,
                "version_predicates": list(project.get("version_predicates") or [])[:8],
                "condition": str(project.get("condition") or ""),
            },
            "scope": {
                "entry_ids": scoped_entry_ids,
                "stages": [],
                "targets": [],
            },
            "evidence": [
                {
                    "id": f"{dependency_id}:clue:{index + 1}",
                    "path": str(clue.get("file_path") or ""),
                    "line": int(clue.get("line_num") or 0),
                    "producer": str(clue.get("source_type") or "enhanced_dep_extractor"),
                    "excerpt": str(clue.get("evidence") or "")[:300],
                }
                for index, clue in enumerate(clues[:12])
            ],
            "providers": providers,
            "status": "resolved" if providers else "unresolved",
        })
    for summary in artifact_summaries or []:
        source = str(summary_value(summary, "path", "") or "")
        summary_kind = str(summary_value(summary, "kind", "") or "")
        summary_status = str(summary_value(summary, "status", "ok") or "ok")
        summary_error = str(summary_value(summary, "error", "") or "")[:300]
        scoped_entry_ids = [
            str(entry.get("entry_id") or "")
            for entry in entries
            if entry.get("entry_id")
            and source
            and source in {
                str(entry.get("entry_path") or ""),
                *(str(path or "") for path in entry.get("evidence_paths") or []),
            }
        ]
        if summary_kind == "debian_control":
            scoped_entry_ids = eligible_entry_ids
        for raw_package in summary_value(summary, "apt_packages", []) or []:
            package_name = str(raw_package or "").strip()
            package_scope_key = tuple(sorted(scoped_entry_ids))
            package_key = (package_name.lower(), package_scope_key, "required" if summary_kind == "debian_control" else "conditional")
            if not package_name or package_key in seen_packages:
                continue
            seen_packages.add(package_key)
            rows.append({
                "dependency_id": "artifact:" + hashlib.sha1(package_name.lower().encode("utf-8")).hexdigest()[:12],
                "entity": {"kind": "apt_package", "name": package_name},
                "requirement": {
                    "requiredness": "required" if summary_kind == "debian_control" else "conditional",
                    "version_predicates": [],
                    "condition": "" if summary_kind == "debian_control" else "artifact_profile",
                },
                "scope": {"entry_ids": scoped_entry_ids, "stages": ["setup"], "targets": []},
                "evidence": [{"id": "artifact:1", "path": source, "line": 0, "producer": "artifact_summary", "excerpt": package_name}],
                "providers": [{"id": f"apt:{package_name.lower()}", "kind": "apt", "name": package_name, "suite": "", "availability": "unknown"}],
                "status": (
                    "resolved"
                    if summary_kind == "debian_control" and summary_status == "ok"
                    else "ambiguous" if summary_status == "ok" else "producer_error"
                ),
                "producer_status": summary_status,
                "producer_error": summary_error,
            })
        for group_index, raw_group in enumerate(summary_value(summary, "apt_package_alternatives", []) or []):
            alternatives = list(dict.fromkeys(
                str(item).strip()
                for item in raw_group
                if re.fullmatch(r"[a-z0-9][a-z0-9.+-]+", str(item).strip())
            )) if isinstance(raw_group, list) else []
            if not alternatives:
                continue
            dependency_id = "artifact-alt:" + hashlib.sha1(
                (source + "\x1f" + "|".join(alternatives)).encode("utf-8")
            ).hexdigest()[:12]
            rows.append({
                "dependency_id": dependency_id,
                "entity": {"kind": "apt_provider_group", "name": " | ".join(alternatives)},
                "requirement": {"requiredness": "required", "version_predicates": [], "condition": ""},
                "scope": {"entry_ids": scoped_entry_ids, "stages": ["setup"], "targets": []},
                "evidence": [{
                    "id": f"{dependency_id}:alternative:{group_index + 1}",
                    "path": source,
                    "line": 0,
                    "producer": "debian_control",
                    "excerpt": " | ".join(alternatives),
                }],
                "providers": [
                    {
                        "id": f"apt:{package}",
                        "kind": "apt",
                        "name": package,
                        "suite": "",
                        "availability": "unknown",
                        "role": "alternative",
                    }
                    for package in alternatives
                ],
                "status": "unbound",
            })
        for raw_dep in summary_value(summary, "source_deps", []) or []:
            if not isinstance(raw_dep, dict):
                raw_dep = {"name": str(raw_dep or "")}
            name = str(raw_dep.get("name") or "").strip()
            if not name:
                continue
            dep_key = "\x1f".join((name.lower(), str(raw_dep.get("url") or raw_dep.get("repository") or "").lower()))
            dependency_id = "source:" + hashlib.sha1(dep_key.encode("utf-8")).hexdigest()[:12]
            rows.append({
                "dependency_id": dependency_id,
                "entity": {
                    "kind": "source_dependency",
                    "name": name,
                    "url": str(raw_dep.get("url") or raw_dep.get("repository") or ""),
                },
                "requirement": {
                    "requiredness": "conditional",
                    "version_predicates": [],
                    "condition": "artifact_profile",
                },
                "scope": {"entry_ids": scoped_entry_ids, "stages": ["setup", "compile"], "targets": []},
                "evidence": [{
                    "id": f"{dependency_id}:artifact",
                    "path": source,
                    "line": 0,
                    "producer": "artifact_summary",
                    "excerpt": str(raw_dep.get("reason") or name)[:300],
                }],
                "providers": [{
                    "kind": "source",
                    "url": str(raw_dep.get("url") or raw_dep.get("repository") or ""),
                    "build_commands": list(raw_dep.get("build_commands") or [])[:8],
                }],
                "status": "ambiguous" if summary_status == "ok" else "producer_error",
                "producer_status": summary_status,
                "producer_error": summary_error,
            })
    return {"schema_version": 2, "dependencies": rows}


def _render_repo_context(repo_context: dict) -> str:
    lines = [
        "=== Repository Build Context ===",
        f"build_system: {repo_context.get('build_system') or 'unknown'}",
        f"build_file_path: {repo_context.get('build_file_path') or ''}",
        "",
        "=== Build Configuration Files ===",
    ]

    build_files = repo_context.get("build_files") or []
    if build_files:
        for item in build_files[:10]:
            lines.append(f"- {item['file']} at {item['path']} (modified: {item['modified']})")
    else:
        lines.append("- none")

    lines.extend(["", "=== Apt Package Signals ==="])
    apt_packages = repo_context.get("apt_packages") or []
    if apt_packages:
        for package in apt_packages:
            lines.append(f"- {package}")
    else:
        lines.append("- none")

    lines.extend(["", "=== Dependency Projects (preserved evidence) ==="])
    dependency_projects = repo_context.get("dependency_projects") or []
    if dependency_projects:
        for project in dependency_projects[:20]:
            if not isinstance(project, dict):
                continue
            project_id = project.get("project_id") or project.get("name") or "unknown"
            lines.append(f"- {project_id}")
            if project.get("clues"):
                lines.append(f"  clues: {json.dumps(project.get('clues')[:5], ensure_ascii=False)}")
            if project.get("version_predicates"):
                lines.append(f"  version_predicates: {json.dumps(project.get('version_predicates')[:5], ensure_ascii=False)}")
            if project.get("unmatched"):
                lines.append("  unmatched: true")
    else:
        lines.append("- none")

    lines.extend(["", "=== Source Dependencies ==="])
    source_deps = repo_context.get("source_deps") or []
    if source_deps:
        for dep in source_deps:
            lines.append(f"- {dep.get('name', 'unknown')}")
            if dep.get("reason"):
                lines.append(f"  reason: {dep['reason']}")
            if dep.get("url"):
                lines.append(f"  url: {dep['url']}")
            if dep.get("sources"):
                lines.append(f"  sources: {', '.join(dep['sources'])}")
            commands = dep.get("build_commands") or []
            if commands:
                lines.append("  build_commands:")
                for command in commands:
                    lines.append(f"    - {command}")
    else:
        lines.append("- none")

    lines.extend(["", "=== Primary Build Commands ==="])
    build_commands = repo_context.get("build_commands") or []
    if build_commands:
        for command in build_commands:
            lines.append(f"- {command}")
    else:
        lines.append("- none")

    lines.extend(["", "=== Command Candidates (with source confidence) ==="])
    command_candidates = repo_context.get("command_candidates") or []
    if command_candidates:
        for item in command_candidates[:20]:
            lines.append(
                f"- [{item.get('confidence', 'unknown')}/{item.get('source', 'unknown')}] "
                f"{item.get('command', '')}  (source: {item.get('path', '')})"
            )
    else:
        lines.append("- none")

    lines.extend(["", "=== Alternative Command Groups ==="])
    alternative_groups = repo_context.get("alternative_command_groups") or {}
    if alternative_groups:
        for family in sorted(alternative_groups):
            commands = alternative_groups[family]
            if not commands:
                continue
            lines.append(f"[{family}]")
            for command in commands:
                lines.append(f"- {command}")
    else:
        lines.append("- none")

    lines.extend(["", "=== Notes By Source ==="])
    notes_by_source = repo_context.get("notes_by_source") or []
    if notes_by_source:
        for item in notes_by_source:
            lines.append(f"- [{item['kind']}] {item['source']}: {item['note']}")
    else:
        lines.append("- none")

    lines.extend(["", "=== Git Submodules ==="])
    if repo_context.get("has_git_submodules"):
        lines.append("has_git_submodules: true")
        excerpt = str(repo_context.get("gitmodules_excerpt") or "").strip()
        if excerpt:
            lines.append(excerpt)
    else:
        lines.append("has_git_submodules: false")

    lines.extend(["", "=== Root Scripts (project-provided build helpers) ==="])
    root_scripts = repo_context.get("root_scripts") or []
    if root_scripts:
        for item in root_scripts:
            exec_flag = "x" if item.get("executable") else " "
            shebang = item.get("shebang") or ""
            lines.append(f"  [{exec_flag}] {item['path']}  {shebang}")
        lines.append("If a root script (e.g., build.sh, bootstrap.sh) exists for this build system, prefer it over raw cmake/make commands.")
    else:
        lines.append("  none")

    lines.extend(["", "=== Setup Commands (environment prerequisites, NOT git clones) ==="])
    setup_cmds = repo_context.get("setup_commands") or []
    if setup_cmds:
        for item in setup_cmds:
            name = item.get('name', '')
            commands = item.get('commands', [])
            reason = item.get('reason', '')
            lines.append(f"  {name}: {reason}")
            for cmd in commands:
                lines.append(f"    $ {cmd}")
    else:
        lines.append("  none")

    return "\n".join(lines)


def _url_fallback_reasons(repo_context: dict) -> List[str]:
    """Return the missing build-signal reasons that justify URL fallback."""
    eligible_entries = [
        item
        for item in repo_context.get("build_entry_candidates") or []
        if isinstance(item, dict)
        and item.get("eligible", True)
        and item.get("command_sequence")
    ]
    explicit_commands = [
        str(item).strip()
        for item in repo_context.get("build_commands") or []
        if str(item).strip()
    ]
    if eligible_entries or explicit_commands:
        return []
    return ["missing_eligible_build_entry", "missing_explicit_build_commands"]


def merge_document_sources(
    compile_path: str,
    source_groups: Iterable[Tuple[str, Iterable[str]]],
    *,
    skip_missing: bool = True,
) -> List[DocumentSource]:
    """Merge document candidates in caller-provided order without source ranking.

    The same file can be discovered by multiple scanners.  Keep its first
    position, but accumulate all source labels so later summaries do not treat
    one source bucket as more authoritative than another.
    """
    merged_by_key: "OrderedDict[str, DocumentSource]" = OrderedDict()
    for source_type, paths in source_groups:
        for raw_path in paths or []:
            normalized = _normalize_doc_path(str(raw_path), compile_path)
            if not normalized:
                continue
            try:
                key = os.path.normcase(os.path.realpath(normalized))
            except (OSError, ValueError):
                continue
            if skip_missing and not _safe_is_file(normalized):
                continue
            if key in merged_by_key:
                existing = merged_by_key[key]
                labels = _dedupe_preserve_order(existing.source_type.split("+") + [source_type])
                merged_by_key[key] = DocumentSource(path=existing.path, source_type="+".join(labels))
                continue
            merged_by_key[key] = DocumentSource(path=normalized, source_type=source_type)
    return list(merged_by_key.values())


def get_input_json(project_path: str) -> dict:
    """
    获取预先准备好的文档文件列表
    TODO: 实现从预先准备的 JSON 文件中读取文档列表
    
    Args:
        project_path: 项目路径
        
    Returns:
        dict: 包含文档文件列表的字典
    """
    # 占位函数，后续实现
    return {
        "files": [],
        "description": "Placeholder for document file list"
    }


def _dependency_result_paths(compile_path: str, log_dir: str) -> List[str]:
    repo_name = os.path.basename(os.path.normpath(compile_path))
    candidates = [
        os.path.join(log_dir, "enhanced_dep_result.json"),
        os.path.join(log_dir, f"{repo_name}_enhanced_dep_result.json"),
    ]
    env_paths = os.environ.get("ENHANCED_DEP_RESULT_PATHS", "")
    for path in env_paths.split(os.pathsep):
        if path.strip():
            candidates.append(path.strip())
    return _dedupe_preserve_order(candidates)


def _project_dependency_payload(data: dict, compile_path: str):
    per_project = data.get("per_project")
    if isinstance(per_project, dict):
        for key in _project_lookup_keys(compile_path):
            if key in per_project:
                return per_project[key]

        for key, value in per_project.items():
            key_basename = os.path.basename(os.path.normpath(str(key)))
            if key_basename in _project_repo_keys(compile_path):
                return value
        return None
    return data


def _project_extractor_status(data: dict, compile_path: str) -> dict:
    per_project_status = data.get("per_project_status")
    if not isinstance(per_project_status, dict):
        return {}
    for key in _project_lookup_keys(compile_path):
        value = per_project_status.get(key)
        if isinstance(value, dict):
            return value
    repo_keys = _project_repo_keys(compile_path)
    for key, value in per_project_status.items():
        if os.path.basename(os.path.normpath(str(key))) in repo_keys and isinstance(value, dict):
            return value
    return {}


def _enhanced_result_usable(data: dict, compile_path: str) -> bool:
    """Accept completed v2 results and non-empty legacy results only."""
    payload = _project_dependency_payload(data, compile_path)
    if not isinstance(payload, dict):
        return False
    dependencies = _dependencies_from_enhanced_payload(payload)
    schema_version = int(data.get("schema_version") or 0)
    if schema_version < ENHANCED_DEP_SCHEMA_VERSION:
        return bool(dependencies)
    project_status = _project_extractor_status(data, compile_path)
    return str(project_status.get("status") or "") == "ok"


def _project_repo_keys(compile_path: str) -> List[str]:
    normalized = os.path.normpath(compile_path)
    repo_name = os.path.basename(normalized)
    owner_name = os.path.basename(os.path.dirname(normalized))
    return _dedupe_preserve_order([
        repo_name,
        f"{owner_name}_{repo_name}" if owner_name else repo_name,
        f"{owner_name}##{repo_name}" if owner_name else repo_name,
    ])


def _project_lookup_keys(compile_path: str) -> List[str]:
    return _dedupe_preserve_order([
        compile_path,
        os.path.abspath(compile_path),
        os.path.realpath(compile_path),
    ])


def _project_metadata_payload(data: dict, compile_path: str) -> Optional[dict]:
    per_project_metadata = data.get("per_project_metadata")
    if not isinstance(per_project_metadata, dict):
        return None
    for key in _project_lookup_keys(compile_path):
        value = per_project_metadata.get(key)
        if isinstance(value, dict):
            return value
    repo_keys = _project_repo_keys(compile_path)
    for key, value in per_project_metadata.items():
        if os.path.basename(os.path.normpath(str(key))) in repo_keys and isinstance(value, dict):
            return value
    return None


def selected_distro_from_enhanced_result(data: dict, compile_path: str) -> Optional[str]:
    metadata = _project_metadata_payload(data, compile_path)
    if metadata:
        selected = str(metadata.get("selected_debian_version") or "").strip()
        if selected:
            return selected
    run_config = data.get("run_config")
    if isinstance(run_config, dict) and len(data.get("per_project", {}) or {}) == 1:
        selected = str(
            run_config.get("selected_debian_version")
            or run_config.get("debian_version")
            or ""
        ).strip()
        if selected:
            return selected
    return None


def constraint_details_path_from_enhanced_result(data: dict, compile_path: str, result_path: str = "") -> Optional[str]:
    metadata = _project_metadata_payload(data, compile_path)
    if metadata:
        details_path = str(metadata.get("constraint_details_path") or "").strip()
        if details_path:
            return details_path
    if result_path:
        return result_path[:-5] + "_constraint_details.json" if result_path.endswith(".json") else result_path + "_constraint_details.json"
    return None


def _dependencies_from_enhanced_payload(project_payload) -> List[Dict[str, str]]:
    dependencies = {}
    if not isinstance(project_payload, dict):
        return []

    # ── New structure: dependency_projects ──
    dep_projects = project_payload.get("dependency_projects")
    if isinstance(dep_projects, list):
        for dp in dep_projects:
            if not isinstance(dp, dict):
                continue
            project_id = str(dp.get("project_id") or "")
            clues = dp.get("clues") or []
            packages = dp.get("packages") or []
            version_predicates = dp.get("version_predicates") or []
            unmatched = dp.get("unmatched", False)

            # Extract version from predicates (take the first)
            version_str = "unspecified"
            version_op = ""
            if version_predicates:
                vp = version_predicates[0]
                version_str = str(vp.get("version") or "unspecified")
                version_op = str(vp.get("op") or "")

            # Collect clue evidences as software_names
            software_names = []
            clue_kinds = []
            clue_files = []
            for clue in clues:
                if isinstance(clue, dict):
                    ev = str(clue.get("evidence") or "").strip()
                    if ev and ev not in software_names and len(software_names) < 5:
                        software_names.append(ev)
                    clue_kinds.append(str(clue.get("kind") or ""))
                    clue_files.append(str(clue.get("file_path") or ""))

            for pkg_ref in packages:
                if not isinstance(pkg_ref, dict):
                    continue
                package_name = str(pkg_ref.get("name") or "").strip()
                if not package_name:
                    continue
                key = package_name.lower()
                if key in dependencies:
                    existing = dependencies[key]
                    for sn in software_names:
                        if sn and sn not in existing.get("software_names", []) and len(existing.get("software_names", [])) < 5:
                            existing["software_names"].append(sn)
                    for cf in clue_files:
                        if cf and cf not in existing.get("sources", []) and len(existing.get("sources", [])) < 5:
                            existing["sources"].append(cf)
                    continue

                dependencies[key] = {
                    "name": software_names[0] if software_names else package_name,
                    "unified_name": package_name,
                    "version": version_str,
                    "version_op": version_op,
                    "extractor_type": "enhanced_dep_extractor",
                    "confidence": "High" if not unmatched else "Low",
                    "context": str(project_id)[:100],
                    "sources": clue_files[:5] if clue_files else [project_id],
                    "software_names": software_names[:5],
                    "project_id": project_id,
                    "clue_kinds": clue_kinds[:5],
                    "unmatched": unmatched,
                    "requiredness": str(dp.get("requiredness") or "conditional"),
                    "condition": str(dp.get("condition") or ""),
                    "provider": {
                        "suite": str(pkg_ref.get("suite") or ""),
                        "source_repo": str(pkg_ref.get("source_repo") or ""),
                        "component": str(pkg_ref.get("component") or ""),
                        "channel": str(pkg_ref.get("channel") or ""),
                        "version": str(pkg_ref.get("version") or ""),
                        "availability": str(pkg_ref.get("availability") or "indexed"),
                    },
                }
        return sorted(dependencies.values(), key=lambda x: x["unified_name"].lower())

    # ── Legacy structure: {rel_path: list[dep]} ──
    for rel_path, file_deps in project_payload.items():
        if not isinstance(file_deps, list):
            continue
        for dep in file_deps:
            if not isinstance(dep, dict):
                continue
            software_name = str(dep.get("software_name") or "").strip()
            packages = dep.get("suite_packages") or dep.get("packages") or []
            if isinstance(packages, str):
                packages = [packages]
            if not isinstance(packages, list):
                continue

            for package in packages:
                package_name = str(package or "").strip()
                if not package_name:
                    continue
                key = package_name.lower()
                if key in dependencies:
                    existing = dependencies[key]
                    sources = existing.get("sources", [])
                    if rel_path not in sources and len(sources) < 5:
                        sources.append(rel_path)
                    software_names = existing.get("software_names", [])
                    if software_name and software_name not in software_names and len(software_names) < 5:
                        software_names.append(software_name)
                    continue

                dependencies[key] = {
                    "name": software_name or package_name,
                    "unified_name": package_name,
                    "version": "unspecified",
                    "version_op": "",
                    "extractor_type": "enhanced_dep_extractor",
                    "confidence": "High",
                    "context": str(rel_path)[:100],
                    "sources": [rel_path],
                    "software_names": [software_name] if software_name else [],
                    "requiredness": "conditional",
                    "condition": "legacy_extractor_result_requires_failure_evidence",
                }

    return sorted(dependencies.values(), key=lambda x: x["unified_name"].lower())


_PKGCONFIG_CONSTRAINT_RE = re.compile(
    r"^pkgconfig:\s*(\S+)\s*(>=|>|<=|<|==|=)\s*([0-9][0-9A-Za-z.\-+]*)"
)


def _version_sort_key(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in re.split(r"[.+~\-]", str(version)):
        m = re.match(r"^(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    return tuple(parts)


def _package_core_names(package_lower: str) -> List[str]:
    """Candidate lookup cores for a Debian package name (lowercased).

    Strips common dev-package suffixes so a pkg-config module name like
    ``libtorrent-rasterbar`` matches ``libtorrent-rasterbar-dev``.
    """
    cores = [package_lower]
    stripped = re.sub(r"-(dev|devel|tools|tool|bin|libs|common)$", "", package_lower)
    if stripped != package_lower:
        cores.append(stripped)
    return cores


def _parse_pkgconfig_constraints(metadata: Optional[dict]) -> Dict[str, Tuple[str, str]]:
    """Map normalized pkg-config module name -> (op, version).

    Only ``pkgconfig:`` constraints are considered: the module name and the
    version live in the same ``PKG_CHECK_MODULES`` argument, so attribution is
    reliable. ``cmake:`` constraints are skipped because the constraint scanner
    can misattribute versions across nearby ``find_package()`` calls.
    """
    if not isinstance(metadata, dict):
        return {}
    constraints = metadata.get("constraints")
    if not isinstance(constraints, list):
        return {}
    result: Dict[str, Tuple[str, str]] = {}
    for raw in constraints:
        if not isinstance(raw, str):
            continue
        m = _PKGCONFIG_CONSTRAINT_RE.match(raw.strip())
        if not m:
            continue
        name, op, version = m.group(1), m.group(2), m.group(3)
        key = name.lower()
        prev = result.get(key)
        if prev is None or _version_sort_key(version) > _version_sort_key(prev[1]):
            result[key] = (op, version)
    return result


def _populate_dependency_versions(
    dependencies: List[Dict[str, str]],
    data: dict,
    compile_path: str,
) -> None:
    """Best-effort populate ``version``/``version_op`` on unified deps.

    Matches pkg-config version constraints (parsed from
    ``per_project_metadata.constraints``) to unified package names by checking
    whether the constraint's module name is a substring of any package core
    name. Leaves unmatched deps at ``unspecified``.
    """
    constraint_map = _parse_pkgconfig_constraints(_project_metadata_payload(data, compile_path))
    if not constraint_map:
        return
    for dep in dependencies:
        package = (dep.get("unified_name") or "").lower()
        if not package:
            continue
        cores = _package_core_names(package)
        best: Optional[Tuple[str, str]] = None
        for cname, (op, version) in constraint_map.items():
            if len(cname) < 4:
                continue
            if not any(cname in core for core in cores):
                continue
            if best is None or _version_sort_key(version) > _version_sort_key(best[1]):
                best = (op, version)
        if best is not None:
            dep["version"] = best[1]
            dep["version_op"] = best[0]


def load_enhanced_dependency_result(compile_path: str, log_dir: str) -> Tuple[List[Dict[str, str]], Optional[dict], str, Optional[str]]:
    for result_path in _dependency_result_paths(compile_path, log_dir):
        if not os.path.exists(result_path):
            continue
        try:
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)
            project_payload = _project_dependency_payload(result, compile_path)
            dependencies = _dependencies_from_enhanced_payload(project_payload)
            _populate_dependency_versions(dependencies, result, compile_path)
            selected_distro = selected_distro_from_enhanced_result(result, compile_path)
            if _enhanced_result_usable(result, compile_path):
                return dependencies, result, result_path, selected_distro
            print(f"[EnhancedDepExtractor] Ignoring stale or degraded result in {result_path}")
        except Exception as e:
            print(f"[EnhancedDepExtractor] Failed reading {result_path}: {e}")
    return [], None, "", None


async def call_enhanced_dep_extractor(repopath: str, save_json: str, debian_version: str = "bookworm") -> dict:
    input_file = None
    process = None
    try:
        if not os.path.isdir(repopath):
            return {
                "schema_version": ENHANCED_DEP_SCHEMA_VERSION,
                "status": "error",
                "error": f"project directory does not exist: {repopath}",
            }
        debian_version = _normalize_debian_version(debian_version)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(os.path.abspath(repopath) + "\n")
            input_file = f.name

        env = os.environ.copy()
        command = [
            sys.executable,
            "-m",
            ENHANCED_DEP_EXTRACTOR_MODULE,
            input_file,
            "-o",
            save_json,
            "--workers",
            "1",
            "--debian-version",
            debian_version,
            "--distro-selection",
            "auto",
            "--scan-debian-rules",
            "--disable-llm",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ENHANCED_DEP_EXTRACTOR_PARENT,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=ENHANCED_DEP_EXTRACTOR_TIMEOUT,
            )
        except asyncio.TimeoutError:
            await _terminate_process_group(process)
            print(
                f"Warning: enhanced_dep_extractor timed out after "
                f"{ENHANCED_DEP_EXTRACTOR_TIMEOUT}s"
            )
            return {
                "schema_version": ENHANCED_DEP_SCHEMA_VERSION,
                "status": "error",
                "error": f"enhanced_dep_extractor timed out after {ENHANCED_DEP_EXTRACTOR_TIMEOUT}s",
            }
        # Cancellation propagates to the outer handler below, which performs
        # the (shielded) process-group teardown exactly once.
        result = {}
        if os.path.isfile(save_json):
            try:
                with open(save_json, encoding="utf-8") as f:
                    result = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                result = {
                    "schema_version": ENHANCED_DEP_SCHEMA_VERSION,
                    "status": "error",
                    "error": f"invalid extractor output: {type(exc).__name__}: {exc}",
                }
        else:
            result = {
                "schema_version": ENHANCED_DEP_SCHEMA_VERSION,
                "status": "error",
                "error": "enhanced_dep_extractor produced no output file",
            }
        if process.returncode != 0:
            process_error = stderr.decode(errors="ignore") or stdout.decode(errors="ignore")
            print(f"Warning: enhanced_dep_extractor failed: {process_error}")
            if isinstance(result, dict):
                result["process_error"] = process_error[-4000:]
                result["status"] = "error"
                result["error"] = (
                    f"enhanced_dep_extractor exited with status {process.returncode}"
                )
                return result
            return {
                "schema_version": ENHANCED_DEP_SCHEMA_VERSION,
                "status": "error",
                "error": f"enhanced_dep_extractor exited with status {process.returncode}",
                "process_error": process_error[-4000:],
            }
        return result
    except asyncio.CancelledError:
        await asyncio.shield(_terminate_process_group(process))
        raise
    except Exception as e:
        await _terminate_process_group(process)
        print(f"Error calling enhanced_dep_extractor: {e}")
        return {
            "schema_version": ENHANCED_DEP_SCHEMA_VERSION,
            "status": "error",
            "error": f"enhanced_dep_extractor exception: {type(e).__name__}: {e}",
        }
    finally:
        if input_file:
            try:
                os.unlink(input_file)
            except OSError:
                pass


async def _terminate_process_group(process) -> None:
    """Best-effort cleanup for the extractor and every child it spawned."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        if process.returncode is None:
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
    if process.returncode is None:
        try:
            # Shielded so a cancellation arriving mid-reap cannot skip the
            # wait and leave the killed process group unreaped.
            await asyncio.shield(process.wait())
        except (OSError, ProcessLookupError):
            pass


async def get_enhanced_dependencies(
    compile_path: str,
    log_dir: str,
    save_json: str,
    debian_version: str = "bookworm",
) -> Tuple[List[Dict[str, str]], Optional[dict], str, Optional[str]]:
    dependencies, result, source_path, selected_distro = load_enhanced_dependency_result(compile_path, log_dir)
    if result is not None:
        _write_build_hints(compile_path, log_dir)
        return dependencies, result, source_path, selected_distro

    result = await call_enhanced_dep_extractor(compile_path, save_json, debian_version)
    if not result:
        _write_build_hints(compile_path, log_dir)
        return [], None, "", None

    # Persist failures as well as successes. A failed producer result is still
    # the authoritative outcome for this Phase-1 run and must not trigger a
    # second long-running extractor invocation later in the same run.
    try:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass

    project_payload = _project_dependency_payload(result, compile_path)
    dependencies = _dependencies_from_enhanced_payload(project_payload)
    _populate_dependency_versions(dependencies, result, compile_path)
    selected_distro = (
        selected_distro_from_enhanced_result(result, compile_path)
        if _enhanced_result_usable(result, compile_path)
        else None
    )
    _write_build_hints(compile_path, log_dir)
    if not _enhanced_result_usable(result, compile_path):
        return [], result, save_json, None
    return dependencies, result, save_json, selected_distro


def _write_build_hints(compile_path: str, log_dir: str) -> None:
    """Extract disable_flags + bundled_libraries into log_dir/build_hints.json.

    agent_build_main._phase2_info() reads this file to populate project_info's
    build_options/bundled_libraries fields, so phase2/phase3 can use them
    without re-scanning the repo.
    """
    if _extract_build_hints is None:
        return
    try:
        hints = _extract_build_hints(compile_path)
        if not any(hints.get(k) for k in ("disable_flags", "bundled_libraries", "build_options", "cargo_git_refs")):
            return
        target = os.path.join(log_dir, "build_hints.json")
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hints, f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
    except Exception as e:
        print(f"[build_hints] skip: {e}")


async def fetch_url_with_retry(url: str, max_retries: int = URL_RETRY_TIMES, delay: int = URL_RETRY_DELAY) -> Tuple[bool, str]:
    """
    带重试机制的 URL 抓取
    
    Args:
        url: 要抓取的 URL
        max_retries: 最大重试次数
        delay: 重试延迟（秒）
        
    Returns:
        Tuple[bool, str]: (是否成功, 内容)
    """
    for attempt in range(max_retries):
        try:
            success, content = await Fetch_URL_Content(url, timeout=10)
            if success:
                return True, content
            
            if attempt < max_retries - 1:
                print(f"[URLFetch] Retry {attempt + 1}/{max_retries} for {url}")
                await asyncio.sleep(delay)
        except Exception as e:
            print(f"[URLFetch] Error fetching {url} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
    
    return False, ""


async def read_document_with_extraction(file_path: str, compile_path: str, max_size: int = MAX_DOC_CHUNK_SIZE) -> Tuple[str, List[str], List[str]]:
    """
    读取单个文档文件内容（支持分片），并提取其中的 URL 和文件引用
    
    Args:
        file_path: 文档文件路径
        compile_path: 仓库根路径
        max_size: 单个文档最大字符数
        
    Returns:
        Tuple[str, List[str], List[str]]: (文件内容, URL列表, 文件引用列表)
    """
    path = Path(file_path)
    if not path.is_file():
        return "", [], []
    
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        
        # 如果文档过长，只读取前 max_size 字符
        if len(content) > max_size:
            print(f"[DocReader] Warning: {file_path} is too long ({len(content)} chars), truncating to {max_size}")
            content = content[:max_size] + "\n\n[... content truncated ...]"
        
        # 提取 URL 和文件引用
        urls, files = await regex_get_urls([file_path], compile_path)
        
        return content, urls, files
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return "", [], []


def _coerce_document_sources(
    file_list: Iterable[Union[str, DocumentSource]],
    compile_path: str,
    source_labels: Optional[Dict[str, str]] = None,
) -> List[DocumentSource]:
    sources: List[DocumentSource] = []
    source_labels = source_labels or {}
    for item in file_list:
        if isinstance(item, DocumentSource):
            sources.append(item)
            continue
        normalized = _normalize_doc_path(str(item), compile_path)
        label = (
            source_labels.get(str(item))
            or source_labels.get(normalized)
            or source_labels.get(os.path.relpath(normalized, compile_path) if normalized else "")
            or "repo_doc"
        )
        sources.append(DocumentSource(path=normalized, source_type=label))
    return sources


def _format_document_section(source: DocumentSource, content: str, compile_path: str) -> str:
    try:
        rel_path = os.path.relpath(source.path, compile_path)
    except Exception:
        rel_path = source.path
    try:
        modified = datetime.fromtimestamp(Path(source.path).stat().st_mtime).strftime("%Y-%m-%d")
    except Exception:
        modified = "unknown"
    header = f"=== Source: {source.source_type} | File: {rel_path} | Modified: {modified} ==="
    return f"{header}\n{content}\n"


async def read_multiple_documents_parallel(
    file_list: List[Union[str, DocumentSource]],
    compile_path: str,
    source_labels: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str], List[str]]:
    """
    并行读取多个文档文件并提取 URL 和文件引用
    
    Args:
        file_list: 文档文件路径列表
        compile_path: 仓库根路径
        
    Returns:
        Tuple[str, List[str], List[str]]: (合并的文件内容, 所有URL, 所有文件引用)
    """
    sources = _coerce_document_sources(file_list, compile_path, source_labels)

    # 并行读取所有文档；合并时仍按 sources 顺序输出，避免 set 导致顺序漂移。
    tasks = [read_document_with_extraction(source.path, compile_path) for source in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_contents = []
    all_urls = []
    all_files = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error processing {sources[i].path}: {result}")
            continue
        
        content, urls, files = result
        if content:
            all_contents.append(_format_document_section(sources[i], content, compile_path))
        all_urls.extend(urls)
        all_files.extend(files)
    
    return "\n\n".join(all_contents), _dedupe_preserve_order(all_urls), _dedupe_preserve_order(all_files)


async def smart_summarize_urls_parallel(urls: List[str], compile_path: str) -> Dict[str, str]:
    """
    并行抓取和总结多个 URL，带重试机制
    
    Args:
        urls: URL 列表
        compile_path: 仓库路径
        
    Returns:
        Dict[str, str]: URL -> 摘要的映射
    """
    url_summaries = {}
    
    if not urls:
        return url_summaries
    
    print(f"[URLSummary] Processing {min(len(urls), MAX_URLS_TO_PROCESS)} URLs with retry...")
    
    # 限制 URL 数量
    urls_to_process = urls[:MAX_URLS_TO_PROCESS]
    
    # 并行抓取所有 URL（带重试）
    fetch_tasks = [fetch_url_with_retry(url) for url in urls_to_process]
    fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    
    # 处理抓取结果，并行调用 LLM
    llm_tasks = []
    
    for i, result in enumerate(fetch_results):
        if isinstance(result, Exception):
            print(f"[URLSummary] Exception for {urls_to_process[i]}: {result}")
            continue
        
        success, content = result
        if success and content:
            llm_tasks.append((urls_to_process[i], content))
    
    # 并行调用 LLM 总结（受 semaphore 限制）
    if llm_tasks:
        sem = _get_llm_semaphore()

        async def summarize_single_url(url, content):
            async with sem:
                try:
                    summary = await ask_ds_async(
                        system_prompt=URL_SUMMARY_SYS,
                        user_prompt=URL_SUMMARY_USR.format(content=content[:10000])
                    )
                    if isinstance(summary, list):
                        summary = summary[0] if summary else ""
                    if summary:
                        summary = re.sub(r"```[a-zA-Z]*\n?", "", summary).strip()
                        try:
                            summary_obj = json.loads(summary) if summary.startswith("{") else {"summary": summary, "urls": []}
                            return url, summary_obj.get("summary", "")
                        except:
                            return url, summary
                    return url, ""
                except Exception as e:
                    print(f"[URLSummary] Error summarizing {url}: {e}")
                    return url, ""

        summary_tasks = [summarize_single_url(url, content) for url, content in llm_tasks]
        summary_results = await asyncio.gather(*summary_tasks, return_exceptions=True)
        
        for result in summary_results:
            if isinstance(result, Exception):
                continue
            url, summary = result
            if summary:
                url_summaries[url] = summary
    
    print(f"[URLSummary] Successfully summarized {len(url_summaries)} URLs")
    return url_summaries


async def filter_and_reorganize_docs(docs_content: str, url_summaries: Dict, dependencies: List) -> str:
    """
    过滤和重组文档内容，删除重复和无关信息
    
    Args:
        docs_content: 原始文档内容
        url_summaries: URL 摘要字典
        dependencies: enhanced_dep_extractor 发现的依赖
        
    Returns:
        str: 过滤和重组后的内容
    """
    if not docs_content or len(docs_content) < 100:
        return docs_content
    
    # 如果内容不超长，直接返回
    if len(docs_content) <= MAX_TOTAL_DOC_SIZE:
        return docs_content
    
    print(f"[DocFilter] Filtering and reorganizing {len(docs_content)} chars of documentation...")
    
    try:
        # 分片处理超长文档
        chunks = []
        chunk_size = 20000
        for i in range(0, len(docs_content), chunk_size):
            chunks.append(docs_content[i:i+chunk_size])
        
        # 并行过滤每个片段（受全局 semaphore 限制）
        sem = _get_llm_semaphore()

        async def filter_chunk(chunk):
            async with sem:
                try:
                    filtered = await ask_ds_async(
                        system_prompt="You are a documentation filter for Linux x64 build information. Preserve source section headers, remove test/example instructions, Windows/macOS content, and duplicates. Keep only main library build info.",
                        user_prompt=f"Filter this content:\n\n{chunk}"
                    )
                    if isinstance(filtered, list):
                        filtered = filtered[0] if filtered else ""
                    return filtered
                except Exception as e:
                    print(f"[DocFilter] Error filtering chunk: {e}")
                    return chunk[:5000]

        filtered_chunks = []
        results = await asyncio.gather(*[filter_chunk(c) for c in chunks], return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            if result:
                filtered_chunks.append(result)
        
        # 合并过滤后的内容
        combined = "\n\n".join(filtered_chunks)
        
        # 如果仍然超长，进行最终总结
        if len(combined) > MAX_TOTAL_DOC_SIZE:
            print(f"[DocFilter] Final reorganization needed ({len(combined)} chars)")
            filter_prompt = """
You are a documentation filter. Extract and reorganize ONLY the information relevant to building the main library/executable on X64 Linux (Debian).

Remove:
- Test/example building instructions
- Windows/macOS specific instructions
- Repeated/duplicate information

Keep:
- Main library/executable build instructions
- Dependencies (required and optional)
- Build system setup (CMake, Make, etc.)
- Compilation and installation commands

Reorganize following this structure:
# Build Information
## Dependencies
## Build System
## Build Steps
## Installation

Keep source file names next to the facts they support when possible.
"""
            final_content = await ask_ds_async(
                system_prompt=filter_prompt,
                user_prompt=combined[:MAX_TOTAL_DOC_SIZE]
            )
            if isinstance(final_content, list):
                final_content = final_content[0] if final_content else combined[:MAX_TOTAL_DOC_SIZE]
            return final_content
        
        return combined
        
    except Exception as e:
        print(f"[DocFilter] Error during filtering: {e}")
        # 降级处理：返回截断的原始内容
        return docs_content[:MAX_TOTAL_DOC_SIZE]


async def process_referenced_files(file_paths: List[str], compile_path: str, visited_files: Set[str]) -> Tuple[str, List[str], List[str]]:
    """
    处理引用的文件，读取内容并提取新的 URL 和文件引用
    
    Args:
        file_paths: 文件路径列表
        compile_path: 仓库根路径
        visited_files: 已访问过的文件集合
        
    Returns:
        Tuple[str, List[str], List[str]]: (文件内容, 新发现的URL, 新发现的文件)
    """
    referenced_sources = merge_document_sources(
        compile_path,
        [("referenced_doc", file_paths)],
        skip_missing=True,
    )
    new_sources = [source for source in referenced_sources if source.path not in visited_files]
    if not new_sources:
        return "", [], []
    
    # 标记为已访问
    visited_files.update(source.path for source in new_sources)
    
    # 并行读取文件
    return await read_multiple_documents_parallel(new_sources, compile_path)


EXTRACT_BUILD_INFO_SYSTEM = """
You are a repository analysis specialist. Extract build-related information from the provided documentation.

Your task is to extract and organize **only the information explicitly stated in the text**
that is required to build this project on X64 Linux (Debian specific), including:
- Build systems and their configuration files
- Dependencies and libraries
- Compiler/toolchain requirements
- Concrete compilation and installation commands
- URLs that provide build instructions
- Referenced files that contain build information

Do NOT infer, guess, or add information that does not appear in the text.

The documentation is grouped by source sections such as:
`=== Source: regex_doc | File: README.md | Modified: YYYY-MM-DD ===`.
Follow this merge strategy:
- First extract build facts from each source section independently.
- Deduplicate equivalent commands/packages across sections.
- Separate system dependencies, source dependencies, project build steps, install targets, tests, and dev-helper commands.
- For build steps, choose one coherent minimal main-project build sequence; do NOT union mutually exclusive build families such as Autotools, CMake, Meson, Cargo, Bazel, and Make-only flows.
- Prefer newer or more specific build/install docs when sources conflict, but keep the source path in Referenced Resources.
- Do not put `make install`, `ninja install`, `meson install`, tests, or package-manager commands into the main compilation workflow unless the text explicitly says they are required for building.

## Output Format (Markdown, strictly follow this structure)

# [Project Name or "Unknown"]

## Build System Identification
- Build system: <build system name>
- Build file path: <path to build file>

## Dependency Summary
- Base dependencies:
  - <dependency name or package>
- Optional dependencies:
  - <dependency name or package>
- Compiler / toolchain requirements:
  - <compiler or tool with version if specified>

## Full Build Workflow
```bash
# Step-by-step build or installation commands exactly as they appear in the text
```

## Referenced Resources (if any)
- URLs: <list of URLs mentioned for build instructions>
- Files: <list of file paths referenced for build configuration>

If a section has no relevant information in the provided text, leave it empty.
If no build-related information at all is present, output "None" only.
"""


async def get_relateds_from_docs_and_scanner(BTarget, enhanced_result=None):
    """
    新的仓库分析流程：并行处理文档、URL、enhanced_dep_extractor
    支持递归提取文档中引用的 URL 和文件
    增强的鲁棒性：URL 重试、文档分片、智能过滤
    
    Args:
        BTarget: BuildAutomation 实例
    """
    compile_path = BTarget.compile_path
    print(f"[DocScanner] Analyzing repository: {compile_path}")
    
    # === 并行执行所有独立任务 ===
    print(f"[DocScanner] Parallel execution — discover+summarize, deps, build_detect...")

    commit = getattr(BTarget, 'commit', 'HEAD')

    async def _discover_and_summarize():
        """发现构建文档 → 并发 LLM 摘要"""
        artifacts, discovery_report = discover_build_artifacts(compile_path, include_report=True)
        print(f"[DocScanner] Found {len(artifacts)} build artifacts: {[a.path for a in artifacts]}")
        summaries = await summarize_all_artifacts_async(compile_path, artifacts, semaphore=_get_llm_semaphore())
        return artifacts, summaries, discovery_report

    async def _run_build_detect():
        """检测构建系统和配置文件"""
        return detect_build(compile_path, commit)

    async def _scan_and_summarize_urls():
        """扫描目录 URL → 并发抓取+总结"""
        directory_content = await parse_level2_directory(compile_path)
        initial_files = directory_content.get("files", [])
        urls, file_refs = await regex_get_urls(initial_files, compile_path)
        url_summaries = {}
        if urls:
            url_summaries = await smart_summarize_urls_parallel(urls, compile_path)
        return urls, file_refs, url_summaries

    producer_results = await asyncio.gather(
        _discover_and_summarize(),
        _run_build_detect(),
        return_exceptions=True,
    )
    artifact_result, build_detect_result = producer_results
    producers: Dict[str, dict] = {}
    if isinstance(artifact_result, BaseException):
        artifacts, artifact_summaries, discovery_report = [], [], {
            "candidate_count": 0,
            "selected_count": 0,
            "omitted": [],
        }
        producers["artifact_discovery"] = {
            "status": "error",
            "error": f"{type(artifact_result).__name__}: {artifact_result}"[:500],
        }
    else:
        artifacts, artifact_summaries, discovery_report = artifact_result
        failed_summaries = [summary for summary in artifact_summaries if summary.status not in {"ok", "partial"}]
        partial_summaries = [summary for summary in artifact_summaries if summary.status == "partial"]
        producers["artifact_discovery"] = {
            "status": "degraded" if failed_summaries or partial_summaries else "ok",
            "error": "; ".join(
                f"{summary.path}: {summary.error or summary.status}"
                for summary in (failed_summaries + partial_summaries)[:4]
            )[:500],
            "artifacts": len(artifacts),
        }
    if isinstance(build_detect_result, BaseException):
        producers["build_detect"] = {
            "status": "error",
            "error": f"{type(build_detect_result).__name__}: {build_detect_result}"[:500],
        }
        build_detect_result = {"build_files": []}
    else:
        producers["build_detect"] = {"status": "ok", "error": ""}

    enhanced_dep_save_path = os.path.join(BTarget.agent_log_dir, "enhanced_dep_result.json")
    if enhanced_result is None:
        try:
            enhanced_result = await get_enhanced_dependencies(
                compile_path,
                BTarget.agent_log_dir,
                enhanced_dep_save_path,
                debian_version=BTarget.sysuse,
            )
        except Exception as exc:
            enhanced_result = (
                [],
                {"schema_version": ENHANCED_DEP_SCHEMA_VERSION, "status": "error", "error": f"{type(exc).__name__}: {exc}"},
                enhanced_dep_save_path,
                None,
            )
    dependencies, loaded_enhanced_result, enhanced_dep_source_path, selected_distro = enhanced_result
    extractor_project_status = (
        _project_extractor_status(loaded_enhanced_result, compile_path)
        if isinstance(loaded_enhanced_result, dict)
        else {}
    )
    extractor_status = str(
        extractor_project_status.get("status")
        or (loaded_enhanced_result or {}).get("status")
        or "missing"
    )
    extractor_usable = isinstance(loaded_enhanced_result, dict) and _enhanced_result_usable(
        loaded_enhanced_result,
        compile_path,
    )
    producer_status = (
        "ok"
        if extractor_usable
        else "error" if extractor_status in {"error", "failed", "timeout", "missing"}
        else "degraded"
    )
    producers["enhanced_dep_extractor"] = {
        "status": producer_status,
        "producer_status": extractor_status,
        "error": str(
            extractor_project_status.get("error")
            or (loaded_enhanced_result or {}).get("error")
            or ""
        )[:500],
        "result_path": enhanced_dep_source_path or enhanced_dep_save_path,
    }
    if not hasattr(BTarget, "phase1_producers"):
        BTarget.phase1_producers = {}
    BTarget.phase1_producers.update(deepcopy(producers))

    # 打印结果
    build_related_files = [os.path.join(compile_path, a.path) for a in artifacts]
    for s in artifact_summaries:
        print(f"  [{s.kind:14s}] {s.path} → {len(s.apt_packages)} pkgs, {len(s.build_commands)} cmds")
    build_files = build_detect_result.get("build_files", [])
    if build_files:
        print(f"[DocScanner] build_detect: {len(build_files)} config files found, newest: {build_files[0]['file']}")

    repo_context = _collect_repo_context(
        compile_path,
        artifact_summaries,
        build_detect_result,
        dependencies,
        commit,
    )
    dependency_projects = []
    if isinstance(loaded_enhanced_result, dict):
        payload = _project_dependency_payload(loaded_enhanced_result, compile_path)
        if isinstance(payload, dict) and isinstance(payload.get("dependency_projects"), list):
            dependency_projects = payload.get("dependency_projects") or []
            repo_context["dependency_projects"] = dependency_projects
    repo_context["dependency_inventory"] = _dependency_inventory(
        dependency_projects,
        repo_context.get("build_entry_candidates") or [],
        artifact_summaries,
    )
    repo_context["dependency_inventory"]["producers"] = {
        "enhanced_dep_extractor": deepcopy(producers["enhanced_dep_extractor"]),
        "artifact_summaries": [
            {
                "path": str(getattr(summary, "path", "") or ""),
                "status": str(getattr(summary, "status", "ok") or "ok"),
                "error": str(getattr(summary, "error", "") or "")[:300],
            }
            for summary in artifact_summaries
        ],
    }
    fallback_reasons = _url_fallback_reasons(repo_context)
    all_urls: List[str] = []
    initial_file_refs: List[str] = []
    url_summaries: Dict[str, str] = {}
    if fallback_reasons:
        print(f"[DocScanner] URL fallback triggered: {', '.join(fallback_reasons)}")
        print(f"[DocScanner] Fallback execution — url_scan...")
        try:
            all_urls, initial_file_refs, url_summaries = await _scan_and_summarize_urls()
            producers["url_fallback"] = {"status": "ok", "error": ""}
        except Exception as exc:
            producers["url_fallback"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
            print(f"[DocScanner] URL fallback failed; continuing: {exc}")
    else:
        print("[DocScanner] URL fallback skipped: build signal already sufficient")
        producers["url_fallback"] = {"status": "skipped", "error": ""}
    BTarget.phase1_producers.update(deepcopy(producers))

    print(f"[DocScanner] {len(dependencies)} deps from config files, {len(url_summaries)} URLs summarized")
    if repo_context.get("has_git_submodules"):
        print(f"[DocScanner] .gitmodules detected — submodule init required")
    full_docs_content = _render_repo_context(repo_context)

    # === 保存结果 ===
    print(f"[DocScanner] Saving results...")

    dependency_names = [dep["unified_name"] for dep in dependencies]

    # input_json hook is intentionally disabled in the main flow for now.
    # get_input_json() is still kept as a placeholder function, but it does not
    # provide real document inputs yet and should not affect Phase 1 behavior.
    # input_json_data = get_input_json(compile_path)
    # predefined_files = input_json_data.get("files", [])
    initial_doc_sources = merge_document_sources(
        compile_path,
        [
            ("artifact_discovery", build_related_files),
            # ("input_json", predefined_files),
            ("directory_ref", initial_file_refs),
        ],
        skip_missing=True,
    )
    ordered_files = [source.path for source in initial_doc_sources]

    BTarget.docs = full_docs_content if full_docs_content else "No documentation available"
    BTarget.files = ordered_files
    BTarget.urlsummaries = {
        "build_detect": build_detect_result,
        "repo_context": repo_context,
        "url_summaries": url_summaries,
        "artifact_summaries": [s.to_dict() for s in artifact_summaries],
        "enhanced_dep_dependencies": dependencies,
        "enhanced_dep_dependency_projects": dependency_projects,
        "enhanced_dep_dependency_names": dependency_names,
        "enhanced_dep_result_path": enhanced_dep_source_path or enhanced_dep_save_path,
        "enhanced_dep_status": deepcopy(
            repo_context["dependency_inventory"]["producers"]["enhanced_dep_extractor"]
        ),
        "dependency_constraint_report_path": constraint_details_path_from_enhanced_result(
            loaded_enhanced_result or {},
            compile_path,
            enhanced_dep_source_path or enhanced_dep_save_path,
        ),
        "enhanced_dep_selected_distro": selected_distro,
        "processed_urls": all_urls,
        "processed_files": ordered_files,
        "artifact_discovery_report": discovery_report,
        "phase1_producers": deepcopy(producers),
    }

    BTarget.save_result(
        "enhanced_dep_summary.json",
        json.dumps({
            "dependencies_with_version": dependencies,
            "dependency_names": dependency_names,
            "source_path": enhanced_dep_source_path or enhanced_dep_save_path,
            "constraint_details_path": constraint_details_path_from_enhanced_result(
                loaded_enhanced_result or {},
                compile_path,
                enhanced_dep_source_path or enhanced_dep_save_path,
            ),
            "selected_distro": selected_distro,
            "project_count": len((loaded_enhanced_result or {}).get("per_project", {})),
            "producer_status": deepcopy(
                repo_context["dependency_inventory"]["producers"]["enhanced_dep_extractor"]
            ),
        }, indent=2, ensure_ascii=False)
    )
    BTarget.save_result("url_summaries.json", json.dumps(url_summaries, indent=2, ensure_ascii=False))
    BTarget.save_result("artifact_summaries.json", json.dumps([s.to_dict() for s in artifact_summaries], indent=2, ensure_ascii=False))
    BTarget.save_result("repo_context.json", json.dumps(repo_context, indent=2, ensure_ascii=False))
    BTarget.save_result("full_docs.txt", full_docs_content)

    # Merge summarizer usage into the main tracker
    from llm.ask_llm import llm_usage_tracker
    s_usage = get_summarizer_usage()
    if s_usage["calls"] > 0:
        llm_usage_tracker.record(s_usage["input_tokens"], s_usage["output_tokens"], s_usage["elapsed"], caller="artifact_summarizer")

    print(f"[DocScanner] Done: {len(artifacts)} artifacts, {len(dependencies)} deps, {len(url_summaries)} URLs")

    return {
        "docs": full_docs_content,
        "build_detect": build_detect_result,
        "repo_context": repo_context,
        "dependencies": dependencies,
        "dependency_names": dependency_names,
        "artifact_summaries": [s.to_dict() for s in artifact_summaries],
        "url_summaries": url_summaries
    }
