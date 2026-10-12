
import os
import json
import asyncio
import requests
import re
from typing import Any
import httpx
from bs4 import BeautifulSoup
import sys
from typing import List, Dict
import logging
import chardet
from pathlib import Path


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag.get_embedding import VectorDBClient
from rag.query_package_versions import VersionKBQueryService
from core.debian_sysuse import normalize_sysuse, VALID_DEBIAN_CODENAMES

logger = logging.getLogger("autogen.tool")

VALID_RELEASES = sorted(VALID_DEBIAN_CODENAMES)

_pkglist_cache: dict[str, set] = {}
_pkgpath_cache: dict[str, dict] = {}


async def Scan_Directory(paths: list[str]) -> dict[str, dict[str, list[str]]]:
    """
    Scans multiple directories and returns non-C/C++ files and subdirectories.

    Args:
        paths (List[str]): A list of directory paths to scan.

    Returns:
        Dict[str, Union[Dict[str, List[str]], Dict[str, str]]]: 
            A dictionary where each key is a directory path, and the value is:
            
            - If the directory is accessible:
                {
                    "file": List of relative non-C/C++ file paths,
                    "directory": List of relative subdirectory paths
                }
            
            - If the directory is not accessible:
                {
                    "error": Error message string
                }
    """
    result = {}
    for path in paths:
        files_and_dirs = {"file": [], "directory": []}
        try:
            for entry in os.scandir(path):
                if entry.is_symlink():
                    continue
                try:
                    rel_path = os.path.relpath(entry.path, path)
                except Exception:
                    continue
                if entry.is_file():
                    name, ext = os.path.splitext(entry.name)
                    if ext.lower() in [".cc", ".c", ".c++", ".h", ".cpp", ".hpp", ".cxx", ".hxx", ".hh"]:
                        continue
                    files_and_dirs['file'].append(rel_path)
                elif entry.is_dir():
                    files_and_dirs['directory'].append(rel_path)
        except Exception as e:
            files_and_dirs = {"error":str(e)}
        result[path] = files_and_dirs
    return json.dumps(result)


async def Cat_File(path: str, index: int = 1) -> dict[str, int | str]:
    """Read a file in pages of up to 10 lines and return the requested page.

    Args:
        path (str): Path to the file to be read.
        index (int, optional): 1-based page number to return. Defaults to 1.

    Returns:
        dict[str, int | str]: A dictionary with the following keys:
            file_path (str): The path of the file that was opened.
            current_page (int): The page number that was returned.
            content (str):
            left_index (int): Number of pages remaining after the current one.
                                Zero if on the last page or if `index` is invalid.
    """
    
    result = {
        "file_path": path,
        "current_page": index,
        "content": "",
        "left_index": 0
    }

    page_size = 50
    try:
        if not os.path.exists(path):
            result["content"] = f"[Error] File not found: {path}"
            return result

        if not os.path.isfile(path):
            result["content"] = f"[Error] Path is not a file: {path}"
            return result

        with open(path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前 10KB 进行检测
        
        if len(raw_data) == 0:
            result["content"] = f"File is empty"
            return result
        detection = chardet.detect(raw_data)
        encoding = detection.get('encoding', 'utf-8')
        # try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            lines = f.readlines()
        # except Exception:
        #     # fallback encoding
        #     with open(path, "r", encoding="latin-1", errors="ignore") as f:
        #         lines = f.readlines()

        total_pages = (len(lines) + page_size - 1) // page_size

        if total_pages == 0:
            result["content"] = "[Empty file]"
            return result
        if index < 1 or index > total_pages:
            result["content"] = "[Error] Invalid page index"
            return result
            

        start = (index - 1) * page_size
        end = min(index * page_size, len(lines))
        page_text = "".join(lines[start:end])
        left_pages = max(total_pages - index, 0)
        result["content"] = str(page_text)
        result["left_index"] = left_pages
        return result
    except Exception as e:
        result["content"] = f"[Exception] {type(e).__name__}: {e}"
        result["left_index"] = 0
        return result



async def Fetch_URL_Content(url:str, timeout:int=10) -> tuple[bool, str | None]:
    """
    Fetches the content of a web page from a given URL.

    Args:
        url (str): The URL of the web page to fetch.
        timeout (int, optional): The request timeout in seconds. Defaults to 10.

    Returns:
        tuple[bool, str | None]: 
            A tuple containing:
                - bool: Whether the request was successful.
                - str | None: The content of the page as plain text if successful, otherwise None.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        texts = soup.get_text(separator="\n",strip=True)
        if response.ok:
            if len(texts)<=320000:
                return (True, texts)
            else:
                return (True, texts[:320000])
        else:
            return (False, None)
    except requests.exceptions.RequestException as e:
        return (False, None)


async def Search_PkgName(queries: List[str], sysuse: str) -> Dict:
    """
    Search for package names matching the given queries in the vector database.

    Args:
        queries (List[str]): A list of pure package name strings to search for (no extra content).
        sysuse (str): Debian release name, e.g., 'bookworm', 'buster', 'jessie', etc.

    Returns:
        Dict: A result message describing the search outcome:

        - If `sysuse` is invalid or DB connection fails:
            {"status": "error", "data": [], "error_message": "..."}
        - Otherwise:
            {
              "status": "success",
              "data": [
                {"query": "pkg1", "matches": ["name - description", ...]},
                ...
              ],
              "error_message": None
            }
    """
    sysuse = normalize_sysuse(sysuse)
    # os.environ["HTTP_PROXY"] set from environment if needed
    # os.environ["HTTPS_PROXY"] set from environment if needed
    db = await VectorDBConnect(sysuse, "pkglist")
    if not db:
        return {"status": "error", "data": [], "error_message": "Invalid sysuse parameter or failed DB connect."}
    # db.ingest_if_needed()
    results = []
    try:
        for q in queries:
            res = await db.query(q, top_k=10)
            matches = [f"{r['pkg_id']} - {r['description']}" for r in res] if res else ["No matching package found."]
            results.append({"query": q, "matches": matches})
    except Exception as e:
        return {"status": "error", "data": [], "error_message": f"Search failed: {repr(e)}"}
    return {"status": "success", "data": results, "error_message": None}

async def Get_False_Pkgname(llmpkgs: List[str], sysuse: str) -> Dict:
    """
    Verifies whether LLM-generated package names exist in the specified Debian package list,
    and queries for valid package names as candidates.

    Args:
        llmpkgs (List[str]): A list of APT package names generated by the LLM.
        sysuse (str): Debian release name, e.g., 'bookworm', 'buster', 'jessie', etc.

    Returns:
        Dict: A result message describing the verification outcome:

        - If `sysuse` is invalid:
            {"status": "error", "invalid": [], "suggestions": {}, "error_message": "..."}
        - If `llmpkgs` is not a list:
            {"status": "error", "invalid": [], "suggestions": {}, "error_message": "..."}
        - If some package names are invalid:
            {
              "status": "success",
              "invalid": ["pkg1", ...],
              "suggestions": {
                "pkg1": ["suggest1 - description", ...],
                ...
              },
              "error_message": None
            }
        - If all package names are valid:
            {"status": "success", "invalid": [], "suggestions": {}, "error_message": None}
    """

    sysuse = normalize_sysuse(sysuse)
    if sysuse not in VALID_RELEASES:
        return {"status": "error", "invalid": [], "suggestions": {}, "error_message": "Invalid sysuse parameter."}
    if not isinstance(llmpkgs, list):
        return {"status": "error", "invalid": [], "suggestions": {}, "error_message": "Invalid llmpkgs type; expected list."}

    if sysuse in _pkglist_cache:
        all_aptpkg = _pkglist_cache[sysuse]
    else:
        try:
            clean_path = Path(f"./rag/clean_pkglist/{sysuse}_clean_pkglist_amd64.json")
            pkglist_path = Path(f"./rag/pkglist/{sysuse}_pkglist_amd64.json")
            with open(clean_path if clean_path.exists() else pkglist_path, "r", encoding="utf-8") as f:
                pkg_data = json.load(f)
                all_aptpkg = set(pkg_data.keys())
            _pkglist_cache[sysuse] = all_aptpkg
        except Exception as e:
            return {"status": "error", "invalid": [], "suggestions": {}, "error_message": f"Failed to read package list: {e}"}
    llmpkgs = [pkg.strip() for pkg in llmpkgs if isinstance(pkg, str) and pkg.strip()]
    invalid = [pkg for pkg in llmpkgs if pkg not in all_aptpkg]

    suggestions = {}
    if invalid:
        search_result = await Search_PkgName(invalid, sysuse,)
        if search_result["status"] == "success":
            for item in search_result["data"]:
                suggestions[item["query"]] = item["matches"]
        else:
            return {"status": "error", "invalid": invalid, "suggestions": {}, 
                    "error_message": "Search tool error: " + (search_result.get("error_message") or "")}
    
    return {"status": "success", "invalid": invalid, "suggestions": suggestions, "error_message": None}


async def VectorDBConnect(sysuse, db_type):
    sysuse = normalize_sysuse(sysuse)
    if sysuse not in VALID_RELEASES:
        return None
    db = VectorDBClient(
        persist_dir=f"./rag/pkglist/{sysuse}_pkglist_amd64_db",
        json_path=f"./rag/pkglistdb/{sysuse}_pkglist_amd64.json",
        # persist_dir="./chroma_db",
        # json_path="tmp.json",
        api_base="https://api.siliconflow.cn/v1/",
        api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        model_name="Qwen/Qwen3-Embedding-4B",
        collection_name=f"{sysuse}_db"
        # collection_name="squeeze_pkgs"
    )
    return db


# TODO:传入db?


async def Search_PkgFile(query: str, sysuse: str) -> str:
    """Search Debian package-file metadata for packages that ship a path/header.

    Args:
        query: Header name or file path to look up, e.g. ``openssl/ssl.h``.
        sysuse: Debian release name.

    Returns:
        A compact text list of matching ``path -> package`` entries.
    """
    result = await Search_PkgFile_Matches(query, sysuse)
    if result.get("status") != "success":
        return str(result.get("error_message") or "Package-file lookup failed.")
    matches = result.get("matches") or []
    if not matches:
        return "No matching package file found."
    return "These packages ship matched files:\n" + "\n".join(
        f"{item['path']} -> {item['package']}" for item in matches
    )


async def Search_PkgFile_Matches(query: str, sysuse: str) -> Dict[str, Any]:
    """Return structured package providers from the local suite/arch file KB."""
    normalized_sysuse = normalize_sysuse(sysuse)
    if normalized_sysuse not in VALID_RELEASES:
        return {"status": "error", "matches": [], "error_message": "Invalid sysuse parameter."}
    normalized_query = (query or "").strip().lower().lstrip("/")
    if not normalized_query:
        return {"status": "error", "matches": [], "error_message": "No query provided."}
    if normalized_sysuse in _pkgpath_cache:
        file_to_pkg = _pkgpath_cache[normalized_sysuse]
    else:
        data_path = f"./rag/clean_pkgpath/{normalized_sysuse}_clean_pkgpath_amd64.json"
        try:
            with open(data_path, "r", encoding="utf-8") as handle:
                file_to_pkg = json.load(handle)
            _pkgpath_cache[normalized_sysuse] = file_to_pkg
        except Exception as exc:
            return {
                "status": "error",
                "matches": [],
                "error_message": f"Failed to read package-file database: {exc}",
            }
    basename = os.path.basename(normalized_query)
    matches = []
    canonical_include = f"usr/include/{normalized_query}"
    macro_file = ""
    if normalized_query.endswith(".m4"):
        macro_file = basename
    elif re.fullmatch(r"(?:ax|ac|am|lt|pkg)_[a-z0-9_]+", basename):
        macro_file = f"{basename}.m4"
    for file_path, package in file_to_pkg.items():
        path_lower = str(file_path).lower().lstrip("/")
        quality = ""
        rank = 99
        is_m4_provider_path = (
            path_lower.endswith(".m4")
            and (
                path_lower.startswith("usr/share/aclocal")
                or "/aclocal/" in "/" + path_lower
                or "/m4/" in "/" + path_lower
            )
        )
        if path_lower == normalized_query:
            quality, rank = "exact_path", 0
        elif path_lower == canonical_include:
            quality, rank = "exact_include_path", 0
        elif macro_file and is_m4_provider_path and path_lower.endswith("/" + macro_file):
            quality, rank = "m4_macro_provider_path", 1
        elif path_lower.endswith("/" + normalized_query) and "/include/" in "/" + path_lower:
            quality, rank = "include_suffix", 1
        elif basename and path_lower.endswith("/" + basename):
            # A basename-only match is useful for queries such as ``nasm`` or
            # ``zlib.h``, but restrict it to executable/include locations. A
            # raw substring match would select documentation and unrelated
            # files that happen to contain the same token.
            is_include = "/include/" in "/" + path_lower or path_lower.startswith("usr/include/")
            is_bin = path_lower.startswith(("usr/bin/", "usr/sbin/", "bin/", "sbin/"))
            if is_include or is_bin or (macro_file and is_m4_provider_path):
                quality, rank = "basename_provider_path", 2
        if not quality:
            continue
        matches.append({
            "path": str(file_path),
            "package": str(package),
            "query": normalized_query,
            "sysuse": normalized_sysuse,
            "architecture": "amd64",
            "source": "clean_pkgpath",
            "match_quality": quality,
            "match_rank": rank,
        })
        if len(matches) >= 400:
            matches.sort(key=lambda item: (
                int(item.get("match_rank") or 0),
                len(str(item.get("path") or "")),
                str(item.get("package") or ""),
            ))
            del matches[200:]
    matches.sort(key=lambda item: (
        int(item.get("match_rank") or 0),
        len(str(item.get("path") or "")),
        str(item.get("package") or ""),
    ))
    return {"status": "success", "matches": matches[:20], "error_message": None}


async def Docker_Run_Command(command: str) -> Dict[str, Any]:
    """Run one command inside the configured Docker build container."""
    from tools.Docker_Interact import Docker_Shell_Interact
    return await Docker_Shell_Interact(command)


async def Verify_Package(package_names: List[str], sysuse: str) -> Dict:
    """Verify Debian APT package names for the requested suite."""
    return await Get_False_Pkgname(package_names, sysuse)


async def Read_Build_Log(log_dir: str, log_name: str = "list_recent", max_chars: int = 20000) -> Dict[str, Any]:
    """Read Docker command logs.

    Args:
        log_name:
          - "list_recent": show the 5 most recent command logs (filename + command + exit_code)
          - "list_all": show all command logs (filename + command + exit_code)
          - "command_NNN.log": read full content of a specific log file

    Returns:
        For list modes: table of filename -> command summary + exit_code
        For read mode: full log content (timestamp, command, exit_code, stdout, stderr)
    """
    base_dir = Path(log_dir or ".").resolve()
    max_chars_int = max(1000, min(int(max_chars or 20000), 100000))
    if not base_dir.exists():
        return {"status": "error", "error_message": f"Log directory not found: {base_dir}"}

    all_logs = sorted(base_dir.glob("*.log"))

    if log_name == "list_recent":
        logs_to_show = all_logs[-5:]
    elif log_name == "list_all":
        logs_to_show = all_logs
    else:
        logs_to_show = None

    if logs_to_show is not None:
        entries = []
        for log_file in logs_to_show:
            content = log_file.read_text(encoding="utf-8", errors="replace")
            cmd_match = re.search(r'^\$ (.+)$', content, re.MULTILINE)
            exit_match = re.search(r'^exit_code: (.+)$', content, re.MULTILINE)
            cmd = cmd_match.group(1)[:80] if cmd_match else "?"
            exit_code = exit_match.group(1) if exit_match else "?"
            entries.append(f"{log_file.name}: [{exit_code}] {cmd}")
        return {"status": "success", "logs": entries}

    # Read specific file
    target = (base_dir / Path(log_name).name).resolve()
    if base_dir not in target.parents and target != base_dir:
        return {"status": "error", "error_message": "Log path escapes the command log directory."}
    if not target.exists() or not target.is_file():
        return {"status": "error", "error_message": f"Log not found: {Path(log_name).name}"}

    content = target.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars_int:
        content = "...\n" + content[-max_chars_int:]
    return {"status": "success", "log_name": target.name, "content": content}


async def Search_Package_File(query: str, sysuse: str) -> str:
    """Find Debian packages that ship a requested header or file path."""
    return await Search_PkgFile(query, sysuse)


async def Query_Package_Versions(query: str, sysuse: str) -> Dict[str, Any]:
    """Query the Debian source-version KB for a package name."""
    suite = normalize_sysuse(sysuse)

    def _query() -> Dict[str, Any]:
        service = None
        try:
            service = VersionKBQueryService()
            return {"status": "success", "data": service.query(query, distro_filter=suite)}
        except Exception as exc:
            return {"status": "error", "error_message": str(exc)}
        finally:
            if service is not None:
                try:
                    service.close()
                except Exception:
                    pass

    return await asyncio.to_thread(_query)

# TODO:
# 网页搜索报错信息
# RAG 搜索/rapidfuzz/apt-file search： done
# 提取头文件

# 测试用 main 函数
async def main():
    paths = ["./data/env_mapping/sudo-project__sudo__ce0ec8ddca9e1910157937917468808ae4359a62/INSTALL.md"]  # 这里你可以换成要测试的目录
    result = await Cat_File(paths[0], 1)
    print(result)
    # for path, info in result.items():
    #     print(f"\n目录: {path}")
    #     print(info)

if __name__ == "__main__":
    asyncio.run(main())
