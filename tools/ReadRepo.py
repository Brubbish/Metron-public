import sys
import asyncio
import uuid
import json
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# from MCPBench import Scan_Directory, Cat_File,
try:
    from tools import MCPBench
except ImportError:  # pragma: no cover - script execution fallback
    import tools.MCPBench as MCPBench

mcp = FastMCP("ReadRepo", log_level="ERROR")

# 所有需要注册的函数名（字符串）
tool_names = [
    "Scan_Directory",
    "Cat_File",
    "Fetch_URL_Content"
]


def get_readrepo_tools():
    """Return tools available to the ReadRepo agent."""
    return [MCPBench.Cat_File, MCPBench.Scan_Directory, MCPBench.Fetch_URL_Content]


def get_zero_entry_tools(repo_root: str, inventory: dict):
    """Return repository-scoped tools for bounded zero-entry discovery."""
    root = Path(repo_root).resolve()
    allowed_files = {
        str(item.get("path") or "")
        for item in inventory.get("files") or []
        if isinstance(item, dict) and str(item.get("path") or "")
    }

    async def List_Repository_Depth2() -> str:
        """Return the controller-provided root and depth-two repository inventory."""
        return json.dumps(inventory, ensure_ascii=False)

    async def Read_Repository_File(path: str, page: int = 1) -> str:
        """Read one page from a file present in the bounded repository inventory."""
        relative = str(path or "").strip().replace("\\", "/").removeprefix("./")
        if relative not in allowed_files:
            return "ERROR: path is not present in the bounded repository inventory"
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            return "ERROR: path is outside the repository or unavailable"
        if candidate.is_symlink() or not resolved.is_file():
            return "ERROR: only regular repository files may be read"
        page_number = max(1, min(int(page or 1), 20))
        start = (page_number - 1) * 80
        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"ERROR: {exc}"
        excerpt = "\n".join(lines[start:start + 80])
        return excerpt[:12000]

    List_Repository_Depth2.__name__ = "List_Repository_Depth2"
    Read_Repository_File.__name__ = "Read_Repository_File"
    return [List_Repository_Depth2, Read_Repository_File]

# 批量注册
for name in tool_names:
    func = getattr(MCPBench, name)
    mcp.tool()(func)

# Scan_Directory = getattr(MCPBench, "Scan_Directory")


# def heartbeat(interval=60):
#     while True:
#         try:
#             # 这里用一个轻量的工具调用作为心跳（如 Get_False_Pkgname）
#             Scan_Directory(['/tmp'])
#         except Exception as e:
#             print(f"Heartbeat failed: {e}")
#         asyncio.sleep(interval)
        
if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # 启动心跳协程
    # loop.create_task(heartbeat(interval=60))
    mcp.run(transport='stdio')
