import os
import requests
import asyncio
import subprocess
from typing import Any
from mcp.server.fastmcp import FastMCP
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core import CancellationToken
import httpx
import traceback
import uuid



async def heartbeat(interval=60):
    while True:
        try:
            # 这里用一个轻量的工具调用作为心跳（如 Get_False_Pkgname）
            await Get_False_Pkgname([""], "buster")
        except Exception as e:
            print(f"Heartbeat failed: {e}")
        await asyncio.sleep(interval)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # 启动心跳协程
    loop.create_task(heartbeat(interval=60))
    mcp.run(transport='stdio')