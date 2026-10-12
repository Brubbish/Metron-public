import os
import json
import asyncio
import time
from pathlib import Path

import openai
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from chromadb.utils.batch_utils import create_batches
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    OpenAIEmbeddingFunctionConfig,
    SentenceTransformerEmbeddingFunctionConfig,
    CustomEmbeddingFunctionConfig,
    HttpChromaDBVectorMemoryConfig,
    DefaultEmbeddingFunctionConfig
    # 或者使用 Qwen‑3 在线嵌入
)
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 如果有 Qwen‑3 client 支持，可用 autogen_ext.models.qwen.QwenEmbeddingFunctionConfig
# from autogen_ext.models.qwen import QwenEmbeddingFunctionConfig


import requests
from typing import List

import openai
from typing import List

type="qwen3",                            # embedding 类型标识
model_name="netease-youdao/bce-embedding-base_v1",       # OpenAI embedding 模型
api_base="https://api.siliconflow.cn/v1/",            # 例如 https://api.openai.com/v1
api_key=os.environ.get("EMBEDDING_API_KEY", ""),             # 填入你的 API Key,


def create_openai_embedder(api_key: str, model_name: str,  base_url: str):
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    return OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name, api_base=base_url)

def make_memory(persist_path: str, collection_name: str = "my_collection") -> ChromaDBVectorMemory:
    """
    创建并返回一个 ChromaDBVectorMemory 实例，
    使用 OpenAIEmbeddingFunctionConfig 进行在线 embedding。
    """
    embedding_cfg = DefaultEmbeddingFunctionConfig()
    cfg = PersistentChromaDBVectorMemoryConfig(
        collection_name=collection_name,
        persistence_path=persist_path,
        k=100,
        score_threshold=0.3,
        embedding_function_config=embedding_cfg,
    )
    return ChromaDBVectorMemory(config=cfg)

async def ingest_if_needed(
    mem: ChromaDBVectorMemory,
    pkg_json: dict[str, str],
    batch_size: int = 512
) -> None:
    """
    清空 collection 后，批量将 pkg_id 做为 document 和主键入库，
    并把 description 存入 metadata。
    """
    # 1. 清空已有数据并触发底层 collection 初始化
    # await mem.clear()
    a=MemoryContent(content="__init__", mime_type=MemoryMimeType.TEXT, metadata={})
    print(a)
    await mem.add(a)

    # 2. 构造要插入的数据和 metadata
    docs = [(pkg_id, pkg_id) for pkg_id in pkg_json.keys()]
    metadatas = [{"pkg_id": pkg_id, "description": pkg_json[pkg_id]} for pkg_id in pkg_json.keys()]

    print(f"[Info] 准备插入 {len(docs)} 条记录")

    # 3. 批量生成 embedding 并分批添加到 collection
    client = mem._collection._client
    batches = create_batches(
        client,
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
        embeddings=None,  # None 表示使用配置中的 OpenAI 模型
        metadatas=metadatas,
        # max_batch_size=batch_size
    )

    for i, (batch_ids, batch_emb, batch_meta, batch_docs) in enumerate(batches, start=1):
        print(f">>> 插入 batch {i}，数量 {len(batch_ids)}，示例 ID：{batch_ids[:3]}")
        mem._collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_emb,
            metadatas=batch_meta
        )

    # 4. 持久化（PersistentClient 会自动写入，无需额外调用）
    print("[Success] 数据插入完成，已自动持久化到磁盘。")


async def query_vector_db(
    query: str,
    vector_db: ChromaDBVectorMemory,
    top_k: int = 3
) -> list[MemoryContent]:
    """
    对输入的 query 进行检索，返回 top_k 条 MemoryContent。
    """
    return await vector_db.query(query, top_k)

async def main():
    # 1. 准备存储目录
    persist_dir = "./chroma_db"
    os.makedirs(persist_dir, exist_ok=True)

    # 2. 初始化 memory（使用 OpenAI online embedding）
    mem = make_memory(persist_dir, collection_name="squeeze_pkgs")

    # 3. 读取待插入的 pkg_json
    json_path = Path("tmp.json")
    pkg_json = json.loads(json_path.read_text(encoding="utf-8"))

    # 4. 重建索引并持久化
    await ingest_if_needed(mem, pkg_json)

    # 5. 执行示例查询
    user_query = "The package dradio-dev is wrong"
    results = await query_vector_db(user_query, mem, top_k=10)
    print(f"\n查询结果 for '{user_query}':")
    for item in results:
        # print(f"- pkg_id: {item.content}, score: {item.metadata['score']:.4f}, desc: {item.metadata['description']}")
        print(item)



if __name__ == "__main__":
    asyncio.run(main())
