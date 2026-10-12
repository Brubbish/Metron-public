import os
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils.batch_utils import create_batches



class VectorDBClient:
    def __init__(
        self,
        persist_dir: str,
        json_path: str,
        api_base: str,
        api_key: str,
        model_name: str,
        collection_name: str = "squeeze_pkgs",
        batch_size: int = 1024,
        max_items: int = 20,
    ):
        """
        :param persist_dir: ChromaDB 数据库存放目录
        :param json_path: 待索引的 JSON 文件路径
        :param api_base: OpenAI API base URL
        :param api_key: OpenAI API key
        :param model_name: OpenAI embedding 模型名
        :param collection_name: ChromaDB collection 名称
        :param batch_size: 批量写入大小
        :param max_items: 条目数上限，超过则跳过写入
        """
        self.persist_dir = persist_dir
        self.json_path = Path(json_path)
        self.batch_size = batch_size
        self.max_items = max_items

        # 初始化客户端
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 初始化 embedding 函数
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_base=api_base, api_key=api_key, model_name=model_name
        )

        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    async def ingest_if_needed(self):
        """
        如果 collection 中条目未超过 max_items，则从 json_path 读取数据并批量写入。
        """
        if self.collection.count() > self.max_items:
            return

        # 读取 JSON
        pkg_json = json.loads(self.json_path.read_text(encoding="utf-8"))
        ids = list(pkg_json.keys())
        metas = [{"pkg_id": pid, "description": pkg_json[pid]} for pid in ids]

        # 生成初步批次
        all_batches = create_batches(
            api=self.collection._client,
            ids=ids,
            documents=ids,
            embeddings=None,
            metadatas=metas,
        )

        # 拆分并写入
        for batch_idx, (batch_ids, _, batch_meta, batch_docs) in enumerate(all_batches):
            print(f"Batch #{batch_idx + 1}: total_records={len(batch_ids)}")
            for start in range(0, len(batch_ids), self.batch_size):
                sub_ids = batch_ids[start : start + self.batch_size]
                sub_meta = batch_meta[start : start + self.batch_size]
                sub_docs = batch_docs[start : start + self.batch_size]
                print(f" → Adding sub-batch: ids[{start}:{start + len(sub_ids)}] count={len(sub_ids)}")
                self.collection.add(
                    ids=sub_ids, documents=sub_docs, metadatas=sub_meta
                )
                import time
                # time.sleep(30)

    async def query(self, query_text: str, top_k: int = 5) -> list:
        """
        向量检索查询。
        :param query_text: 查询文本
        :param top_k: 返回条目数
        :return: 包含 content、metadata、score 的字典列表
        """
        try:
            resp = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=[ "documents", "metadatas", "distances"]
            )
        except Exception as e:
            return "Embedding 调用失败，错误："+repr(e)
        docs = resp["documents"][0]
        metas = resp["metadatas"][0]
        dists = resp["distances"][0]

        return [
            # {"content": doc, "metadata": meta, "score": dist}
            meta
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    async def reload(self):
        """
        强制清空并重新索引 JSON 文件中的所有数据。
        """
        # 删除旧数据
        self.collection.delete(where={})
        # 重新 ingest
        await self.ingest_if_needed()


# 使用示例
if __name__ == "__main__":

    model_name="Qwen/Qwen3-Embedding-4B"
    api_base="https://api.siliconflow.cn/v1/"
    api_key=os.environ.get("EMBEDDING_API_KEY", "")
    import asyncio
    async def run():
        # for sysuse in  ['bookworm', 'buster', 'jessie', 'squeeze', 'wheezy', 'bullseye',  'lenny', 'stretch']:
        for sysuse in ['trixie']:
        # sysuse = 'jessie'
            # if os.path.exists(f"./pkglistdb/{sysuse}_pkglist_amd64_db"):
            #     continue
            client = VectorDBClient(
                persist_dir=f"./pkglistdb/{sysuse}_pkglist_amd64_db",
                json_path=f"./pkglist/{sysuse}_pkglist_amd64.json",
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                collection_name=f"{sysuse}_db"
            )
            # 第一次写入
            await client.ingest_if_needed()

            # 查询示例
            result = await client.query("libpng", top_k=10)
            # for item in result:
            #     print(f"- pkg: {item['content']}, metadata: {item['metadata']}")
            # print(json.dumps(result, indent=2, ensure_ascii=False))
            print(result)
            
    asyncio.run(run())