import logging
import asyncio
from typing import List, Dict, Tuple, Union
import numpy as np
from zhipuai import ZhipuAI
from research_agent.core.config import Config
import aiohttp
from pyaml_env import parse_config
from research_agent.core.utils import chunking
import random

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel_speed:
    def __init__(self):
        """初始化嵌入模型"""
        self.configs = parse_config(Config.YAML_CONFIG)
        self.client = ZhipuAI(api_key=self.configs[Config.DEFAULT_MODEL]["API_KEY"])
        # 初始化缓存和配置
        self._embedding_cache = {}
        self.max_cache_size = 10000
        self.batch_size = 5  # 批处理大小
        self.embedding_dim = 2048  # 向量维度
        self.max_retries = 3  # 最大重试次数

    async def get_embeddings(self, texts: Union[str, List[str]]) -> Tuple[List[np.ndarray], List[int]]:
        """批量处理embedding请求"""
        if isinstance(texts, str):
            texts = [texts]

        # 创建结果列表和待处理列表
        results = [None] * len(texts)
        texts_to_process = []
        process_indices = []

        # 首先检查缓存
        for idx, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self._embedding_cache:
                results[idx] = self._embedding_cache[cache_key]
            else:
                texts_to_process.append(text)
                process_indices.append(idx)

        # 处理未缓存的文本
        if texts_to_process:
            # 批量处理未缓存的文本
            for i in range(0, len(texts_to_process), self.batch_size):
                batch_texts = texts_to_process[i:i + self.batch_size]
                batch_indices = process_indices[i:i + self.batch_size]

                retry_count = 0
                while retry_count < self.max_retries:
                    try:
                        # logger.info(
                        #     f"处理批次: {i // self.batch_size + 1}, 文本: {batch_texts}")
                        response = await asyncio.to_thread(
                            self.client.embeddings.create,
                            model=Config.EMBEDDING_MODEL,
                            input=batch_texts
                        )

                        # 将结果放入对应位置
                        for j, (data, idx) in enumerate(zip(response.data, batch_indices)):
                            embedding = np.array(data.embedding)
                            cache_key = hash(texts_to_process[i + j])
                            self._embedding_cache[cache_key] = embedding
                            results[idx] = embedding
                            # logger.debug(
                            #     f"成功生成向量: {texts_to_process[i + j][:50]}... (索引: {idx})")
                        break  # 成功后退出重试循环

                    except Exception as e:
                        logger.error(f"批处理生成向量失败: {e}")
                        if "429" in str(e):
                            retry_count += 1
                            # 初始延迟为0.5秒，最大延迟为5秒
                            delay = min(0.5 * (2 ** retry_count), 5)
                            logger.warning(
                                f"遇到请求限制，等待 {delay} 秒后重试 (重试 {retry_count}/{self.max_retries})")
                            await asyncio.sleep(delay)
                        else:
                            # 并行处理失败的批次
                            async def process_single_text(text: str, idx: int):
                                try:
                                    embedding = await self.generate_embedding(text)
                                    results[idx] = embedding
                                except Exception as e:
                                    # logger.error(
                                    #     f"单个文本向量生成失败: {text[:50]}... 错误: {e}")
                                    zero_vector = np.zeros(self.embedding_dim)
                                    cache_key = hash(text)
                                    self._embedding_cache[cache_key] = zero_vector
                                    results[idx] = zero_vector

                            # 创建并发任务
                            tasks = [
                                process_single_text(text, idx)
                                for text, idx in zip(batch_texts, batch_indices)
                            ]
                            # 并行执行所有任务
                            await asyncio.gather(*tasks)
                        break  # 退出重试循环

        # 管理缓存大小
        await self._manage_cache_size()

        # 统计失败的索引
        zero_vector = np.zeros(self.embedding_dim)
        failed_ids = [i for i, res in enumerate(results)
                      if res is None or np.array_equal(res, zero_vector)]

        # 记录处理结果和失败信息
        logger.info(f"向量生成完成，总数: {len(texts)}, 缓存命中: {len(texts) - len(texts_to_process)}, "
                    f"新处理: {len(texts_to_process)}, 失败: {len(failed_ids)}")
        # if failed_ids:
        #     logger.warning(
        #         f"失败的文本: {[texts[i][:50] + '...' for i in failed_ids]}")

        return results, failed_ids

    async def get_embedding(self, text: str) -> np.ndarray:
        """获取单个文本的向量表示"""
        # 检查缓存
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            # 如果没有缓存，生成新的embedding
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=Config.EMBEDDING_MODEL,
                input=text
            )
            embedding = np.array(response.data[0].embedding)

            # 存入缓存
            self._embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            # 失败时返回零向量并缓存
            zero_vector = np.zeros(self.embedding_dim)
            self._embedding_cache[cache_key] = zero_vector
            return zero_vector

    async def generate_embedding(self, text: str) -> np.ndarray:
        """生成单个文本的向量表示（带重试机制）"""
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        max_retries = 2
        retry_count = 0
        reduce_length = 0

        while retry_count < max_retries:
            try:
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    model=Config.EMBEDDING_MODEL,
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
                self._embedding_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"生成向量失败 (第 {retry_count+1} 次): {e}")
                retry_count += 1

                # 尝试减少文本长度后重新生成
                papers_content = await asyncio.get_running_loop().run_in_executor(
                    None, chunking, [text], reduce_length
                )
                reduce_length += 100 * retry_count
                text = '\n'.join(papers_content)

        # 超过重试次数，返回零向量并存入缓存
        # logger.error(f"生成向量最终失败，存储零向量到缓存: {text[:50]}...")
        zero_vector = np.zeros(self.embedding_dim)
        self._embedding_cache[cache_key] = zero_vector
        return zero_vector

    async def get_cos_scores(self, statement: str, papers_content: List[str]) -> np.ndarray:
        """计算余弦相似度分数"""
        doc_embeddings, failed_embeding_ids = await self.get_embeddings(papers_content)

        query_embedding = await self.get_embedding(statement)

        # 确保输入是 numpy 数组
        query_embedding = np.array(query_embedding)
        doc_embeddings = [np.array(de) for de in doc_embeddings]

        # 计算余弦相似度
        scores = np.array([self.cosine_similarity(query_embedding, de)
                          for de in doc_embeddings])

        # 记录相似度分数
        logger.info(f"计算得到的余弦相似度分数: {scores}")

        return scores

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算余弦相似度"""
        # 确保输入是非零向量
        if np.all(vec_a == 0) or np.all(vec_b == 0):
            return 0.0

        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # 避免除零错误
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def rerank_documents(self, query: str, documents: List[str], top_n=0, return_docs=True, return_raw=True):
        """异步调用 Rerank API"""
        url = "https://open.bigmodel.cn/api/paas/v4/rerank"
        headers = {
            "Authorization": f"Bearer {self.configs[Config.DEFAULT_MODEL]['RERANK_API_KEY']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "rerank",
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_docs,
            "return_raw_scores": return_raw
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    result = await response.json()
                    if "error" in result:
                        raise Exception(f"API Error: {result['error']}")
                    return result
            except Exception as e:
                logger.error(f"Rerank请求失败: {e}")
                return None

    async def _manage_cache_size(self):
        """管理缓存大小"""
        if len(self._embedding_cache) >= self.max_cache_size:
            # 随机删除10%的缓存项
            remove_keys = random.sample(
                list(self._embedding_cache.keys()),
                int(self.max_cache_size * 0.1)
            )
            for key in remove_keys:
                del self._embedding_cache[key]
            logger.info(f"清理缓存 {len(remove_keys)} 项")
