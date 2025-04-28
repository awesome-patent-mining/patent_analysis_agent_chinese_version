import numpy as np
from typing import List, Tuple
import os
import requests
from pyaml_env import parse_config
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
import asyncio
from typing import List, Dict


class RerankByLLM:
    """
    处理文档重排序的类，提供大模型的重排序功能
    """

    def __init__(self, threshold: float = 0.3):
        """
        初始化重排序处理器

        参数:
            threshold (float): 相似度过滤阈值，默认0.3
        """
        absolute_path = os.path.abspath(Config.YAML_CONFIG)
        self.configs = parse_config(absolute_path)
        self.llm = LLM(config=self.configs[Config.DEFAULT_MODEL])
        self.threshold = threshold
        self.batch_size = Config.RERANK_BATCH_SIZE

    async def rerank_batched_documents(self, query, documents, top_n, return_docs=True, return_raw=True):
        """重新排序文档"""
        url = "https://open.bigmodel.cn/api/paas/v4/rerank"
        headers = {
            "Authorization": f"Bearer {self.configs[Config.DEFAULT_MODEL]['API_KEY']}",
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

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 检查HTTP错误

            result = response.json()

            # 检查API返回的错误
            if "error" in result:
                raise Exception(f"API Error: {result['error']}")

            return result

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    async def async_rerank_batch(self,query: str,
                                 documents: List[str],
                                 ) -> List[Dict]:
        """异步分块调用rerank接口的核心函数"""

        # 分块处理文档集
        batches = [documents[i:i + self.batch_size]
                   for i in range(0, len(documents), self.batch_size)]

        # 创建异步任务集合
        tasks = []
        for batch in batches:
            task = self.rerank_batched_documents(query, batch, len(batch), return_docs=True, return_raw=True)
            tasks.append(task)

        # 并行执行异步请求
        rerank_results = await asyncio.gather(*tasks)

        # 解析并聚合结果
        all_ranked = []
        for idx,result_i in enumerate(rerank_results):
            for response in result_i['results']:
                response['index'] = idx * self.batch_size + response['index']
                all_ranked.append(response)
        return all_ranked


    # 同步调用入口函数
    async def rerank_documents(self,query: str,
                         documents: List[str],
                         ) -> List[Dict]:
        """文档重排序主函数"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.async_rerank_batch(query, documents)
        )


if __name__ == "__main__":
    query = "如何学习人工智能"
    documents = [
        "机器学习的基本概念介绍",
        "Python编程入门教程",
        "深度学习的数学基础",
        "人工智能发展历史概述",
        "如何准备AI面试"
    ]
    glm4_rerank = RerankByLLM()
    result = glm4_rerank.rerank_documents(query, documents, top_k=len(documents))
    if result:
        print("排序结果：")
        for item in result.get('results', []):
            print(f"文档 {item['index']} - 得分 {item['score']:.4f}: {documents[item['index']]}")