import numpy as np
from typing import List, Tuple
import os
import requests
from pyaml_env import parse_config
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
import asyncio
from typing import List, Dict


class WebSearch:
    """
    处理文档重排序的类，提供大模型的重排序功能
    """

    def __init__(self, ):
        """
        初始化重排序处理器

        参数:
            threshold (float): 相似度过滤阈值，默认0.3
        """
        self.web_search_top_n = Config.WEB_SEARCH_TOP_N

    async def query(self, query:str, top_n: int = None ):
        """
        在互联网上执行搜索查询，返回结果列表

        :param query: 查询语句列表
        :param top_n: 返回排名前top_n的结果
        :return: 按顺序排列的结果列表
        """
        if top_n is None:
            top_n = self.web_search_top_n
        url = "https://api.bochaai.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {Config.WEB_SEARCH_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "summary": True,
            "count": top_n,
            "page": 1
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

    async def batch_async_queries(
            self,
            queries: List[str],
            batch_size: int = 10
    ) -> List[dict]:
        """
        批量异步执行互联网搜索查询

        :param queries: 查询语句列表
        :param batch_size: 每批次并发数量，默认是10
        :return: 按顺序排列的结果列表
        """

        results = []
        # 分批处理查询请求
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            # 并发执行当前批次
            batch_results = await asyncio.gather(
                *(self.query(q,self.web_search_top_n) for q in batch)
            )
            for result in batch_results:
                context_results = []
                related_paper_results = []
                for content in result['data']['webPages']['value']:
                    context_results.append(content["summary"])
                    related_paper_results.append(content["name"])
                result = {
                    'contexts': context_results,
                    'related_papers': related_paper_results
                }
                results.append(result)

        return results



if __name__ == "__main__":

    import nest_asyncio

    nest_asyncio.apply()

    async def main():
        queries= [
            "均胜汽车安全系统公司","本田公司"
        ]
        web_search = WebSearch()
        results = await web_search.batch_async_queries(queries,5)
        if results:
            for result in results:
                print(result)
    asyncio.run(main())
