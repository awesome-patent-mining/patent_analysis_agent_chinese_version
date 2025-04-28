import numpy as np
from typing import List, Tuple
import os
import requests
from pyaml_env import parse_config
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
import asyncio
from typing import List, Dict
from zhipuai import ZhipuAI


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
        self.client = ZhipuAI(api_key=Config.WEB_SEARCH_API_KEY)

    async def query(self, query:str, top_n: int = None ):
        """
        在互联网上执行搜索查询，返回结果列表

        :param query: 查询语句列表
        :param top_n: 返回排名前top_n的结果
        :return: 按顺序排列的结果列表
        """
        if top_n is None:
            top_n = self.web_search_top_n
        #query = self.truncate_long_text(query)
        print(query)
        #print(len(query.split()))
        response = self.client.web_search.web_search(
            search_engine="search-std",
            search_query=query
        )
        search_result = response.search_result
        result_num = min(len(search_result),top_n)
        return search_result[:result_num]

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
                for web_i in result:
                    context_results.append(web_i.content)
                    related_paper_results.append(web_i.title)
                result = {
                    'contexts': context_results,
                    'related_papers': related_paper_results
                }
                results.append(result)

        return results

    @staticmethod
    def truncate_long_text(text: str) -> str:
        """截断超过75个单词的字符串，保留前75个单词"""
        words = text.split()  # 按空格分割单词（自动处理连续空格）
        return " ".join(words[:75]) if len(words) > 75 else text



if __name__ == "__main__":

    import nest_asyncio

    nest_asyncio.apply()

    async def main():
        queries= [
            "The fundamental principles of AI include problem-solving, learning, and adaptability. Design concepts involve modularity, generalization, and efficiency","本田公司"
        ]
        web_search = WebSearch()
        results = await web_search.batch_async_queries(queries,5)
        if results:
            for result in results:
                print(result)
    asyncio.run(main())
