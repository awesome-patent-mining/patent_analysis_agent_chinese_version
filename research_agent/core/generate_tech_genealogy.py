import importlib.resources as pkg_resources
import json
from typing import List, Tuple, Dict
from typing import Optional
import logging
import os
import asyncio
from functools import wraps
from jinja2 import Environment
from json_repair import repair_json
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from pyaml_env import parse_config
from pathlib import Path
from research_agent.core.generate_hyde import HYDEGenerator
from research_agent.core.propose_question import QuestionProposer
from research_agent.core.combine_genealogies import Genealogy_combiner
from research_agent.core.resolve_HYDE_via_WebSearch import HYDEResolver_via_WebSearch
from research_agent.core.resolve_HYDE_via_PatentSearch import HYDEResolver_PatentSearch
import json_repair

def async_retry(retries=3, delay=1):
    """
    异步重试装饰器
    Args:
        retries (int): 最大重试次数
        delay (int): 重试间隔时间(秒)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        await asyncio.sleep(delay)
                    logging.warning(f"第 {attempt + 1} 次尝试失败: {str(e)}")
            raise last_exception
        return wrapper
    return decorator

class Tech_Gene_Generator:
    """A class that proposes research questions based on a given topic and context.

    This class uses an LLM to generate relevant research questions by processing
    a topic, optional context, and related papers.

    Attributes:
        llm: An instance of the LLM class for generating completions
        iteration: Current iteration number for question generation
        prompt_template: Jinja template for generating prompts
    """

    def __init__(self):
        """Initialize the QuestionProposer.

        Args:
            iteration (int, optional): Iteration number for question generation. Defaults to 0.
        """
        absolute_path = os.path.abspath(Config.YAML_CONFIG)
        configs = parse_config(absolute_path)
        self.language = Config.LANGUAGE
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])
        self.question_proposer = QuestionProposer()
        self.hydeResolver_webSearch = HYDEResolver_via_WebSearch()
        self.hydeResolver_patentSearch = HYDEResolver_PatentSearch()
        self.HYDE_generator = HYDEGenerator()
        self.genealogy_combiner = Genealogy_combiner()
        try:
            base_path = Path(__file__).parent / "prompts"

            patentSearch_based_genealogy_write_prompt_file = base_path / "write_genealogy_based_on_patentsearch.jinja"
            with open(patentSearch_based_genealogy_write_prompt_file, "r", encoding="utf-8") as f:
                self.patentSearch_based_genealogy_write_prompt_template = Environment().from_string(f.read())

            webSearch_based_genealogy_write_prompt_file = base_path / "write_genealogy_based_on_websearch.jinja"
            with open(webSearch_based_genealogy_write_prompt_file, "r", encoding="utf-8") as f:
                self.webSearch_based_genealogy_write_prompt_template = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")
    def set_language(self, language):
        self.language = language
    def get_language(self,):
        return self.language

    def _prepare_prompts_for_search_results(
            self, topic: str, context: List[str]
    ) -> List[dict]:
        system_prompt = self.patentSearch_based_genealogy_write_prompt_template.render(role="system",language=self.language)
        user_prompt = self.patentSearch_based_genealogy_write_prompt_template.render(
            role="user",
            topic=topic,
            language=self.language,
            context=context
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _process_results(self, results: List[dict]) -> List[str]:
        """Process question resolution results into context and related papers.

        Args:
            results: List of resolution results containing context and related papers

        Returns:
            Tuple containing processed context string and related papers string
        """
        try:
            context = []

            for result in results:
                if not isinstance(result, dict) or 'context' not in result:
                    #logger.warning(f"Malformed result detected: {result}")
                    continue
                context.extend(result["context"])

            return context
        except Exception as e:
            #logger.error(f"Error processing results: {str(e)}")
            raise



    async def generate_tech_genealogy(self, topic:str,genealogy_type:int) -> List[dict]:
        """write conclusion of the survey.
        Args:
            topic (str): The topic of patent analysis report.
            genealogy_type (int): The type of tech genealogy, 1 means based on webpage
                                                              2 means based on patent
                                                              3 means hybrid type of tech genealogy

        Returns:
            dict: tech spectrum.

        Raises:
            ValueError: If question is empty or not a string
            RuntimeError: If LLM completion fails
        """
        try:
            # 使用重试装饰器包装关键操作
            @async_retry(retries=3, delay=1)
            async def generate_questions(topic):
                return await self.question_proposer.propose_question(topic)

            @async_retry(retries=3, delay=1)
            async def resolve_HYDEs_generate_genealogy_via_WebSearch(hydes):
                web_search_results = await self.hydeResolver_webSearch.resolve_HYDEs(hydes)
                result_genealogy = await self.generate_genealogy(topic, web_search_results)
                return result_genealogy
            @async_retry(retries=3, delay=1)
            async def resolve_HYDEs_generate_genealogy_via_PatentSearch(hydes):
                patent_search_results = await self.hydeResolver_patentSearch.resolve_HYDEs(hydes)
                result_genealogy = await self.generate_genealogy(topic, patent_search_results)
                return result_genealogy

            @async_retry(retries=3, delay=1)
            async def generate_HYDEs(questions):
                hydes = await self.HYDE_generator.generate_HYDEs(questions)
                return hydes

            # 先生成相关问题
            genealogy = None
            question_list = await generate_questions(topic=topic)
            # 再准备伪答案
            hydes = await generate_HYDEs(question_list)
            # 拿着伪答案到web_search上检索
            hydes = [d['answer'] if isinstance(d, dict) and 'answer' in d else d for d in hydes]

            if genealogy_type==1:
                genealogy = await resolve_HYDEs_generate_genealogy_via_WebSearch(hydes)
            elif genealogy_type==2:
                genealogy = await resolve_HYDEs_generate_genealogy_via_PatentSearch(hydes)
            elif genealogy_type==3:
                tasks = [resolve_HYDEs_generate_genealogy_via_WebSearch(hydes),
                         resolve_HYDEs_generate_genealogy_via_PatentSearch(hydes)]
                genealogy_based_on_web, genealogy_based_on_patent = await asyncio.gather(*tasks,
                                                                                             return_exceptions=True)
                # 将技术谱系合并，形成最终的技术谱系
                genealogy = await self.genealogy_combiner.combine_genealogies(topic=topic,genealogy_webpage=genealogy_based_on_web,genealogy_patent=genealogy_based_on_patent)


            # 返回技术谱系
            return genealogy
        except Exception as e:
            #logger.error(f"Error in draft generation: {str(e)}")
            raise

    async def generate_genealogy(self, topic: str, search_results: List[dict], max_retries: int = 3) -> List[dict]:
        """
        根据给定的语句调用LLM生成相关的引用。

        :param topic: 研究的主题或文本内容
        :param search_results: 上下文信息
        :param max_retries: 最大重试次数
        :return: 模型返回的技术图谱
        """

        prompt_messages_for_search_result = self._prepare_prompts_for_search_results(topic=topic,
                                                                             context=self._process_results(
                                                                                 search_results))

        for attempt in range(max_retries):
            try:
                response = await self.llm.completion(prompt_messages_for_search_result)
                response_data = json_repair.loads(response)
                return response_data
            except Exception as e:
                #logger.error(f"调用LLM模型时出错：{e}，尝试次数：{attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # 可选：在重试之间等待一段时间
                else:
                    #logger.error("达到最大重试次数，操作失败。")
                    return []


if __name__ == "__main__":
    async def main():
        # Initialize token manager
        # Initialize with your actual API key and token manager
        topic = "第四代半导体材料"
        tech_genealogy_generator = Tech_Gene_Generator()
        tech_genealogy = await tech_genealogy_generator.generate_tech_genealogy(topic=topic,genealogy_type=1)
        print(tech_genealogy)

    asyncio.run(main())