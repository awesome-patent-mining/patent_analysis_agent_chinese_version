import importlib.resources as pkg_resources
import json
import asyncio
from pathlib import Path
from jinja2 import Environment
import json_repair
from research_agent.core.config import Config
from research_agent.core.web_search_bocha import WebSearch
from pyaml_env import parse_config
from research_agent.core.query import Query


class HYDEResolver_via_WebSearch:
    """A class to resolve research questions using LLM and document retrieval.

    This class combines document retrieval and LLM-based question answering to provide
    responses based on research paper content.

    Args:
        top_k (int, optional): Number of top documents to retrieve. Defaults to 5.

    Attributes:
        query (Query): Document retrieval interface
        top_k (int): Number of documents to retrieve
        llm (LLM): Language model interface
        prompt_template: Jinja template for prompt generation
    """

    def __init__(self, top_k: int = 5):
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.top_k = top_k

        configs = parse_config(Config.YAML_CONFIG)
        self.webSearch = WebSearch()  # 加上括号变成实例化调用
        try:
            # 修改文件路径的获取方式
            base_path = Path(__file__).parent / "prompts"
            answer_question_prompt_file = base_path / "answer_question.jinja"
            with open(answer_question_prompt_file, "r") as f:
                self.answer_question_prompt_template = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

    async def resolve_HYDEs(self, questions: list[str]) -> list:
        """
        批量异步调用 query_by_content 检索专利
        """
        if not isinstance(questions, list) or not questions:
            raise ValueError("Questions must be a non-empty list of strings")
        # 批量异步检索
        results = await self.webSearch.batch_async_queries(questions)
        return results


