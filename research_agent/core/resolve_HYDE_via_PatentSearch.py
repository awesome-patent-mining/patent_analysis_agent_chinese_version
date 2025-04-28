import importlib.resources as pkg_resources
import json
import asyncio
from pathlib import Path

from jinja2 import Environment
import json_repair
from research_agent.core.config import Config
from research_agent.core.web_search_bocha import WebSearch
from pyaml_env import parse_config
from research_agent.core.query_1 import Query


class HYDEResolver_PatentSearch:
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
        self.patentSearch = Query()

        try:
            # 修改文件路径的获取方式
            base_path = Path(__file__).parent / "prompts"
            answer_question_prompt_file = base_path / "answer_question.jinja"
            with open(answer_question_prompt_file, "r") as f:
                self.answer_question_prompt_template = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")
    async def resolve_HYDE(self, question: str) -> dict:
        """Resolve a single research question.

        Args:
            question (str): The research question to be answered.

        Returns:
            dict: Contains 'answer' and 'related_papers' keys with corresponding values.

        Raises:
            ValueError: If question is empty or not a string
            RuntimeError: If LLM completion fails
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        retrieved_chunks = await self.patentSearch.query_13_columns_by_content(question, limit=self.top_k)
        if not retrieved_chunks:
            return {"context": [], "related_papers": []}

        context = [chunk["abstract"] for chunk in retrieved_chunks]

        related_papers = [chunk["title"] for chunk in retrieved_chunks]


        result = {
            "related_papers": related_papers,
            "context":context
        }
        return result
    async def resolve_HYDEs(self, questions: list[str]) -> list[dict]:
        """Resolve multiple research questions concurrently.

        Args:
            questions (list[str]): List of research questions to be answered.

        Returns:
            list[dict]: List of resolution results for each question.

        Raises:
            ValueError: If questions is not a list or contains invalid questions
        """
        if not isinstance(questions, list) or not questions:
            raise ValueError("Questions must be a non-empty list of strings")

        tasks = [self.resolve_HYDE(question) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any failed results
        valid_results = [
            result for result in results
            if not isinstance(result, Exception)
        ]
        return valid_results

