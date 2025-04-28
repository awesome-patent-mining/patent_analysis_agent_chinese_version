import importlib.resources as pkg_resources
import json
import asyncio
from pathlib import Path

from jinja2 import Environment
import json_repair
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from pyaml_env import parse_config
from research_agent.core.query import Query


class HYDEGenerator:
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

    def __init__(self, ):
        configs = parse_config(Config.YAML_CONFIG)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])
        self.language = Config.LANGUAGE

        try:
            # 修改文件路径的获取方式
            base_path = Path(__file__).parent / "prompts"
            answer_question_prompt_file = base_path / "generate_HYDE.jinja"
            with open(answer_question_prompt_file, "r") as f:
                self.answer_question_prompt_template = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")
    def set_language(self, language: str):
        self.language = language
    def get_language(self) -> str:
        return self.language
    async def generate_HYDE(self, question: str) -> dict:
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

        prompt_messages = self._prepare_prompts(question)

        try:
            response = await self.llm.completion(prompt_messages)
            response = json_repair.loads(response)
            result = {
                "answer": response["answer"]
            }
            return result
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Failed to parse LLM response: {str(e)}")

    async def generate_HYDEs(self, questions: list[str]) -> list[dict]:
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

        tasks = [self.generate_HYDE(question) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any failed results
        valid_results = [
            result for result in results 
            if not isinstance(result, Exception)
        ]
        return valid_results

    def _prepare_prompts(self, question: str) -> list[dict]:
        """Prepare system and user prompts for LLM completion.

        Args:
            question (str): The research question

        Returns:
            list[dict]: List of prompt messages with roles and content
        """
        system_prompt = self.answer_question_prompt_template.render(role="system",language=self.language)
        user_prompt = self.answer_question_prompt_template.render(
            role="user", question=question, language=self.language
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
