import importlib.resources as pkg_resources
import json
from pathlib import Path
from typing import Optional

from jinja2 import Environment
from json_repair import repair_json
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from pyaml_env import parse_config

class OutlineWriter:
    """A class that proposes research questions based on a given topic and context.

    This class uses an LLM to generate relevant research questions by processing
    a topic, optional context, and related papers.

    Attributes:
        llm: An instance of the LLM class for generating completions
        iteration: Current iteration number for question generation
        prompt_template: Jinja template for generating prompts
    """

    def __init__(self, iteration: int = 0):
        """Initialize the QuestionProposer.

        Args:
            iteration (int, optional): Iteration number for question generation. Defaults to 0.
        """

        configs = parse_config(Config.YAML_CONFIG)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])
        self.language = None
        self.iteration = iteration
        self.prompt_template = []
        try:
            base_path = Path(__file__).parent / "prompts"
            for i in range(4):
                write_outline_prompt_file = base_path / f"write_outline_type_{i+1}.jinja"
                with open(write_outline_prompt_file, "r",encoding="utf-8") as f:
                    self.prompt_template.append(Environment().from_string(f.read()))
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

    def _prepare_prompts(
        self,
        topic: str,
        survey_type: int,
        context: Optional[str] = None,
        related_papers: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Prepare system and user prompts for the LLM.

        Args:
            topic (str): The main research topic
            context (Optional[str], optional): Additional context. Defaults to None.
            related_papers (Optional[list[str]], optional): List of related papers. Defaults to None.

        Returns:
            list[dict[str, str]]: List of prompt messages in the format required by the LLM
        """
        system_prompt = self.prompt_template[survey_type-1].render(
            role="system", language = self.language,iteration=self.iteration
        )
        user_prompt = self.prompt_template[survey_type-1].render(
            role="user",
            iteration=self.iteration,
            topic=topic,
            language=self.language,
            context=context,
            related_papers=related_papers
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    def set_language(self, language: str):
        self.language = language
    def get_language(self):
        return self.language
    async def write_outline(
        self,
        topic: str,
        survey_type: int,
        context: Optional[str] = None,
        related_papers: Optional[str] = None,
    ) -> str:
        """Generate research questions based on the provided inputs.

        Args:
            topic (str): The main research topic
            survey_type (int): The type of survey to generate questions for
            context (Optional[str], optional): Additional context. Defaults to None.
            related_papers (Optional[list[str]], optional): List of related papers. Defaults to None.

        Returns:
            list[str]: List of generated research questions

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON
            json.JSONDecodeError: If the response JSON is malformed
        """
        prompt_messages = self._prepare_prompts(topic,survey_type, context, related_papers)
        response = await self.llm.completion(prompt_messages)
        return response
