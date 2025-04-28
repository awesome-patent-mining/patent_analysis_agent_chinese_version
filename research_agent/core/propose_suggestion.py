import importlib.resources as pkg_resources
import json
from pathlib import Path
from typing import Optional

from jinja2 import Environment
from json_repair import repair_json
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from pyaml_env import parse_config

class SuggestionProposer:
    """A class that proposes suggestions for paper improvement based on a given topic, context and an old version of draft.

    This class uses an LLM to generate relevant research questions by processing
    a topic, optional context, related papers, and an old version of draft.

    Attributes:
        llm: An instance of the LLM class for generating completions
        iteration: Current iteration number for question generation
        prompt_template: Jinja template for generating prompts
    """

    def __init__(self, iteration: int = 0):
        """Initialize the SuggestionProposer.

        Args:
            iteration (int, optional): Iteration number for suggestion generation. Defaults to 0.
        """

        configs = parse_config(Config.YAML_CONFIG)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])

        self.iteration = iteration

        try:
            # 修改文件路径的获取方式
            base_path = Path(__file__).parent / "prompts"
            propose_suggestion_prompt_file = base_path / "propose_suggestion.jinja"
            with open(propose_suggestion_prompt_file, "r", encoding='utf-8') as f:
                self.propose_suggestion_prompt_template = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

    def _prepare_prompts(
        self,
        topic: str,
        draft: str = None,
        context: str = None,
        related_papers: str = None,
    ) -> list[dict[str,str]]:
        """Prepare system and user prompts for the LLM.

        Args:
            topic (str): The main research topic
            context (Optional[str], optional): Additional context. Defaults to None.
            related_papers (Optional[list[str]], optional): List of related papers. Defaults to None.

        Returns:
            list[dict[str, str]]: List of prompt messages in the format required by the LLM
        """
        system_prompt = self.propose_suggestion_prompt_template.render(
            role="system", iteration=self.iteration
        )
        user_prompt = self.propose_suggestion_prompt_template.render(
            role="user",
            topic=topic,
            draft=draft,
            context=context,
            related_papers=related_papers,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def propose_suggestion(
        self,
        topic: str,
        draft: str,
        context: str = None,
        related_papers: str = None,
    ) -> list[str]:
        """Generate research questions based on the provided inputs.

        Args:
            draft:
            topic (str): The main research topic
            draft (str): The old version of the draft paper
            context (Optional[str], optional): Additional context. Defaults to None.
            related_papers (Optional[list[str]], optional): List of related papers. Defaults to None.

        Returns:
            list[str]: List of generated research questions

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON
            json.JSONDecodeError: If the response JSON is malformed
        """
        prompt_messages = self._prepare_prompts(topic, draft, context, related_papers)
        response = await self.llm.completion(prompt_messages)
        try:
            response = repair_json(response)
            parsed_response = json.loads(response)
            if not isinstance(parsed_response, dict) or "suggestions" not in parsed_response:
                raise ValueError("Invalid response format from LLM")
            return parsed_response["suggestions"]
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse LLM response as JSON: {str(e)}", e.doc, e.pos
            )
