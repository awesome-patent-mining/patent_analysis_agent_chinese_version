import asyncio
from typing import Dict, List, Optional
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from research_agent.core.query import Query
from pyaml_env import parse_config
from jinja2 import Environment
import json_repair
import json
from json_repair import repair_json
from pathlib import Path
from AI_agent.utils.query import Query


class SummaryExtractor:
    """A class to extract and generate summaries from paper chunks using LLM.

    Attributes:
        llm (LLM): Language model instance for generating summaries
        query (Query): Query instance for retrieving paper chunks
    """

    def __init__(self, ):
        configs = parse_config(Config.YAML_CONFIG)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])
        self.query = Query()
        try:
            # 修改文件路径的获取方式
            base_path = Path(__file__).parent / "prompts"
            chunk_summary_prompt_file = base_path / "chunk_summary.jinja"
            with open(chunk_summary_prompt_file, "r") as f:
                self.chunk_summary_prompt_template = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

    async def summarize_chunks(self, chunks: List[str],topic,language) -> List[str]:
        """Summarize multiple chunks concurrently.

        Args:
            chunks (List[Dict]): List of paper chunks to summarize
            topic (str): the topic of the survey
            language (str): the language of the chunks

        Returns:
            List[str]: List of generated summaries for each chunk

        Raises:
            Exception: If chunk summarization fails
        """
        try:
            tasks = [self.summarize_chunk(topic,chunk,language) for chunk in chunks]
            return await asyncio.gather(*tasks)
        except Exception as e:
            raise Exception(f"Failed to summarize chunks: {str(e)}")

    async def summarize_chunk(self, topic:str,chunk: str,language:str) -> str:
        """Summarize a single chunk of text.

        Args:
            chunk (str): containing chunk text
            language(str): language of the chunk
            topic（str）:the topic of the survey

        Returns:
            str: Generated summary for the chunk

        Raises:
            Exception: If chunk summarization fails
        """
        prompt_messages = self._prepare_prompts(chunk,topic,language)

        try:
            response = await self.llm.completion(prompt_messages)
            response = json_repair.loads(response)
            result =  response["summary"]
            return result
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Failed to parse LLM response: {str(e)}")
    def _prepare_prompts(self, chunk: str,topic:str, language: str) -> list[dict]:
        """Prepare system and user prompts for LLM completion.

        Args:
            question (str): The research question
            context (str): Retrieved context for answering the question

        Returns:
            list[dict]: List of prompt messages with roles and content
        """
        system_prompt = self.chunk_summary_prompt_template.render(role="system")
        user_prompt = self.chunk_summary_prompt_template.render(
            role="user", topic = topic, chunk=chunk, language=language
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


if __name__ == "__main__":
    llm = LLM()
    summary_extractor = SummaryExtractor(llm)
    summary = asyncio.run(
        summary_extractor.summarize_paper("651b7dbc3fda6d7f06304579", None)
    )
    print(summary)
