import importlib.resources as pkg_resources
import json
from typing import Optional
import os
import asyncio
import requests
import time
from jinja2 import Environment
from json_repair import repair_json
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from pyaml_env import parse_config
from pathlib import Path

class TokenManager:
    def __init__(self):
        self.token = None
        self.last_refresh_time = 0
        self.expires_in = 1800  # 30分钟（单位：秒）

    def get_token(self):
        # 如果 token 不存在或已过期，则刷新
        if not self.token or (time.time() - self.last_refresh_time) >= self.expires_in:
            self.refresh_token()
        return self.token

    def refresh_token(self):
        url = "https://9EmfQHAac0MyPmtx0gXseNZCkGfGf7GKFnv2NGPMyTshhKQy:BENnYXk3O15ExmcGG0opU10dWZWAW1KT01oqzXEcGNX7mHyoqgzxLqK6ry7q996d@connect.zhihuiya.com/oauth/token"
        payload = "grant_type=client_credentials"
        headers = {"content-type": "application/x-www-form-urlencoded"}

        response = requests.post(url, data=payload, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        if response.status_code == 200:
            data = response.json()
            self.token = data["data"].get("token")
            self.last_refresh_time = time.time()
            self.expires_in = int(data["data"].get("expires_in", 1800))  # 转换为整数类型
        else:
            raise Exception(f"Failed to refresh token: {response.text}")

class QuestionProposer:
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
        absolute_path = os.path.abspath(Config.YAML_CONFIG)
        configs = parse_config(absolute_path)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])

        self.iteration = iteration
        self.propose_question_prompt_template = ""
        try:
            # 修改文件路径的获取方式
            base_path = Path(__file__).parent / "prompts"
            propose_question_prompt_file = base_path / f"create_questions.jinja"
            with open(propose_question_prompt_file, "r",encoding="utf-8") as f:
                self.propose_question_prompt_template= Environment().from_string(f.read())

            # prompt_file = pkg_resources.files("research_agent.core.prompts").joinpath(
            #     "create_questions.jinja"
            # )
            # prompt_env = Environment()
            # with open(prompt_file, "r", encoding='utf-8') as f:
            #     self.prompt_template = prompt_env.from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

    def _prepare_prompts(
        self,
        topic: str,
        context: Optional[str] = None,
        related_papers: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        """Prepare system and user prompts for the LLM.

        Args:
            topic (str): The main research topic
            context (Optional[str], optional): Additional context. Defaults to None.
            related_papers (Optional[list[str]], optional): List of related papers. Defaults to None.

        Returns:
            list[dict[str, str]]: List of prompt messages in the format required by the LLM
        """
        system_prompt = self.propose_question_prompt_template.render(
            role="system", iteration=self.iteration
        )
        user_prompt = self.propose_question_prompt_template.render(
            role="user",
            iteration=self.iteration,
            topic=topic,
            context=context,
            related_papers=related_papers,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def propose_question(
        self,
        topic: str,
        context: Optional[str] = None,
        related_papers: Optional[list[str]] = None,
    ) -> list[str]:
        """Generate research questions based on the provided inputs.

        Args:
            topic (str): The main research topic
            context (Optional[str], optional): Additional context. Defaults to None.
            related_papers (Optional[list[str]], optional): List of related papers. Defaults to None.

        Returns:
            list[str]: List of generated research questions

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON
            json.JSONDecodeError: If the response JSON is malformed
        """
        prompt_messages = self._prepare_prompts(topic,context, related_papers)
        response = await self.llm.completion(prompt_messages)
        try:
            response = repair_json(response)
            parsed_response = json.loads(response)
            if not isinstance(parsed_response, dict) or "questions" not in parsed_response:
                raise ValueError("Invalid response format from LLM")
            return parsed_response["questions"]
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse LLM response as JSON: {str(e)}", e.doc, e.pos
            )

if __name__ == "__main__":
    async def main():
        # Initialize token manager
        token_manager = TokenManager()
        # Initialize with your actual API key and token manager
        topic = "第四代半导体材料"
        question_proposer = QuestionProposer()
        question_list = await question_proposer.propose_question(topic=topic)
        print(question_list)

        # Example queries
        # search_text = "The invention discloses an automobile front-view based wireless video transmission system and method. The system comprises a front-view camera, a wireless video transmitting module, a wireless video receiving module, a display screen, a display triggering device, a first controller, a wireless command transmitting module, a wireless command receiving module, a second controller and an automobile starting detecting module, wherein the display screen is connected with the wireless video receiving module; the front-view camera is connected with the wireless video transmitting module; the wireless video transmitting module is wirelessly connected with the wireless video receiving module and wirelessly transmits a video shot by the front-view camera; and the wireless video receiving module receives and sends the video and displays the video on the display screen, so that the mounting time of the front-view camera is shortened greatly, no damage can be caused to an original automobile, the front-view camera can be mounted without a threading manner, and great convenience is brought to the owner of the automobile."
        # patent_data = await query.query_by_content(search_text)
        # insert_patent_to_db(patent_data)
        #
        # patent_id = "b053642f-3108-4ea9-b629-420b0ab959e3"
        # patent_data = await query.query_by_id(patent_id)
        # insert_patent_to_db(patent_data)
        #

        #patent_data = await query.query_by_title(title)
        #insert_patent_to_db(patent_data)
        #
        # keyword = "wireless"
        # patent_data = await query.query_by_keyword(keyword)
        # insert_patent_to_db(patent_data)
        #
        # patent_number = "CN106185468A"
        # patent_data = await query.query_by_patent_number(patent_number)
        # insert_patent_to_db(patent_data)
        #
        # assignee = "Apple, Inc."
        # patent_data = await query.query_by_assignee(assignee)
        # insert_patent_to_db(patent_data)
        #
        # application = "Samsung Electronics Co., Ltd."
        # patent_data = await query.query_by_application(application)
        # insert_patent_to_db(patent_data)


    asyncio.run(main())