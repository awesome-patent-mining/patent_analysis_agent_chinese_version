import logging
from pathlib import Path
from typing import List, Optional
import asyncio
from jinja2 import Environment
from research_agent.core.general_llm import LLM
from research_agent.core.config import Config
from pyaml_env import parse_config
import json_repair

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Genealogy_combiner:
    """
    通过调用LLM模型，根据给定的研究主题和文本内容生成相关的语句引用。
    """

    def __init__(self, base_path: Optional[str] = None, max_concurrent: int = 15):
        """
        初始化 GenStatementHyde 类，加载 LLM 模型配置和提示模板，并设置并发控制信号量。

        :param base_path: 提示模板所在的基础路径，默认为当前文件所在目录下的 "prompts" 文件夹
        :param max_concurrent: 最大并发数
        """
        # 加载配置文件并获取默认模型配置
        configs = parse_config(Config.YAML_CONFIG)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])
        self.language = Config.LANGUAGE
        self.topic = ""
        # 确定模板文件路径
        if base_path is None:
            base_path = Path(__file__).parent / "prompts"
        else:
            base_path = Path(base_path)

        prompt_file = base_path / "combine_genealogies.jinja"
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                template_content = f.read()
        except Exception as e:
            logger.error(f"加载提示模板文件失败：{prompt_file}，错误信息：{e}")
            raise

        # 使用 Jinja2 加载模板
        self.prompt_template = Environment().from_string(template_content)

    async def combine_genealogies(self, topic:str,genealogy_webpage: List[dict], genealogy_patent: List[dict],max_retries:int=3) -> List[dict]:
        """
        根据给定的语句调用LLM生成相关的引用。
        :param topic: 研究主题
        :param genealogy_webpage: 第一个技术图谱的内容
        :param genealogy_patent: 第二个技术图谱的内容
        :param max_retries: 最大重试次数
        :return: 合并后的技术图谱内容
        """
        self.topic = topic
        prompt_messages = self._prepare_prompt_messages(genealogy_webpage, genealogy_patent)

        for attempt in range(max_retries):
            try:
                response = await self.llm.completion(prompt_messages)
                response_data = json_repair.loads(response)
                print(response_data)
                return response_data
            except Exception as e:
                logger.error(f"调用LLM模型时出错：{e}，尝试次数：{attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # 可选：在重试之间等待一段时间
                else:
                    logger.error("达到最大重试次数，操作失败。")
                    return []

    def _prepare_prompt_messages(self, genealogy_webpage: List[dict], genealogy_patent: List[dict]) -> List[dict]:
        """
        准备生成查找语句引用所需的提示消息。

        :param genealogy_1: 第一个技术图谱的内容
        :param genealogy_2: 第二个技术图谱的内容
        :return: 包含系统和用户提示信息的字典列表
        """
        system_prompt = self.prompt_template.render(
            role="system",language=self.language)
        user_prompt = self.prompt_template.render(
            role="user", genealogy_webpage=genealogy_webpage, genealogy_patent=genealogy_patent,language=self.language,topic=self.topic)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]