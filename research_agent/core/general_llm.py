import logging

import httpx
# import litellm
# from litellm import max_tokens
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 1600
# litellm.set_verbose = True


def assemble_headers_and_payload(config, messages, json_schema=None, tools=None):
    headers = {
        "Authorization": f"Bearer {config['API_KEY']}",
        "Content-Type": "application/json",
    }
    headers.update(config.get("additional_headers", {}))

    payload = {
        "model": config["model"],
        "messages": messages,
        "max_tokens": config["max_tokens"],
    }
    if tools:
        payload["tools"] = tools
    payload.update(config.get("additional_parameters", {}))
    if json_schema:
        payload.update(
            {
                "response_format": {
                    "type": config.get("json_schema_type", "json_object"),
                    "schema": json_schema,
                },
            }
        )

    return headers, payload


async def chat_completion_async(
    httpx_client, config, messages, json_schema=None, tools=None, timeout=120.0
):
    headers, payload = assemble_headers_and_payload(
        config, messages, json_schema=json_schema, tools=tools
    )

    llm_response = None
    try:
        llm_response = await httpx_client.post(
            config["url"], json=payload, headers=headers, timeout=timeout
        )
    except Exception as e:
        logger.error(e)
    return llm_response


class LLM(BaseModel):
    """
    Base class for LLM clients that handles communication with a single LLM provider.

    Attributes:
        config (dict): Configuration dictionary containing LLM provider settings
        timeout (float): Maximum time in seconds to wait for LLM response
    """

    config: dict
    timeout: float = Field(default=DEFAULT_TIMEOUT)
    # max_tokens: int = Field(default=MAX_TOKENS)

    async def completion(self, messages: list[dict], tools=None, **kwargs):
        """
        Call the LLM with the given messages for chat completion.

        Args:
            messages (list[dict]): List of message dictionaries containing the conversation history.
                Each message should have 'role' and 'content' keys.

        Returns:
            LLMOutput: Object containing the model's response text

        Raises:
            httpx.TimeoutException: If the request times out
            httpx.RequestError: If there's a network-related error
            ValueError: If the response format is invalid
            Exception: For other LLM-related errors
        """
        logger.info("Starting LLM chat completion request with single provider")
        try:
            async with httpx.AsyncClient() as _httpx_client:
                logger.debug(f"Sending request with {len(messages)} messages")
                response = await chat_completion_async(
                    _httpx_client,
                    self.config,
                    messages,
                    tools=tools,
                    timeout=self.timeout
                )

            if not response:
                logger.error("Empty response received from LLM")
                raise Exception("Empty response received from LLM")

            if response.status_code != 200:
                logger.error(
                    f"LLM request failed with status code: {response.status_code}"
                )
                raise Exception(
                    f"LLM request failed with status code: {response.status_code}"
                )

            try:
                response_json = response.json()

                content = response_json["choices"][0]["message"]["content"]
                if "web_search" in response_json.keys():
                    logger.info(
                        "Successfully received and parsed LLM response")
                    web_search = response_json["web_search"]
                    return content, web_search
                logger.info("Successfully received and parsed LLM response")
                return content
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse LLM response: {str(e)}")
                raise ValueError(f"Invalid response format from LLM: {str(e)}")

        except httpx.TimeoutException:
            logger.error("LLM request timed out")
            raise httpx.TimeoutException("LLM request timed out")
        except httpx.RequestError as e:
            logger.error(f"Network error during LLM request: {str(e)}")
            raise httpx.RequestError(
                f"Network error during LLM request: {str(e)}")


if __name__ == "__main__":
    import asyncio
    from pyaml_env import parse_config
    configs = parse_config(
        r"D:\GoodStudy\AI_Agents_Hackathon\awesome_patent_mining\patent_analysis_agent\research_agent\core\llm_config.yaml")
    print(configs["glm-4"])
    # 定义工具参数示例
    tools = [{
        "type": "web_search",
        "web_search": {
            "enable": True,
            "search_engine": "search_std",
            "search_result": True,
            "search_prompt": "你是一名分析师，请用简洁的语言总结网络搜索中：{{search_result}}中的关键信息，简要介绍该公司在氧化镓材料领域的技术布局和发展"
        }
    }]

    llm = LLM(config=configs["glm-4"])
    # messages = [{'content': '''
    #       You are a research assistant tasked with writing a comprehensive related work section for a research survey.
    # Your goal is to synthesize the provided context, introduction, and related papers to create a detailed overview of the current state of research.
    # The detail_analysis_output must be in valid JSON format with a single key "related_work" containing a string.

    # Guidelines for the related work:
    # 1. Ensure consistency with themes and research directions mentioned in the introduction
    # 2. Organize the content into 3-4 distinct research directions, aligned with the introduction's focus
    # 3. For each research direction, cite and discuss relevant papers from the provided list
    # 4. Compare and contrast different approaches within each direction
    # 5. Maintain an academic tone throughout
    # 6. Length should be 5-6 paragraphs
    # 7. Use citations in the format[paper title]
    # 8. Identify key limitations and research gaps in current approaches
    # 9. Discuss potential future research directions and open challenges
    # 10. Maintain narrative continuity with the introduction section

    # Example format:
    #     {
    #         "related_work": "string"
    #     }
    # ''',
    #              'role': 'system'},
    #             {'content': '''
    # Topic: What does the technology development roadmap for multi - modal large models look like?
    # Context:
    # Related Papers:
    # Introduction: This research survey delves into the technology development roadmap for multi - modal large models, an area of increasing significance within artificial intelligence and machine learning.These models, capable of processing and generating data across various modalities such as text, images, and audio, are poised to revolutionize a multitude of fields.The survey's goal is to discern the key trends and challenges in the development and application of such models, highlighting their potential impact on future technologies. Our scope includes an extensive review of multi-modal learning architectures, training methodologies, and the implications for natural language processing, computer vision, and beyond.Drawing from a selection of key papers, this survey outlines the evolution of multi-modal large models and identifies the pivotal themes including interoperability, scalability, and ethical considerations.The structure of this survey is organized to provide a cohesive journey through the state of the art, beginning with foundational concepts and culminating in a discussion of future research directions and potential roadblocks.

    # Write a structured related work section for the research survey following the system guidelines.Output only the JSON structure, no additional text or explanations.
    # ''', 'role': 'user'}]
    # messages = [{"role": "user", "content": "介绍一下openai最近开源的项目有哪些？"}]
    messages = [
        {"role": "user", "content": "    100字简要介绍北京铭镓半导体有限公司在'制备技术_衬底加工技术'上的技术布局和发展\n        "}]

    print(asyncio.run(llm.completion(messages, tools=tools)))  # 使用工具
    print(asyncio.run(llm.completion(messages)))  # 不使用工具
