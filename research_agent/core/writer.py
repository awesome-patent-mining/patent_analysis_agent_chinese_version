import importlib.resources as pkg_resources
import json
from pathlib import Path
from typing import List
import json_repair
import asyncio
import re
from json_repair import repair_json
from jinja2 import Environment
from research_agent.core.survey import Survey
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from pyaml_env import parse_config
from research_agent.core.rerank_with_embedding import RankingProcessor
from research_agent.core.rerank_with_LLM import RerankByLLM
from research_agent.core.generate_embedding import EmbeddingGenerator
import logging
from functools import wraps

logger = logging.getLogger(__name__)
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

class Writer:
    def __init__(self):
        configs = parse_config(Config.YAML_CONFIG)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])
        self.survey = None
        self.language = None
        base_path = Path(__file__).parent / "prompts"
        rewrite_outline_prompt_file = base_path / "rewrite_outline.jinja"
        with open(rewrite_outline_prompt_file, "r",encoding="utf-8") as f:
            self.rewrite_outline_prompt_template = Environment().from_string(f.read())

        rewrite_section_prompt_file = base_path / "rewrite_section.jinja"
        with open(rewrite_section_prompt_file, "r",encoding="utf-8") as f:
            self.rewrite_section_prompt_template = Environment().from_string(f.read())

        write_section_prompt_file = base_path / "write_section.jinja"
        with open(write_section_prompt_file, "r",encoding="utf-8") as f:
            self.write_section_prompt_template = Environment().from_string(f.read())

        write_title_prompt_file = base_path / "write_title.jinja"
        with open(write_title_prompt_file, "r", encoding="utf-8") as f:
            self.write_title_prompt_template = Environment().from_string(f.read())

        write_introduction_prompt_file = base_path / "write_introduction.jinja"
        with open(write_introduction_prompt_file, "r",encoding="utf-8") as f:
            self.write_introduction_prompt_template = Environment().from_string(f.read())

        write_conclusion_prompt_file = base_path / "write_conclusion.jinja"
        with open(write_conclusion_prompt_file, "r",encoding="utf-8") as f:
            self.write_conclusion_prompt_template = Environment().from_string(f.read())

        self.embedding_gen = EmbeddingGenerator()
        self.section_rag_top_k = EmbeddingGenerator(Config.SECTION_RAG_TOP_K)
        self.ranking_processor = RankingProcessor(
            threshold=Config.THRESHOLD)  # 重排序相似度阈值,默认0.35
        self.rerank_by_llm = RerankByLLM()

    def set_language(self, language):
        self.language = language
    def get_language(self):
        return self.language

    async def write_initial_draft(self, topic: str, outline:str, context: List[str],context_embedding:List[List[float]],related_papers:List[str])-> str:
        parsed_outline = self.parse_outline(outline=outline)
        # parse related_work_outline and transform into networkx.DiGraph object
        self.survey = Survey(topic=topic)
        self.survey.transfer_parsed_outline_into_nx(parsed_outline)
        # rewrite content of related work section by section
        # 得到根节点
        root = self.survey.get_root()
        # 从根节点得到section节点
        section_nodes = self.survey.full_content.successors(root)
        section_nodes = list(section_nodes)

        if not isinstance(section_nodes, list) or not section_nodes:
            raise ValueError("section_nodes must be a non-empty list of strings")

        tasks = [self.write_section(node_section,topic,outline,context,context_embedding,related_papers) for node_section in section_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 按照顺序，输出每个section的内容
        return '\n'.join([section_content for section_content in results if isinstance(section_content, str)])

    async def write_section(self, node_section,topic,outline,context,context_embedding,related_papers) -> str:
        """Resolve a single research question.
        Args:
            node_section (str): The section code of the survey.
            topic (str): The topic of the research survey.
            outline (str): The outline of the research survey.
            context (list[str]): The context used for writing the section.
            context_embedding (list): The embedding of the context.
            related_papers (list[str]): The related papers for writing the section.

        Returns:
            str: the content of the section.

        Raises:
            ValueError: If question is empty or not a string
            RuntimeError: If LLM completion fails
        """
        if not isinstance(node_section, int):
            raise ValueError("section node number must be a int number")

        previous_content = ""
        section_code = self.survey.full_content.nodes[node_section]['code']
        section_title = self.survey.full_content.nodes[node_section]['title']
        section_description = self.survey.full_content.nodes[node_section]['description']

        # previous_content = previous_content + '\n' + "## " + section_code + " " + section_title
        logger.info(
            f"write section {section_code}:{self.survey.full_content.nodes[node_section]['title']}……")
        subsection_nodes = self.survey.full_content.successors(node_section)
        subsection_nodes = list(subsection_nodes)

        if len(subsection_nodes) == 0:
            # 准备特定的context
            # 两种rerank同时进行，一种是余弦相似度计算embedding，另一种是大模型的rerank函数
            # 这样做的目的是，如果余弦相似度高出阈值的文献不少，那么再看看rerank的结果，两者融合一下，给出最后的filtered_context
            #section_info_embedding = \
            #self.embedding_gen.convert_texts_to_embeddings(section_title + '\n' + section_description)[0]
            #reranked_context = self.rerank_by_llm.rerank_documents(self, section_title + '\n' + section_description,
            #                                                       context)
            convert_text_to_embedding_task = asyncio.create_task(self.embedding_gen.convert_texts_to_embeddings(section_title + '\n' + section_description))
            rerank_documents_task = asyncio.create_task(self.rerank_by_llm.rerank_documents(section_title + '\n' + section_description,
                                                                   context))
            section_info_embedding = await convert_text_to_embedding_task
            reranked_context = await rerank_documents_task
            # 按照reranked_context中index的顺序从小到大排序
            reranked_context = sorted(reranked_context,
                   key=lambda x: x['index'],
                   reverse=False)

            section_info_embedding = section_info_embedding[0]
            filtered_context_similarities = self.ranking_processor.filter_and_return_similarity(
                query_embedding=section_info_embedding,
                doc_embeddings=context_embedding)

            reranked_relevant_score = [context_i['relevance_score'] for context_i in  reranked_context]
            # 先看一下高出阈值的context有多少
            min_top_k = min(len(filtered_context_similarities)-filtered_context_similarities.count(0),Config.SECTION_RAG_TOP_K)

            # 如果低于等于阈值，就不管了
            # 如果高于阈值，就基于llm rerank的结果，再找出前30篇
            combined = [
                {
                    "document": doc,
                    "paper_title":title,
                    "index": idx,
                    "score": llm_score
                }
                for idx, (doc, title, cos_score,llm_score) in enumerate(zip(context,related_papers, filtered_context_similarities, reranked_relevant_score))
                if cos_score != 0
            ]
            # 按相似度降序排序
            sorted_quad = sorted(combined, key=lambda x: x["score"], reverse=True)
            filtered_context = '\n\n'.join([item['document'] for item in sorted_quad[:min_top_k]])
            filtered_related_papers = '\t'.join([item['paper_title'] for item in sorted_quad[:min_top_k]])
            logger.info(
                f"prepared context and related paper for section {section_code}:{self.survey.full_content.nodes[node_section]['title']}……")

            prompt_messages = self._prepare_write_section_prompt(
                topic=topic, outline=outline, subsection_code=section_code, code_level='section', context=filtered_context,
                related_papers=filtered_related_papers, previous_content=previous_content
            )
            response = await self.llm.completion(prompt_messages)
            section_content = json_repair.loads(response)["subsection"]
            # 清除标题
            section_content = Writer.remove_markdown_headers(section_content)
            section_content = f'## {section_code} {section_title}\n\n{section_content}'
        else:
            if not isinstance(subsection_nodes, list) or not subsection_nodes:
                raise ValueError("subsection_nodes must be a non-empty list of strings")
            tasks = []
            for idx,node_subsection in enumerate(subsection_nodes):
                previous_content = ""
                if idx == 0:
                    tasks.append(self.write_subsection(previous_content + '\n' + "## " + section_code + " " + section_title,node_subsection, topic, outline, context, context_embedding, related_papers) )
                else:
                    tasks.append(self.write_subsection(previous_content, node_subsection,topic, outline, context, context_embedding, related_papers))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            section_content = '\n'.join([subsection_content for subsection_content in results if isinstance(subsection_content, str)])
        return section_content
    async def write_title(self, topic, outline, paper_body_draft) -> str:
        """write introduction of the survey.
        Args:
            topic (str): The topic of the research survey.
            outline (str): The outline of the research survey.
            paper_body_draft (str): The body part of the research survey.

        Returns:
            str: introduction content.

        Raises:
            ValueError: If question is empty or not a string
            RuntimeError: If LLM completion fails
        """
        prompt_messages = self._prepare_write_title_prompt(
            topic=topic, outline=outline, paper_body=paper_body_draft
        )
        response = await self.llm.completion(prompt_messages)
        return json_repair.loads(response)["title"]

    async def write_introduction(self, topic, outline, paper_body_draft) -> str:
        """write introduction of the survey.
        Args:
            topic (str): The topic of the research survey.
            outline (str): The outline of the research survey.
            paper_body_draft (str): The body part of the research survey.

        Returns:
            str: introduction content.

        Raises:
            ValueError: If question is empty or not a string
            RuntimeError: If LLM completion fails
        """
        prompt_messages = self._prepare_write_introduction_prompt(
            topic=topic, outline=outline, paper_body=paper_body_draft
        )
        response = await self.llm.completion(prompt_messages)
        return json_repair.loads(response)["introduction"]
    async def write_conclusion(self, topic, outline, paper_body_draft) -> str:
        """write conclusion of the survey.
        Args:
            topic (str): The topic of the research survey.
            outline (str): The outline of the research survey.
            paper_body_draft (str): The body part of the research survey.

        Returns:
            str: introduction content.

        Raises:
            ValueError: If question is empty or not a string
            RuntimeError: If LLM completion fails
        """
        prompt_messages = self._prepare_write_conclusion_prompt(
            topic=topic, outline=outline, paper_body=paper_body_draft
        )
        response = await self.llm.completion(prompt_messages)
        return json_repair.loads(response)["conclusion"]

    async def write_subsection(self,previous_content,node_subsection, topic, outline, context, context_embedding, related_papers):
        # 按照顺序，输出每个section的内容
        subsection_code = self.survey.full_content.nodes[node_subsection]['code']
        subsection_title = self.survey.full_content.nodes[node_subsection]['title']
        subsection_description = self.survey.full_content.nodes[node_subsection]['description']

        convert_text_to_embedding_task = asyncio.create_task(
            self.embedding_gen.convert_texts_to_embeddings(subsection_title + '\n' + subsection_description))
        rerank_documents_task = asyncio.create_task(
            self.rerank_by_llm.rerank_documents(subsection_title + '\n' + subsection_description,
                                                context))
        subsection_info_embedding = await convert_text_to_embedding_task
        reranked_context = await rerank_documents_task

        reranked_context = sorted(reranked_context,
                                  key=lambda x: x['index'],
                                  reverse=False)
        subsection_info_embedding = subsection_info_embedding[0]
        filtered_context_similarities = self.ranking_processor.filter_and_return_similarity(
            query_embedding=subsection_info_embedding,
            doc_embeddings=context_embedding)

        reranked_relevant_score = [context_i['relevance_score'] for context_i in reranked_context]
        # 先看一下高出阈值的context有多少
        min_top_k = min(len(filtered_context_similarities) - filtered_context_similarities.count(0),
                        Config.SECTION_RAG_TOP_K)

        # 如果低于等于阈值，就不管了
        # 如果高于阈值，就基于llm rerank的结果，再找出前30篇
        combined = [
            {
                "document": doc,
                "paper_title": title,
                "index": idx,
                "score": llm_score
            }
            for idx, (doc, title, cos_score,llm_score) in
            enumerate(zip(context, related_papers, filtered_context_similarities, reranked_relevant_score))
            if cos_score != 0
        ]
        # 按相似度降序排序
        sorted_quad = sorted(combined, key=lambda x: x["score"], reverse=True)
        filtered_context = '\n\n'.join([item['document'] for item in sorted_quad[:min_top_k]])
        filtered_related_papers = '\t'.join([item['paper_title'] for item in sorted_quad[:min_top_k]])

        logger.info(
            f"prepared context and related paper for subsection {subsection_code}:{self.survey.full_content.nodes[node_subsection]['title']}……")
        # 准备特定的context
        prompt_messages = self._prepare_write_section_prompt(
            topic=topic, outline=outline, subsection_code=subsection_code, code_level='subsection',
            context=filtered_context,
            related_papers=filtered_related_papers, previous_content=previous_content
        )
        logger.info(
            f"write subsection {subsection_code}:{self.survey.full_content.nodes[node_subsection]['title']}……")
        response = await self.llm.completion(prompt_messages)
        # 先得把json_repair.loads(response)["subsection"]中的标题清除一下
        subsection_content = json_repair.loads(response)["subsection"]
        # 清除标题
        subsection_content = Writer.remove_markdown_headers(subsection_content)
        subsection_content = f'{previous_content}\n\n### {subsection_code} {subsection_title}\n\n{subsection_content}'

        return subsection_content

    @staticmethod
    def remove_markdown_headers(md_text):
        """
        移除Markdown文本中带数字编号的标题行（如 # 1.1 标题），保留其他内容
        """
        # 正则匹配规则：以1个或多个#开头，后接空格+数字编号（如1, 1.1, 1.1.1等），最后接空格或结尾
        pattern = r'^#+\s+\d+(?:\.\d+)*(?:\s+|$).*'
        lines = md_text.split('\n')
        # 过滤掉符合标题模式的行
        filtered_lines = [line for line in lines if not re.match(pattern, line)]
        return '\n'.join(filtered_lines)
    async def rewrite_draft(
        self,
        topic: str,
        context: str,
        context_embedding,
        suggestions: str,
        related_papers: str,
        raw_draft: str
    ):
        # generate outline of new draft
        prompt_messages = self._prepare_rewrite_outline_prompt(
            topic, context,suggestions, related_papers, raw_draft
        )
        outline = await self.llm.completion(prompt_messages)
        logger.info(
            f"new outline:{outline}")

        parsed_outline = self.parse_outline(outline=outline)
        # parse related_work_outline and transform into networkx.DiGraph object
        self.survey = Survey(topic=topic)
        self.survey.transfer_parsed_outline_into_nx(parsed_outline)
        # rewrite content of related work section by section
        # 得到根节点
        root = self.survey.get_root()
        # 从根节点得到section节点
        section_nodes = self.survey.full_content.successors(root)
        section_nodes = list(section_nodes)
        if not isinstance(section_nodes, list) or not section_nodes:
            raise ValueError("Questions must be a non-empty list of strings")

        tasks = [self.rewrite_section(node_section, topic, outline, context,context_embedding, related_papers,raw_draft) for node_section in
                 section_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return '\n'.join([section_content for section_content in results if isinstance(section_content, str)])

    async def rewrite_section(
        self,
        node_section:str,
        topic: str,
        outline: str,
        context: str,
        context_embedding,
        related_papers: str,
        raw_draft: str):

        section_code = self.survey.full_content.nodes[node_section]['code']
        section_title = self.survey.full_content.nodes[node_section]['title']
        previous_content = ""
        logger.info(
            f"rewrite section {section_code}:{self.survey.full_content.nodes[node_section]['title']}……")
        subsection_nodes = self.survey.full_content.successors(node_section)
        subsection_nodes = list(subsection_nodes)

        section_title = self.survey.full_content.nodes[node_section]['title']
        section_description = self.survey.full_content.nodes[node_section]['description']

        section_info_embedding = self.embedding_gen.get_embedding(section_title+'\n'+section_description)
        if len(subsection_nodes) == 0:

            filtered_context, filtered_related_papers, sorted_triplets = self.ranking_processor.filter_and_rerank(
                query_embedding=section_info_embedding,
                doc_embeddings=context_embedding,
                documents=context,
                related_docs=related_papers)
            prompt_messages = self._prepare_rewrite_section_prompt(
                topic=topic, outline=outline, subsection_code=section_code, context=filtered_context,
                related_papers=filtered_related_papers, raw_draft=raw_draft,
                previous_content=previous_content
            )
            response = await self.llm.completion(prompt_messages)
            previous_content = previous_content + '\n' + json_repair.loads(response)["subsection"]
        else:
            for idx, node_subsection in enumerate(subsection_nodes):
                if idx == 0:
                    previous_content = previous_content + '\n' + "## " + section_code + " " + section_title
                subsection_code = self.survey.full_content.nodes[node_subsection]['code']
                subsection_title = self.survey.full_content.nodes[node_subsection]['title']
                subsection_description = self.survey.full_content.nodes[node_subsection]['description']
                subsection_info_embedding = self.embedding_gen.get_embedding(
                    subsection_title + '\n' + subsection_description)
                # 进行reranking
                filtered_context, filtered_related_papers, sorted_triplets = self.ranking_processor.filter_and_rerank(
                    query_embedding=subsection_info_embedding,
                    doc_embeddings=context_embedding,
                    documents=context,
                    related_docs=related_papers)

                prompt_messages = self._prepare_rewrite_section_prompt(
                    topic=topic, outline=outline, subsection_code=subsection_code, context=filtered_context,
                    related_papers=filtered_related_papers,
                    raw_draft=raw_draft, previous_content=previous_content
                )
                logger.info(
                    f"rewrite subsection {subsection_code}:{self.survey.full_content.nodes[node_subsection]['title']}……")
                response = await self.llm.completion(prompt_messages)
                previous_content = previous_content + '\n' + json_repair.loads(response)["subsection"]

        return previous_content
    def _prepare_rewrite_outline_prompt(
        self, topic: str, context: str, suggestions:str,related_papers: str, raw_draft:str
    ):
        system_prompt = self.rewrite_outline_prompt_template.render(role="system",language = self.language)
        user_prompt = self.rewrite_outline_prompt_template.render(
            role="user",
            topic=topic,
            language=self.language,
            context=context,
            suggestions=suggestions,
            related_papers=related_papers,
            raw_draft=raw_draft
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _prepare_write_section_prompt(
            self, topic: str, outline: str, subsection_code: str,code_level:str, context: str, related_papers: str,
            previous_content: str
    ):
        system_prompt = self.write_section_prompt_template.render(role="system",code_level=code_level)
        user_prompt = self.write_section_prompt_template.render(
            role="user",
            topic=topic,
            language=self.language,
            outline=outline,
            subsection_code=subsection_code,
            code_level=code_level,
            context=context,
            related_papers=related_papers,
            previous_content=previous_content
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _prepare_write_introduction_prompt(
            self, topic: str, outline: str, paper_body: str

    ):
        system_prompt = self.write_introduction_prompt_template.render(role="system")
        user_prompt = self.write_introduction_prompt_template.render(
            role="user",
            topic=topic,
            language=self.language,
            outline=outline,
            paper_body=paper_body
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    def _prepare_write_title_prompt(
            self, topic: str, outline: str, paper_body: str

    ):
        system_prompt = self.write_title_prompt_template.render(role="system")
        user_prompt = self.write_title_prompt_template.render(
            role="user",
            topic=topic,
            language=self.language,
            outline=outline,
            paper_body=paper_body
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _prepare_write_conclusion_prompt(
            self, topic: str, outline: str, paper_body: str
    ):
        system_prompt = self.write_conclusion_prompt_template.render(role="system")
        user_prompt = self.write_conclusion_prompt_template.render(
            role="user",
            topic=topic,
            language=self.language,
            outline=outline,
            paper_body=paper_body
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _prepare_rewrite_section_prompt(
        self, topic: str, outline: str,subsection_code:str,context: str, related_papers: str, raw_draft:str,previous_content: str
    ):
        system_prompt = self.rewrite_section_prompt_template.render(role="system")
        user_prompt = self.rewrite_section_prompt_template.render(
            role="user",
            topic=topic,
            language = self.language,
            outline=outline,
            subsection_code=subsection_code,
            context=context,
            related_papers=related_papers,
            original_survey=raw_draft,
            previous_content=previous_content
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_outline(self, outline):
        result = {
            "title": "",
            "sections": [],
            "section_descriptions": [],
            "subsections": [],
            "subsection_descriptions": []
        }

        # Split the outline into lines
        lines = outline.split('\n')
        # 将lines中的空行去掉
        lines = [line.strip() for line in lines if line.strip() != '']

        for i, line in enumerate(lines):
            line = line.strip()
            # Match title, sections, subsections and their descriptions
            if line.startswith('# '):
                result["title"] = line[2:].strip()
            elif line.startswith('## '):
                result["sections"].append(line[3:].strip())
                # Extract the description in the next line
                if i + 1 < len(lines):
                    if self.language == 'Chinese':
                        if lines[i + 1].startswith('描述：'):
                            result["section_descriptions"].append(
                                lines[i + 1].split('描述：', 1)[1].strip())
                        else:
                            result["section_descriptions"].append("")
                        result["subsections"].append([])
                        result["subsection_descriptions"].append([])
                    else:
                        if lines[i + 1].startswith('Description:'):
                            result["section_descriptions"].append(
                                lines[i + 1].split('Description:', 1)[1].strip())
                        else:
                            result["section_descriptions"].append("")
                        result["subsections"].append([])
                        result["subsection_descriptions"].append([])
            elif line.startswith('### '):
                if result["subsections"]:
                    result["subsections"][-1].append(line[4:].strip())
                    # Extract the description in the next line
                    if self.language=="Chinese":
                        if i + 1 < len(lines) and lines[i + 1].startswith('描述：'):
                            result["subsection_descriptions"][-1].append(
                                lines[i + 1].split('描述：', 1)[1].strip())
                    else:
                        if i + 1 < len(lines) and lines[i + 1].startswith('Description:'):
                            result["subsection_descriptions"][-1].append(
                                lines[i + 1].split('Description:', 1)[1].strip())

        return result


outline = '''
<format>
    
    ## 1 Introduction
    Description: Introduce the topic of the technological roadmap of multi-model large models and the context of the survey.

    ## 2 Background and Scope
    Description: Define the temporal, thematic, and methodological scope of the survey, including the context provided in the paper.

    ### 2.1 Transformer-based Reinforcement Learning
    Description: Discuss the development of Transformer-based reinforcement learning methods and the challenges in generalization and adaptability.

    ### 2.2 Working Memory Module
    Description: Introduce the working memory module and its role in enhancing model efficiency and generalization.

    ### 2.3 Limitations in Current Roadmap
    Description: Outline the limitations identified in the current technological roadmap for multi-model large models.

    ## 3 Pre-trained Model Features and Generalization
    Description: Analyze the issues with pre-trained model features and their impact on out-of-distribution generalization.

    ### 3.1 Inherent Issues in Pre-trained Features
    Description: Discuss the inherent problems in pre-trained features and their limitations.

    ### 3.2 Fine-tuning Generalization Approaches
    Description: Review previous approaches to fine-tuning generalization and their inadequacies.

    ## 4 Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs)
    Description: Explore the development of LLMs and MLLMs, their applications in sequential recommendation systems, and the introduction of LLaRA.

    ### 4.1 Versatility of LLMs
    Description: Highlight the versatility of LLMs and their integration with other modalities in MLLMs.

    ### 4.2 LLaRA and its Impact
    Description: Discuss the pioneering work of LLaRA in the field of recommendation systems.

    ## 5 Advancements in Large Language Models (LLMs) and Vision Language Models (VLMs)
    Description: Review advancements in LLMs and VLMs, including Transformer-based models and the challenges in processing long video sequences.

    ### 5.1 Evolution of Transformer-based Models
    Description: Outline the evolution of Transformer-based models.

    ### 5.2 GPT and its Variants
    Description: Discuss the impact of GPT and its variants on the field.

    ### 5.3 LLaMA-VID Framework
    Description: Introduce LLaMA-VID and its role in enhancing LLMs for image and video understanding.

    ## 6 Multi-model Large Language Models in Natural Language Processing
    Description: Discuss the technological roadmap of multi-model LLMs in natural language processing and their potential in transforming embodied AI/robotics.

    ### 6.1 Object Detection Systems
    Description: Highlight the limitations of current object detection systems.

    ### 6.2 Reasoning-based Object Detection
    Description: Introduce the new research task of reasoning-based object detection and the proposed multimodal model.

    ## 7 AI in Biomedicine
    Description: Explore the rapid evolution of AI in biomedicine, the integration of large language models and multimodal machine learning, and the challenges in clinical settings.

    ### 7.1 Integration in Healthcare
    Description: Discuss the challenges in implementing AI models in clinical settings.

    ### 7.2 Data Sharing and Ethical Considerations
    Description: Address the importance of data sharing and ethical considerations in AI applications.

    ## 8 Future of Multi-model Large Models in Brain Graph Analysis
    Description: Outline the future of multi-model large models in brain graph analysis and the need for improved diagnosis of brain disorders.

    ### 8.1 Graph Neural Networks
    Description: Discuss the potential of graph neural networks for synthesizing brain graphs.

    ### 8.2 Multimodal Brain Graph Synthesis
    Description: Suggest areas for advancement in multimodal brain graph synthesis frameworks.

    ## 9 Multi-scale Modeling in Time Series Forecasting
    Description: Discuss the challenges of multi-scale modeling in time series forecasting with Transformers and the introduction of Pathformer.

    ### 9.1 Limitations of Current Models
    Description: Address the limitations of incomplete multi-scale modeling and fixed multi-scale processes.

    ### 9.2 Pathformer Architecture
    Description: Introduce the Pathformer architecture and its adaptive pathways.

    ## 10 Conclusion
    Description: Summarize the key findings and implications of the survey on the technological roadmap of multi-model large models.
</format>
'''