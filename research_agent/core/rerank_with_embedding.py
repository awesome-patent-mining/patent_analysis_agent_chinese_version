import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging
logger = logging.getLogger(__name__)

class RankingProcessor:
    """
    处理文档重排序的类，提供基于相似度的过滤和重排序功能
    """

    def __init__(self, threshold: float = 0.3):
        """
        初始化重排序处理器

        参数:
            threshold (float): 相似度过滤阈值，默认0.3
        """
        self.threshold = threshold

    def filter_and_rerank(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        documents: List[str],
        related_docs: List[str]
    ) -> Tuple[List[str],List[str], List[Tuple[float, str,str]]]:
        """
        对文档计算相似度，并返回重新排序的结果

        参数:
            query_embedding: 查询文本(如HYDE生成文本)的embedding向量
            doc_embeddings: 所有文档的embedding向量列表
            documents: 文档文本列表
            related_docs: 相关文档的标题列表

        返回:
            Tuple[List[str], List[Tuple[float, str]]]: 
                - 过滤后的文档列表
                - (相似度,文档)对的有序列表
        """
        # 计算相似度
        similarities = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(doc_embeddings)
        )[0]

        # 创建(相似度,文档)对的列表
        similarity_doc_triplets = list(zip(similarities, documents,related_docs))

        # 按相似度降序排序
        sorted_triplets = sorted(
            similarity_doc_triplets,
            key=lambda x: x[0],
            reverse=True
        )

        # 过滤掉相似度低于阈值的结果
        filtered_docs = [
            doc for sim, doc,_ in sorted_triplets
            if sim >= self.threshold
        ]
        filtered_titles = [
            title for sim, _,title in sorted_triplets
            if sim >= self.threshold
        ]
        logger.info(f"重排序完成，过滤后文档数量: {len(filtered_docs)}，过滤篇数: {len(sorted_triplets) - len(filtered_docs)}")
        return filtered_docs,filtered_titles, sorted_triplets

    def filter_and_return_similarity(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]]
    ) -> List[float]:
        """
        对文档计算相似度，并返回相似度数值，如果相似度低于阈值，直接返回0

        参数:
            query_embedding: 查询文本(如HYDE生成文本)的embedding向量
            doc_embeddings: 所有文档的embedding向量列表
            documents: 文档文本列表
            related_docs: 相关文档的标题列表

        返回:
            Tuple[List[str], List[Tuple[float, str]]]:
                - 过滤后的文档列表
                - (相似度,文档)对的有序列表
        """
        # 计算相似度
        similarities = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(doc_embeddings)
        )[0]

        # 创建(相似度,文档)对的列表
        #similarity_doc_triplets = list(zip(similarities, documents,related_docs))
        # 过滤掉相似度低于阈值的结果
        filtered_sims = []

        for sim in similarities:
            if sim >= self.threshold:
                filtered_sims.append(sim)
            else:
                filtered_sims.append(0)

        logger.info(f"重排序完成，过滤后文档数量: {len(filtered_sims) - filtered_sims.count(0)}，过滤篇数: {filtered_sims.count(0)}")
        return filtered_sims

    def get_similarity_scores(
        self,
        sorted_pairs: List[Tuple[float, str]]
    ) -> List[float]:
        """
        获取排序后的相似度分数列表

        参数:
            sorted_pairs: (相似度,文档)对的有序列表

        返回:
            List[float]: 相似度分数列表
        """
        return [sim for sim, _ in sorted_pairs]

    def get_top_k_docs(
        self,
        sorted_pairs: List[Tuple[float, str]],
        k: int
    ) -> List[str]:
        """
        获取相似度最高的k个文档

        参数:
            sorted_pairs: (相似度,文档)对的有序列表
            k: 需要返回的文档数量

        返回:
            List[str]: 前k个最相关的文档
        """
        return [doc for _, doc in sorted_pairs[:k]]
