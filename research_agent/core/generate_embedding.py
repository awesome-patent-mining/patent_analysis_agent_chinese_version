from typing import List, Union
from zhipuai import ZhipuAI
from research_agent.core.config import Config
import re
import asyncio
import os
from pyaml_env import parse_config
import tiktoken
from typing import Tuple
import logging

logger = logging.getLogger(__name__)
class EmbeddingGenerator:
    """
    文本向量生成器类
    用于将文本转换为向量表示，支持批量处理和自定义维度。
    """

    def __init__(
        self,
        model: str = Config.EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,

    ):
        """
        初始化向量生成器

        Args:
            api_key: ZhipuAI的API密钥
            model: 使用的embedding模型，默认为'embedding-3'
            dimensions: 向量维度，建议选择256、512、1024或2048
        """

        absolute_path = os.path.abspath(Config.YAML_CONFIG)
        configs = parse_config(absolute_path)
        self.client = ZhipuAI(api_key=configs[Config.DEFAULT_MODEL]['API_KEY'])
        self.model = model
        self.dimensions = dimensions

    def _validate_input(self, texts: Union[str, List[str]]) -> List[str]:
        """
        验证输入文本

        Args:
            texts: 单个文本或文本列表

        Returns:
            List[str]: 标准化后的文本列表

        Raises:
            ValueError: 如果输入为空或包含非字符串数据
        """
        if isinstance(texts, str):
            texts = [texts]
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("所有输入必须是字符串类型")
        if not texts:
            raise ValueError("输入文本不能为空")
        return texts

    def split_paragraphs(self,content)->List[str]:

        # 匹配任意数量空白字符构成的分隔行
        paragraph_separator = r'\n\s*\n'
        raw_paragraphs = re.split(paragraph_separator, content)

        # 清理段落首尾空白字符，过滤空段落
        cleaned_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        return cleaned_paragraphs
    async def generate_embedding(self, text: str) -> Tuple[List[float],str]:
        """
        将文本转换为向量表示
        Args:
            text: 文本
        Returns:
            List[float]: 文本转化成的向量
        """
        # 将text拆分成段落
        paragraphs = self.split_paragraphs(text)
        # 判断text转换成token后长度是否大于3072，如果不超过，先试试直接转embedding
        enc = tiktoken.get_encoding("cl100k_base")
        enc_output = enc.encode(text)
        # 如果enc_output长度大于3060，将text的最后一段去掉,看转化成的token长度是否小于3060，如果仍然大于3060，则继续去掉倒数第二段，直至成功
        while len(enc_output) > 3060:
            paragraphs = paragraphs[:-1]
            text = '\n'.join(paragraphs)
            enc_output = enc.encode(text)
        # 调用ZhipuAI的embedding接口，并获取embedding结果,如果报错，将text的最后一段去掉，重新转化embedding,如果仍然报错，继续去掉倒数第二段，直至成功
        while True:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding,text
            except Exception as e:
                if len(paragraphs) <= 1:
                    raise RuntimeError(f"无法继续截断文本，原始错误：{str(e)}")
                # 截断最后100个字符并保留提示信息
                paragraphs = paragraphs[:-1]
                text = '\n'.join(paragraphs)

    async def generate_embeddings(self, texts: Union[str, List[str]]) -> Tuple[List[List[float]],List[int]]:
        """
        生成文本的向量表示

        Args:
            texts: 单个文本或文本列表

        Returns:
            List[List[float]]: 向量列表

        Raises:
            ValueError: 输入验证失败时
            Exception: 生成向量过程中发生错误时

        Note:
            - embedding-2 的单条请求最多支持 512 个Tokens，数组总长度不得超过8K
            - embedding-3 的单条请求最多支持 3072 个Tokens，数组总长度不得超过8K
            - 数组最大不得超过 64 条
        """
        idx = 0
        try:
            texts = self._validate_input(texts)
            #batches = self._batch_texts(texts)
            tasks = [self.generate_embedding(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out any failed results

            context_embedding_tuples = [
                result for result in results
                if not isinstance(result, Exception)
            ]
            failed_ids = [idx for idx,result in enumerate(results)
                if isinstance(result, Exception)]
            logger.info(f"生成向量完成，共生成{len(context_embedding_tuples)}个向量")
            return context_embedding_tuples,failed_ids

        except ValueError as ve:
            raise ValueError(f"输入验证失败: {str(ve)}")
        except Exception as e:
            raise Exception(f"在第{idx}个batch生成向量时发生错误: {str(e)}")

    def _batch_texts(self, texts: List[str], batch_size: int = Config.BATCH_SIZE) -> List[List[str]]:
        """
        将文本列表分割成批次

        Args:
            texts: 文本列表
            batch_size: 每批次的最大数量，默认64

        Returns:
            List[List[str]]: 分批次的文本列表
        """
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    async def convert_texts_to_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        生成文本的向量表示

        Args:
            texts: 单个文本或文本列表

        Returns:
            List[List[float]]: 向量列表

        Raises:
            ValueError: 输入验证失败时
            Exception: 生成向量过程中发生错误时

        Note:
            - embedding-2 的单条请求最多支持 512 个Tokens，数组总长度不得超过8K
            - embedding-3 的单条请求最多支持 3072 个Tokens，数组总长度不得超过8K
            - 数组最大不得超过 64 条
        """
        try:
            texts = self._validate_input(texts)
            batches = self._batch_texts(texts)
            all_embeddings = []

            for batch in batches:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            logger.info(f"生成向量完成，共生成{len(all_embeddings)}个向量")
            return all_embeddings

        except ValueError as ve:
            raise ValueError(f"输入验证失败: {str(ve)}")
        except Exception as e:
            raise Exception(f"生成向量时发生错误: {str(e)}")
# 当直接运行此文件时的测试代码
if __name__ == "__main__":
    # 创建实例
    async def main():

        embedding_generator = EmbeddingGenerator("96a8cfcb206bea29b8f8fc8306290752.SMM5dc6IIYRNFjD4")

        # 生成单个文本的向量
        text = r'''# 3 MULTI-MODALITY
In real life, humans often perceive the environment in a multi-modal cognitive way. Similarly, multi-modal machine learning is a modeling approach aiming to process and relate the sensory information from multiple modalities [358 ]. By aggregating the advantages and capabilities of various data modalities, multi-modal machine learning can often provide more robust and accurate HAR. There are two main types of multi-modality learning methods, namely, fusion and co-learning. Fusion refers to the integration of information from two or more modalities for training and inference, while co-learning refers to the transfer of knowledge between different data modalities.

# 3.1 FUSION
As discussed in Section 2 , different modalities can have different strengths. Thus it becomes a natural choice to take advantage of the complementary strengths of different data modalities via fusion, so as to achieve enhanced HAR performance. There are two widely used multi-modality fusion schemes in HAR, namely, score fusion and feature fusion. Generally, the score fusion [ 362 ] integrates the decisions that are separately made based on different modalities (e.g., by weighted averaging [ 382 ] or by learning a score fusion model [ 361 ]) to produce the final classification results. On the other hand, the feature fusion [ 379 ] generally combines the features from different modalities to yield aggregated features that are often very discriminative and powerful for HAR. Note that data fusion, i.e., fusing the multimodality input data before feature extraction [ 404 ], has also been exploited. Since the input data can be treated as the original raw features, here we simply categorize the data fusion approaches under feature fusion. Table 5 gives the results of multi-modality fusion-based HAR methods on the MSRDailyActivity3D [ 284 ], UTD-MHAD [ 355 ], and NTU $\mathrm{RGB+D}$ [195 ] benchmark datasets.  

TABLE 5 Performance comparison of multi-modality fusion-based HAR methods on MSRDailyActivity3D (M), UTD-MHAD (U), and NTU $\mathsf{R G B+D}$ (N) datasets. S: Skeleton, D: Depth, IR: Infrared, PC: Point Cloud, Ac: Acceleration, Gyr: Gyroscope.   


<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2">Modality</td><td rowspan="2">Fusion</td><td rowspan="2"></td><td colspan="3">Dataset</td><td rowspan="2"></td></tr><tr><td>U</td><td></td><td>N</td></tr><tr><td></td><td>2016</td><td></td><td>Score</td><td>M</td><td>91.2</td><td>CS</td><td>CV</td></tr><tr><td>Imran et al. [359] SFAM[360]</td><td>2017</td><td></td><td>Feature,Score</td><td>- -</td><td>-</td><td>-</td><td>-</td></tr><tr><td>c-ConvNet[58]</td><td>2018</td><td></td><td>Feature,Score</td><td></td><td></td><td>86.4</td><td>89.1</td></tr><tr><td>GMVAR [361]</td><td>2019</td><td></td><td>Score</td><td></td><td>-</td><td>=</td><td>=</td></tr><tr><td>Dhiman et al. [362]</td><td>2020</td><td>RGB,D</td><td>Score</td><td></td><td>-</td><td>79.4</td><td>84.1</td></tr><tr><td></td><td>2020</td><td></td><td>Feature</td><td>-</td><td>-</td><td>89.5</td><td>91.7</td></tr><tr><td>Wang et al. [363]</td><td>2021</td><td></td><td>Feature</td><td>-</td><td>-</td><td></td><td></td></tr><tr><td>Trear [364]</td><td></td><td></td><td>Score</td><td>-</td><td>-</td><td>89.7</td><td>93.0</td></tr><tr><td>Ren et al. [365]</td><td>2021</td><td></td><td>Feature,Score</td><td></td><td>-</td><td>94.2</td><td></td></tr><tr><td>CAPF [366]</td><td>2022</td><td></td><td></td><td></td><td>-</td><td>73.2</td><td>97.3</td></tr><tr><td>ST-LSTM[8]</td><td>2017</td><td></td><td>Feature</td><td></td><td>-</td><td>80.8</td><td>80.6</td></tr><tr><td>Chain-MS[367]</td><td>2017</td><td></td><td>Feature</td><td></td><td></td><td>82.5</td><td></td></tr><tr><td>GRU+STA-Hands[368]</td><td>2017</td><td></td><td>Score</td><td>-</td><td>-</td><td>83.7</td><td>88.6</td></tr><tr><td>Zhao et al. [369]</td><td>2017</td><td></td><td>Feature</td><td></td><td></td><td>84.8</td><td>93.7</td></tr><tr><td>Baradel et al. [370]</td><td>2017</td><td></td><td>Score</td><td>90.0</td><td></td><td>92.6</td><td>90.6</td></tr><tr><td>SI-MM [371]</td><td>2018</td><td>RGB,S</td><td>Feature</td><td>91.9</td><td></td><td>92.2</td><td>97.9</td></tr><tr><td>SeparableSTA[372]</td><td>2019</td><td></td><td>Feature</td><td></td><td>-</td><td>89.1</td><td>94.6</td></tr><tr><td>SGM-Net [373]</td><td>2020</td><td></td><td>Score</td><td></td><td>-</td><td>95.5</td><td>95.9</td></tr><tr><td>VPN (RNX3D101)[374]</td><td>2020</td><td></td><td>Feature</td><td></td><td>-</td><td>89.9</td><td>98.0</td></tr><tr><td>Luvizon et al. [375]</td><td>2020</td><td></td><td>Feature</td><td>-</td><td>-</td><td></td><td>=</td></tr><tr><td>JOLO-GCN [376]</td><td>2021</td><td></td><td>Score</td><td>-</td><td>-</td><td>93.8</td><td>98.1</td></tr><tr><td>TP-ViT [377]</td><td>2022</td><td></td><td>Feature,Score Feature,Score</td><td>-</td><td>-</td><td>97.0</td><td>99.6</td></tr><tr><td>RGBPose-Conv3D[378]</td><td>2022</td><td></td><td>Feature</td><td>- -</td><td>- -</td><td>75.2</td><td>83.1</td></tr><tr><td>Rahmani et al.[379]</td><td>2017 2018</td><td>S,D</td><td>Score</td><td>88.1</td><td></td><td></td><td></td></tr><tr><td>Kamel et al. [380]</td><td>2019</td><td></td><td>Score</td><td>-</td><td>95.3</td><td></td><td>- -</td></tr><tr><td>3DSTCNN [381]</td><td>2021</td><td></td><td>Score</td><td>-</td><td>88.5</td><td></td><td></td></tr><tr><td>Rani et al. [382]</td><td></td><td></td><td>Feature</td><td>97.5</td><td>-</td><td>74.9</td><td>-</td></tr><tr><td>DSSCA-SSLM[383]</td><td>2017 2018</td><td></td><td>Feature</td><td></td><td></td><td>85.4</td><td>= 90.7</td></tr><tr><td>Deep Bilinear [384]</td><td></td><td>RGB,S,D</td><td>Feature</td><td></td><td>94.6</td><td></td><td></td></tr><tr><td>Cardenas et al. [23]</td><td>2018</td><td></td><td>Score</td><td></td><td>95.1</td><td></td><td>-</td></tr><tr><td>Khaire et al. [385] Khaire et al. [386]</td><td>2018 2018</td><td></td><td>Score</td><td>-</td><td>95.4</td><td>-</td><td>-</td></tr><tr><td>Ye et al. [387]</td><td>2015</td><td>RGB,S,D,PC</td><td>Score</td><td>-</td><td>-</td><td></td><td>-</td></tr><tr><td>Ardianto et al. [388]</td><td>2018</td><td>RGB,D,IR</td><td>Score</td><td>-</td><td>-</td><td></td><td>- -</td></tr><tr><td>FUSION-CPA [389]</td><td>2020</td><td>S,IR</td><td>Feature</td><td></td><td>-</td><td>91.6</td><td>94.5</td></tr><tr><td>ActAR[390]</td><td>2022</td><td>S,IR</td><td>Feature</td><td>- -</td><td>-</td><td></td><td>-</td></tr><tr><td>Wang et al.[391]</td><td>2016</td><td></td><td>Feature,Score</td><td></td><td></td><td></td><td></td></tr><tr><td>Owens et al. [315]</td><td>2018</td><td></td><td>Feature</td><td></td><td>-</td><td></td><td>-</td></tr><tr><td>TBN [25]</td><td>2019</td><td></td><td>Feature</td><td>-</td><td>-</td><td></td><td>-</td></tr><tr><td>TSN+audio stream[25]</td><td>2019</td><td></td><td>Score</td><td></td><td>-</td><td></td><td>-</td></tr><tr><td>AVSlowFast[317]</td><td>2020</td><td></td><td>Feature</td><td></td><td>-</td><td>-</td><td>-</td></tr><tr><td>Gao et al. [316]</td><td>2020</td><td>RGB,Au</td><td>Feature</td><td>-</td><td>-</td><td></td><td>-</td></tr><tr><td>MAFnet [392]</td><td>2021</td><td></td><td>Feature</td><td>-</td><td>-</td><td></td><td>-</td></tr><tr><td>RNA-Net [393]</td><td>2021</td><td></td><td>Feature,Score</td><td></td><td>-</td><td></td><td>-</td></tr><tr><td>IMD-B [394]</td><td>2022</td><td></td><td>Feature</td><td>-</td><td>-</td><td></td><td>-</td></tr><tr><td>MM-ViT [395]</td><td>2022</td><td></td><td>Feature</td><td></td><td>-</td><td></td><td>-</td></tr><tr><td>Zhang et al. [396]</td><td>2022</td><td></td><td>Feature,Score</td><td>- - - - - - - - - -</td><td>- 95.6 99.3 96.8 97.0 - - 93.3 97.9 - -</td><td>- 89.2 - - -</td><td>- - - - - - - - - -</td></tr><tr><td>Dawar et al. [397] Dawar et al. [398] Wei et al. [399] Ahmad et al. [400] MGAF [401] Pham et al. [402] Ijaz et al. [403] DCNN [404] DeepConvLSTM [405] WiVi [341] Zou et al. [406] Memmesheimer et al [407]2020 Imran et al [282] Multi-Modal GCN[408] VATT [409] Doughty et al. [410]</td><td>2018 2018 2019 2019 2020 2021 2022 2015 2016 2019 2020 2020 2022</td><td>D,Ac,Gyr D,Ac,Gyr RGB,Ac,Gyr D,Ac,Gyr D,Ac,Gyr S,Ac S,Ac Ac,Gyr Ac,Gyr RGB,WiFi Ac,Gyr S,WiFi,etc. RGB,S,Gyr 2021RGB,Audio,Text RGB,Text</td><td>Score Score Score Feature,Score Feature Feature Feature,Score Feature Feature Score Feature Feature Feature 2021RGB,Audio,TextFeature,Score Feature Feature</td></table></body></html>'''


        embedding = await embedding_generator.generate_embeddings(text)
        print(embedding)
        # 生成多个文本的向量
        texts = ["文本1", "文本2", "文本3"]
        embeddings = await embedding_generator.generate_embeddings(texts)
        print(embeddings)


    asyncio.run(main())


