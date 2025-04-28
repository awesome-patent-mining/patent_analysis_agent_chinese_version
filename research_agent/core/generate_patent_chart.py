import matplotlib.dates as mdates
import pandas as pd
import os
from jinja2 import Environment
from json_repair import repair_json
from research_agent.core.config import Config
from research_agent.core.general_llm import LLM
from pyaml_env import parse_config
from pathlib import Path
import json_repair
import matplotlib.pyplot as plt
import pymysql
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False

from typing import List, Dict
import pandas as pd



class Patent_Chart_Generator:
    """该类用于将统计数据转化为折线图,需要的输入内容是markdown格式的文本，然后进行指定国家的专利申请趋势分析
    This class uses an LLM to generate relevant research questions by processing
    a topic, optional context, and related papers.

    Attributes:
        llm: An instance of the LLM class for generating completions
        prompt_template: Jinja template for generating prompts
    """

    def __init__(self):
        """Initialize the QuestionProposer.

        Args:
            iteration (int, optional): Iteration number for question generation. Defaults to 0.
        """
        absolute_path = os.path.abspath(Config.YAML_CONFIG)
        configs = parse_config(absolute_path)
        self.language = ""
        self.ipc_dict = self.parse_ipc_txt_to_dict(Config.IPC_DICT_PATH)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])
        try:
            base_path = Path(__file__).parent / "prompts"

            generate_patent_application_trend_prompt_file = base_path / "generate_patent_application_trend_analysis.jinja"
            with open(generate_patent_application_trend_prompt_file, "r", encoding="utf-8") as f:
                self.generate_patent_application_trend_prompt = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")
    def set_language(self, language):
        self.language = language
    def get_language(self,):
        return self.language
    def _prepare_prompts(
            self, patent_stat: str
    ):
        system_prompt = self.generate_patent_application_trend_prompt.render(role="system")
        user_prompt = self.generate_patent_application_trend_prompt.render(
            role="user",
            patent_statistics=patent_stat
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def get_ipc_description(self, ipc_code):
        """
        根据IPC分类号获得对应的技术描述信息
        """
        return self.ipc_dict[ipc_code]['Description']


    @staticmethod
    def plot_patent_trends(patent_stat_df, file_dir,figsize=(14, 7)):
        """
        绘制专利申请趋势对比图
        参数：
            patent_stat_df: DataFrame 需包含年份索引和CN/US/全球三列
            figsize : 图表尺寸，默认(14,7)
        """
        plt.figure(figsize=figsize)

        # 转换索引为日期格式
        years = patent_stat_df.index.astype(str)
        x = pd.to_datetime(years, format='%Y')

        # 绘制趋势线
        plt.plot(x, patent_stat_df['CN'], 'r-o', linewidth=2, markersize=6, label='中国')
        plt.plot(x, patent_stat_df['US'], 'b--s', linewidth=2, markersize=6, label='美国')
        plt.plot(x, patent_stat_df['全球'], 'g-.^', linewidth=2, markersize=6, label='全球')

        # 坐标轴格式设置
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator(5))  # 5年刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        # 图表元素
        plt.title('实验动物领域专利申请趋势对比（1989-2025）', fontsize=14, pad=20)
        plt.xlabel('申请年份', fontsize=12)
        plt.ylabel('专利申请量', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)

        # 数值标注
        for col in ['CN', 'US', '全球']:
            for x_val, y_val in zip(x, patent_stat_df[col]):
                if y_val > 0:  # 仅标注非零值
                    plt.text(x_val, y_val + 2, str(y_val),
                             ha='center', va='bottom',
                             fontsize=8 if col == 'US' else 9,
                             color='blue' if col == 'US' else 'black')

        plt.tight_layout()
        ax.figure.savefig(file_dir, dpi=300)
        return None


    def parse_ipc_txt_to_dict(self,file_path):
        """
        Parse the IPC info from a txt file into a dictionary format.
        该函数返回一个词典，其中键是IPC代码，值是dict，包含了该IPC的对应信息，其中Parent_IPC键对应着IPC的父亲IPC、Level对应该IPC的层次、
        Description对应该IPC的中文描述，Description_EN对应该该IPC的英语描述。
        样例如下：
        {
        'A01B1/10': {
        'Parent_IPC': 'A01B1/06',
        'Level': 5,
        'Description': '带双铲刀或多铲刀的',
        'Description_EN': 'with two or more blades'
    }
        ...
        }
        Parameters:
            file_path (str): Path to the txt file.

        Returns:
            dict: A dictionary where the keys are IPC codes and the values are dictionaries
                  containing Parent_IPC, Level, Description, and Description_EN.
        """

        ipc_dict = {}  # Initialize the dictionary to store results

        try:
            # Open the file with the appropriate encoding (try 'utf-8', fallback to 'gbk')
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read all lines from the file
                lines = file.readlines()
        except UnicodeDecodeError:
            # If utf-8 encoding fails, fallback to 'gbk'
            with open(file_path, 'r', encoding='gbk') as file:
                lines = file.readlines()

        # Extract the header (first line) and skip it
        headers = lines[0].strip().split('\t')  # Split the header line by tabs

        # Iterate over the remaining lines to process the data
        for line in lines[1:]:
            # Split the line by tabs
            values = line.strip().split('\t')

            # Map the IPC data to the corresponding keys
            ipc_code = values[0]  # IPC is the first column
            parent_ipc = values[1]  # Parent_IPC is the second column
            level = int(values[2])  # Level is the third column (convert to integer)
            description = values[3]  # Description is the fourth column
            description_en = values[4]  # Description_EN is the fifth column

            # Construct the dictionary entry for the IPC code
            ipc_dict[ipc_code] = {
                "Parent_IPC": parent_ipc,
                "Level": level,
                "Description": description,
                "Description_EN": description_en
            }

        return ipc_dict

    @staticmethod
    def get_top_applicants(patent_info: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        获取专利申请量前N名的申请人及其逐年申请量

        参数：
        patent_info - 包含专利信息的DataFrame
        n - 需要返回的申请人数量

        返回：
        包含申请人排名及逐年申请量的DataFrame，列结构：
        [申请人名称, 总申请量, 年份1申请量, 年份2申请量...]
        """
        # 数据预处理
        valid_data = patent_info.dropna(subset=['[标]当前申请(专利权)人', '申请年'])

        # 统计申请人年度分布
        annual_counts = (
            valid_data.groupby(['[标]当前申请(专利权)人', '申请年'])
            .size()
            .unstack(fill_value=0)
        )

        # 计算总申请量并取前N名
        total_counts = valid_data['[标]当前申请(专利权)人'].value_counts().head(n)
        top_applicants = total_counts.index.tolist()

        # 合并总申请量和年度数据
        result_df = pd.concat([
            total_counts.rename('总申请量'),
            annual_counts.loc[top_applicants]
        ], axis=1)

        # 重置索引并重命名列
        result_df = result_df.reset_index().rename(
            columns={'index': '申请人名称'})

        # 按总申请量降序排序
        return result_df.sort_values('总申请量', ascending=False)

    @staticmethod
    def get_top_ipc_applicants(patent_df: pd.DataFrame,
                               applicants: List[str],
                               top_n: int) -> Dict[str, Dict[str, Dict[int, int]]]:
        """
        统计指定申请人在Top N技术分类号上的逐年专利申请量

        参数：
        patent_df: 包含专利信息的DataFrame，需含'[标]当前申请(专利权)人'、'IPC分类号'、'公开(公告)年'列
        applicants: 需要统计的专利申请人列表
        top_n: 需要统计的技术分类号数量

        返回：
        嵌套字典结构：申请人->分类号->年份->申请量
        """
        # 数据预处理
        df = patent_df.dropna(subset=['[标]当前申请(专利权)人', 'IPC分类号', '公开(公告)年']).copy()
        df['IPC分类号'] = df['IPC分类号'].str.split('; ')

        # 展开分类号并筛选有效数据
        exploded_df = df.explode('IPC分类号')
        exploded_df['公开(公告)年'] = exploded_df['公开(公告)年'].astype(int)

        # 获取Top N分类号
        top_ips = exploded_df['IPC分类号'].value_counts().head(top_n).index.tolist()

        # 生成所有可能年份
        all_years = sorted(exploded_df['公开(公告)年'].unique())

        # 构建结果结构
        result = {}
        for applicant in applicants:
            # 筛选申请人数据
            applicant_df = exploded_df[exploded_df['[标]当前申请(专利权)人'] == applicant]

            # 筛选Top分类号数据
            ipc_df = applicant_df[applicant_df['IPC分类号'].isin(top_ips)]

            # 初始化申请人记录
            result[applicant] = {}

            # 遍历每个分类号
            for ipc in top_ips:
                # 生成年份计数字典
                year_counts = (
                    ipc_df[ipc_df['IPC分类号'] == ipc]
                    .groupby('公开(公告)年')
                    .size()
                    .reindex(all_years, fill_value=0)
                    .to_dict()
                )
                result[applicant][ipc] = year_counts

        return result

    @staticmethod
    def retrieve_patent_trends_info(patent_info_df, country_col='当前申请(专利权)人国家',
                              year_col='申请年', target_countries=['CN', 'US']):
        """
        从智慧芽获得的数据集中采集指定国家及全球的专利申请年度趋势

        参数：
            df: 包含专利数据的DataFrame
            country_col: 国家字段列名（默认：'当前申请(专利权)人国家'）
            year_col: 申请年份列名（默认：'申请年'）
            target_countries: 待分析国家列表（默认：中国、美国）

        返回：
            包含三列数据的DataFrame（年份、国家计数、全球计数）
        """
        # 预处理年份数据
        valid_df = patent_info_df.copy()
        valid_df[year_col] = pd.to_numeric(valid_df[year_col], errors='coerce')
        valid_df = valid_df.dropna(subset=[year_col]).astype({year_col: 'int'})

        # 国家维度统计
        country_counts = (
            valid_df[valid_df[country_col].isin(target_countries)]
            .groupby([year_col, country_col])
            .size()
            .unstack(fill_value=0)
        )

        # 全球维度统计
        global_counts = valid_df.groupby(year_col).size()

        # 合并结果
        result = country_counts.join(global_counts.rename('全球'), how='outer')
        return result.fillna(0).astype(int)

    import pymysql

    def execute_query_to_markdown(sql, host, user, password, database, port=3306):
        """
        执行MySQL查询并返回Markdown格式结果

        参数：
        sql -- 要执行的SQL查询语句
        host -- 数据库主机地址
        user -- 数据库用户名
        password -- 数据库密码
        database -- 数据库名称
        port -- 数据库端口，默认3306

        返回：
        Markdown格式的查询结果（字符串）
        """
        try:
            # 建立数据库连接
            conn = pymysql.connect(
                host=host,
                port=port,
                user=user,
                passwd=password,
                database=database,
                charset='utf8mb4'
            )

            with conn.cursor() as cursor:
                # 执行查询语句
                cursor.execute(sql)

                # 获取结果集和列名
                result = cursor.fetchall()
                columns = [col[0] for col in cursor.description]

                # 构建Markdown表格
                markdown = []
                # 表头
                header = "| " + " | ".join(columns) + " |"
                markdown.append(header)
                # 分隔线
                separator = "| " + " | ".join(["---"] * len(columns)) + " |"
                markdown.append(separator)
                # 数据行
                for row in result:
                    formatted_row = [str(item).replace('\n', ' ') if item is not None else 'NULL' for item in row]
                    markdown.append("| " + " | ".join(formatted_row) + " |")

                return '\n'.join(markdown)

        except pymysql.Error as e:
            return f"数据库操作异常: {str(e)}"
        finally:
            if 'conn' in locals() and conn.open:
                conn.close()

    async def generate_patent_application_trend_analysis(self, pat_statistics) -> str:
        """write conclusion of the survey.
        Args:
            topic (str): The topic of the research survey.
            language (str): The language of the topic.

        Returns:
            str: topic type and other info.

        Raises:
            ValueError: If question is empty or not a string
            RuntimeError: If LLM completion fails
        """
        prompt_messages = self._prepare_prompts(
            patent_stat=pat_statistics,
        )
        response = await self.llm.completion(prompt_messages)
        #response = json_repair.repair_json(response)
        return json_repair.loads(response)['answer']

# 示例用法
if __name__ == "__main__":
    pass

