import nest_asyncio
import asyncio
import pymysql
from research_agent.core.general_llm import LLM
from research_agent.core.config import Config
from research_agent.core.generate_patent_chart import Patent_Chart_Generator
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import json_repair
from pathlib import Path
from pyaml_env import parse_config
from json_repair import repair_json
from jinja2 import Environment, FileSystemLoader
import matplotlib.dates as mdates
import pandas as pd
import sys
import os

# nest_asyncio.apply()
# sys.path.append(os.path.abspath('../..'))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False


class PatentTrendAnalyzer:
    def __init__(self):
        """Initialize the PatentTrendAnalyzer.

        Sets up database connection parameters and initializes components for patent trend analysis.
        """
        # Database connection parameters
        self.host = Config.MYSQL_HOST
        self.port = Config.MYSQL_PORT
        self.user = Config.MYSQL_USERNAME
        self.passwd = Config.MYSQL_PASSWORD
        self.database = Config.MYSQL_DB
        self.charset = Config.MYSQL_CHARSET
        self.table_name = Config.patent_table

        # LLM and prompt initialization
        absolute_path = os.path.abspath(Config.YAML_CONFIG)
        configs = parse_config(absolute_path)
        self.language = ""
        # self.ipc_dict = self.parse_ipc_txt_to_dict(Config.IPC_DICT_PATH)
        self.llm = LLM(config=configs[Config.DEFAULT_MODEL])

    def set_language(self, language):
        self.language = language

    def get_language(self):
        return self.language

    def _prepare_prompts(
            self, patent_stat: str
    ):
        system_prompt = self.generate_patent_application_trend_prompt.render(
            role="system")
        user_prompt = self.generate_patent_application_trend_prompt.render(
            role="user",
            patent_statistics=patent_stat
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def retrieve_patent_trends_info(self, country_col='当前申请(专利权)人国家',
                                    year_col='申请年', target_countries=['CN', 'US'],
                                    table_name='patent_info'):
        """
        从数据库中采集指定国家及全球的专利申请年度趋势

        参数：
            country_col: 国家字段列名（默认：'当前申请(专利权)人国家'）
            year_col: 申请年份列名（默认：'申请年'）
            target_countries: 待分析国家列表（默认：中国、美国）
            table_name: 数据库表名（默认：patent_table）

        返回：
            包含三列数据的DataFrame（年份、国家计数、全球计数）
        """
        # 创建数据库连接
        # sql_host = os.getenv("MYSQL_HOST")
        # sql_user = os.getenv("MYSQL_USERNAME")
        # sql_password = os.getenv("MYSQL_PASSWORD")
        # sql_db = os.getenv("MYSQL_DB")
        # sql_charset = os.getenv("MYSQL_CHARSET")
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # 获取所有年份的数据
                query = f"""
                SELECT `{year_col}`, `{country_col}`, COUNT(*) as count
                FROM {self.table_name}
                WHERE `{year_col}` IS NOT NULL
                GROUP BY `{year_col}`, `{country_col}`
                """
                cursor.execute(query)
                results = cursor.fetchall()

                # 将结果转换为DataFrame
                df = pd.DataFrame(results, columns=[
                                  year_col, country_col, 'count'])

                # 预处理年份数据
                df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
                df = df.dropna(subset=[year_col]).astype({year_col: 'int'})

                # 国家维度统计
                country_counts = (
                    df[df[country_col].isin(target_countries)]
                    .pivot_table(index=year_col, columns=country_col, values='count', fill_value=0)
                )

                # 全球维度统计
                global_counts = df.groupby(year_col)['count'].sum()

                # 合并结果
                result = country_counts.join(
                    global_counts.rename('全球'), how='outer')
                return result.fillna(0).astype(int)

        finally:
            connection.close()

    @staticmethod
    def plot_patent_trends(patent_stat_df, file_dir, figsize=(14, 7), phases=None):
        """
        绘制专利申请趋势对比图
        参数：
            patent_stat_df: DataFrame 需包含年份索引和CN/US/全球三列
            figsize : 图表尺寸，默认(14,7)
            phases: 阶段列表，每个阶段包含period（阶段名称）、start_year（起始年）、
                    end_year（结束年）和description（描述），格式为：
                    [{'period': '阶段1', 'start_year': 2000, 'end_year': 2010, 'description': '...'},
                     {'period': '阶段2', 'start_year': 2010, 'end_year': 2020, 'description': '...'}]
        """
        plt.figure(figsize=figsize)

        # 转换索引为日期格式
        years = patent_stat_df.index.astype(str)
        x = pd.to_datetime(years, format='%Y')

        # 绘制趋势线
        plt.plot(x, patent_stat_df['CN'], 'r-o',
                 linewidth=2, markersize=6, label='中国')
        plt.plot(x, patent_stat_df['US'], 'b--s',
                 linewidth=2, markersize=6, label='美国')
        plt.plot(x, patent_stat_df['全球'], 'g-.^',
                 linewidth=2, markersize=6, label='全球')

        # 坐标轴格式设置
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator(5))  # 5年刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        # 设置y轴范围
        y_max = patent_stat_df.max().max() * 1.3
        plt.ylim(0, y_max)

        # 添加发展阶段分隔线和标注
        if phases:
            # 计算每个阶段的宽度，用于精准定位标注位置
            start_year = pd.to_datetime(
                str(patent_stat_df.index[0]), format='%Y')
            end_year = pd.to_datetime(
                str(patent_stat_df.index[-1]), format='%Y')

            for phase in phases:
                phase_start = pd.to_datetime(
                    str(phase['start_year']), format='%Y')
                phase_end = pd.to_datetime(str(phase['end_year']), format='%Y')
                phase_mid = phase_start + (phase_end - phase_start) / 2

                # 添加竖线
                plt.axvline(x=phase_start, color='gray',
                            linestyle='--', alpha=0.7)
                plt.axvline(x=phase_end, color='gray',
                            linestyle='--', alpha=0.7)

                # 添加阶段标注，使用phase['period']作为阶段名称
                plt.text(phase_mid, y_max * 0.9, phase['period'],
                         ha='center', va='center', fontsize=20)

        # 图表元素
        plt.title('第四代半导体材料领域专利申请趋势对比（1990-2025）', fontsize=14, pad=20)
        plt.xlabel('申请年份', fontsize=12)
        plt.ylabel('专利申请量', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # 修改图例位置
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0, 1))

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

    async def generate_patent_application_trend_analysis(self, pat_statistics) -> str:
        """Generate a patent application trend analysis based on statistics.

        Args:
            pat_statistics (str): Patent application statistics data.

        Returns:
            str: Analysis result text.

        Raises:
            RuntimeError: If prompt template loading fails
            json_repair.JSONDecodeError: If response parsing fails
        """
        try:
            # Load prompt template
            base_path = Path(__file__).parent / "prompts"
            prompt_file = base_path / "0_generate_patent_trend_chart.jinja"
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.generate_patent_application_trend_prompt = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

        # Generate completion
        prompt_messages = self._prepare_prompts(
            patent_stat=pat_statistics,
        )
        response = await self.llm.completion(prompt_messages)

        # Parse and return result
        result = json_repair.loads(response)
        return result

    async def generate_patent_trend_part_analysis(self, pat_statistics, period_info, top5_applicants_info) -> str:
        """
        生成专利趋势分析的部分内容，重点关注某一时期的技术发展趋势。

        参数：
            pat_statistics (str): 专利申请统计数据，markdown格式
            period_info (dict): 趋势阶段信息，包含阶段名称、起始年、结束年和描述
            top5_applicants_info (list): 5个重要申请主体在该阶段的专利申请信息

        返回：
            str: 分析结果文本，格式为JSON

        异常：
            RuntimeError: 如果提示词模板加载失败
            json_repair.JSONDecodeError: 如果响应解析失败
        """
        try:
            # 加载提示词模板
            base_path = Path(__file__).parent / "prompts"
            prompt_file = base_path / "1_generate_patent_trend_part.jinja"
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.generate_patent_application_trend_prompt = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

        # 将所有输入格式化为单个字符串
        combined_stats = f"""
                            Patent Statistics:
                            {pat_statistics}

                            Period Info:
                            {period_info}

                            Top 5 Applicants Info:
                            {top5_applicants_info}
                            """

        # 生成完成
        prompt_messages = self._prepare_prompts(
            patent_stat=combined_stats
        )
        response = await self.llm.completion(prompt_messages)

        # 解析并返回结果
        result = json_repair.loads(response)
        return result

    def retrieve_top5_applicants_info(self, start_year: int, end_year: int, table_name: str = "patent_info") -> list:
        """
        查询指定年份区间内top5当前申请(专利权)人及其专利信息

        参数：
            start_year: 起始年份
            end_year: 结束年份
            table_name: 数据库表名（默认：patent_info）

        返回：
            包含top5申请人及其专利信息的列表
        """
        # 创建数据库连接
        # sql_host = os.getenv("MYSQL_HOST")
        # sql_user = os.getenv("MYSQL_USERNAME")
        # sql_password = os.getenv("MYSQL_PASSWORD")
        # sql_db = os.getenv("MYSQL_DB")
        # sql_charset = os.getenv("MYSQL_CHARSET")
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # 查询top5申请人
                query_top5 = f"""
                SELECT `[标]当前申请(专利权)人`, COUNT(*) AS patent_count
                FROM `{self.table_name}`
                WHERE `[标]当前申请(专利权)人` IS NOT NULL
                  AND `申请年` BETWEEN %s AND %s
                GROUP BY `[标]当前申请(专利权)人`
                ORDER BY patent_count DESC
                LIMIT 5
                """
                cursor.execute(query_top5, (start_year, end_year))
                top5_applicants = cursor.fetchall()

                # 查询每个申请人的专利信息
                result = []
                for applicant, _ in top5_applicants:
                    query_patents = f"""
                    SELECT `公开(公告)号`, `标题(译)(简体中文)`, `摘要`, `申请年`, `当前申请(专利权)人国家`
                    FROM `{table_name}`
                    WHERE `[标]当前申请(专利权)人` = %s
                      AND `申请年` BETWEEN %s AND %s
                    """
                    cursor.execute(
                        query_patents, (applicant, start_year, end_year))
                    patents = cursor.fetchall()
                    result.append({
                        "applicant": applicant,
                        "patents": [{
                            "publication_number": patent[0],
                            "title": patent[1],
                            "abstract": patent[2],
                            "application_year": patent[3],
                            "applicant_country": patent[4]
                        } for patent in patents]
                    })

                return result

        finally:
            connection.close()

    async def generate_patent_trend_part_analysis_concurrent(self, pat_statistics, period_info):
        """
        并发执行 generate_patent_trend_part_analysis 函数，处理 period_info 列表中的每个元素。

        参数：
            pat_statistics (str): 专利申请统计数据，markdown格式
            period_info (list): 趋势阶段信息列表，每个元素包含阶段名称、起始年、结束年和描述
            top5_applicants_info (list): 5个重要申请主体在该阶段的专利申请信息

        返回：
            list: 包含所有并发任务结果的列表
        """
        # 创建并发任务列表
        tasks = [
            self.generate_patent_trend_part_analysis(
                pat_statistics=pat_statistics,
                period_info=period,
                top5_applicants_info=self.retrieve_top5_applicants_info(
                    period['start_year'], period['end_year'])
            )
            for period in period_info
        ]

        # # 并发执行所有任务
        results = await asyncio.gather(*tasks)

        return results

    def write_analysis_to_markdown(self, file_path, title, alt_text, image_path, title_text, overall_trend, tasks):
        """
        将专利趋势分析结果写入 Markdown 文件。

        参数：
            file_path (str): 文件路径
            title (str): 文件标题
            alt_text (str): 图片的替代文本
            image_path (str): 图片路径
            title_text (str): 图片标题
            overall_trend (str): 整体趋势分析
            tasks (list): 包含每个阶段分析结果的列表
        """
        # markdown_image = f'![{alt_text}]({image_path} "{title_text}")\n'
        # trand_report = title +"\n"+markdown_image+"\n"+overall_trend + '\n'
        trand_report = ""
        with open(file_path, 'w', encoding='utf-8') as file:
            markdown_image = f'![{alt_text}]({image_path} "{title_text}")\n'
            file.write(title + '\n')
            file.write(markdown_image + '\n')
            file.write(overall_trend + '\n')
            a = title + "\n" + markdown_image + "\n" + overall_trend + '\n'
            trand_report += a
            for i, period in enumerate(tasks, 1):
                # 添加两个换行符，确保段落之间有空白行
                file.write(f'### ({i}){period["period_title"]}\n\n')
                file.write(period['country_compare'] + '\n\n')  # 添加两个换行符
                file.write(period['company_compare'] + '\n\n')  # 添加两个换行符
                b = f'### ({i}){period["period_title"]}\n\n' + \
                    period['country_compare'] + '\n\n' + \
                    period['company_compare'] + '\n\n'
                trand_report += b
        return trand_report

    async def main(self):
        """
        主方法，执行专利趋势分析流程并将结果写入 Markdown 文件。
        """
        start_time = time.time()

        # 获取当前时间并格式化为文件夹名称
        folder_name = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('./patent_trend_analysis', folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # 获取专利趋势数据
        patent_application_trend_info = self.retrieve_patent_trends_info()
        patent_application_trend_info_md = patent_application_trend_info.to_markdown(
            index=True)

        tech_trend_analysis = await self.generate_patent_application_trend_analysis(patent_application_trend_info_md)
        # 定义阶段信息
        period_info = tech_trend_analysis['period_info']
        overall_trend = tech_trend_analysis['overall_trend']

        # 绘制趋势图
        image_path = os.path.join(output_dir, '趋势图.png')
        self.plot_patent_trends(
            patent_application_trend_info, image_path, phases=period_info)

        # 并发执行分析任务
        results = await self.generate_patent_trend_part_analysis_concurrent(
            pat_statistics=patent_application_trend_info_md,
            period_info=period_info)

        # 将分析结果写入 Markdown 文件
        self.write_analysis_to_markdown(
            file_path=os.path.join(output_dir, 'patent_analysis.md'),
            title='## （一）专利申请趋势分析',
            alt_text='趋势图',
            image_path='./趋势图.png',
            title_text='专利申请趋势图',
            overall_trend=overall_trend,
            tasks=results
        )

        end_time = time.time()
        print(f"程序运行时间：{end_time - start_time}秒")


# 如果直接运行该模块，则执行主方法
if __name__ == "__main__":
    patent_trend = PatentTrendAnalyzer()
    asyncio.run(patent_trend.main())
