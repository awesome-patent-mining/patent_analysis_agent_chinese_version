from research_agent.core.storage import PatentDatabase
import asyncio
from collections import deque
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List
import re
import matplotlib.pyplot as plt
from research_agent.core.query_1 import Query
from research_agent.core.storage import PatentDatabase
from research_agent.core.generate_tech_genealogy import Tech_Gene_Generator
import base64
from io import BytesIO
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PatentTechAnalyzer:
    def __init__(self):
        self.tech_gene_generator = Tech_Gene_Generator()
        self.technology_map = None
        self.patent_num = 0
        self.query = Query()
        self.current_year = datetime.now().year
    def set_technology_map(self, technology_map):
        self.technology_map = technology_map
    def get_technology_map(self):
        return self.technology_map
    async def search_by_tech(self, sub_tech, limit=50):
        """带数据清洗的专利搜索方法
        :param sub_tech: 子技术名称
        :param limit: 限制返回数量，默认为100
        """
        results = []
        batch_size = 50
        for offset in range(0, limit, batch_size):
            batch = await self.query.query_by_content(
                sub_tech,
                limit=min(batch_size, limit - offset),
                offset=offset
            )
            if batch:
                valid_batch = [
                    {k: v for k, v in item.items()
                     if not (k == 'pbdt' and self._is_invalid_date(v))}
                    for item in batch
                ]
                results.extend(valid_batch)
            if not batch or len(batch) < batch_size:
                break
        return results[:limit]

    def _is_invalid_date(self, value):
        """判断无效日期格式"""
        if pd.isna(value) or value in ['', 'NaN', 'NaT']:
            return True
        return False

    def _extract_year(self, df, year_col):
        """多功能年份提取方法"""
        # 第一阶段：基础清洗
        clean_df = df[df[year_col].notna()].copy()
        clean_series = clean_df[year_col].astype(str)

        # 第二阶段：格式标准化
        def format_converter(val):
            # 处理数值型日期 (20211001.0 -> 20211001)
            if re.match(r'^\d{8}\.?\d*$', val):
                return val.split('.')[0]
            # 处理时间戳 (2025-04-26 00:17:19 -> 20250426)
            if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', val):
                return val[:10].replace('-', '')
            return val

        formatted_series = clean_series.apply(format_converter)

        # 第三阶段：日期解析
        years = pd.to_datetime(
            formatted_series.str[:8],
            format='%Y%m%d',
            errors='coerce'
        ).dt.year

        # 第四阶段：容错处理
        fallback_years = pd.to_numeric(
            formatted_series.str[:4],
            errors='coerce'
        )

        # 合并结果并重新索引
        final_years = years.where(years.notna(), fallback_years)
        return final_years.reindex(df.index)

    def _generate_year_plot(self,save_dir,year_count):
        """生成年份趋势柱状图"""
        if year_count.empty:
            return None

        plt.figure(figsize=(12, 6))
        year_count.plot(kind='bar', color='steelblue')
        plt.title("Annual Patent Trend Analysis", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Patent Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        #buf = BytesIO()
        plt.savefig(fname = save_dir+'/year_count.png', bbox_inches='tight', dpi=120)

        print(save_dir+'/year_count.png')
        return save_dir+'/year_count.png'

    def _generate_country_plot(self, save_dir,country_count):
        """生成国家分布图"""
        if country_count.empty:
            return None

        plt.figure(figsize=(10, 6))
        country_count.head(10).plot(kind='barh', color='darkgreen')
        plt.title("Countries/Regions Distribution", fontsize=14)
        plt.xlabel("Patent Count", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.savefig(fname=save_dir + '/country_count.png', bbox_inches='tight', dpi=120)
        plt.close()
        print(save_dir + '/country_count.png')
        return save_dir + '/country_count.png'

    def _detect_column(self, df, possible_columns, default_name):
        """鲁棒的列名检测方法"""
        # 精确匹配优先
        for col in possible_columns:
            if col in df.columns:
                return col

        # 模糊匹配使用正则表达式
        pattern = '|'.join([f'^{col}$' for col in possible_columns])
        matched = df.columns[df.columns.str.contains(pattern, case=False, na=False)]
        return matched[0] if not matched.empty else default_name

    def _generate_overall_stats(self,save_dir, all_results):
        """生成总体统计数据
        :param save_dir: 保存目录
        :param all_results: 所有技术对应的专利列表
        :return: 技术数量统计、年份趋势、国家分布图、年份趋势图、国家分布图
        """
        all_patents = []
        tech_counts = {}

        for sub_tech, patents in all_results.items():
            if patents:
                df = pd.DataFrame(patents)
                patent_id_col = self._detect_column(
                    df,
                    ['patent_id', 'pn', 'apno'],
                    None
                )
                if patent_id_col:
                    df = df.drop_duplicates(subset=[patent_id_col])

                tech_counts[sub_tech] = len(df)
                all_patents.extend(patents)

        if not all_patents:
            return None, None, None

        all_df = pd.DataFrame(all_patents)
        patent_id_col = self._detect_column(
            all_df,
            ['patent_id', 'id', 'publication_number', 'application_number'],
            None
        )
        if patent_id_col:
            all_df = all_df.drop_duplicates(subset=[patent_id_col])

        # 年份分析
        year_col = self._detect_column(all_df, ['pbdt'], None)
        if year_col:
            valid_years = self._extract_year(all_df, year_col)
            valid_years = valid_years.dropna().astype(int)
            valid_years = valid_years[(valid_years >= 1990) &
                                      (valid_years <= self.current_year + 2)]
            year_count = valid_years.value_counts().sort_index()
            year_img = self._generate_year_plot(save_dir=save_dir,year_count=year_count)
        else:
            year_count = pd.Series(dtype='int')
            year_img = None

        # 国家分析
        country_col = self._detect_column(
            all_df,
            ['country', 'patent_office', 'patent_office_code', 'patent_country'],
            None
        )
        if country_col and country_col in all_df.columns:
            country_count = all_df[country_col].value_counts()
            country_img = self._generate_country_plot(save_dir,country_count)
        else:
            country_count = pd.Series(dtype='int')
            country_img = None

        return tech_counts, year_count, country_count, year_img, country_img

    def report(self, save_dir,all_results,unique_patent_df):
        """生成最终报告
        :param save_dir: 保存文件夹路径
        :param all_results: 所有技术领域和对应的专利列表
        :param unique_patent_df: 去重后的专利数据集"""
        report_lines = []

        # 生成总体统计数据
        tech_counts, year_counts, country_counts, year_img, country_img = self._generate_overall_stats(save_dir,all_results)

        report_lines.append("### 专利数据总览")


        if tech_counts:
            # 各技术专利数量 - 优化后的表格
            #report_lines.append("#### Overview---")
            report_lines.append(f"- **相关专利数量**: {unique_patent_df.shape[0]}\n\n")
            report_lines.append("---")
            #report_lines.append("#### 各技术领域专利数量")
            '''
            tech_df = pd.DataFrame({
                '技术领域': list(tech_counts.keys()),
                '专利数量': list(tech_counts.values())
            })
            # 按专利数量降序排列
            #tech_df = tech_df.sort_values('专利数量', ascending=False)
            markdown_table = tech_df.to_markdown(index=False,tablefmt="github")
            lines = markdown_table.split("\n")

            # 修改分隔符行：将默认的 "---" 替换为 ":---:"
            new_separator = "|".join([":---:"] * (tech_df.shape[1]))
            new_separator = '|' + new_separator + '|'  # +1 是考虑索引列
            lines[1] = new_separator

            # 重新拼接表格
            centered_table = "\n".join(lines)
            report_lines.append(centered_table)
            '''

        # 总体年份趋势
        if year_counts is not None and not year_counts.empty:
            report_lines.append("#### 专利申请量年度变化趋势")
            if year_img:
                report_lines.append(f'![](year_count.png)')
            #report_lines.append("**详细数据:**")

            #markdown_table = year_counts.to_markdown(index=False, tablefmt="github")
            #lines = markdown_table.split("\n")

            # 修改分隔符行：将默认的 "---" 替换为 ":---:"
            #new_separator = "|".join([":---:"] * 2)
            #new_separator = '|' + new_separator + '|'  # +1 是考虑索引列
            #lines[1] = new_separator

            # 重新拼接表格
            #centered_table = "\n".join(lines)
            #report_lines.append(centered_table)

            #report_lines.append(year_counts.to_markdown())

        # 总体国家分布
        if country_counts is not None and not country_counts.empty:
            report_lines.append("#### 不同国家/地区的专利数量分布")
            if country_img:
                report_lines.append(f'![](country_count.png)')
            #report_lines.append("**Top 10 国家/地区:**")

            #markdown_table = country_counts.to_markdown(index=False, tablefmt="github")
            #lines = markdown_table.split("\n")

            # 修改分隔符行：将默认的 "---" 替换为 ":---:"
            #new_separator = "|".join([":---:"] * 2)
            #new_separator = '|' + new_separator + '|'  # +1 是考虑索引列
            #lines[1] = new_separator

            # 重新拼接表格
            #centered_table = "\n".join(lines)
            #report_lines.append(centered_table)

        # 保存报告
        with open(save_dir+'/patent_report.md', 'w', encoding='utf-8',newline="\n") as f:
            for line in report_lines:
                f.write(line+'\n\n')
        print(f"Patent analysis report generated:{save_dir}/patent_report.md")

    async def run(self,save_dir:str,tech_map:List[dict]):
        """主运行逻辑，统一去重存库
        :param save_dir: 保存目录
        :param tech_map: 技术图谱的字典
        """
        all_patents = []
        all_results = {}

        self.set_technology_map(tech_map)
        patent_result_deque = await self.query.query_by_tech_map_zh(tech_map)
        for patent_result_i in patent_result_deque:
            # 获取当前元素的 'patent' 列表，并将其扩展到结果列表中
            all_patents.extend(patent_result_i.get('patents', []))

        # 统一去重（以patent_id为唯一键）
        df_all = pd.DataFrame(all_patents)
        patent_id_col = self._detect_column(
            df_all,
            ['patent_id', 'pn', 'apno'],
            None
        )
        if patent_id_col:
            df_all = df_all.drop_duplicates(subset=[patent_id_col])
        # ---------- 为df_all增加字段 ----------

        added_columns = await self.query.batch_query_simple_bibliography(deque(df_all['patent_id']))
        df_added_columns = pd.DataFrame(added_columns)
        merged_df = pd.merge(df_all, df_added_columns, on='patent_id', how='inner')
        #print(merged_df)
        #merged_df中有个字段apdt和pbdt,这两个字段是整数，请将其处理成字符串，并取其前四位更新到apdt和pbdt中
        merged_df['apdt'] = merged_df['apdt'].astype(str).str[:4]
        merged_df['pbdt'] = merged_df['pbdt'].astype(str).str[:4]
        #请新建一个字段'app_country',将‘patent_office’字段对应的内容赋予'app_country'字段
        merged_df['app_country'] = merged_df['patent_office']
        #在patent_result_deque中增加新添加的字段
        for idx,patent_result_i in enumerate(patent_result_deque):
            patent_list_i = patent_result_i.get('patents', [])
            tech_point_i = patent_result_i.get('tech_point', None)
            full_patent_list_i = self.get_full_patent_data(patent_list_i, merged_df)
            all_results[tech_point_i]=full_patent_list_i


        # ---------- 新增数据预处理代码开始 ----------
        # 定义数据库表字段
        field_list = [
            'patent_id', 'pn', 'apno', 'title', 'original_assignee',
            'current_assignee', 'inventor', 'apdt', 'pbdt', 'abstract',
            'ipc', 'patent_office', 'relevancy','app_country'
        ]

        # 确保字段对齐并填充缺失列
        for col in field_list:
            if col not in merged_df.columns:
                merged_df[col] = None
        merged_df = merged_df[field_list]

        # 过滤无效patent_id
        merged_df = merged_df[merged_df['patent_id'].notnull() & (df_all['patent_id'] != '')]

        # 转换NaN为None（兼容数据库）
        merged_df = merged_df.where(pd.notnull(merged_df), None)
        # ---------- 新增数据预处理代码结束 ----------

        # 转回dict批量入库
        unique_patents = merged_df.to_dict(orient='records')

        # 批量入库
        db = PatentDatabase()
        try:
            db.connect()
            db.create_patents_table()
            db.insert_patents_batch(unique_patents)
            print(f"共批量入库{len(unique_patents)}条专利。")
            self.patent_num = len(unique_patents)
        except Exception as e:
            print(f"数据库操作出错: {e}")
        finally:
            db.disconnect()

        # 继续原有报告等后续功能
        self.report(save_dir,all_results,merged_df)

    @staticmethod
    def get_full_patent_data(patent_list, full_patent_df):
        """
        根据 patent_list 中的专利 ID，从 full_patent_df 中提取全字段数据
        :param patent_list: 包含专利 ID 的 deque，每个元素是一个 dict，dict 中有 'patent_id' 键
        :param full_patent_df: pandas 的 DataFrame，包含全量的专利数据（包括 'patent_id' 字段）
        :return: 包含全字段专利数据的 list，每个元素是一个 dict
        """
        # 提取所有专利 ID 列表
        patent_ids = [element['patent_id'] for element in patent_list]

        # 根据专利 ID 过滤 full_patent_df，得到需要的专利数据
        filtered_df = full_patent_df[full_patent_df['patent_id'].isin(patent_ids)]

        # 将过滤后的 DataFrame 转换为 dict 的列表
        full_patent_data_list = filtered_df.to_dict(orient='records')

        return full_patent_data_list
# 使用示例

if __name__ == "__main__":
    # 示例数据（实际使用时从Query类获取）
    analyzer = PatentTechAnalyzer()
    asyncio.run(analyzer.run('../general_analysis_output/1', [{'Primary Technology': ['Text Preprocessing Technology'], 'Secondary Technology': ['Tokenization Technology', 'Stemming and Lemmatization Technology', 'Stop Words Removal Technology']}, {'Primary Technology': ['Text Representation Technology'], 'Secondary Technology': ['Bag of Words Technology', 'TF-IDF Technology', 'Word Embedding Technology']}, {'Primary Technology': ['Machine Learning Models for NLP'], 'Secondary Technology': ['Supervised Learning Models', 'Unsupervised Learning Models', 'Reinforcement Learning Models']}, {'Primary Technology': ['Natural Language Understanding Technology'], 'Secondary Technology': ['Named Entity Recognition Technology', 'Sentiment Analysis Technology', 'Question Answering Technology']}, {'Primary Technology': ['Natural Language Generation Technology'], 'Secondary Technology': ['Text Summarization Technology', 'Machine Translation Technology', 'Dialogue Generation Technology']}]
))

    '''
    sample_patents = [
        {
            "relevancy": "84%",
            "patent_id": "ee598175-195d-47bc-aa42-a3a6f9efaf1d",
            "pn": "CN111314673A",
            "apno": "CN202010205283.9",
            "title": "一种实时智能视频传输和运动控制系统及方法",
            "original_assignee": "河北师范大学",
            "current_assignee": "河北师范大学",
            "inventor": "赵佳|赵佳娣|赵华|吕清",
            "apdt": 20200323,
            "pbdt": 20200619,
            "abstract": "本发明公开一种实时智能视频传输和运动控制系统...",
            "ipc": "H04N7/18",
            "patent_office": "CN"
        }
    ]

    # 创建数据库实例并存储数据
    db = PatentDatabase()
    try:
        db.connect()
        db.create_patents_table()
        db.insert_patents_batch(sample_patents)
    except Exception as e:
        db.logger.error(f"操作数据库时发生错误: {e}")
    finally:
        db.disconnect()
    '''
