import asyncio
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List
import re
import matplotlib.pyplot as plt
from research_agent.core.query import Query
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

        self.query = Query()
        self.current_year = datetime.now().year
    def set_technology_map(self, technology_map):
        self.technology_map = technology_map
    def get_technology_map(self):
        return self.technology_map
    async def search_by_tech(self, sub_tech, limit=20):
        """带数据清洗的专利搜索方法"""
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
    @staticmethod
    def _generate_year_plot(self, save_dir,year_count, title):
        """生成年份趋势柱状图"""
        if year_count.empty:
            return None

        plt.figure(figsize=(12, 6))
        year_count.plot(kind='bar', color='steelblue')
        plt.title(title, fontsize=14)
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("专利数量", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        #buf = BytesIO()
        plt.savefig(fname = save_dir+'\\year_count.png', bbox_inches='tight', dpi=120)
        plt.close()
        return save_dir+'\\year_count.png'

    def _generate_country_plot(self, save_dir,country_count, title):
        """生成国家分布图"""
        if country_count.empty:
            return None

        plt.figure(figsize=(10, 6))
        country_count.head(10).plot(kind='barh', color='darkgreen')
        plt.title(title, fontsize=14)
        plt.xlabel("专利数量", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.savefig(fname=save_dir + '\\country_count.png', bbox_inches='tight', dpi=120)
        plt.close()
        return save_dir + '\\country_count.png'

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
                    ['patent_id', 'id', 'publication_number', 'application_number'],
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
            year_img = self._generate_year_plot(save_dir,year_count, "专利年度趋势分析")
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
            country_img = self._generate_country_plot(save_dir,country_count, "国家/地区分布（前10）")
        else:
            country_count = pd.Series(dtype='int')
            country_img = None

        return tech_counts, year_count, country_count, year_img, country_img

    def report(self, save_dir,all_results):
        """生成最终报告
        :param save_dir: 保存文件夹路径
        :param all_results: 所有技术领域和对应的专利列表"""
        report_lines = []

        # 生成总体统计数据
        tech_counts, year_counts, country_counts, year_img, country_img = self._generate_overall_stats(save_dir,all_results)

        report_lines.append("# 专利技术总体分析报告")
        report_lines.append("\n---")

        if tech_counts:
            # 各技术专利数量 - 优化后的表格
            report_lines.append("## 各技术领域专利数量")
            tech_df = pd.DataFrame({
                '技术领域': list(tech_counts.keys()),
                '专利数量': list(tech_counts.values())
            })
            # 按专利数量降序排列
            tech_df = tech_df.sort_values('专利数量', ascending=False)
            report_lines.append(tech_df.to_markdown(index=False))

        # 总体年份趋势
        if year_counts is not None and not year_counts.empty:
            report_lines.append("\n## 专利年度趋势分析")
            if year_img:
                report_lines.append(f'<img src="data:image/png;base64,{year_img}">')
            report_lines.append("\n**详细数据:**")
            report_lines.append(year_counts.to_markdown())

        # 总体国家分布
        if country_counts is not None and not country_counts.empty:
            report_lines.append("\n## 国家/地区分布分析")
            if country_img:
                report_lines.append(f'<img src="data:image/png;base64,{country_img}">')
            report_lines.append("\n**Top 10 国家/地区:**")
            report_lines.append(country_counts.head(10).to_markdown())

        # 保存报告
        with open(save_dir+'\\patent_report.md', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(report_lines))
        print(f"专利分析报告已生成：{save_dir}\\patent_report.md")

    async def run(self,save_dir:str,tech_map:List[dict]):
        """主运行逻辑，统一去重存库
        :param save_dir: 保存目录
        :param tech_map: 技术图谱的字典
        """

        all_results = {}
        all_patents = []
        self.set_technology_map(tech_map)
        for tech_node in self.technology_map:
            main_techs = tech_node.get("一级技术", [])
            sub_techs = tech_node.get("二级技术", [])
            for main_tech in main_techs:
                for sub in sub_techs:
                    key = f"{main_tech} - {sub}"
                    print(f"正在检索：{key}...")
                    patents = await self.search_by_tech(sub)
                    all_results[key] = patents
                    all_patents.extend(patents)

        # 统一去重（以patent_id为唯一键）
        df_all = pd.DataFrame(all_patents)
        patent_id_col = self._detect_column(
            df_all,
            ['patent_id', 'id', 'publication_number', 'application_number'],
            None
        )
        if patent_id_col:
            df_all = df_all.drop_duplicates(subset=[patent_id_col])

        # ---------- 新增数据预处理代码开始 ----------
        # 定义数据库表字段
        field_list = [
            'patent_id', 'pn', 'apno', 'title', 'original_assignee',
            'current_assignee', 'inventor', 'apdt', 'pbdt', 'abstract',
            'ipc', 'patent_office', 'relevancy'
        ]

        # 确保字段对齐并填充缺失列
        for col in field_list:
            if col not in df_all.columns:
                df_all[col] = None
        df_all = df_all[field_list]

        # 过滤无效patent_id
        df_all = df_all[df_all['patent_id'].notnull() & (df_all['patent_id'] != '')]

        # 转换NaN为None（兼容数据库）
        df_all = df_all.where(pd.notnull(df_all), None)
        # ---------- 新增数据预处理代码结束 ----------

        # 转回dict批量入库
        unique_patents = df_all.to_dict(orient='records')

        # 批量入库
        db = PatentDatabase()
        try:
            db.connect()
            db.create_patents_table()
            #db.insert_patents_batch(unique_patents)
            #print(f"共批量入库{len(unique_patents)}条专利。")
        except Exception as e:
            print(f"数据库操作出错: {e}")
        finally:
            db.disconnect()

        # 继续原有报告等后续功能
        self.report(save_dir,all_results)


if __name__ == "__main__":
    analyzer = PatentTechAnalyzer()
    asyncio.run(analyzer.run('.\\.\\general_analysis_output\\1',[{'一级技术': ['机器学习'], '二级技术': ['监督学习', '无监督学习', '强化学习']}, {'一级技术': ['深度学习'], '二级技术': ['卷积神经网络', '循环神经网络', '生成对抗网络']} ]))
