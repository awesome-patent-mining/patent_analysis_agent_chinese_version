#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专利申请人分析模块：用于分析专利申请人数据并生成可视化报告
"""
from research_agent.core.generate_patent_trend import PatentTrendAnalyzer
from research_agent.core.general_llm import LLM
from research_agent.core.config import Config
import json_repair
import pymysql
from dotenv import load_dotenv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import asyncio
import json
import os
from datetime import datetime
import time
from pyaml_env import parse_config
import sys

# 将当前目录的父目录添加到Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(current_dir, parent_dir)
# from config import Config
# from .general_llm import LLM
# 导入配置和LLM模块

# 加载环境变量
load_dotenv()

# 加载配置
configs = parse_config(Config.YAML_CONFIG)

# print(configs)


def create_connection():
    """
    创建并返回一个数据库连接

    Returns:
        cursor: 数据库游标对象，如果连接失败则返回 None
    """
    # 从环境变量获取数据库配置
    sql_host = Config.MYSQL_HOST
    sql_user = Config.MYSQL_USERNAME
    sql_password = Config.MYSQL_PASSWORD
    sql_db = Config.MYSQL_DB
    sql_charset = Config.MYSQL_CHARSET

    try:
        # 创建连接
        connection = pymysql.connect(
            host=sql_host,
            user=sql_user,
            password=sql_password,
            database=sql_db,
            charset=sql_charset
        )
        print("数据库连接成功！")
        return connection.cursor()
    except pymysql.InterfaceError as e:
        print(f"数据库连接错误: {e}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None


# 技术图谱定义
MAP_TECH = '''
### 制备技术
    - 单晶生长技术
    - 衬底加工技术
    - 外延生长、薄膜制备技术
    - 器件工艺
### 器件及应用
    - 日盲紫外光电探测器
    - 红外焦平面阵列
    - X射线探测器
    - 气敏传感器
    - 功率电子器件
'''


def get_applicant_data(top_n=5):
    """
    获取申请人专利数据

    Args:
        top_n (int): 返回排名前几的申请人数据

    Returns:
        tuple: (申请人排名数据, 申请人详细专利信息)
    """
    my_sql = create_connection()
    if not my_sql:
        return [], []

    # 获取申请人及其专利数量统计
    query = f"""
    SELECT 
        `申请人`,
        GROUP_CONCAT(
            CONCAT(`国家`, '(', `分国专利数`, ')') 
            SEPARATOR ', '
        ) AS `受理局及数量`,
        CAST(SUM(`分国专利数`) AS UNSIGNED) AS `总专利数`
    FROM (
        SELECT 
            `[标]当前申请(专利权)人` AS `申请人`,
            `受理局` AS `国家`,
            CAST(COUNT(*) AS UNSIGNED) AS `分国专利数`
        FROM 
            `{Config.patent_table}`
        WHERE 
            `[标]当前申请(专利权)人` IS NOT NULL
        GROUP BY 
            `申请人`, `国家`
    ) AS subquery
    GROUP BY 
        `申请人`
    ORDER BY 
        `总专利数` DESC;
    """
    my_sql.execute(query)
    result = my_sql.fetchall()

    # 如果没有结果，返回空列表
    if not result:
        return [], []

    # 获取排名前N的机构
    applicants = [x for x, y, z in result[:top_n]]

    # 如果没有申请人，返回空列表
    if not applicants:
        return result[:top_n], []

    # 构建参数占位符
    placeholders = ','.join(['%s'] * len(applicants))

    # 获取这些申请人的专利详细信息
    query = f"""
    SELECT `公开(公告)号`, `标题(译)(简体中文)`, `摘要(译)(简体中文)`,`[标]当前申请(专利权)人`
    FROM `{Config.patent_table}`
    WHERE `[标]当前申请(专利权)人` IN ({placeholders})
    ORDER BY `[标]当前申请(专利权)人`
    """

    my_sql.execute(query, applicants)
    applicant_data = my_sql.fetchall()

    return result[:top_n], applicant_data


async def analysis_classification(deepseek_llm, company_name, data, map_tech):
    """
    对公司专利数据进行分类分析

    Args:
        deepseek_llm: LLM实例
        company_name (str): 公司名称
        data (DataFrame): 包含专利信息的DataFrame

    Returns:
        dict: 分析结果的JSON数据
    """
    # 初始化结果字符串列表
    result_strings = []

    # 遍历数据，构建专利信息字符串
    for idx, row in data.iterrows():
        # 使用f-string格式化每个专利的信息
        patent_info = f"id:{row['id']};专利标题：{row['title']}；专利摘要：{row['abstract']}"
        result_strings.append(patent_info)

    # 构建提示内容
    prompt_a = [{"role": "user",
                 "content": f"""
                **角色**：资深半导体技术专利分析师，擅长技术路线图绘制与创新点挖掘
                **输入数据**：以下专利申请人{company_name}的{len(result_strings)}项专利数据（编号0-{len(result_strings)}）：
                {result_strings}

                **核心任务**：
                1. 专利分类（确保处理所有编号）：
                    - ###为一级分类；-为二级分类
                    - 以下是包含具体的一级分类及二级分类的**技术图谱**
                        ```markdown
                        {map_tech}
                        ```
                    - 每项专利必须输出["一级分类", "二级分类"]
                    - 输出的一级分类、二级分类需要严格按照**技术图谱**中的命名

                2. **综合技术挖掘**（基于所有专利）：
                    ■ 核心技术方向：提炼专利共同聚焦的3-5个技术方向, 
                    ■ 技术问题解决分析：
                        - 针对哪些技术痛点，提出了哪些解决方案，效果指标如何
                    ■ 典型案例说明：
                        - 选择出具有代表性的3-5篇专利，论证这些专利是如何实现对技术问题的解决的。

                **处理规则**：
                    1. 分类阶段逐项处理，输出后需验证："已确认处理0-{len(result_strings)}号专利"
                    2. 技术挖掘必须引用具体专利内容，禁止臆测
                    3. 引用专利的时候需要明确其专利id，比如：CN117558758A
                    4. 请输出有效的json格式
                **输出示例**：
                    ```json
                    {{
                    "分类结果":{{"CN117558758A":["一级分类","二级分类"],}}
                    "验证状态":"完成多少项分类",
                    "综合技术挖掘":"在这里写下你基于所有专利进行的**综合技术挖掘**"
                    }}
                    ```
                """
                 }]

    # 调用LLM生成分析结果
    result = await deepseek_llm.completion(prompt_a)

    # 解析并修复JSON格式的结果
    try:
        result_json = json_repair.loads(result)
        result_json["company_name"] = company_name
        return result_json
    except Exception as e:
        print(f"JSON解析错误: {e}")
        return {"company_name": company_name, "分类结果": {}, "验证状态": "处理失败", "综合技术挖掘": ""}


async def search_applicants(search_llm, company_name, domain):
    """
    搜索公司在特定领域的信息

    Args:
        search_llm: LLM实例
        company_name (str): 公司名称
        domain (str): 技术领域

    Returns:
        str: 搜索结果
    """
    # 定义搜索工具参数
    tools = [{
        "type": "web_search",
        "web_search": {
            "enable": True,
            "search_engine": "search_std",
            "search_result": True,
            "search_prompt": "你是一名专利分析师，请用简洁的语言总结网络搜索中：{{search_result}}中的关键信息"
        }
    }]

    # 定义搜索请求
    messages = [{
        "role": "user",
        "content": f"""
        100字简要介绍{company_name}在{domain}上的技术布局和发展
        """
    }]

    # 执行搜索
    result = await search_llm.completion(messages=messages, tools=tools)
    return result


def heatmap_visualization(save_dir, analysis_result):
    """
    生成专利技术分布热力图

    Args:
        save_dir (str): 保存目录
        analysis_result (list): 分析结果列表

    Returns:
        str: 保存的文件路径
    """
    # 从分析结果中提取数据
    result_dict = {r["company_name"]: r["分类结果"] for r in analysis_result}
    companies = list(result_dict.keys())
    secondary_types = set()

    # 收集所有二级分类
    for company, patents in result_dict.items():
        for _, types in patents.items():
            if len(types) > 1:
                secondary_types.add(f"{types[0]}_{types[1]}")

    secondary_types = sorted(list(secondary_types))

    # 如果没有数据，返回None
    if not companies or not secondary_types:
        return None

    # 创建数据矩阵
    data = np.zeros((len(companies), len(secondary_types)))

    # 填充数据
    for i, company in enumerate(companies):
        for _, types in result_dict[company].items():
            if len(types) > 1:
                j = secondary_types.index(f"{types[0]}_{types[1]}")
                data[i, j] += 1

    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.Blues
    im = ax.imshow(data, cmap=cmap)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("专利数量", rotation=-90, va="bottom", fontsize=12)

    # 设置坐标轴刻度和标签
    ax.set_xticks(np.arange(len(secondary_types)))
    ax.set_yticks(np.arange(len(companies)))
    ax.set_xticklabels(secondary_types)
    ax.set_yticklabels(companies)

    # 旋转X轴标签
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # 在每个单元格中添加文本
    for i in range(len(companies)):
        for j in range(len(secondary_types)):
            if data[i, j] > 0:  # 只显示非零值
                text = ax.text(j, i, int(data[i, j]),
                               ha="center", va="center",
                               color="black" if data[i, j] < np.max(data)/2 else "white")

    # 添加网格线
    ax.set_xticks(np.arange(len(secondary_types)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(companies)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # 设置标题
    ax.set_title("专利二级分类分布热力图", fontsize=16)

    # 调整布局并保存
    fig.tight_layout()
    output_path = os.path.join(save_dir, '专利主体-专利技术热力图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return "./专利主体-专利技术热力图.png"


def bar_visualization(save_dir, applicant_rank):
    """
    生成专利申请人排名柱状图

    Args:
        save_dir (str): 保存目录
        applicant_rank (list): 申请人排名数据

    Returns:
        str: 保存的文件路径
    """
    # 检查输入数据
    if not applicant_rank:
        return None

    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 分离数据
    names = [f"{item[0]}({item[1]})" for item in applicant_rank]
    counts = [item[2] for item in applicant_rank]

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    plt.barh(names, counts, color='skyblue')
    plt.xlabel('申请数量')
    plt.title('专利申请人及其申请数量')
    plt.gca().invert_yaxis()  # 反转y轴，使得申请数量最多的在顶部

    # 保存到指定文件夹
    output_path = os.path.join(save_dir, '专利主体-专利数量柱状图.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

    return "./专利主体-专利数量柱状图.png"


def visualization(save_dir, applicant_rank, analysis_result):
    """
    生成可视化图表

    Args:
        save_dir (str): 保存目录
        applicant_rank (list): 申请人排名数据
        analysis_result (list): 分析结果列表

    Returns:
        tuple: (柱状图路径, 热力图路径)
    """
    # 确保输出文件夹存在
    os.makedirs(save_dir, exist_ok=True)

    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 生成柱状图和热力图
    bar_path = bar_visualization(save_dir, applicant_rank)
    heatmap_path = heatmap_visualization(save_dir, analysis_result)

    return bar_path, heatmap_path


def data_to_json(analysis_result, applicant_rank):
    """
    将分析结果转换为JSON格式

    Args:
        analysis_result (list): 分析结果列表
        applicant_rank (list): 申请人排名数据

    Returns:
        tuple: (技术分布JSON, 申请人排名JSON)
    """
    # 检查输入数据
    if not analysis_result or not applicant_rank:
        return {}, "[]"

    # 提取技术分类结果
    result_dict = {r["company_name"]: r["分类结果"] for r in analysis_result}

    # 创建技术分类统计字典
    tech_stats = defaultdict(lambda: defaultdict(int))

    # 遍历每家公司
    for company, patents in result_dict.items():
        for patent_id, categories in patents.items():
            primary_cat = categories[0]  # 一级分类
            secondary_cat = categories[1]  # 二级分类
            tech_stats[company][primary_cat] += 1
            tech_stats[company][f"{primary_cat}-{secondary_cat}"] += 1

    # 转换为DataFrame并确保所有数值为Python原生int
    df = pd.DataFrame.from_dict(tech_stats, orient='index')
    df = df.fillna(0).applymap(lambda x: int(x))

    # 获取所有二级分类并按字母顺序排序
    secondary_categories = sorted([col for col in df.columns if '-' in col])
    df = df[secondary_categories]

    # 转换为结构化的JSON格式
    def df_to_structured_json(df):
        result = {
            "metadata": {
                "data_type": "企业专利技术分布",
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "companies_count": len(df),
                "technology_categories_count": len(df.columns)
            },
            "companies": []
        }

        for company in df.index:
            company_data = {
                "company_name": company,
                "patent_counts": {
                    "by_category": {k: int(v) for k, v in df.loc[company].items()},
                }
            }
            result["companies"].append(company_data)

        return result

    # 转换申请人排名为字典列表
    json_data = [
        {
            "当前申请(专利权)人": company,
            "专利受理局及专利数量": country,
            "专利总数量": patent_count
        }
        for company, country, patent_count in applicant_rank
    ]

    # 输出JSON
    applicant_rank_json = json.dumps(json_data, ensure_ascii=False, indent=2)
    return df_to_structured_json(df), applicant_rank_json


async def generate_applicant_report(deepseek_llm, applicant_rank_json, bar_dir, company_tech_json, heatmap_dir):
    """
    生成专利申请人分析报告

    Args:
        deepseek_llm: LLM实例
        applicant_rank_json (str): 申请人排名JSON
        bar_dir (str): 柱状图路径
        company_tech_json (dict): 公司技术分布JSON
        heatmap_dir (str): 热力图路径

    Returns:
        str: 生成的报告
    """
    prompt_applicant_rank = [{"role": "user",
                              "content": f"""
                **角色:** 你是一位资深的专利分析专家。
                **任务:** 根据以下输入数据，用{Config.language}撰写一份专利申请人分析报告。
                **输入数据:**
                1.  **专利申请人排名分析:**
                    * **排名JSON数据 (需分析):** `{applicant_rank_json}`
                        * *说明:* 包含申请人及其专利数量的排名。
                    * **排名柱状图路径 (直接整合):** `{bar_dir}`
                        * *说明:* 指向排名柱状图。**作为占位符直接插入。**
                2.  **专利申请人-技术分布分析:**
                    * **技术分布JSON数据 (需分析):** `{company_tech_json}`
                        * *说明:* 包含主要申请人在不同技术分类下的专利分布。
                    * **技术分布热力图路径 (直接整合):** `{heatmap_dir}`
                        * *说明:* 指向技术分布热力图。**作为占位符直接插入。**

                2.  **内容生成:**
                    * **专利申请人分析:** 根据 `applicant_rank_json` 数据，撰写一段文字，总结申请人排名情况和主要发现。
                    * **专利申请人技术分布分析:** 实现对`company_tech_json` 的分析，识别出主要申请人的技术布局特点。。
                    * **插入图表占位符:** 在合适位置插入占位符：`![专利申请人排名柱状图]({bar_dir})`、`![专利申请人技术分布热力图]({heatmap_dir})`并附简短图注。
                中文格式:
                ## 二、专利申请人分析报告
                ### (1) 专利申请人排名分析
                ### (2) 专利申请人技术分布
                英文格式:
                ## 2. Patent Applicant Analysis Report
                ### (1) Patent Applicant Ranking Analysis
                ### (2) Patent Applicant Technical Distribution
                """}]

    # 获取报告部分一和二
    report_part1 = await deepseek_llm.completion(prompt_applicant_rank)

    return report_part1


async def generate_applicant_tech_report(deepseek_llm, company_tech_json, company_info, patent_miner):
    """
    生成申请人技术布局分析报告

    Args:
        deepseek_llm: LLM实例
        company_tech_json (dict): 公司技术分布JSON
        company_info (list): 公司背景信息
        patent_miner (list): 专利挖掘结果

    Returns:
        str: 生成的报告
    """
    prompt_applicant_tech = [
        {"role": "system",
         "content": f"""
        **角色:** 你是一位经验丰富的 AI 专利分析师，专长是将复杂的专利数据和公司信息整合成富有洞察力的叙述性报告。
        **任务目标:** 基于提供的输入数据，生成一份全面、专业、结构清晰的专利分析报告。报告应重点突出申请人的技术分布、核心创新焦点和关键技术成就。
        **输入数据及处理指令:** 你将获得以下结构化数据，请严格按照以下指示进行处理和利用：
        **输出语言:** {Config.language}
        1.  **`company_info` (申请人背景信息):**
            * **内容:** 提供目标申请人的背景信息，包括但不限于其历史、市场地位、关键业务领域、战略举措（如并购）、研发理念或整体使命。
            * **你的任务:** 利用此数据作为报告的引入部分。
                * 清晰地识别目标申请人。
                * 简要介绍其背景、核心业务以及在相关行业中的总体定位。
                * 这些背景信息应为后续的技术分析提供必要的语境支撑，帮助读者理解申请人为何聚焦某些技术领域。

        2.  **`company_tech_json` (技术分布 JSON 数据):**
            * **内容:** 包含目标申请人在不同技术分类（例如 IPC、CPC 或自定义类别）下的专利分布结构化数据（例如 JSON 格式）。
            * **你的任务:** 对此数据进行深入分析，并构成报告中关于“技术分布与焦点”部分的核心内容。
                * 识别并明确指出申请人专利活动最集中、投入最多的主要技术领域或分类。
                * 描述不同技术领域的相对专利数量或强度，揭示技术投入的优先顺序和集中度。
                * 分析是否存在显著的技术重点模式、随着时间推移的技术转移或新兴的重点领域（如果数据允许）。
                * 结合 `company_info` 中的背景信息，阐述为何这些集中的技术领域可能对申请人的业务战略、市场地位或长期发展至关重要。

        3.  **`patent_miner` (技术细节与代表性专利):**
            * **内容:** 包含通过分析申请人特定专利得出的精选见解。这包括：在其主要技术领域内追求的核心技术方向、旨在解决的具体技术挑战或痛点、其专利中提出的创新解决方案或方法、声称的效果、益处或性能改进（例如，提高效率、降低成本、增强安全性）、以及用于说明这些创新的具体的、有代表性的专利号（或标识符）。
            * **你的任务:** 这是你分析的**核心证据和细节来源**。利用此数据构建报告中关于“创新焦点与关键成就”的详细部分。
                * 深入阐述申请人在其主要技术领域内（从 `company_tech_json` 识别）的具体研发重点和创新方向。
                * 明确指出他们通过专利旨在解决的**具体技术问题或行业痛点**，这些应源自 `patent_miner` 中的描述。
                * 详细描述其专利中提出的**创新性解决方案、技术方案或实现方法**，这些也应基于 `patent_miner` 的内容。
                * 阐述这些创新带来的**声称效果、技术优势或商业益处**（例如，性能提升、效率改进、成本降低、安全性增强等）。
                * **必须引用 `patent_miner` 中提供的具有代表性的专利号（或标识符）**来具体说明上述创新点。
                * 对于引用的每个代表性专利，提供一个简明扼要的分析，清晰地梳理出该专利所解决的**问题**、提出的**解决方案**以及声称带来的**益处**。模仿均胜汽车安全系统范例中的引用方式，突出“问题 -> 解决方案 -> 益处”的逻辑链条。

        **输出报告要求:**

        1.  **结构与风格:** 逐个专利申请人进行分析，每个专利申请人的分析需要包含如下内容。
            * **开篇:** 基于 `company_info` 介绍申请人及其背景。
            * **第一部分:** 基于 `company_tech_json` 的分析，详细描述申请人的技术分布和主要技术焦点领域，并结合 `{company_info}` 解释战略重要性。
            * **第二部分:** 基于 `patent_miner` 的详细数据，阐述申请人的具体创新战略、解决的技术问题、提出的解决方案、声称的益处，并通过引用代表性专利进行例证说明（问题 -> 解决方案 -> 益处）。
            * **结尾:** 对申请人在其关键技术领域的整体专利布局、创新能力和行业地位进行总结。
        2.  **内容要求:**
            * 清晰明确地指明报告分析的申请人。
            * 准确呈现基于 `company_tech_json` 分析得出的技术分布重点。
            * 有机地融入 `company_info` 中的背景信息，为技术分析提供战略视角。
            * 详尽描述基于 `patent_miner` 的技术问题、解决方案和益处，并引用代表性专利进行佐证。
            * 确保所有引用的代表性专利都有简要的、结构化的（问题 -> 解决方案 -> 益处）分析。
            * 将所有信息整合成一个连贯、有深度、能够讲述申请人创新故事的叙述。
        3.  **语气与语调:** 专业、分析性、客观且信息丰富。避免主观臆断或夸大。
        4.  **数据使用原则:** **严格仅使用提供的输入数据 (`company_tech_json`, `company_info`, `patent_miner`)。**你的工作是分析、综合、转述和解释这些数据，提取其中的深层洞察，而不是简单地罗列原始信息点。

        中文格式：
        ### (3) 专利申请人技术布局分析
        (在这里插入专利申请人技术布局分析报告，以下是一个范例：大赛璐公司作为日本化工领域具有影响力的企业，在产气剂制备方面拥有较高技术实力，长期以来，其对车用安全气囊气体发生器进行了大量投入，研究和开发了多种新型产气剂配方，并与其他汽车制造企业紧密合作，不断优化产品设计，在气体发生器小型化、轻量化以及效率提升等方面取得了显著突破，并积累了大量创新成果。该公司相关专利通过采用硝酸铵、氮化物、聚合物复合物等新型燃料，减少毒性气体的排放，通过采用缓燃型燃料和双四唑化合物、金属氢氧化物等高燃烧效率组合物，优化气体产率并降低燃烧温度。如在专利EP2910536B1中提出了一种包括三嗪化合物或胍化合物的燃料、包括碱金属硝酸盐或金属碳酸盐的氧化剂的气体发生剂，其可长期维持稳定点火性能；在专利JP5481723B2中公开了一种含有作为燃料的含氮有机化合物和作为氧化剂的硝酸铵的气体发生剂组合物，其压力指数小、燃烧速度的压力依赖性低；在专利CN117412886A中公开了一种利用隔离壁有效地将点火器与气体发生剂隔离的点火器组件，提升了燃烧室的密封性，并简化了焊接工序，减少了对焊接热量的影响。)
        英文格式：
        ### (3) Patent Applicant Technical Layout Analysis
        (在这里插入专利申请人技术布局分析报告，以下是一个范例)
        """},
        {"role": "user",
         "content": f"""
        **输入数据:**
        * **申请人背景信息:** `{company_info}`
        * **技术分布 JSON 数据:** `{company_tech_json}`
        * **专利申请人及其技术细节与代表性专利:** `{patent_miner}`
        """}]

    # 获取报告部分三
    report_part2 = await deepseek_llm.completion(prompt_applicant_tech)

    return report_part2


async def generate_full_report(save_dir=None, top_n=5, map_tech=None):
    """
    生成完整的专利申请人分析报告

    Args:
        save_dir (str): 报告保存目录
        top_n (int): 分析前几名申请人

    Returns:
        tuple: (完整报告, 报告路径)
    """
    print("开始生成专利申请人分析报告...")
    if map_tech is None:
        map_tech = MAP_TECH
    # 创建保存目录
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "detail_analysis_output")
    os.makedirs(save_dir, exist_ok=True)
    print(f"报告将保存至: {save_dir}")

    # 初始化LLM实例
    print("正在初始化LLM实例...")
    deepseek_llm = LLM(config=configs["volcengine-deepseek-chat"])
    search_llm = LLM(config=configs["glm-4"])

    # 获取申请人数据
    print("正在获取申请人数据...")
    applicant_rank, applicant_data = get_applicant_data(top_n)

    # 创建日志文件
    log_path = os.path.join(save_dir, "analysis_log.json")
    log_data = {
        "applicant_rank": applicant_rank,
        "applicant_data": applicant_data
    }

    # 如果没有数据，返回空报告
    if not applicant_data:
        print("未找到专利申请人数据")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"error": "未找到专利申请人数据"}, f, ensure_ascii=False, indent=2)
        return "未找到专利申请人数据", None

    # 转换为DataFrame
    print("正在处理申请人数据...")
    df = pd.DataFrame(applicant_data, columns=[
                      "id", "title", "abstract", "company"])
    log_data["dataframe"] = df.to_dict(orient="records")
    a = time.time()
    # 分析每个申请人的专利分类
    print("正在进行：1.分析专利分类；2.专利申请人趋势周期分析")
    # --------------舒磊------------
    patent_trend = PatentTrendAnalyzer()
    # 获取专利趋势数据
    patent_application_trend_info = patent_trend.retrieve_patent_trends_info()
    patent_application_trend_info_md = patent_application_trend_info.to_markdown(
        index=True)
    # ====================任务并发进行（包括专利申请人-技术布局分析+专利趋势阶段分析）======================================
    a = time.time()
    tech_trand_tasks = [
        analysis_classification(deepseek_llm, a, df[df["company"] == a], map_tech=map_tech) for a, _, x in applicant_rank[:top_n]
    ] + [
        patent_trend.generate_patent_application_trend_analysis(
            patent_application_trend_info_md)
        # 分析专利趋势阶段
    ]
    # 将两个任务的结果分开存储
    tech_trand_results = await asyncio.gather(*tech_trand_tasks, return_exceptions=True)
    b = time.time()
    print(f"一阶段并发消耗：{b-a}s")
    # 专利-技术分类、每个申请人技术方向、难点的挖掘
    log_data["applicant-tech-class"] = tech_trand_results[:-1]
    analysis_result = tech_trand_results[:-1]
    tech_trend_analysis = tech_trand_results[-1]
    log_data["tech_trend_analysis"] = tech_trend_analysis
    # ====================任务并发进行（包括专利申请人-技术布局分析+专利趋势阶段分析）======================================


# ===================sl===================
    # 定义阶段信息
    period_info = tech_trend_analysis['period_info']
    overall_trend = tech_trend_analysis['overall_trend']

    # 绘制趋势图
    image_path = os.path.join(save_dir, '趋势图.png')
    patent_trend.plot_patent_trends(
        patent_application_trend_info, image_path, phases=period_info)
# ===================sl===================

# ===================lzy===================

    # 提取每个公司的技术领域
    print("正在提取技术领域信息...")
    company_tech = {}
    for x in analysis_result:
        tech = set()
        for _, yy in x["分类结果"].items():
            tech.add(yy[1])
        company_tech[x["company_name"]] = list(tech)
    log_data["company_tech"] = company_tech

    # 搜索公司背景信息
    print("正在搜索公司背景信息...")
    a = time.time()
    search_tasks = [search_applicants(search_llm, company_name, domain)
                    for company_name, domain in company_tech.items()]
    search_result = await asyncio.gather(*search_tasks)
    print("公司背景信息搜索完成")
    b = time.time()
    print(f"挖掘部分耗时：{b-a}s")
    log_data["search_result"] = search_result

    # 生成可视化图表
    print("正在生成可视化图表...")
    bar_dir, heatmap_dir = visualization(
        save_dir, applicant_rank, analysis_result)
    print(f"图表已保存至: {bar_dir}, {heatmap_dir}")
    log_data["visualization"] = {
        "bar_dir": bar_dir,
        "heatmap_dir": heatmap_dir
    }

    # 转换数据为JSON格式
    print("正在转换数据格式...")
    company_tech_json, applicant_rank_json = data_to_json(
        analysis_result, applicant_rank)
    log_data["json_data"] = {
        "company_tech_json": company_tech_json,
        "applicant_rank_json": applicant_rank_json
    }

    # 提取专利挖掘结果
    patent_miner = [
        f"{a['company_name']}：{a['综合技术挖掘']}" for a in analysis_result]
    log_data["patent_miner"] = patent_miner

    # 提取公司信息
    company_info = [s[0] if isinstance(s, tuple) else s for s in search_result]
    log_data["company_info"] = company_info
# ===================lzy===================


# =========================并发（生专利-技术布局分析）+（生成每一阶段的趋势分析）=============================
    # 生成报告
    print("正在生成分析报告...")

    # 使用asyncio.gather并发执行两个报告生成任务
    a = time.time()
    # 1. 创建第一个包含固定任务的列表
    initial_tasks = [
        generate_applicant_report(
            deepseek_llm, applicant_rank_json, bar_dir, company_tech_json, heatmap_dir),
        generate_applicant_tech_report(
            deepseek_llm, company_tech_json, company_info, patent_miner)
    ]

    # 2. 创建趋势分析任务列表 (列表推导式)
    #    (确保 period_info 是一个包含字典或元组的列表)
    trend_analysis_tasks = [
        patent_trend.generate_patent_trend_part_analysis(
            pat_statistics=patent_application_trend_info_md,
            period_info=period,  # 传递当前的 period 对象
            top5_applicants_info=patent_trend.retrieve_top5_applicants_info(  # 这是同步调用
                period['start_year'], period['end_year'])  # 确保 period 中有这些键
        )
        for period in period_info
    ]

    # 3. 使用 extend 将趋势任务列表添加到初始任务列表的末尾
    generate_report_tasks = initial_tasks
    generate_report_tasks.extend(trend_analysis_tasks)

    generate_report_tasks_results = await asyncio.gather(*generate_report_tasks, return_exceptions=True)
    # 分离结果
    applicant_report_result = None
    applicant_tech_report_result = None
    trend_part_analysis_results = []

    # 检查第一个任务的结果 (generate_applicant_report)
    if len(generate_report_tasks_results) > 0:
        if isinstance(generate_report_tasks_results[0], Exception):
            print(
                f"Error in generate_applicant_report: {generate_report_tasks_results[0]}")
            # 或其他错误处理
            applicant_report_result = f"Error: {generate_report_tasks_results[0]}"
        else:
            applicant_report_result = generate_report_tasks_results[0]

    # 检查第二个任务的结果 (generate_applicant_tech_report)
    if len(generate_report_tasks_results) > 1:
        if isinstance(generate_report_tasks_results[1], Exception):
            print(
                f"Error in generate_applicant_tech_report: {generate_report_tasks_results[1]}")
            # 或其他错误处理
            applicant_tech_report_result = f"Error: {generate_report_tasks_results[1]}"
        else:
            applicant_tech_report_result = generate_report_tasks_results[1]

    # 处理趋势分析任务的结果 (从第三个结果开始)
    if len(generate_report_tasks_results) > 2:
        trend_results_raw = generate_report_tasks_results[2:]
        for i, res in enumerate(trend_results_raw):
            # 获取对应的周期描述
            period_desc = f"{period_info[i]['start_year']}-{period_info[i]['end_year']}"
            if isinstance(res, Exception):
                print(
                    f"Error in patent_trend_part_analysis for period {period_desc}: {res}")
                trend_part_analysis_results.append(
                    f"Error for period {period_desc}: {res}")
            else:
                trend_part_analysis_results.append(res)

    # # --- 使用分离后的结果 ---
    # print("\n--- Separated Results ---")
    # print("Applicant Report Result:", applicant_report_result)
    # print("Applicant Tech Report Result:", applicant_tech_report_result)
    # print("Trend Part Analysis Results:", trend_part_analysis_results)
    b = time.time()
    print(f"二阶段并发耗时：{b-a}s")
# =========================并发（生专利-技术布局分析）+（生成每一阶段的趋势分析）=============================

# ===========================sl===========================
    trand_report = patent_trend.write_analysis_to_markdown(file_path=os.path.join(save_dir, 'patent_trand_analysis.md'),
                                                           title='## （一）专利申请趋势分析',
                                                           alt_text='趋势图',
                                                           image_path='./趋势图.png',
                                                           title_text='专利申请趋势图',
                                                           overall_trend=overall_trend,
                                                           tasks=trend_part_analysis_results)

# ==================================lzy=========================================

    # 合并报告
    full_report = f"{trand_report}\n\n{applicant_report_result}\n\n{applicant_tech_report_result}"
    log_data["report"] = {
        "trand_part": trand_report,
        "tech_part1": applicant_report_result,
        "tech_part2": applicant_tech_report_result,
        "full_report": full_report
    }

    # 保存报告
    report_path = os.path.join(save_dir, "专利分析报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"报告已保存至: {report_path}")

    # 保存日志数据
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"分析日志已保存至: {log_path}")

    print("专利申请人分析报告生成完成")
    return full_report, report_path


# 主函数


async def main():
    """主函数"""
    start_time = time.time()  # 记录开始时间
    save_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "detail_analysis_output")
    # 创建基于时间的子目录
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式示例: 20231225_143022
    time_dir = os.path.join(save_dir, time_str)
    report, report_path = await generate_full_report(time_dir)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时（秒）

    print(f"报告已生成并保存至: {report_path}")
    print(f"总耗时: {elapsed_time:.2f} 秒")  # 保留2位小数
    return report, report_path


# 程序入口
if __name__ == "__main__":
    # 执行异步主函数
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        print(f"运行出错: {e}")
