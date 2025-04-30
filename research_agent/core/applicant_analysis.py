#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patent Applicant Analysis Module: Used to analyze patent applicant data and generate visual reports
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

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(current_dir, parent_dir)

# Load environment variables
load_dotenv()

# Load configuration
configs = parse_config(Config.YAML_CONFIG)

def create_connection():
    """
    Create and return a database connection

    Returns:
        cursor: Database cursor object, returns None if connection fails
    """
    # Get database configuration from environment variables
    sql_host = Config.MYSQL_HOST
    sql_user = Config.MYSQL_USERNAME
    sql_password = Config.MYSQL_PASSWORD
    sql_db = Config.MYSQL_DB
    sql_charset = Config.MYSQL_CHARSET

    try:
        # Create connection
        connection = pymysql.connect(
            host=sql_host,
            user=sql_user,
            password=sql_password,
            database=sql_db,
            charset=sql_charset
        )
        print("Database connection successful!")
        return connection.cursor()
    except pymysql.InterfaceError as e:
        print(f"Database connection error: {e}")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Technology map definition
MAP_TECH = '''
### Preparation Technology
    - Single Crystal Growth Technology
    - Substrate Processing Technology
    - Epitaxial Growth and Thin Film Preparation Technology
    - Device Process
### Devices and Applications
    - Solar Blind UV Photodetector
    - Infrared Focal Plane Array
    - X-ray Detector
    - Gas Sensor
    - Power Electronic Devices
'''

def get_applicant_data(top_n=5):
    """
    Get patent applicant data

    Args:
        top_n (int): Number of top applicants to return

    Returns:
        tuple: (Applicant ranking data, Detailed patent information of applicants)
    """
    my_sql = create_connection()
    if not my_sql:
        return [], []

    # Get applicant and patent count statistics
    query = f"""
    SELECT 
        `applicant`,
        GROUP_CONCAT(
            CONCAT(`country`, '(', `patent_count_by_country`, ')') 
            SEPARATOR ', '
        ) AS `office_and_count`,
        CAST(SUM(`patent_count_by_country`) AS UNSIGNED) AS `total_patent_count`
    FROM (
        SELECT 
            `current_assignee` AS `applicant`,
            `patent_office` AS `country`,
            CAST(COUNT(*) AS UNSIGNED) AS `patent_count_by_country`
        FROM 
            `{Config.patent_table}`
        WHERE 
            `current_assignee` IS NOT NULL
        GROUP BY 
            `applicant`, `country`
    ) AS subquery
    GROUP BY 
        `applicant`
    ORDER BY 
        `total_patent_count` DESC;
    """
    print(query)
    my_sql.execute(query)
    result = my_sql.fetchall()


    # If no results, return empty lists
    if not result:
        return [], []

    # Get top N applicants
    applicants = [x for x, y, z in result[:top_n]]

    # If no applicants, return empty lists
    if not applicants:
        return result[:top_n], []

    # Build parameter placeholders
    placeholders = ','.join(['%s'] * len(applicants))

    # Get detailed patent information for these applicants
    query = f"""
    SELECT `patent_number`, `title`, `abstract`, `current_assignee`
    FROM `{Config.patent_table}`
    WHERE `current_assignee` IN ({placeholders})
    ORDER BY `current_assignee`
    """

    my_sql.execute(query, applicants)
    applicant_data = my_sql.fetchall()

    return result[:top_n], applicant_data

async def analysis_classification(deepseek_llm, company_name, data, map_tech):
    """
    Analyze and classify company patent data

    Args:
        deepseek_llm: LLM instance
        company_name (str): Company name
        data (DataFrame): DataFrame containing patent information

    Returns:
        dict: JSON data of analysis results
    """
    # Initialize result string list
    result_strings = []

    # Iterate through data to build patent information strings
    for idx, row in data.iterrows():
        # Format each patent's information using f-string
        patent_info = f"id:{row['id']};Patent Title: {row['title']};Patent Abstract: {row['abstract']}"
        result_strings.append(patent_info)

    # Build prompt content
    prompt_a = [{"role": "user",
                 "content": f"""
                **Role**: Senior semiconductor technology patent analyst, specializing in technology roadmap drawing and innovation point mining
                **Input Data**: The following {len(result_strings)} patent data items (numbered 0-{len(result_strings)}) from patent applicant {company_name}:
                {result_strings}

                **Core Tasks**:
                1. Patent Classification (ensure processing all numbers):
                    - ### for primary classification; - for secondary classification
                    - Below is the **technology map** containing specific primary and secondary classifications
                        ```markdown
                        {map_tech}
                        ```
                    - Each patent must output ["primary classification", "secondary classification"]
                    - Output primary and secondary classifications must strictly follow naming in **technology map**

                2. **Comprehensive Technology Mining** (based on all patents):
                    ■ Core Technology Directions: Extract 3-5 technology directions commonly focused on in patents
                    ■ Technical Problem Solving Analysis:
                        - What technical pain points are addressed, what solutions are proposed, and what are the effect indicators
                    ■ Representative Case Description:
                        - Select 3-5 representative patents to demonstrate how these patents solve technical problems

                **Processing Rules**:
                    1. Classification stage processes items one by one, output must verify: "Confirmed processing patents 0-{len(result_strings)}"
                    2. Technology mining must cite specific patent content, no speculation allowed
                    3. When citing patents, must specify patent id, e.g.: CN117558758A
                    4. Please output valid json format
                **Output Example**:
                    ```json
                    {{
                    "classification_results":{{"CN117558758A":["primary_classification","secondary_classification"],}}
                    "verification_status":"Number of classifications completed",
                    "comprehensive_technology_mining":"Write your **comprehensive technology mining** based on all patents here"
                    }}
                    ```
                """
                 }]

    # Call LLM to generate analysis results
    result = await deepseek_llm.completion(prompt_a)

    # Parse and repair JSON format results
    try:
        result_json = json_repair.loads(result)
        result_json["company_name"] = company_name
        return result_json
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return {"company_name": company_name, "classification_results": {}, "verification_status": "Processing failed", "comprehensive_technology_mining": ""}

async def search_applicants(search_llm, company_name, domain):
    """
    Search for company information in specific domain

    Args:
        search_llm: LLM instance
        company_name (str): Company name
        domain (str): Technology domain

    Returns:
        str: Search results
    """
    # Define search tool parameters
    tools = [{
        "type": "web_search",
        "web_search": {
            "enable": True,
            "search_engine": "search_std",
            "search_result": True,
            "search_prompt": "You are a patent analyst, please summarize key information from web search: {{search_result}} in concise language"
        }
    }]

    # Define search request
    messages = [{
        "role": "user",
        "content": f"""
        Provide a 100-word brief introduction to {company_name}'s technology layout and development in {domain}
        """
    }]

    # Execute search
    result = await search_llm.completion(messages=messages, tools=tools)
    return result

def heatmap_visualization(save_dir, analysis_result):
    """
    Generate patent technology distribution heatmap

    Args:
        save_dir (str): Save directory
        analysis_result (list): Analysis result list

    Returns:
        str: Saved file path
    """
    # Extract data from analysis results
    result_dict = {r["company_name"]: r["classification_results"] for r in analysis_result}
    companies = list(result_dict.keys())
    secondary_types = set()

    # Collect all secondary classifications
    for company, patents in result_dict.items():
        for _, types in patents.items():
            if len(types) > 1:
                secondary_types.add(f"{types[1]}")

    secondary_types = sorted(list(secondary_types))

    # If no data, return None
    if not companies or not secondary_types:
        return None

    # Create data matrix
    data = np.zeros((len(companies), len(secondary_types)))

    # Fill data
    for i, company in enumerate(companies):
        for _, types in result_dict[company].items():
            if len(types) > 1:
                j = secondary_types.index(f"{types[0]}_{types[1]}")
                data[i, j] += 1

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.Blues
    im = ax.imshow(data, cmap=cmap)

    # Add color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Patent Count", rotation=-90, va="bottom", fontsize=12)

    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(secondary_types)))
    ax.set_yticks(np.arange(len(companies)))
    ax.set_xticklabels(secondary_types)
    ax.set_yticklabels(companies)

    # Rotate X-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Add text to each cell
    for i in range(len(companies)):
        for j in range(len(secondary_types)):
            if data[i, j] > 0:  # Only show non-zero values
                text = ax.text(j, i, int(data[i, j]),
                               ha="center", va="center",
                               color="black" if data[i, j] < np.max(data)/2 else "white")

    # Add grid lines
    ax.set_xticks(np.arange(len(secondary_types)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(companies)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Set title
    ax.set_title("Patent Secondary Classification Distribution Heatmap", fontsize=16)

    # Adjust layout and save
    fig.tight_layout()
    output_path = os.path.join(save_dir, 'patent_entity_technology_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return "./patent_entity_technology_heatmap.png"

def bar_visualization(save_dir, applicant_rank):
    """
    Generate patent applicant ranking bar chart

    Args:
        save_dir (str): Save directory
        applicant_rank (list): Applicant ranking data

    Returns:
        str: Saved file path
    """
    # Check input data
    if not applicant_rank:
        return None

    # Set Chinese font
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Separate data
    names = [f"{item[0]}({item[1]})" for item in applicant_rank]
    counts = [item[2] for item in applicant_rank]

    # Draw bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(names, counts, color='skyblue')
    plt.xlabel('Application Count')
    plt.title('Patent Applicants and Their Application Counts')
    plt.gca().invert_yaxis()  # Invert y-axis to show highest count at top

    # Save to specified folder
    output_path = os.path.join(save_dir, 'patent_entity_count_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to release memory

    return "./patent_entity_count_bar.png"

def visualization(save_dir, applicant_rank, analysis_result):
    """
    Generate visualization charts

    Args:
        save_dir (str): Save directory
        applicant_rank (list): Applicant ranking data
        analysis_result (list): Analysis result list

    Returns:
        tuple: (Bar chart path, Heatmap path)
    """
    # Ensure output folder exists
    os.makedirs(save_dir, exist_ok=True)

    # Set Chinese font
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Generate bar chart and heatmap
    bar_path = bar_visualization(save_dir, applicant_rank)
    heatmap_path = heatmap_visualization(save_dir, analysis_result)

    return bar_path, heatmap_path

def data_to_json(analysis_result, applicant_rank):
    """
    Convert analysis results to JSON format

    Args:
        analysis_result (list): Analysis result list
        applicant_rank (list): Applicant ranking data

    Returns:
        tuple: (Technology distribution JSON, Applicant ranking JSON)
    """
    # Check input data
    if not analysis_result or not applicant_rank:
        return {}, "[]"

    # Extract technology classification results
    result_dict = {r["company_name"]: r["classification_results"] for r in analysis_result}

    # Create technology classification statistics dictionary
    tech_stats = defaultdict(lambda: defaultdict(int))

    # Iterate through each company
    for company, patents in result_dict.items():
        for patent_id, categories in patents.items():
            primary_cat = categories[0]  # Primary classification
            secondary_cat = categories[1]  # Secondary classification
            tech_stats[company][primary_cat] += 1
            tech_stats[company][f"{primary_cat}-{secondary_cat}"] += 1

    # Convert to DataFrame and ensure all values are Python native int
    df = pd.DataFrame.from_dict(tech_stats, orient='index')
    df = df.fillna(0).applymap(lambda x: int(x))

    # Get all secondary classifications and sort alphabetically
    secondary_categories = sorted([col for col in df.columns if '-' in col])
    df = df[secondary_categories]

    # Convert to structured JSON format
    def df_to_structured_json(df):
        result = {
            "metadata": {
                "data_type": "Company Patent Technology Distribution",
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

    # Convert applicant ranking to list of dictionaries
    json_data = [
        {
            "current_assignee": company,
            "patent_office_and_count": country,
            "total_patent_count": patent_count
        }
        for company, country, patent_count in applicant_rank
    ]

    # Output JSON
    applicant_rank_json = json.dumps(json_data, ensure_ascii=False, indent=2)
    return df_to_structured_json(df), applicant_rank_json

async def generate_applicant_report(deepseek_llm, applicant_rank_json, bar_dir, company_tech_json, heatmap_dir):
    """
    Generate patent applicant analysis report

    Args:
        deepseek_llm: LLM instance
        applicant_rank_json (str): Applicant ranking JSON
        bar_dir (str): Bar chart path
        company_tech_json (dict): Company technology distribution JSON
        heatmap_dir (str): Heatmap path

    Returns:
        str: Generated report
    """
    prompt_applicant_rank = [{"role": "user",
                              "content": f"""
                **Role:** You are a senior patent analysis expert.
                **Task:** Write a patent applicant analysis report in {Config.language} based on the following input data.
                **Input Data:**
                1.  **Patent Applicant Ranking Analysis:**
                    * **Ranking JSON Data (to analyze):** `{applicant_rank_json}`
                        * *Description:* Contains applicant and their patent count ranking.
                    * **Ranking Bar Chart Path (to integrate):** `{bar_dir}`
                        * *Description:* Points to ranking bar chart. **Insert directly as placeholder.**
                2.  **Patent Applicant-Technology Distribution Analysis:**
                    * **Technology Distribution JSON Data (to analyze):** `{company_tech_json}`
                        * *Description:* Contains patent distribution of major applicants across different technology classifications.
                    * **Technology Distribution Heatmap Path (to integrate):** `{heatmap_dir}`
                        * *Description:* Points to technology distribution heatmap. **Insert directly as placeholder.**

                2.  **Content Generation:**
                    * **Patent Applicant Analysis:** Write a paragraph summarizing applicant ranking and key findings based on `applicant_rank_json` data.
                    * **Patent Applicant Technology Distribution Analysis:** Analyze `company_tech_json` to identify technology layout characteristics of major applicants.
                    * **Insert Chart Placeholders:** Insert placeholders at appropriate positions: `![Patent Applicant Ranking Bar Chart]({bar_dir})`, `![Patent Applicant Technology Distribution Heatmap]({heatmap_dir})` with brief captions.
                Format:
                ## 2. Patent Applicant Analysis Report
                ### (1) Patent Applicant Ranking Analysis
                ### (2) Patent Applicant Technical Distribution
                """}]

    # Get report parts one and two
    report_part1 = await deepseek_llm.completion(prompt_applicant_rank)

    return report_part1

async def generate_applicant_tech_report(deepseek_llm, company_tech_json, company_info, patent_miner):
    """
    Generate applicant technology layout analysis report

    Args:
        deepseek_llm: LLM instance
        company_tech_json (dict): Company technology distribution JSON
        company_info (list): Company background information
        patent_miner (list): Patent mining results

    Returns:
        str: Generated report
    """
    prompt_applicant_tech = [
        {"role": "system",
         "content": f"""
        **Role:** You are an experienced AI patent analyst, specializing in integrating complex patent data and company information into insightful narrative reports.
        **Task Objective:** Generate a comprehensive, professional, and well-structured patent analysis report based on the provided input data. The report should highlight the applicant's technology distribution, core innovation focus, and key technical achievements.
        **Input Data and Processing Instructions:** You will receive the following structured data, please process and utilize it strictly according to the following instructions:
        **Output Language:** {Config.language}
        1.  **`company_info` (Applicant Background Information):**
            * **Content:** Provides background information about target applicants, including but not limited to their history, market position, key business areas, strategic initiatives (such as M&A), R&D philosophy, or overall mission.
            * **Your Task:** Use this data as the introduction part of the report.
                * Clearly identify target applicants.
                * Briefly introduce their background, core business, and overall positioning in the relevant industry.
                * This background information should provide necessary context for subsequent technical analysis, helping readers understand why applicants focus on certain technology areas.

        2.  **`company_tech_json` (Technology Distribution JSON Data):**
            * **Content:** Contains structured data (e.g., JSON format) of target applicants' patent distribution across different technology classifications (e.g., IPC, CPC, or custom categories).
            * **Your Task:** Conduct in-depth analysis of this data, forming the core content of the "Technology Distribution and Focus" section in the report.
                * Identify and clearly point out the main technology areas or classifications where applicants' patent activities are most concentrated and invested.
                * Describe the relative patent quantity or intensity across different technology areas, revealing the priority order and concentration of technology investment.
                * Analyze if there are significant technology focus patterns, technology shifts over time, or emerging focus areas (if data allows).
                * Combine background information from `company_info` to explain why these concentrated technology areas might be crucial for applicants' business strategy, market position, or long-term development.

        3.  **`patent_miner` (Technical Details and Representative Patents):**
            * **Content:** Contains selected insights derived from analyzing applicants' specific patents. This includes: core technology directions pursued within their main technology areas, specific technical challenges or pain points they aim to solve, innovative solutions or methods proposed in their patents, claimed effects, benefits, or performance improvements (e.g., improved efficiency, reduced costs, enhanced safety), and specific, representative patent numbers (or identifiers) used to illustrate these innovations.
            * **Your Task:** This is your **core evidence and detail source**. Use this data to build the detailed section of the report about "Innovation Focus and Key Achievements".
                * Elaborate on applicants' specific R&D focus and innovation directions within their main technology areas (identified from `company_tech_json`).
                * Clearly point out the **specific technical problems or industry pain points** they aim to solve through patents, which should be derived from descriptions in `patent_miner`.
                * Detail the **innovative solutions, technical approaches, or implementation methods** proposed in their patents, which should also be based on `patent_miner` content.
                * Explain the **claimed effects, technical advantages, or commercial benefits** brought by these innovations (e.g., performance improvement, efficiency enhancement, cost reduction, safety enhancement).
                * **Must cite representative patent numbers (or identifiers)** provided in `patent_miner` to specifically illustrate these innovation points.
                * For each cited representative patent, provide a concise analysis, clearly outlining the **problem** solved by the patent, the **solution** proposed, and the **benefits** claimed. Follow the example of JunSheng Automotive Safety System, highlighting the "problem -> solution -> benefit" logical chain.

        **Output Report Requirements:**

        1.  **Structure and Style:** Analyze each patent applicant individually, each applicant's analysis should include the following content.
            * **Introduction:** Introduce applicant and their background based on `company_info`.
            * **Part One:** Based on `company_tech_json` analysis, detail the applicant's technology distribution and main technology focus areas, and explain strategic importance in combination with `{company_info}`.
            * **Part Two:** Based on `patent_miner` detailed data, elaborate on applicant's specific innovation strategy, technical problems solved, solutions proposed, benefits claimed, and provide examples through cited representative patents (problem -> solution -> benefit).
            * **Conclusion:** Summarize applicant's overall patent layout, innovation capability, and industry position in their key technology areas.
        2.  **Content Requirements:**
            * Clearly specify the applicant being analyzed in the report.
            * Accurately present technology distribution focus based on `company_tech_json` analysis.
            * Organically integrate background information from `company_info` to provide strategic perspective for technical analysis.
            * Detail technical problems, solutions, and benefits based on `patent_miner`, and support with cited representative patents.
            * Ensure all cited representative patents have brief, structured (problem -> solution -> benefit) analysis.
            * Integrate all information into a coherent, in-depth narrative that tells the applicant's innovation story.
        3.  **Tone and Style:** Professional, analytical, objective, and informative. Avoid subjective speculation or exaggeration.
        4.  **Data Usage Principle:** **Strictly use only the provided input data (`company_tech_json`, `company_info`, `patent_miner`).** Your work is to analyze, synthesize, paraphrase, and interpret this data, extracting deep insights, not simply listing original information points.

        Format:
        ### (3) Patent Applicant Technical Layout Analysis
        (Insert patent applicant technical layout analysis report here)
        """},
        {"role": "user",
         "content": f"""
        **Input Data:**
        * **Applicant Background Information:** `{company_info}`
        * **Technology Distribution JSON Data:** `{company_tech_json}`
        * **Patent Applicant and Their Technical Details with Representative Patents:** `{patent_miner}`
        """}]

    # Get report part three
    report_part2 = await deepseek_llm.completion(prompt_applicant_tech)

    return report_part2

async def generate_full_report(save_dir=None, top_n=5, map_tech=None):
    """
    Generate complete patent applicant analysis report

    Args:
        save_dir (str): Report save directory
        top_n (int): Number of top applicants to analyze

    Returns:
        tuple: (Complete report, Report path)
    """
    print("Starting to generate patent applicant analysis report...")
    if map_tech is None:
        map_tech = MAP_TECH
    # Create save directory
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "detail_analysis_output")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Report will be saved to: {save_dir}")

    # Initialize LLM instance
    print("Initializing LLM instance...")
    deepseek_llm = LLM(config=configs["volcengine-deepseek-chat"])
    search_llm = LLM(config=configs["glm-4"])

    # Get applicant data
    print("Getting applicant data...")
    applicant_rank, applicant_data = get_applicant_data(top_n)

    # Create log file
    log_path = os.path.join(save_dir, "analysis_log.json")
    log_data = {
        "applicant_rank": applicant_rank,
        "applicant_data": applicant_data
    }

    # If no data, return empty report
    if not applicant_data:
        print("No patent applicant data found")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"error": "No patent applicant data found"}, f, ensure_ascii=False, indent=2)
        return "No patent applicant data found", None

    # Convert to DataFrame
    print("Processing applicant data...")
    df = pd.DataFrame(applicant_data, columns=[
                      "id", "title", "abstract", "company"])
    log_data["dataframe"] = df.to_dict(orient="records")
    a = time.time()
    # Analyze patent classification for each applicant
    print("In progress: 1. Analyzing patent classification; 2. Patent applicant trend period analysis")
    # --------------Shu Lei------------
    patent_trend = PatentTrendAnalyzer()
    # Get patent trend data
    patent_application_trend_info = patent_trend.retrieve_patent_trends_info()
    patent_application_trend_info_md = patent_application_trend_info.to_markdown(
        index=True)
    # ====================Concurrent tasks (including patent applicant-technology layout analysis + patent trend phase analysis)======================================
    a = time.time()
    tech_trand_tasks = [
        analysis_classification(deepseek_llm, a, df[df["company"] == a], map_tech=map_tech) for a, _, x in applicant_rank[:top_n]
    ] + [
        patent_trend.generate_patent_application_trend_analysis(
            patent_application_trend_info_md)
        # Analyze patent trend phases
    ]
    # Store results of two tasks separately
    tech_trand_results = await asyncio.gather(*tech_trand_tasks, return_exceptions=True)
    b = time.time()
    print(f"First phase concurrent consumption: {b-a}s")
    # Patent-technology classification, each applicant's technology direction, difficulty mining
    log_data["applicant-tech-class"] = tech_trand_results[:-1]
    analysis_result = tech_trand_results[:-1]
    tech_trend_analysis = tech_trand_results[-1]
    log_data["tech_trend_analysis"] = tech_trend_analysis
    # ====================Concurrent tasks (including patent applicant-technology layout analysis + patent trend phase analysis)======================================

    # Define phase information
    period_info = tech_trend_analysis['period_info']
    overall_trend = tech_trend_analysis['overall_trend']

    # Draw trend chart
    image_path = os.path.join(save_dir, 'trend_chart.png')
    patent_trend.plot_patent_trends(
        patent_application_trend_info, image_path, phases=period_info)

    # Extract technology areas for each company
    print("Extracting technology area information...")
    company_tech = {}
    for x in analysis_result:
        tech = set()
        for _, yy in x["classification_results"].items():
            tech.add(yy[1])
        company_tech[x["company_name"]] = list(tech)
    log_data["company_tech"] = company_tech

    # Search company background information
    print("Searching company background information...")
    a = time.time()
    search_tasks = [search_applicants(search_llm, company_name, domain)
                    for company_name, domain in company_tech.items()]
    search_result = await asyncio.gather(*search_tasks)
    print("Company background information search completed")
    b = time.time()
    print(f"Mining part time consumption: {b-a}s")
    log_data["search_result"] = search_result

    # Generate visualization charts
    print("Generating visualization charts...")
    bar_dir, heatmap_dir = visualization(
        save_dir, applicant_rank, analysis_result)
    print(f"Charts saved to: {bar_dir}, {heatmap_dir}")
    log_data["visualization"] = {
        "bar_dir": bar_dir,
        "heatmap_dir": heatmap_dir
    }

    # Convert data to JSON format
    print("Converting data format...")
    company_tech_json, applicant_rank_json = data_to_json(
        analysis_result, applicant_rank)
    log_data["json_data"] = {
        "company_tech_json": company_tech_json,
        "applicant_rank_json": applicant_rank_json
    }

    # Extract patent mining results
    patent_miner = [
        f"{a['company_name']}: {a['comprehensive_technology_mining']}" for a in analysis_result]
    log_data["patent_miner"] = patent_miner

    # Extract company information
    company_info = [s[0] if isinstance(s, tuple) else s for s in search_result]
    log_data["company_info"] = company_info

    # Generate report
    print("Generating analysis report...")

    # Use asyncio.gather to concurrently execute two report generation tasks
    a = time.time()
    # 1. Create first list containing fixed tasks
    initial_tasks = [
        generate_applicant_report(
            deepseek_llm, applicant_rank_json, bar_dir, company_tech_json, heatmap_dir),
        generate_applicant_tech_report(
            deepseek_llm, company_tech_json, company_info, patent_miner)
    ]

    # 2. Create trend analysis task list (list comprehension)
    #    (Ensure period_info is a list containing dictionaries or tuples)
    trend_analysis_tasks = [
        patent_trend.generate_patent_trend_part_analysis(
            pat_statistics=patent_application_trend_info_md,
            period_info=period,  # Pass current period object
            top5_applicants_info=patent_trend.retrieve_top5_applicants_info(  # This is synchronous call
                period['start_year'], period['end_year'])  # Ensure period has these keys
        )
        for period in period_info
    ]

    # 3. Use extend to add trend task list to end of initial task list
    generate_report_tasks = initial_tasks
    generate_report_tasks.extend(trend_analysis_tasks)

    generate_report_tasks_results = await asyncio.gather(*generate_report_tasks, return_exceptions=True)
    # Separate results
    applicant_report_result = None
    applicant_tech_report_result = None
    trend_part_analysis_results = []

    # Check first task result (generate_applicant_report)
    if len(generate_report_tasks_results) > 0:
        if isinstance(generate_report_tasks_results[0], Exception):
            print(
                f"Error in generate_applicant_report: {generate_report_tasks_results[0]}")
            # Or other error handling
            applicant_report_result = f"Error: {generate_report_tasks_results[0]}"
        else:
            applicant_report_result = generate_report_tasks_results[0]

    # Check second task result (generate_applicant_tech_report)
    if len(generate_report_tasks_results) > 1:
        if isinstance(generate_report_tasks_results[1], Exception):
            print(
                f"Error in generate_applicant_tech_report: {generate_report_tasks_results[1]}")
            # Or other error handling
            applicant_tech_report_result = f"Error: {generate_report_tasks_results[1]}"
        else:
            applicant_tech_report_result = generate_report_tasks_results[1]

    # Process trend analysis task results (from third result)
    if len(generate_report_tasks_results) > 2:
        trend_results_raw = generate_report_tasks_results[2:]
        for i, res in enumerate(trend_results_raw):
            # Get corresponding period description
            period_desc = f"{period_info[i]['start_year']}-{period_info[i]['end_year']}"
            if isinstance(res, Exception):
                print(
                    f"Error in patent_trend_part_analysis for period {period_desc}: {res}")
                trend_part_analysis_results.append(
                    f"Error for period {period_desc}: {res}")
            else:
                trend_part_analysis_results.append(res)

    b = time.time()
    print(f"Second phase concurrent consumption: {b-a}s")

    trand_report = patent_trend.write_analysis_to_markdown(file_path=os.path.join(save_dir, 'patent_trend_analysis.md'),
                                                           title='## (1) Patent Application Trend Analysis',
                                                           alt_text='Trend Chart',
                                                           image_path='./trend_chart.png',
                                                           title_text='Patent Application Trend Chart',
                                                           overall_trend=overall_trend,
                                                           tasks=trend_part_analysis_results)

    # Merge reports
    full_report = f"{trand_report}\n\n{applicant_report_result}\n\n{applicant_tech_report_result}"
    log_data["report"] = {
        "trand_part": trand_report,
        "tech_part1": applicant_report_result,
        "tech_part2": applicant_tech_report_result,
        "full_report": full_report
    }

    # Save report
    report_path = os.path.join(save_dir, "patent_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Report saved to: {report_path}")

    # Save log data
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"Analysis log saved to: {log_path}")

    print("Patent applicant analysis report generation completed")
    return full_report, report_path

async def main():
    """Main function"""
    start_time = time.time()  # Record start time
    save_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "detail_analysis_output")
    # Create time-based subdirectory
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format example: 20231225_143022
    time_dir = os.path.join(save_dir, time_str)
    report, report_path = await generate_full_report(time_dir)
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate time consumption (seconds)

    print(f"Report generated and saved to: {report_path}")
    print(f"Total time consumption: {elapsed_time:.2f} seconds")  # Keep 2 decimal places
    return report, report_path

# Program entry
if __name__ == "__main__":
    # Execute asynchronous main function
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        print(f"Runtime error: {e}")
