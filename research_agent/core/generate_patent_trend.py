import os
import sys
sys.path.append(os.path.abspath('../..')) 

import asyncio
import pymysql
from research_agent.core.general_llm import LLM
from research_agent.core.config import Config
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import json_repair
from pathlib import Path
from pyaml_env import parse_config
from jinja2 import Environment
import matplotlib.dates as mdates
import pandas as pd
import os

# nest_asyncio.apply()
# sys.path.append(os.path.abspath('../..'))

plt.rcParams['font.sans-serif'] = ['SimHei']  # Set global font
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

    def retrieve_patent_trends_info(self, country_col='app_country',
                                    year_col='application_date', target_countries=['CN', 'US']):
        """
        Retrieve annual patent application trends for specified countries and globally from the database.

        Args:
            country_col: Country field column name (default: 'app_country')
            year_col: Application year column name (default: 'application_date')
            target_countries: List of countries to analyze (default: a, US)

        Returns:
            DataFrame containing three columns (year, country count, global count)
        """
        # Create database connection
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # Get data for all years
                query = f"""
                SELECT `{year_col}`, `{country_col}`, COUNT(*) as count
                FROM {self.table_name}
                WHERE `{year_col}` IS NOT NULL
                GROUP BY `{year_col}`, `{country_col}`
                """
                cursor.execute(query)
                results = cursor.fetchall()

                # Convert results to DataFrame
                df = pd.DataFrame(results, columns=[
                                  year_col, country_col, 'count'])

                # Preprocess year data
                df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
                df = df.dropna(subset=[year_col]).astype({year_col: 'int'})

                # Country dimension statistics
                country_counts = (
                    df[df[country_col].isin(target_countries)]
                    .pivot_table(index=year_col, columns=country_col, values='count', fill_value=0)
                )

                # Global dimension statistics
                global_counts = df.groupby(year_col)['count'].sum()

                # Merge results
                result = country_counts.join(
                    global_counts.rename('Global'), how='outer')
                return result.fillna(0).astype(int)

        finally:
            connection.close()

    @staticmethod
    def plot_patent_trends(patent_stat_df, file_dir, figsize=(14, 7), phases=None):
        """
        Plot patent application trend comparison chart.

        Args:
            patent_stat_df: DataFrame containing year index and CN/US/Global columns
            figsize: Chart size, default (14,7)
            phases: List of phases, each containing period (phase name), start_year, end_year,
                   and description, format:
                   [{'period': 'Phase 1', 'start_year': 2000, 'end_year': 2010, 'description': '...'},
                    {'period': 'Phase 2', 'start_year': 2010, 'end_year': 2020, 'description': '...'}]
        """
        plt.figure(figsize=figsize)

        # Convert index to date format
        years = patent_stat_df.index.astype(str)
        x = pd.to_datetime(years, format='%Y')

        # Plot trend lines
        plt.plot(x, patent_stat_df['CN'], 'r-o',
                 linewidth=2, markersize=6, label='China')
        plt.plot(x, patent_stat_df['US'], 'b--s',
                 linewidth=2, markersize=6, label='USA')
        plt.plot(x, patent_stat_df['Global'], 'g-.^',
                 linewidth=2, markersize=6, label='Global')

        # Axis format settings
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator(5))  # 5-year intervals
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        # Set y-axis range
        y_max = patent_stat_df.max().max() * 1.3
        plt.ylim(0, y_max)

        # Add development phase dividers and labels
        if phases:
            # Calculate width of each phase for precise label positioning
            start_year = pd.to_datetime(
                str(patent_stat_df.index[0]), format='%Y')
            end_year = pd.to_datetime(
                str(patent_stat_df.index[-1]), format='%Y')

            for phase in phases:
                phase_start = pd.to_datetime(
                    str(phase['start_year']), format='%Y')
                phase_end = pd.to_datetime(str(phase['end_year']), format='%Y')
                phase_mid = phase_start + (phase_end - phase_start) / 2
                
                # Calculate phase width in years
                phase_width = (phase_end - phase_start).days / 365.25
                
                # Calculate base font size (adjust these values as needed)
                base_font_size = 16
                min_font_size = 10
                max_font_size = 16
                
                # Calculate dynamic font size based on phase width
                # Longer phases get larger font, shorter phases get smaller font
                font_size = min(max_font_size, max(min_font_size, base_font_size * (phase_width / 5)))
                
                # Add vertical lines
                plt.axvline(x=phase_start, color='gray',
                            linestyle='--', alpha=0.7)
                plt.axvline(x=phase_end, color='gray',
                            linestyle='--', alpha=0.7)

                # Add phase labels with dynamic font size
                plt.text(phase_mid, y_max * 0.9, phase['period'],
                         ha='center', va='center', fontsize=font_size)

        # Chart elements
        plt.title(' Patent Application Trend Comparison', fontsize=14, pad=20)
        plt.xlabel('Application Year', fontsize=12)
        plt.ylabel('Number of Patent Applications', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Modify legend position
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0, 1))

        # Value labels
        for col in ['CN', 'US', 'Global']:
            for x_val, y_val in zip(x, patent_stat_df[col]):
                if y_val > 0:  # Only label non-zero values
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
        Generate part of the patent trend analysis, focusing on technical development trends in a specific period.

        Args:
            pat_statistics (str): Patent application statistics data in markdown format
            period_info (dict): Trend phase information containing phase name, start year, end year, and description
            top5_applicants_info (list): Patent application information of 5 important applicants in this phase

        Returns:
            str: Analysis result text in JSON format

        Raises:
            RuntimeError: If prompt template loading fails
            json_repair.JSONDecodeError: If response parsing fails
        """
        try:
            # Load prompt template
            base_path = Path(__file__).parent / "prompts"
            prompt_file = base_path / "1_generate_patent_trend_part.jinja"
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.generate_patent_application_trend_prompt = Environment().from_string(f.read())
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Failed to load prompt template: {str(e)}")

        # Format all inputs into a single string
        combined_stats = f"""
                            Patent Statistics:
                            {pat_statistics}

                            Period Info:
                            {period_info}

                            Top 5 Applicants Info:
                            {top5_applicants_info}
                            """

        # Generate completion
        prompt_messages = self._prepare_prompts(
            patent_stat=combined_stats
        )
        response = await self.llm.completion(prompt_messages)

        # Parse and return result
        result = json_repair.loads(response)
        return result

    def retrieve_top5_applicants_info(self, start_year: int, end_year: int, table_name: str = "patent_info") -> list:
        """
        Query top 5 current patent applicants and their patent information within specified year range.

        Args:
            start_year: Start year
            end_year: End year
            table_name: Database table name (default: patent_info)

        Returns:
            list: List containing top 5 applicants and their patent information
        """
        # Create database connection
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # Query top 5 applicants
                query_top5 = f"""
                SELECT `current_assignee`, COUNT(*) AS patent_count
                FROM `{self.table_name}`
                WHERE `current_assignee` IS NOT NULL
                  AND `application_date` BETWEEN %s AND %s
                GROUP BY `current_assignee`
                ORDER BY patent_count DESC
                LIMIT 5
                """
                cursor.execute(query_top5, (start_year, end_year))
                top5_applicants = cursor.fetchall()

                # Query patent information for each applicant
                result = []
                for applicant, _ in top5_applicants:
                    query_patents = f"""
                    SELECT `patent_number`, `title`, `abstract`, `application_date`, `app_country`
                    FROM `{table_name}`
                    WHERE `current_assignee` = %s
                      AND `application_date` BETWEEN %s AND %s
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
                            "application_date": patent[3],
                            "applicant_country": patent[4]
                        } for patent in patents]
                    })

                return result

        finally:
            connection.close()

    async def generate_patent_trend_part_analysis_concurrent(self, pat_statistics, period_info):
        """
        Execute generate_patent_trend_part_analysis function concurrently for each element in period_info list.

        Args:
            pat_statistics (str): Patent application statistics data in markdown format
            period_info (list): List of trend phase information, each containing phase name, start year, end year, and description

        Returns:
            list: List containing results of all concurrent tasks
        """
        # Create concurrent task list
        tasks = [
            self.generate_patent_trend_part_analysis(
                pat_statistics=pat_statistics,
                period_info=period,
                top5_applicants_info=self.retrieve_top5_applicants_info(
                    period['start_year'], period['end_year'])
            )
            for period in period_info
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        return results

    def write_analysis_to_markdown(self, file_path, title, alt_text, image_path, title_text, overall_trend, tasks):
        """
        Write patent trend analysis results to Markdown file.

        Args:
            file_path (str): File path
            title (str): File title
            alt_text (str): Image alternative text
            image_path (str): Image path
            title_text (str): Image title
            overall_trend (str): Overall trend analysis
            tasks (list): List containing analysis results for each phase
        """
        trend_report = ""
        with open(file_path, 'w', encoding='utf-8') as file:
            markdown_image = f'![{alt_text}]({image_path} "{title_text}")\n'
            file.write(title + '\n')
            file.write(markdown_image + '\n')
            file.write(overall_trend + '\n')
            a = title + "\n" + markdown_image + "\n" + overall_trend + '\n'
            trend_report += a
            for i, period in enumerate(tasks, 1):
                # Add two newlines to ensure blank lines between paragraphs
                file.write(f'### ({i}){period["period_title"]}\n\n')
                file.write(period['country_compare'] + '\n\n')  # Add two newlines
                file.write(period['company_compare'] + '\n\n')  # Add two newlines
                b = f'### ({i}){period["period_title"]}\n\n' + \
                    period['country_compare'] + '\n\n' + \
                    period['company_compare'] + '\n\n'
                trend_report += b
        return trend_report

    async def main(self):
        """
        Main method to execute patent trend analysis process and write results to Markdown file.
        """
        start_time = time.time()

        # Get current time and format as folder name
        folder_name = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('./general_analysis_output', folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get patent trend data
        patent_application_trend_info = self.retrieve_patent_trends_info()
        patent_application_trend_info_md = patent_application_trend_info.to_markdown(
            index=True)

        tech_trend_analysis = await self.generate_patent_application_trend_analysis(patent_application_trend_info_md)
        # Define phase information
        period_info = tech_trend_analysis['period_info']
        overall_trend = tech_trend_analysis['overall_trend']

        # Plot trend chart
        image_path = os.path.join(output_dir, 'trend_chart.png')
        self.plot_patent_trends(
            patent_application_trend_info, image_path, phases=period_info)

        # Execute analysis tasks concurrently
        results = await self.generate_patent_trend_part_analysis_concurrent(
            pat_statistics=patent_application_trend_info_md,
            period_info=period_info)

        # Write analysis results to Markdown file
        self.write_analysis_to_markdown(
            file_path=os.path.join(output_dir, 'patent_analysis.md'),
            title='## (1) Patent Application Trend Analysis',
            alt_text='Trend Chart',
            image_path='./trend_chart.png',
            title_text='Patent Application Trend Chart',
            overall_trend=overall_trend,
            tasks=results
        )

        end_time = time.time()
        print(f"Program execution time: {end_time - start_time} seconds")


# Execute main method if module is run directly
if __name__ == "__main__":
    patent_trend = PatentTrendAnalyzer()
    asyncio.run(patent_trend.main())
