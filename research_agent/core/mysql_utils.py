from research_agent.core.config import Config
from research_agent.core.generate_patent_chart import Patent_Chart_Generator
import pymysql
import pandas as pd

class MySQL:
    def __init__(self):
        """Initialize the QuestionProposer.

        Args:
            iteration (int, optional): Iteration number for question generation. Defaults to 0.
        """
        self.host = Config.MYSQL_HOST
        self.port = Config.MYSQL_PORT
        self.user = Config.MYSQL_USERNAME
        self.passwd = Config.MYSQL_PASSWORD
        self.database = Config.MYSQL_DB
        self.charset = Config.MYSQL_CHARSET
        self.table_name = None
        self.patent_chart = Patent_Chart_Generator()

    def execute_query_to_markdown(self,sql):
        """
        执行SQL查询并返回Markdown格式的查询结果

        参数：
        sql -- 要执行的SQL查询语句

        返回：
        Markdown格式的查询结果（字符串）
        """
        try:
            # 建立数据库连接
            connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                passwd=self.passwd,
                database=self.database,
                charset=self.charset
            )

            with connection.cursor() as cursor:
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
            if 'connection' in locals() and connection.open:
                connection.close()

    def create_table_and_import_excel_to_mysql(self, excel_file, table_name):
        """
        Create a MySQL table based on Excel headers and import Excel data into the table.
        If the table already exists, it will be dropped and recreated.
        If any text exceeds the column length limit, it will be truncated.

        Parameters:
            excel_file (str): Path to the Excel file.
            table_name (str): Name of the MySQL table to create and import data into.

        Returns:
            str: Success message or error message.
        """
        connection = None
        MAX_VARCHAR_LENGTH = 255  # Set the maximum length for VARCHAR fields

        try:
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(excel_file)

            # Connect to the MySQL database
            connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.passwd,
                database=self.database,
                charset=self.charset
            )
            cursor = connection.cursor()

            # Check if the table already exists and delete it if it does
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`;")

            # Dynamically generate table creation SQL based on DataFrame columns and data types
            column_definitions = []
            varchar_columns = []  # Keep track of VARCHAR columns for truncation later
            for column in df.columns:
                # Escape column names to handle special characters
                escaped_column = f"`{column.replace('`', '``')}`"  # Escape backticks inside column names
                # Infer SQL column type from pandas DataFrame column data types
                if pd.api.types.is_integer_dtype(df[column]):
                    column_definitions.append(f"{escaped_column} INT")
                elif pd.api.types.is_float_dtype(df[column]):
                    column_definitions.append(f"{escaped_column} FLOAT")
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    column_definitions.append(f"{escaped_column} DATETIME")
                else:
                    column_definitions.append(f"{escaped_column} VARCHAR({MAX_VARCHAR_LENGTH})")
                    varchar_columns.append(column)  # Track columns that are VARCHAR

            # Create table SQL
            create_table_query = f"""
            CREATE TABLE `{table_name}` (
                {', '.join(column_definitions)}
            );
            """

            # Execute the table creation query
            cursor.execute(create_table_query)

            # Truncate text values in DataFrame to fit VARCHAR column limits
            for column in varchar_columns:
                df[column] = df[column].astype(str).str.slice(0, MAX_VARCHAR_LENGTH)

            # Generate SQL for inserting data
            columns = [f"`{col.replace('`', '``')}`" for col in df.columns]  # Escape column names
            placeholders = ', '.join(['%s'] * len(columns))
            insert_query = f"INSERT INTO `{table_name}` ({', '.join(columns)}) VALUES ({placeholders})"

            # Insert data row by row
            for _, row in df.iterrows():
                cursor.execute(insert_query, tuple(row))

            # Commit the transaction
            connection.commit()

            return f"Table '{table_name}' created and data from '{excel_file}' has been successfully imported."

        except Exception as e:
            return f"Error occurred: {str(e)}"

        finally:
            if 'connection' in locals() and connection.open:
                connection.close()

    def get_top_ipc_yearly_patent_counts(
            self,
            country: str = "",
            applicant: str = "",
            table_name: str = "patent_info"
    ) -> str:
        """
        该函数用于分析全部专利，或者当前申请(专利权)人国家为country，或者当前申请(专利权)人为applicant的名下，每年排名前十IPC技术类别的专利数量情况，并借此分析每年的技术布局情况，并以 Markdown 格式返回专利相关信息。
        获取当前申请(专利权)人国家为country，或者当前申请(专利权)人为applicant的名下，专利数量排名前十的 IPC，然后查询这些 IPC 每年的专利申请数量，并以 Markdown 格式返回。
        如果当前申请(专利权)人国家或当前申请(专利权)人没有指定，则默认查询所有专利，并获取专利数量排名前十的 IPC，然后查询这些 IPC 每年的专利申请数量，并以 Markdown 格式返回。

        参数:
            country (str):  当前申请(专利权)人国家
            applicant (str): 当前申请(专利权)人
            table_name (str): 存储专利数据的表名称，默认为 "patent_info"

        返回:
            str: Markdown 表格形式的查询结果
        """
        # 创建数据库连接
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # 1. 获取排名前十的 IPC 主分类号
                if country != "" and applicant == "":
                    query_top_ipc = f"""
                    SELECT `IPC主分类号`, COUNT(*) AS patent_count
                    FROM `{table_name}`
                    WHERE `IPC主分类号` IS NOT NULL
                      AND `当前申请(专利权)人国家` LIKE %s
                    GROUP BY `IPC主分类号`
                    ORDER BY patent_count DESC
                    LIMIT 10
                    """
                    cursor.execute(query_top_ipc, ("%" + country + "%",))
                elif country == "" and applicant != "":
                    query_top_ipc = f"""
                    SELECT `IPC主分类号`, COUNT(*) AS patent_count
                    FROM `{table_name}`
                    WHERE `IPC主分类号` IS NOT NULL
                      AND `当前申请(专利权)人` LIKE %s
                    GROUP BY `IPC主分类号`
                    ORDER BY patent_count DESC
                    LIMIT 10
                    """
                    cursor.execute(query_top_ipc, ("%" + applicant + "%",))
                elif country == "" and applicant == "":
                    query_top_ipc = f"""
                    SELECT `IPC主分类号`, COUNT(*) AS patent_count
                    FROM `{table_name}`
                    WHERE `IPC主分类号` IS NOT NULL
                    GROUP BY `IPC主分类号`
                    ORDER BY patent_count DESC
                    LIMIT 10
                    """
                    cursor.execute(query_top_ipc)
                else:
                    raise ValueError("请勿同时指定一个国家和一个申请人，或者两者均为空，或者一者为空")
                top_ipc_results = cursor.fetchall()
                top_ipc = [row[0] for row in top_ipc_results]

                # 2. 查询这些 IPC 每年的专利申请数量
                ipc_placeholders = ",".join(["%s"] * len(top_ipc))

                if country != "" and applicant == "":
                    query_ipc_yearly = f"""
                    SELECT `申请年`, `IPC主分类号`, COUNT(*) AS patent_count
                    FROM `{table_name}`
                    WHERE `IPC主分类号` IN ({ipc_placeholders})
                      AND `当前申请(专利权)人国家` LIKE %s
                      AND `申请年` IS NOT NULL
                    GROUP BY `申请年`, `IPC主分类号`
                    ORDER BY `申请年` ASC
                    """
                    cursor.execute(query_ipc_yearly, tuple(top_ipc) + ("%" + country + "%",))
                elif country == "" and applicant != "":
                    query_ipc_yearly = f"""
                    SELECT `申请年`, `IPC主分类号`, COUNT(*) AS patent_count
                    FROM `{table_name}`
                    WHERE `IPC主分类号` IN ({ipc_placeholders})
                      AND `当前申请(专利权)人` LIKE %s
                      AND `申请年` IS NOT NULL
                    GROUP BY `申请年`, `IPC主分类号`
                    ORDER BY `申请年` ASC
                    """
                    cursor.execute(query_ipc_yearly, tuple(top_ipc) + ("%" + applicant + "%",))
                elif country == "" and applicant == "":
                    query_ipc_yearly = f"""
                    SELECT `申请年`, `IPC主分类号`, COUNT(*) AS patent_count
                    FROM `{table_name}`
                    WHERE `IPC主分类号` IN ({ipc_placeholders})
                      AND `申请年` IS NOT NULL
                    GROUP BY `申请年`, `IPC主分类号`
                    ORDER BY `申请年` ASC
                    """
                    cursor.execute(query_ipc_yearly, tuple(top_ipc))
                else:
                    raise ValueError("请勿同时指定一个国家和一个申请人，或者两者均为空，或者一者为空")

                yearly_results = cursor.fetchall()

                # 3. 数据处理：按年份整理数据，每个 IPC 做为一列
                data_by_year = {}
                for row in yearly_results:
                    year = row[0]
                    ipc = row[1]
                    count = row[2]
                    if year not in data_by_year:
                        data_by_year[year] = {ipc: count}
                    else:
                        data_by_year[year][ipc] = count

                # 4. 构造 Markdown 表格
                markdown_lines = []
                # 将top_ipc中的IPC号码换成ipc描述
                top_ipc_desc = []
                for ipc in top_ipc:
                    description = self.patent_chart.get_ipc_description(ipc)
                    top_ipc_desc.append(description)
                header = ["申请年"] + top_ipc_desc
                markdown_lines.append("| " + " | ".join(header) + " |")
                markdown_lines.append("| " + " | ".join(["------"] * len(header)) + " |")

                # 按年份生成数据行
                for year in sorted(data_by_year.keys()):
                    row = [str(year)]
                    for ipc in top_ipc:
                        count = data_by_year[year].get(ipc, 0)  # 如果某 IPC 某年没有数据，则为 0
                        row.append(str(count))
                    markdown_lines.append("| " + " | ".join(row) + " |")

                return "\n".join(markdown_lines)

        finally:
            connection.close()

    def get_top_n_countries_by_patent_count(self,
            table_name: str = "patent_info",
            top_n: int = 10
    ) -> list:
        """
        获取专利申请量排名前N的当前申请(专利权)人国家列表。
        :param table_name: 存储专利数据的表名称，默认值为 "patent_info"
        :param top_n: 返回排名前N的当前申请(专利权)人国家，默认值为10
        :return: 一个包含当前申请(专利权)人国家的列表 [当前申请(专利权)人国家]
        """
        # 创建数据库连接
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # 查询排名前N的当前申请(专利权)人及专利申请量
                query = f"""
                SELECT `当前申请(专利权)人国家`, COUNT(*) AS patent_count
                FROM `{table_name}`
                WHERE `当前申请(专利权)人国家` IS NOT NULL
                GROUP BY `当前申请(专利权)人`
                ORDER BY patent_count DESC
                LIMIT %s
                """
                cursor.execute(query, (top_n,))
                results = cursor.fetchall()

                # 返回结果列表 [申请人国家]
                return [row[0] for row in results]

        except Exception as e:
            print(f"Error occurred: {e}")
            return []

        finally:
            connection.close()
    def get_top_n_applicants_by_patent_count(self,
            table_name: str = "patent_info",
            top_n: int = 10
    ) -> list:
        """
        获取专利申请量排名前N的当前申请(专利权)人列表。
        :param table_name: 存储专利数据的表名称，默认值为 "patent_info"
        :param top_n: 返回排名前N的当前申请(专利权)人，默认值为10
        :return: 一个包含当前申请(专利权)人的列表 [申请人]
        """
        # 创建数据库连接
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # 查询排名前N的当前申请(专利权)人及专利申请量
                query = f"""
                SELECT `当前申请(专利权)人`, COUNT(*) AS patent_count
                FROM `{table_name}`
                WHERE `当前申请(专利权)人` IS NOT NULL
                GROUP BY `当前申请(专利权)人`
                ORDER BY patent_count DESC
                LIMIT %s
                """
                cursor.execute(query, (top_n,))
                results = cursor.fetchall()

                # 返回结果列表 [(申请人, 专利申请量)]
                return [row[0] for row in results]

        except Exception as e:
            print(f"Error occurred: {e}")
            return []

        finally:
            connection.close()
    def get_patent_application_number_given_countries_and_global(
            self,
            countries: list,
            table_name: str
    ) -> str:
        """
        从数据库中获取指定国家逐年的专利申请量，并以 Markdown 格式返回，每个国家作为一列，增加一列表示全球申请量。

        参数:
            countries (list): 要查询的国家列表，例如 ["CN", "US", "JP"]
            table_name (str): 专利数据所在的表名，默认为 "patent_info"

        返回:
            str: Markdown 表格形式的查询结果
        """
        # 创建数据库连接
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.database,
            charset=self.charset
        )

        try:
            with connection.cursor() as cursor:
                # 构造 SQL 查询
                placeholders = ",".join(["%s"] * len(countries))  # 防止 SQL 注入
                sql = f"""
                SELECT 
                    `申请年`, 
                    `当前申请(专利权)人国家`, 
                    COUNT(*) AS patent_count
                FROM `{table_name}`
                WHERE `当前申请(专利权)人国家` IN ({placeholders})
                  AND `申请年` IS NOT NULL
                GROUP BY `申请年`, `当前申请(专利权)人国家`
                ORDER BY `申请年` ASC, `当前申请(专利权)人国家` ASC
                """

                # 执行 SQL 查询
                cursor.execute(sql, countries)
                results = cursor.fetchall()

                # 数据处理 - 将查询结果转换为年份为键的字典
                data_by_year = {}
                for row in results:
                    year = row[0]
                    country = row[1]
                    count = row[2]
                    if year not in data_by_year:
                        data_by_year[year] = {country: count}
                    else:
                        data_by_year[year][country] = count

                # 构造 Markdown 表格
                header = ["申请年"] + countries + ["全球"]
                markdown_lines = ["| " + " | ".join(header) + " |"]
                markdown_lines.append("| " + " | ".join(["------"] * len(header)) + " |")

                # 按年份生成数据行
                for year, country_data in data_by_year.items():
                    row = [year]
                    global_total = 0
                    for country in countries:
                        count = country_data.get(country, 0)  # 如果某国家某年无数据，则为 0
                        row.append(count)
                        global_total += count
                    row.append(global_total)
                    markdown_lines.append("| " + " | ".join(map(str, row)) + " |")

                return "\n".join(markdown_lines)

        finally:
            connection.close()

if __name__ == "__main__":
    #query = "SELECT * FROM patents LIMIT 5"
    mysql = MySQL()
    #
    result = mysql.create_table_and_import_excel_to_mysql(excel_file="C:\\Users\\admin\\patent_survey_draft\\20250320012345555.XLSX", table_name="patent_table")
    print(result)
    '''
    markdown_output = execute_query_to_markdown(
        sql=query,
        host="localhost",
        user="root",
        password="your_password",
        database="your_database"
    )
    print(markdown_output)
    '''