from mysql import connector
from typing import List, Dict
import logging
from research_agent.core.config import Config


class PatentDatabase:
    """专利数据存储类，用于将专利数据保存到MySQL数据库"""

    def __init__(self):
        """初始化数据库连接"""
        self.host = Config.MYSQL_HOST
        self.user = Config.MYSQL_USERNAME
        self.password = Config.MYSQL_PASSWORD
        self.database = Config.MYSQL_DB
        self.connection = None
        self.cursor = None

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                ssl_disabled=True
            )
            self.cursor = self.connection.cursor()
            self.logger.info("成功连接到MySQL数据库")
        except connector.Error as err:
            self.logger.error(f"数据库连接失败: {err}")
            raise

    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            if self.cursor:
                self.cursor.close()
            self.connection.close()
            self.logger.info("数据库连接已关闭")

    def create_patents_table(self):
        """创建专利数据表（如果不存在）"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS patents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            patent_id VARCHAR(36) NOT NULL UNIQUE,
            patent_number VARCHAR(20),
            application_number VARCHAR(30),
            title TEXT,
            original_assignee TEXT,
            current_assignee TEXT,
            inventor TEXT,
            application_year INT,
            publication_year INT,
            abstract LONGTEXT,
            ipc VARCHAR(50),
            patent_office VARCHAR(10),
            app_country VARCHAR(10),
            relevancy VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("专利表创建成功或已存在")
        except connector.Error as err:
            self.logger.error(f"创建表失败: {err}")
            raise

    def insert_patent(self, patent_data: Dict):
        """插入单条专利数据到数据库"""
        insert_query = """
        INSERT INTO patents (
            patent_id, patent_number, application_number, title, 
            original_assignee, current_assignee, inventor, 
            application_date, publication_date, abstract, 
            ipc, patent_office, relevancy
        ) VALUES (
            %(patent_id)s, %(pn)s, %(apno)s, %(title)s, 
            %(original_assignee)s, %(current_assignee)s, %(inventor)s, 
            %(apdt)s, %(pbdt)s, %(abstract)s, 
            %(ipc)s, %(patent_office)s, %(relevancy)s
        ) ON DUPLICATE KEY UPDATE
            patent_number = VALUES(patent_number),
            application_number = VALUES(application_number),
            title = VALUES(title),
            original_assignee = VALUES(original_assignee),
            current_assignee = VALUES(current_assignee),
            inventor = VALUES(inventor),
            application_date = VALUES(application_date),
            publication_date = VALUES(publication_date),
            abstract = VALUES(abstract),
            ipc = VALUES(ipc),
            patent_office = VALUES(patent_office),
            relevancy = VALUES(relevancy)
        """
        try:
            self.cursor.execute(insert_query, patent_data)
            self.connection.commit()
            self.logger.info(f"成功插入/更新专利: {patent_data.get('patent_id')}")
        except connector.Error as err:
            self.logger.error(f"插入专利数据失败: {err}")
            raise

    def insert_patents_batch(self, patents: list):
        data_to_insert = []
        for patent in patents:
            # 过滤掉 patent_id 为空或全空格或None的条目
            pid = patent.get('patent_id')
            if not pid or str(pid).strip() == "":
                continue
            patent_tuple = (
                pid,
                patent.get('pn'),
                patent.get('apno'),
                patent.get('title'),
                patent.get('original_assignee'),
                patent.get('current_assignee'),
                patent.get('inventor'),
                patent.get('apdt'),
                patent.get('pbdt'),
                patent.get('abstract'),
                patent.get('ipc'),
                patent.get('patent_office'),
                patent.get('relevancy')
            )
            data_to_insert.append(patent_tuple)
        if not data_to_insert:
            print("本批次没有有效的 patent_id，不进行插入。")
            return
        insert_query = """
        INSERT INTO patents (
            patent_id, patent_number, application_number, title,
            original_assignee, current_assignee, inventor,
            application_date, publication_date, abstract,
            ipc, patent_office, relevancy
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            patent_number = VALUES(patent_number),
            application_number = VALUES(application_number),
            title = VALUES(title),
            original_assignee = VALUES(original_assignee),
            current_assignee = VALUES(current_assignee),
            inventor = VALUES(inventor),
            application_date = VALUES(application_date),
            publication_date = VALUES(publication_date),
            abstract = VALUES(abstract),
            ipc = VALUES(ipc),
            patent_office = VALUES(patent_office),
            relevancy = VALUES(relevancy)
        """
        self.cursor.executemany(insert_query, data_to_insert)
        self.connection.commit()

    def patent_exists(self, patent_id: str) -> bool:
        """检查专利是否已存在"""
        query = "SELECT 1 FROM patents WHERE patent_id = %s LIMIT 1"
        try:
            self.cursor.execute(query, (patent_id,))
            return self.cursor.fetchone() is not None
        except connector.Error as err:
            self.logger.error(f"检查专利存在性失败: {err}")
            raise


# 使用示例
if __name__ == "__main__":
    # 示例数据（实际使用时从Query类获取）
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
