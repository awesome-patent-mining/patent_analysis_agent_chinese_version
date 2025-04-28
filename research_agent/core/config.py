class Config:
    """配置类,统一管理API密钥和模型参数"""
    # API配置
    ZHIHUIYA_API_KEY = "9EmfQHAac0MyPmtx0gXseNZCkGfGf7GKFnv2NGPMyTshhKQy"
    API_KEY = "ebfb3089ec884a8b87375dc442363032.HDbADyAvg8WQ9K6P"
    RERANK_API_KEY = "09bb02e2b48f4ce7827d9c5ca72e7c3c.Ou6h24BnfkoaFDPR"
    WEB_SEARCH_API_KEY = "sk-8dc49d891f2043fe87120bb70dfa8603"
    # DEFAULT_MODEL = "deepseek-chat"                     # 默认文本生成模型
    WEB_SEARCH_TOP_N = 5
    LANGUAGE='Chinese'
    BATCHSIZE_BIBLIOGRAPHY = 100
    #DEFAULT_MODEL = "deepseek-chat"
    CPM_FOR_SIMPLE_BIBLIOGRAPHY =  10
    CPM_FOR_SEMANTIC_SEARCH = 10
    # DEFAULT_MODEL = "deepseek-chat"
    RERANK_BATCH_SIZE = 10   # 使用llm进行重排时候，设定的batch大小
    SECTION_RAG_TOP_K = 30   #撰写一个section最多用K篇文献
    #DEFAULT_MODEL = "glm-4"
    DEFAULT_MODEL = "volcengine-deepseek-chat"                     # 默认文本生成模型
    EMBEDDING_MODEL = "embedding-3"             # embedding模型
    EMBEDDING_DIMENSIONS = 2048                 # embedding维度
    COS_THRESHOLD = 0.5
    RERANK_THRESHOLD = 19
    MAX_TOKENS = 4095
    #YAML_CONFIG = 'C:\\Users\\admin\\patent_analysis_agent\\patent_analysis_agent\\research_agent\\core\\llm_config.yaml'
    #YAML_CONFIG = r"./research_agent/core/llm_config.yaml"            # yaml配置文件路径
    YAML_CONFIG = r"/root/patent_analysis_agent/patent_analysis_agent/research_agent/core/llm_config.yaml"
    # 最大token数
    # 其他配置参数
    TOP_K = 10              # query_by_content返回前多少个相关文档

    THRESHOLD = 0.5        # 重排序相似度阈值,默认0.35
    BATCH_SIZE = 64         # 生成embedding时分批处理数量

    # MYSQL_HOST = "59.110.150.237"
    # MYSQL_PORT = 3306
    # MYSQL_USERNAME = "liziyou"
    # MYSQL_PASSWORD = "[(!Admin123)]"
    # MYSQL_DB = "liziyou_db"
    # MYSQL_CHARSET = "utf8mb4"
    # IPC_DICT_PATH = r"C:\Users\admin\patent_survey_draft\ipc_dictionary_1.txt"
    MYSQL_HOST = "59.110.150.237"
    MYSQL_PORT = 3306
    MYSQL_USERNAME = "chenliang"
    MYSQL_PASSWORD = "[(!Admin123)]"
    MYSQL_DB = "chenliang_db"
    MYSQL_CHARSET = "utf8mb4"
    patent_table = "patent_info"
    language = "中文"
    # Word模版文件路径
    REFERENCE_DOC = r'D:\Users\SUYUNYI\ISTIC\ChenLiang\2025\Microsoft_agent\patent_analysis_agent-main\research_agent\core\reference.docx'
