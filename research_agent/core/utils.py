import re
from typing import List, Union
import tiktoken
import mysql.connector
from nltk.tokenize import sent_tokenize


class TokenCounter:
    """
    num_tokens_from_string 方法，计算单个字符串的 token 数量
    num_tokens_from_list_string 方法，计算字符串列表的总 token 数量
    text_truncation 方法，支持按最大长度截断文本

    """

    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def num_tokens_from_string(self, string: str) -> int:
        """
        计算单个字符串的 token 数量
        """
        try:
            return len(self.encoding.encode(string, disallowed_special=()))
        except Exception as e:
            # 如果出现编码错误，尝试清理文本
            cleaned_string = string.replace('<|endoftext|>', '')
            return len(self.encoding.encode(cleaned_string, disallowed_special=()))

    def num_tokens_from_list_string(self, list_of_string: List[str]) -> int:
        """
        计算字符串列表的总 token 数量
        """
        return sum(self.num_tokens_from_string(s) for s in list_of_string)

    def text_truncation(self, text: str, max_len: int = 1000) -> str:
        """
        按最大长度(token长度)截断文本
        """
        encoded_id = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(encoded_id[:min(max_len, len(encoded_id))])


def tokenize_sentences(text: str) -> str:
    """
    对输入文本进行句子分割，并格式化为包含句号索引的字符串。

    :param text: 原始文本
    :return: 格式化后的字符串，每个句子均以 "sen_id:{索引}" 和 "sentence_text:{句子内容}" 标识
    """
    sentences = sent_tokenize(text)
    return "\n".join(f"sen_id:{sen_id}\nsentence_text:{sentence.replace(chr(10), ' ')}" for sen_id, sentence in enumerate(sentences))


def chunking(chunk_text_list: Union[str, List[str]], reduce_length: int = 0, max_length: int = 3000) -> List[str]:
    """
    将文本分块，确保每个块的 token 数量不超过最大长度。

    :param chunk_text_list: 输入文本或文本列表
    :param reduce_length: 减少的长度
    :param max_length: 最大长度
    :return: 分块后的文本列表
    """
    token_counter = TokenCounter()
    truncation_length = 2500 - reduce_length
    chunk_list = []

    if isinstance(chunk_text_list, str):
        chunk_text_list = [chunk_text_list]

    for chunk_text in chunk_text_list:
        chunk_tokens = token_counter.num_tokens_from_string(chunk_text)
        if chunk_tokens > max_length:
            chunk_text = re.sub("<html>.*?</html>", "",
                                chunk_text, flags=re.DOTALL)
            chunk_tokens = token_counter.num_tokens_from_string(chunk_text)
            if chunk_tokens > max_length:
                chunk_text = token_counter.text_truncation(
                    chunk_text, truncation_length)
        chunk_list.append(chunk_text)

    return chunk_list
def flatten_tech_structure_zh(data):
    result = {"一级技术": [], "二级技术": []}
    for item in data:
        first_level = item["一级技术"][0] if item["一级技术"] else ""
        second_levels = item["二级技术"]
        # 一级技术出现一次，后面空补齐与二级长度相同
        result["一级技术"].extend([first_level] + [""] * (len(second_levels) - 1))
        result["二级技术"].extend(second_levels)
    return result

def flatten_tech_structure_en(data):
    print(data)
    result = {"Primary Technology": [], "Secondary Technology": []}
    for item in data:
        first_level = item["Primary Technology"][0] if item["Primary Technology"] else ""
        second_levels = item["Secondary Technology"]
        # 一级技术出现一次，后面空补齐与二级长度相同
        result["Primary Technology"].extend([first_level] + [""] * (len(second_levels) - 1))
        result["Secondary Technology"].extend(second_levels)
    return result

def transform_data_zh(input_data):
    first_level = []  # 存储处理后的"一级技术"列表
    second_level = []  # 存储处理后的"二级技术"列表

    for block in input_data:
        # 提取当前块的一级技术（假设每个块只有一个一级技术）
        current_first = block["一级技术"][0]
        # 提取当前块的二级技术列表
        current_seconds = block["二级技术"]

        # 记录当前块处理前的二级技术列表长度（即起始索引）
        start_idx = len(second_level)
        # 将当前块的二级技术添加到总列表
        second_level.extend(current_seconds)

        # 扩展一级技术列表长度以匹配二级技术列表，填充空字符串
        while len(first_level) < len(second_level):
            first_level.append("")
        # 在当前块的第一个二级技术位置标记一级技术
        first_level[start_idx] = current_first

    return {
        "一级技术": first_level,
        "二级技术": second_level
    }


def insert_patent_to_db(patent_data):
    """将专利数据插入到 MySQL 数据库中"""
    try:
        mydb = mysql.connector.connect(
            host="59.110.150.237",
            user="wanglekang",
            password="[(!Admin123)]",
            database="wanglekang_db"
        )

        mycursor = mydb.cursor()

        # 检查 patent_data 中是否包含 'data' 字段，以及 'data' 中是否包含 'results' 字段
        if isinstance(patent_data, dict) and 'data' in patent_data and 'results' in patent_data['data']:
            results = patent_data['data']['results']
            for patent in results:
                patent_id = patent.get('patent_id', '')
                title = patent.get('title', '')
                abstract = ''  # 响应中没有 'abstract' 字段，可根据实际情况处理
                assignee = patent.get('current_assignee', '')
                application_date = patent.get('apdt', None)
                publication_date = patent.get('pbdt', None)

                # 将日期格式从 YYYYMMDD 转换为 YYYY-MM-DD
                if application_date:
                    application_date = str(application_date)
                    application_date = f"{application_date[:4]}-{application_date[4:6]}-{application_date[6:]}"
                if publication_date:
                    publication_date = str(publication_date)
                    publication_date = f"{publication_date[:4]}-{publication_date[4:6]}-{publication_date[6:]}"

                sql = "INSERT INTO patents (patent_id, title, abstract, assignee, application_date, publication_date) VALUES (%s, %s, %s, %s, %s, %s)"
                val = (patent_id, title, abstract, assignee, application_date, publication_date)
                mycursor.execute(sql, val)
        else:
            print("Invalid patent data structure. Skipping...")

        mydb.commit()
        mycursor.close()
        mydb.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    # token_counter = TokenCounter()
    # print(token_counter.num_tokens_from_string("tiktoken is great!"))
    # print(token_counter.num_tokens_from_list_string(
    #     ["tiktoken is great!", "tiktoken is great!"]))
    # print(token_counter.text_truncation("tiktoken is great!", 5))
    print(chunking("tiktoken is great!aaaaaa", 0, 3000))
