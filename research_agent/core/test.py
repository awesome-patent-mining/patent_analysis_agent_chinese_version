import mysql.connector

# 数据库连接信息
mydb = mysql.connector.connect(
    host="59.110.150.237",
    user="wanglekang",
    password="[(!Admin123)]",
    database="wanglekang_db"
)

mycursor = mydb.cursor(dictionary=True)  # 设置 cursor 以字典形式返回结果

# 假设你获取到的专利数据存储在 patent_data 变量中（根据你的代码，实际获取数据的方式不同）
# 这里以模拟数据为例，你需要根据实际情况替换
patent_data = {
    "status": True,
    "error_code": 0,
    "data": {
        "result_count": 5,
        "total_search_result_count": 722,
        "results": [
            {"relevancy": "78%", "patent_id": "2d5713a4-e70b-429b-8453-c2d127c0f408", "pn": "CN114024326B", "apno": "CN202111311212.8", "title": "一种可用于调峰的风光制氢耦合发电和储能系统及方法", "original_assignee": "西安热工研究院有限公司", "current_assignee": "西安热工研究院有限公司", "inventor": "赵元财|徐远纲|王慧青|王国忠|赵永坚|孟勇|赵杰", "apdt": 20211108, "pbdt": 20240123},
            {"relevancy": "78%", "patent_id": "24bea1a4-718d-496f-8d7c-628a9bf765b5", "pn": "CN114024326A", "apno": "CN202111311212.8", "title": "一种可用于调峰的风光制氢耦合发电和储能系统及方法", "original_assignee": "西安热工研究院有限公司", "current_assignee": "西安热工研究院有限公司", "inventor": "赵元财|徐远纲|王慧青|王国忠|赵永坚|孟勇|赵杰", "apdt": 20211108, "pbdt": 20220208},
            {"relevancy": "77%", "patent_id": "dd3857b6-f63f-4e37-a9f6-11ea2b4ff664", "pn": "CN114123521A", "apno": "CN202111381869.1", "title": "一种针对可再生能源的电解氢与压缩二氧化碳联合储能系统", "original_assignee": "清华大学无锡应用技术研究院", "current_assignee": "清华大学无锡应用技术研究院", "inventor": "任晓栋|李雪松|周奥铮|顾春伟|胡博", "apdt": 20211122, "pbdt": 20220301},
            {"relevancy": "76%", "patent_id": "2ed632cd-c293-47cf-ae7d-01c54e64e5e0", "pn": "CN114977228A", "apno": "CN202210515966.3", "title": "一种基于液体甲醇的风光水储能系统及充放电方法", "original_assignee": "广州赛特新能源科技发展有限公司", "current_assignee": "广州赛特新能源科技发展有限公司", "inventor": "龚新金", "apdt": 20220512, "pbdt": 20220830},
            {"relevancy": "74%", "patent_id": "f8cde54a-8f3f-4c2b-9bc2-9cab8359969c", "pn": "CN108487994A", "apno": "CN201810166881.2", "title": "一种微能源网复合储能系统", "original_assignee": "中国科学院广州能源研究所", "current_assignee": "中国科学院广州能源研究所", "inventor": "林仕立|杨昌儒|宋文吉|冯自平|吕杰", "apdt": 20180228, "pbdt": 20180904}
        ]
    }
}

# 从获取到的数据中提取 patent_id 列表
patent_ids = [patent["patent_id"] for patent in patent_data["data"]["results"]]

# 构建 SQL 查询语句，根据 patent_id 进行查询
query = "SELECT * FROM patents WHERE patent_id IN (%s)" % (", ".join(["%s"] * len(patent_ids)))
mycursor.execute(query, patent_ids)

# 获取查询结果
results = mycursor.fetchall()

# 打印查询结果
for row in results:
    print(row)

mycursor.close()
mydb.close()