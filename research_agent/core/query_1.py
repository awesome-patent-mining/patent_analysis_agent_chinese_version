import requests
import time
import asyncio
import aiohttp
import mysql.connector
from aiolimiter import AsyncLimiter
from research_agent.core.config import Config
from typing import Dict, List, Optional,Union

import asyncio
import time
from collections import deque

class RateLimiter:
    """速率限制器：限制在规定时间内的函数调用次数"""
    def __init__(self, max_calls: int, period: int):
        """
        初始化
        :param max_calls: 最大调用次数
        :param period: 时间窗口（秒）
        """
        self.max_calls = max_calls
        self.period = period
        self.call_times = deque()  # 用来记录最近调用的时间戳

    async def acquire(self):
        """
        获取调用权限，超过速率限制时等待
        """
        now = time.time()
        while self.call_times and (now - self.call_times[0] > self.period):
            # 移除超出时间窗口的调用记录
            self.call_times.popleft()

        if len(self.call_times) >= self.max_calls:
            # 如果达到最大调用次数，等待直到可以调用
            sleep_time = self.period - (now - self.call_times[0])
            await asyncio.sleep(sleep_time)

        self.call_times.append(time.time())
class TokenManager:
    def __init__(self):
        self.token = None
        self.last_refresh_time = 0
        self.expires_in = 1800  # 30分钟（单位：秒）

    def get_token(self):
        # 如果 token 不存在或已过期，则刷新
        if not self.token or (time.time() - self.last_refresh_time) >= self.expires_in:
            self.refresh_token()
        return self.token

    def refresh_token(self):
        url = "https://9EmfQHAac0MyPmtx0gXseNZCkGfGf7GKFnv2NGPMyTshhKQy:BENnYXk3O15ExmcGG0opU10dWZWAW1KT01oqzXEcGNX7mHyoqgzxLqK6ry7q996d@connect.zhihuiya.com/oauth/token"
        payload = "grant_type=client_credentials"
        headers = {"content-type": "application/x-www-form-urlencoded"}

        response = requests.post(url, data=payload, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response text: {response.text}")
        if response.status_code == 200:
            data = response.json()
            self.token = data["data"].get("token")
            self.last_refresh_time = time.time()
            self.expires_in = int(data["data"].get("expires_in", 1800))  # 转换为整数类型
        else:
            raise Exception(f"Failed to refresh token: {response.text}")


class Query:
    def __init__(self, *args, **kwargs):
        """Initialize the Query instance with Zhihuiya API credentials.

        Args:
            api_key (str): The API key for Zhihuiya
            token_manager (TokenManager): The token manager for Zhihuiya
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.semantic_search_url = "https://connect.zhihuiya.com/search/patent/semantic-search-patent/v2"
        self.current_search_url = "https://connect.zhihuiya.com/search/patent/current-search-patent/v2"
        self.company_search_url = "https://connect.zhihuiya.com/search/patent/company-search-patent/v2"
        self.api_key = Config.ZHIHUIYA_API_KEY
        self.token_manager = TokenManager()
        self.rateLimiter_for_simple_bibliography = RateLimiter(Config.CPM_FOR_SIMPLE_BIBLIOGRAPHY,60)
        self.rateLimiter_for_query_content = RateLimiter(Config.CPM_FOR_SIMPLE_BIBLIOGRAPHY,60)
        self.default_countries = []  # Default to Chinese patents
        self.default_relevancy = "50%"

    async def _make_request(self, url, payload: Dict, max_retries: int = 3) -> List[Dict]:
        """Make a request to the Zhihuiya API with retry logic for rate limiting.

        Args:
            url (str): The API URL
            payload (Dict): The request payload
            max_retries (int): The maximum number of retries when hitting rate limit

        Returns:
            List[Dict]: The response data or empty list if request fails
        """
        params = {"apikey": self.api_key}
        token = self.token_manager.get_token()
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}"
        }

        retry_count = 0
        while retry_count <= max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            url,
                            params=params,
                            json=payload,
                            headers=headers
                    ) as response:
                        response.raise_for_status()
                        api_response = await response.json()
                        print(f"API Response: {api_response}")  # Print API response
                        error_code = api_response.get("error_code", None)
                        if error_code == 0:
                            return api_response
                        elif error_code == 67200002:
                            retry_count += 1
                            if retry_count > max_retries:
                                print(f"Reached max retries ({max_retries}) for rate limit.")
                                return []
                            print(
                                f"API rate limit hit (error_code 67200002), sleeping for 10 seconds (attempt {retry_count}/{max_retries})...")
                            await asyncio.sleep(10)
                            continue  # Retry
                        else:
                            print(f"API returned error_code: {error_code}")
                            return []
            except Exception as e:
                print(f"Error making request: {e}")
                return []
        return []
    async def batch_query_simple_bibliography(self,queue: deque):
        """
        异步处理队列数据，规定每次最多处理 batch_size 个元素，且一分钟内最多调用 call_per_min 次
        :param queue: 队列，包含需要处理的 m 个元素
        """
        results=[]
        while queue:
            # 从队列中取出最多 b 个元素
            batch = [queue.popleft() for _ in range(min(Config.BATCHSIZE_BIBLIOGRAPHY, len(queue)))]

            # 获取调用权限（满足速率限制）
            await self.rateLimiter_for_simple_bibliography.acquire()

            # 调用异步处理函数
            patent_id_batch_string = ','.join([f"{item}" for item in batch])
            print(f"未添加摘要、IPC和受理局的专利数还剩{len(queue)}条!")
            processed_batch = await self.get_simple_bibliography_async(patent_id_batch_string)
            for patent_item in processed_batch['data']:
                patent_id = patent_item['patent_id']
                abstract_package = patent_item['bibliographic_data'].get('abstracts', [])
                if abstract_package!=[]:
                    abstract = abstract_package[0]['text']
                else:
                    abstract = ''
                #print(patent_item)
                ipc_package = patent_item['bibliographic_data']['classification_data'].get('classification_ipcr', {})
                #ipc = patent_item['bibliographic_data']['classification_data']['classification_ipcr'][
                #    'main']
                if ipc_package!={}:
                    ipc = ipc_package['main']
                else:
                    ipc = ''
                # 专利受理局
                patent_office = patent_item['bibliographic_data']['publication_reference']['country']
                app_country = patent_item['bibliographic_data']['publication_reference']['country']


                patent_info = {'patent_id':patent_id,'abstract': abstract, 'ipc': ipc, 'patent_office': patent_office,'app_country': app_country}

                results.append(patent_info)

        return results

    async def query_by_tech_map_en(self,tech_map: List[dict]):
        """
        直接使用生成的技术图谱进行查询
        异步处理队列数据，规定每次最多处理 batch_size 个元素，且一分钟内最多调用 call_per_min 次
        :param tech_map: 技术图谱
        """
        tech_query_result = []
        all_patent_result = []  # 存储处理结果
        #将技术图谱转换为队列
        for tech_node in tech_map:
            main_techs = tech_node.get("Primary Technology", [])
            sub_techs = tech_node.get("Secondary Technology", [])
            for main_tech in main_techs:
                for sub in sub_techs:
                    key = f"{main_tech} - {sub}"
                    tech_query_result.append({'tech_point':key,'patents':[]})
        queue = deque(tech_query_result)
        # 创建一个新的 deque 来存储处理后的元素
        processed_deque = deque()
        while queue:
            # 从队列中取出最多 b 个元素
            element = queue.popleft()
            # 获取调用权限（满足速率限制）
            await self.rateLimiter_for_query_content.acquire()
            # 调用异步处理函数
            processed_element = await self.query_by_content(element['tech_point'])
            element['patents'] = processed_element
            processed_deque.append(element)

        return processed_deque

    async def query_by_tech_map_zh(self,tech_map: List[dict]):
        """
        直接使用生成的技术图谱进行查询
        异步处理队列数据，规定每次最多处理 batch_size 个元素，且一分钟内最多调用 call_per_min 次
        :param tech_map: 技术图谱
        """
        tech_query_result = []
        all_patent_result = []  # 存储处理结果
        #将技术图谱转换为队列
        for tech_node in tech_map:
            main_techs = tech_node.get("一级技术", [])
            sub_techs = tech_node.get("二级技术", [])
            for main_tech in main_techs:
                for sub in sub_techs:
                    key = f"{main_tech} - {sub}"
                    tech_query_result.append({'tech_point':key,'patents':[]})
        queue = deque(tech_query_result)
        # 创建一个新的 deque 来存储处理后的元素
        processed_deque = deque()
        while queue:
            # 从队列中取出最多 b 个元素
            element = queue.popleft()
            # 获取调用权限（满足速率限制）
            await self.rateLimiter_for_query_content.acquire()
            # 调用异步处理函数
            processed_element = await self.query_by_content(element['tech_point'])
            element['patents'] = processed_element
            processed_deque.append(element)

        return processed_deque

    # 注意这个函数，query_by_content，一分钟内最多调用10次，一次最多返回1000条数据
    async def query_by_content(
            self,
            text: str,
            limit: int = 50,
            apd_from: Optional[int] = 19500101,
            apd_to: Optional[int] = 20250101,
            pbd_from: Optional[int] = 19500101,
            pbd_to: Optional[int] = 20250101,
            offset: int = 0,
            countries: Optional[List[str]] = None,
            relevancy: Optional[str] = None
    ) -> List[Dict]:
        if not text or not isinstance(text, str) or text.strip() == "":
            print("ERROR: text 参数为空或不是字符串")
            return []
        payload = {
            "text": text,
            "limit": limit,
            "apd_from": apd_from,
            "apd_to": apd_to,
            "pbd_from": pbd_from,
            "pbd_to": pbd_to,
            "offset": offset,
            "country": countries or self.default_countries,
            "relevancy": relevancy or self.default_relevancy
        }
        # print payload for debug
        print(f"DEBUG: payload={payload}")
        raw_result = await self._make_request(self.semantic_search_url, payload)
        # ---- 加入容错判断 ----
        if not raw_result or 'data' not in raw_result or 'results' not in raw_result.get('data', {}):
            return []
        return raw_result['data']['results']

    async def query_by_id(self, patent_id: str) -> List[Dict]:
        """Search patents based on patent ID.

        Args:
            patent_id (str): The ID of the patent to search for.

        Returns:
            List[Dict]: A list containing the patent information.
        """
        payload = {
            "text": patent_id,
            "limit": 1,
            "search_field": "patent_id",
            "country": self.default_countries,
        }
        raw_result = await self._make_request(self.semantic_search_url, payload)
        # raw_result中只包含10个专利字段，需要再加上ipc和摘要
        raw_patents = raw_result['data']['results']
        result = await self.add_abstract_ipc_async_batch(raw_patents)
        return result

    async def query_by_title(self, title: str, limit: int = 5) -> List[Dict]:
        """Search patents based on patent title.

        Args:
            title (str): The title of the patent to search for.
            limit (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict]: A list of dictionaries containing patent information.
        """
        payload = {
            "text": title,
            "limit": limit,
            "search_field": "title",
            "country": self.default_countries,
        }
        raw_result = await self._make_request(self.semantic_search_url, payload)
        # raw_result中只包含10个专利字段，需要再加上ipc和摘要
        raw_patents = raw_result['data']['results']
        result = await self.add_abstract_ipc_async_batch(raw_patents)
        return result

    async def query_by_keyword(
            self,
            keyword: str,
            limit: int = 5,
            field: str = "claims"
    ) -> List[Dict]:
        """Search patents based on keyword in specified field.

        Args:
            keyword (str): The keyword to search for.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
            field (str, optional): Field to search in (title, claims, description). Defaults to "claims".

        Returns:
            List[Dict]: A list of dictionaries containing patent information.
        """
        payload = {
            "text": keyword,
            "limit": limit,
            "search_field": field,
            "country": self.default_countries,
        }
        raw_result = await self._make_request(self.semantic_search_url, payload)
        # raw_result中只包含10个专利字段，需要再加上ipc和摘要
        raw_patents = raw_result['data']['results']
        result = await self.add_abstract_ipc_async_batch(raw_patents)
        return result

    async def query_by_patent_number(self, patent_number: str) -> List[Dict]:
        """Search patents based on patent number.

        Args:
            patent_number (str): The patent number to search for.

        Returns:
            List[Dict]: A list of dictionaries containing patent information.
        """
        payload = {
            "text": patent_number,
            "limit": 10,
            "search_field": "patent_number",
            "country": self.default_countries,
        }
        raw_result = await self._make_request(self.semantic_search_url, payload)
        # raw_result中只包含10个专利字段，需要再加上ipc和摘要
        raw_patents = raw_result['data']['results']
        result = await self.add_abstract_ipc_async_batch(raw_patents)
        return result

    async def query_by_assignee(self, assignee: str) -> List[Dict]:
        """Search patents based on assignee.

        Args:
            assignee (str): The assignee of the patent.

        Returns:
            List[Dict]: A list of dictionaries containing patent information.
        """
        payload = {
            "sort": [
                {
                    "field": "SCORE",
                    "order": "DESC"
                }
            ],
            "limit": 10,
            "offset": 0,
            "assignee": assignee,
            "collapse_by": "PBD",
            "collapse_type": "ALL",
            "collapse_order": "LATEST",
            "collapse_order_authority": [
                "CN",
                "US",
                "EP",
                "JP",
                "KR"
            ],
            "country": self.default_countries,
        }
        raw_result = await self._make_request(self.current_search_url, payload)
        # raw_result中只包含10个专利字段，需要再加上ipc和摘要
        raw_patents = raw_result['data']['results']
        result = await self.add_abstract_ipc_async_batch(raw_patents)
        return result

    async def get_simple_bibliography_async(self, patent_id: str = None, patent_number: str = None) -> dict:
        url = "https://connect.zhihuiya.com/basic-patent-data/simple-bibliography"
        token = self.token_manager.get_token()
        params = {"apikey": self.api_key}
        if patent_id:
            params["patent_id"] = patent_id
        if patent_number:
            params["patent_number"] = patent_number
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                try:
                    return await resp.json()
                except Exception:
                    return {"error": await resp.text()}

    async def get_patent_abstract_translated_async(self, patent_id: str = None, patent_number: str = None, lang='en',
                                                   replace_by_related='0') -> dict:
        url = "https://connect.zhihuiya.com/basic-patent-data/abstract-data-translated"
        token = self.token_manager.get_token()
        params = {
            "lang": lang,
            "replace_by_related": replace_by_related,
            "apikey": self.api_key
        }
        if patent_id:
            params["patent_id"] = patent_id
        if patent_number:
            params["patent_number"] = patent_number
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as resp:
                try:
                    return await resp.json()
                except Exception:
                    return {"error": await resp.text()}
    async def query_by_application(self, application: str) -> List[Dict]:
        """Search patents based on application.

        Args:
            application (str): The application of the patent.

        Returns:
            List[Dict]: A list of dictionaries containing patent information.
        """
        payload = {
            "sort": [
                {
                    "field": "SCORE",
                    "order": "DESC"
                }
            ],
            "limit": 10,
            "offset": 0,
            "application": application,
            "collapse_by": "PBD",
            "collapse_type": "ALL",
            "collapse_order": "LATEST",
            "collapse_order_authority": [
                "CN",
                "US",
                "EP",
                "JP",
                "KR"
            ],
            "country": self.default_countries,
        }
        raw_result = await self._make_request(self.company_search_url, payload)
        # raw_result中只包含10个专利字段，需要再加上ipc和摘要
        raw_patents = raw_result['data']['results']
        result = await self.add_abstract_ipc_async_batch(raw_patents)
        return result

    async def add_abstract_ipc_async_batch(
            self,
            patent_list: List[Dict[str, Union[str, int, float, None]]],
            batch_size: int = 50
    ) -> List[Dict]:
        """
        批量查询专利简单书目信息。
        patent_list: 列表，每个元素为字典，value 可为 None、字符串、数字。
        batch_size: 每批查询数量
        返回: 结果列表，顺序与输入一致
        """

        async def single_query(patent_item):
            params = {"apikey": self.api_key}
            # 支持 patent_id 和 patent_number, 自动类型处理
            if "patent_id" in patent_item and patent_item["patent_id"] is not None:
                params["patent_id"] = str(patent_item["patent_id"])
            if "pn" in patent_item and patent_item["pn"] is not None:
                params["pn"] = str(patent_item["pn"])
            # 增加对键值为 None/数字/字符串的健壮性支持

            if not params.get("patent_id") and not params.get("pn"):
                return {"error": "缺少 patent_id 或 pn", "input": patent_item}

            headers = {
                "Content-Type": "application/json",
                "authorization": f"Bearer {self.token_manager.get_token()}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://connect.zhihuiya.com/basic-patent-data/simple-bibliography",
                        params=params,
                        headers=headers
                ) as response:
                    try:
                        result = await response.json()
                        abstract = result['data'][0]['bibliographic_data']['abstracts'][0]['text']
                        ipc = result['data'][0]['bibliographic_data']['classification_data']['classification_ipcr'][
                            'main']

                        #专利受理局
                        patent_office = result['data'][0]['bibliographic_data']['publication_reference']['country']

                        patent_item.update({'abstract': abstract, 'ipc': ipc,'patent_office': patent_office})
                        return patent_item
                    except Exception as e:
                        return {"error": str(e), "raw": await response.text(), "input": patent_item}

        results = []
        for i in range(0, len(patent_list), batch_size):
            batch = patent_list[i:i + batch_size]
            # 对每个专利进行批量异步处理
            batch_results = await asyncio.gather(*[single_query(item) for item in batch])
            results.extend(batch_results)
        return results

    async def query_13_columns_by_content(
            self,
            text: str,
            limit: int = 10,
            apd_from: Optional[int] = 20200101,
            apd_to: Optional[int] = 20220101,
            pbd_from: Optional[int] = 20200101,
            pbd_to: Optional[int] = 20220101,
            offset: int = 0,
            countries: Optional[List[str]] = None,
            relevancy: Optional[str] = None
    ) -> List[Dict]:
        if not text or not isinstance(text, str) or text.strip() == "":
            print("ERROR: text 参数为空或不是字符串")
            return []
        payload = {
            "text": text,
            "limit": limit,
            "apd_from": apd_from,
            "apd_to": apd_to,
            "pbd_from": pbd_from,
            "pbd_to": pbd_to,
            "offset": offset,
            "country": countries or self.default_countries,
            "relevancy": relevancy or self.default_relevancy
        }
        # print payload for debug
        print(f"DEBUG: payload={payload}")
        raw_result = await self._make_request(self.semantic_search_url, payload)
        # ---- 加入容错判断 ----
        if not raw_result or 'data' not in raw_result or 'results' not in raw_result.get('data', {}):
            return []
        # ---- end ----
        raw_patents = raw_result['data']['results']
        result = await self.add_abstract_ipc_async_batch(raw_patents)
        return result

if __name__ == "__main__":
    async def main():
        # Initialize token manager
        # Initialize with your actual API key and token manager
        query = Query()

        # Example queries
        #search_text = "The invention discloses an automobile front-view based wireless video transmission system and method. The system comprises a front-view camera, a wireless video transmitting module, a wireless video receiving module, a display screen, a display triggering device, a first controller, a wireless command transmitting module, a wireless command receiving module, a second controller and an automobile starting detecting module, wherein the display screen is connected with the wireless video receiving module; the front-view camera is connected with the wireless video transmitting module; the wireless video transmitting module is wirelessly connected with the wireless video receiving module and wirelessly transmits a video shot by the front-view camera; and the wireless video receiving module receives and sends the video and displays the video on the display screen, so that the mounting time of the front-view camera is shortened greatly, no damage can be caused to an original automobile, the front-view camera can be mounted without a threading manner, and great convenience is brought to the owner of the automobile."
        #patent_data = await query.query_by_content(search_text)
        # insert_patent_to_db(patent_data)
        #
        patent_id = "b053642f-3108-4ea9-b629-420b0ab959e3,122fc785-3e61-4b34-9f7b-5cf7d76e621a,e07104fb-8283-4c43-a2f1-bd85feb64de6"
        patent_data = await query.get_simple_bibliography_async(patent_id)
        #print(patent_data)
        # insert_patent_to_db(patent_data)
        #
        #title = "储能"
        #pn = 'CN1070866C'
        #id = '24bea1a4-718d-496f-8d7c-628a9bf765b5'
        #patent_data = await query.query_by_patent_number(pn)
        #patent_data = await query.query_by_title(title)
        print(patent_data)


    asyncio.run(main())
