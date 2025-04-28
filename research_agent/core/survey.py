import asyncio
import os
from dotenv import load_dotenv
# from litellm import completion
import networkx as nx
import json
import pickle
from typing import List
import re
from json_repair import repair_json
load_dotenv()

# DEFAULT_MODEL = "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"


class Survey:
    def __init__(self, topic:str):
        self.full_content = nx.DiGraph()
        self.topic = topic
        self.title = None
        self.section_map = {}
        params = {'level':0,'leaf':False,'root':True,'text':None,'code':None,'title':None,'description':None,'content':None,'questions':[]}
        num_nodes = self.full_content.number_of_nodes()
        self.full_content.add_node(num_nodes+1,**params)
        self.root = list(self.full_content.nodes())[0]

    def check_root(self,node):
        #如果node不是根节点，那么返回False，否则返回True
        if node != self.root:
            return False
        else:
            return True

    def check_leaf(self,code):
        #先检查code处于第几层，如果处于第二层，返回True，否则返回False
        if self.check_level(code) == 2:
            return True
        else:
            return False

    def check_level(self,code):
        #如果code中没有句点，说明处于第1层次，如果code中包含句点，那么它包含多少句点，其所在层次即句点数量+1
        if '.' not in code:
            return 1
        else:
            return code.count('.') + 1

    def extractCodeAndTitle(self,text: str) -> [str, str]:
        """find code and title from text and detail_analysis_output them."""
        # 将text中的#去除
        if '#' in text:
            text = text.replace('#', '')
        text = text.strip()
        # text是章节标题，其格式一般是"1 title name"或者"1.1 title name",其中1是章节编号，title name是章节标题，使用正则表达式将章节编号和章节标题分离出来
        # 使用正则表达式匹配章节编号和标题
        match = re.match(r'^(\d+(?:\.\d+)*)\s+(.*)$', text)
        if match:
            chapter_number = match.group(1)  # 提取章节编号
            chapter_title = match.group(2)  # 提取章节标题
            return chapter_number, chapter_title
        else:
            return None, None  # 如果没有匹配到，返回None

    def set_title(self, title):
        self.title = title
    def get_root(self):
        return self.root

    def set_root(self, root):
        pass

    def add_node(self, current_node,node_for_adding,**kwargs):
        # 为当前节点添加一个子节点
        self.full_content.add_node(node_for_adding,**kwargs)
        self.full_content.add_edge(current_node,node_for_adding)
        return None
    def getNodeByCode(self,code):
        # 通过代码获取节点
        for node in self.full_content.nodes():
            if self.full_content.nodes[node]['code'] == code:
                return node
        return None

    def get_whole_outline(self,):
        # 获取问卷的outline，即从根节点开始，按照深度优先遍历的方式，将所有节点的代码、文本、级别等信息收集起来，形成一个列表
        outline = []
        stack = [self.root]
        while len(stack) > 0:
            curr_node = stack.pop()
            outline.append({'code':self.full_content.nodes[curr_node]['code'],'title':self.full_content.nodes[curr_node]['title']})
            children = list(self.full_content.successors(curr_node))
            #将children倒序排列，这样可以保证先遍历子节点，再遍历父节点
            children.sort(reverse=True)
            stack.extend(children)
        return outline
    def get_max_section_code(self,):
        # 获取outline中最大一级标题号码
        section_code_list = []
        stack = [self.root]
        while len(stack) > 0:
            curr_node = stack.pop()
            section_code_list.append(self.full_content.nodes[curr_node]['code'])
            children = list(self.full_content.successors(curr_node))
            #将children倒序排列，这样可以保证先遍历子节点，再遍历父节点
            children.sort(reverse=True)
            stack.extend(children)
        return section_code_list[-1]
    def get_paper_content(self,except_sections:List[str]=[]):
        # 获得综述内容，即从根节点开始，按照深度优先遍历的方式，将所有节点的代码、文本、级别等信息收集起来，形成一个列表
        # 如果不愿意获得个别章节，可以将其标号放在except_sections中，这样就不会被加入
        except_sections = [str(section) for section in except_sections]
        paper_draft = ""
        stack = [self.root]
        while len(stack) > 0:
            curr_node = stack.pop()
            code = self.full_content.nodes[curr_node]['code']
            title = self.full_content.nodes[curr_node]['title']
            content = self.full_content.nodes[curr_node]['content']
            level = self.full_content.nodes[curr_node]['level']
            if code not in except_sections:
                if level==1:
                    if content.strip()!="":
                        paper_draft = paper_draft + "## " + code +' ' +title + "\n\n"+content + "\n\n"
                    else:
                        paper_draft = paper_draft + "## " + code + ' ' + title + "\n\n"
                elif level==2:
                    paper_draft = paper_draft + "### " + code +' ' +title + "\n\n"+content + "\n\n"
                children = list(self.full_content.successors(curr_node))
                #将children倒序排列，这样可以保证先遍历子节点，再遍历父节点
                children.sort(reverse=True)
                for child in children:
                    if child not in except_sections:
                        stack.append(child)
        return paper_draft
    def get_section_outline(self,code):
        # 获取问卷的outline，即从根节点开始，按照深度优先遍历的方式，将所有节点的代码、文本、级别等信息收集起来，形成一个列表
        outline = []
        outline_str = ""
        stack = [self.getNodeByCode(code)]
        while len(stack) > 0:
            curr_node = stack.pop()
            outline.append({'code':self.full_content.nodes[curr_node]['code'],'title':self.full_content.nodes[curr_node]['title'],'description':self.full_content.nodes[curr_node]['description'],'level':self.full_content.nodes[curr_node]['level']})
            children = list(self.full_content.successors(curr_node))
            #将children倒序排列，这样可以保证先遍历子节点，再遍历父节点
            children.sort(reverse=True)
            stack.extend(children)
        #将outline中的大纲转换成字符串形式
        for i in outline:
            if i['level']==1:
                outline_str=outline_str + "## " +i['code'] +' ' +i['title'] + "\nDescription:" +i['description'] + "\n\n"
            elif i['level']==2:
                outline_str=outline_str + "### " +i['code'] +' ' +i['title'] + "\nDescription:" +i['description'] + "\n\n"
        return outline_str

    def transfer_dict_2_text(self,outline_list:list[str]):
        # 将大纲列表转换成字符串形式，用于生成markdown文件
        if type(outline_list)!=list:
            raise TypeError("The input should be a list")
        outline_str = ''
        for i in outline_list:
            if i['code']==None:
                outline_str="# "+self.title+"\n\n"
            elif self.check_level(i['code'])==1:
                outline_str = outline_str + "## " + i['code']+ ' '+ i['title'] + "\n\n"
            elif self.check_level(i['code'])==2:
                outline_str = outline_str + "### " + i['code']+ ' '+ i['title'] + "\n\n"
        return outline_str
    def transfer_parsed_outline_into_nx(self,parsed_outline):
        # 将解析后的大纲转换成网络图结构
        for section_idx, [section, desc] in enumerate(
                zip(parsed_outline['sections'], parsed_outline['section_descriptions'])):
            code_i,title_i = self.extractCodeAndTitle(section)
            node_i = self.full_content.number_of_nodes() + 1
            root_i = False
            leaf_i = self.check_leaf(code_i)
            level_i = self.check_level(code_i)
            self.add_node(current_node=self.get_root(), node_for_adding=node_i, code=code_i, title=title_i,
                                description=desc, root=root_i, leaf=leaf_i, level=level_i, content="")
            for sub_idx, [sub, desc] in enumerate(zip(parsed_outline['subsections'][section_idx],
                                                      parsed_outline['subsection_descriptions'][section_idx])):
                code_j,title_j = self.extractCodeAndTitle(sub)
                node_j = self.full_content.number_of_nodes() + 1
                root_j = False
                leaf_j = self.check_leaf(code_j)
                level_j = self.check_level(code_j)
                self.add_node(current_node=node_i, node_for_adding=node_j, code=code_j, title=title_j,
                                    description=desc,
                                    root=root_j, leaf=leaf_j, level=level_j, content=None)
    def update_full_content(self,code,new_content):

        node = self.getNodeByCode(code)
        if isinstance(new_content, str):
            section_content = new_content
        elif isinstance(new_content, List):
            section_content = '\n'.join([content_i for content_i in new_content if content_i != ""])
        else:
            raise ValueError("new_content must be a string or a list of strings")
        self.full_content.nodes[node]['content'] = section_content

