import re
import streamlit as st
import os
import pypandoc
from research_agent.core.config import Config


# 参考文件路径
reference_doc = Config.REFERENCE_DOC


def parse_markdown_with_images(markdown_content):
    """
    解析 Markdown 内容，提取文本和图片信息。
    
    参数:
        markdown_content (str): Markdown 文件内容。
    
    返回:
        list: 包含文本和图片信息的列表，图片信息为字典格式。
    """
    # 正则表达式匹配 Markdown 图片语法
    # 支持两种形式：
    # 1. ![alt text](path "title")
    # 2. ![alt text](path)
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)(?:\s*"(.*?)")?\)')
    
    # 分割 Markdown 内容
    parts = []
    last_end = 0
    
    for match in image_pattern.finditer(markdown_content):
        # 添加图片之前的文本
        if last_end < match.start():
            parts.append({'type': 'text', 'content': markdown_content[last_end:match.start()]})
        
        # 添加图片信息
        alt_text = match.group(1)
        image_path = match.group(2)
        title = match.group(3)  # 标题可能为 None
        parts.append({'type': 'image', 'alt_text': alt_text, 'image_path': image_path, 'title': title})
        
        # 更新 last_end
        last_end = match.end()
    
    # 添加最后一段文本
    if last_end < len(markdown_content):
        parts.append({'type': 'text', 'content': markdown_content[last_end:]})
    
    return parts

def display_markdown_with_images_from_file(markdown_file_path,time_dir):
    """
    从文件路径读取 Markdown 内容并在 Streamlit 上显示。
    
    参数:
        markdown_file_path (str): Markdown 文件的路径。
        time_dir (str): 图片文件所在的目录路径。
    """
    # 读取 Markdown 文件
    #markdown_file_path = os.path.join(time_dir, '专利分析报告.md')
    #Word_file_path = os.path.join(time_dir, '专利分析报告.docx')
    Word_file_path = markdown_file_path.replace('.md', '.docx')
    # 转换为 docx
    pypandoc.convert_file(
    markdown_file_path, 'docx', outputfile=Word_file_path,
    extra_args=[f'--resource-path={time_dir}',
                '--reference-doc=' + reference_doc]
)

    with open(markdown_file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    parts = parse_markdown_with_images(markdown_content)
    
    # 在 Streamlit 上显示
    for part in parts:
        if part['type'] == 'text':
            st.markdown(part['content'])
        elif part['type'] == 'image':
            # 将相对路径与 time_dir 结合，生成完整路径
            full_image_path = os.path.join(time_dir, part['image_path'])
            # 如果标题为 None，使用 alt_text 作为 caption
            caption = part['title'] if part['title'] else part['alt_text']
            st.image(full_image_path, caption=caption, use_container_width=True)

        # 添加 Word 下载按钮
    with open(Word_file_path, "rb") as file:
        st.download_button(
            label="📥 下载专利报告 (Word)",
            data=file,
            file_name="专利分析报告.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ) 

# 示例使用
if __name__ == "__main__":
    # 传入 Markdown 文件和图片目录路径
    time_dir = "research_agent/patent_annalysis_report/20250425_092217"
    display_markdown_with_images_from_file(time_dir)
