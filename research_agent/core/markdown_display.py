import re
import streamlit as st
import os
import pypandoc
from research_agent.core.config import Config


# Reference file path
reference_doc = Config.REFERENCE_DOC


def parse_markdown_with_images(markdown_content):
    """
    Parse Markdown content and extract text and image information.
    
    Args:
        markdown_content (str): Markdown file content.
    
    Returns:
        list: List containing text and image information, with image information in dictionary format.
    """
    # Regular expression to match Markdown image syntax
    # Supports two formats:
    # 1. ![alt text](path "title")
    # 2. ![alt text](path)
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)(?:\s*"(.*?)")?\)')
    
    # Split Markdown content
    parts = []
    last_end = 0
    
    for match in image_pattern.finditer(markdown_content):
        # Add text before image
        if last_end < match.start():
            parts.append({'type': 'text', 'content': markdown_content[last_end:match.start()]})
        
        # Add image information
        alt_text = match.group(1)
        image_path = match.group(2)
        title = match.group(3)  # Title may be None
        parts.append({'type': 'image', 'alt_text': alt_text, 'image_path': image_path, 'title': title})
        
        # Update last_end
        last_end = match.end()
    
    # Add last text segment
    if last_end < len(markdown_content):
        parts.append({'type': 'text', 'content': markdown_content[last_end:]})
    
    return parts

def display_markdown_with_images_from_file(markdown_file_path,time_dir):

    """
    Read Markdown content from file path and display on Streamlit.
    
    Args:
        time_dir (str): Directory path where image files are located.
    """

    # Read Markdown file
    word_file_path = markdown_file_path.replace('.md', '.docx')
    
    # Convert to docx

    # 读取 Markdown 文件
    #markdown_file_path = os.path.join(time_dir, '专利分析报告.md')
    #Word_file_path = os.path.join(time_dir, '专利分析报告.docx')
    Word_file_path = markdown_file_path.replace('.md', '.docx')
    # 转换为 docx
    pypandoc.convert_file(
        markdown_file_path, 'docx', outputfile=word_file_path,
        extra_args=[f'--resource-path={time_dir}',
                   '--reference-doc=' + reference_doc]
    )

    with open(markdown_file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    parts = parse_markdown_with_images(markdown_content)
    
    # Display on Streamlit
    for part in parts:
        if part['type'] == 'text':
            st.markdown(part['content'])
        elif part['type'] == 'image':
            # Combine relative path with time_dir to generate full path
            full_image_path = os.path.join(time_dir, part['image_path'])
            # Use alt_text as caption if title is None
            caption = part['title'] if part['title'] else part['alt_text']
            st.image(full_image_path, caption=caption, use_container_width=True)

        # Add Word download button
    with open(word_file_path, "rb") as file:
        st.download_button(
            label="📥 Download Patent Report (Word)",
            data=file,
            file_name="patent_analysis_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ) 

# Example usage
if __name__ == "__main__":
    # Pass Markdown file and image directory path
    time_dir = "research_agent/patent_analysis_report/20250425_092217"
    display_markdown_with_images_from_file(time_dir)
