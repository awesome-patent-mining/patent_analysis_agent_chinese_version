from datetime import datetime
import os
import time
import streamlit as st
import pandas as pd
import asyncio
from research_agent.core.generate_tech_genealogy import Tech_Gene_Generator
from research_agent.core.utils import transform_data_zh, flatten_tech_structure_zh
from research_agent.core.markdown_display import display_markdown_with_images_from_file
from research_agent.core.applicant_analysis import generate_full_report
from research_agent.core.patent_tech_analysis_1 import PatentTechAnalyzer

# Initialize the current step
general_report_generator = PatentTechAnalyzer()
tech_genealogy_generator = Tech_Gene_Generator()

if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = 1  # Default: Web
if 'last_used_data_source_type' not in st.session_state:
    st.session_state.last_used_data_source_type = 1

data_source_options = {
    1: "网页",
    2: "专利",
    3: "网页 + 专利"
}

step_name_dict = {1: "输入技术主题", 2: "生成技术图谱", 3: "检索专利数据",
                  4: "生成专利报告", 5: "完成"}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Initialize technical topic
if 'tech_topic' not in st.session_state:
    st.session_state.tech_topic = ""
if 'patent_data_generated' not in st.session_state:
    st.session_state.patent_data_generated = False  # To track whether patent data has already been generated

if 'tech_genealogy' not in st.session_state:
    st.session_state.tech_genealogy = None

if 'last_used_topic' not in st.session_state:
    st.session_state.last_used_topic = ""  # Record the last topic used to generate the technical map

#记录分析用到的专利数量
if 'patent_num' not in st.session_state:
    st.session_state.patent_num = None

# 初始化步骤计时
for i in range(1, 6):
    if f'step{i}_time' not in st.session_state:
        st.session_state[f'step{i}_time'] = None


# Custom CSS styles
st.markdown("""
    <style>
    /* Logo styles */
    .logo-container {
        margin-bottom: 30px;
        padding: 15px;
        border-bottom: 2px solid #4A90E2;
    }
    .logo-text {
        font-size: 20px;
        font-weight: bold;
        color: #2B5876;
        margin-left: 10px;
    }
    /* Step styles */
    .step {
        font-size: 16px;
        margin: 12px 0;
        padding: 15px;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .step:hover {
        transform: translateX(5px);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    # Add logo and system name
    st.markdown("""
        <div class="logo-container">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 28px;">📑</span>
                <span class="logo-text">专利分析系统</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.header("分析流程")

    # Generate steps
    for step in range(1, 6):
        if step == st.session_state.current_step:
            style = """
                background-color: #E6F3FF;
                color: #2B5876 !important;
                font-weight: bold;
                border-left: 4px solid #4A90E2;
            """
        else:
            style = """
                background-color: #F8F9FA;
                color: #666666;
            """

        st.markdown(
            f"<div class='step' style='{style}'>📌 步骤 {step}: {step_name_dict[step]}</div>",
            unsafe_allow_html=True
        )

# Main interface content
content = st.container()
tech_topic = ""

# Show different content based on the current step
with content:
    if st.session_state.current_step == 1:
        st.header("📤 步骤1 - 输入技术主题")
        tech_topic = st.text_input(
            label="请输入技术主题（如：人工智能、云计算）：",
            placeholder="人工智能",
            value=st.session_state.tech_topic  # Retain input consistency
        )
        st.session_state.tech_topic = tech_topic

        selected_data_source = st.selectbox(
            "请选择生成技术图谱时使用的数据源：",
            options=list(data_source_options.keys()),
            format_func=lambda x: data_source_options[x],
            index=list(data_source_options.keys()).index(st.session_state.data_source_type)
        )
        st.session_state.data_source_type = selected_data_source

    elif st.session_state.current_step == 2:
        st.header("⚙️ 步骤2 - 生成技术图谱")
        st.write("---")
        st.session_state.map_tech = st.session_state.tech_genealogy
        initial_data = flatten_tech_structure_zh(st.session_state.tech_genealogy)

        if "df" not in st.session_state:
            st.session_state.df = pd.DataFrame(initial_data)
        st.write(f"**{st.session_state.tech_topic}的技术图谱**")
        st.dataframe(st.session_state.df, use_container_width=True)

    elif st.session_state.current_step == 3:
        st.header("🚀 步骤3 - 检索专利数据")
        if not st.session_state.patent_data_generated:  # Check if data has been generated
            report_container = st.container()
            with report_container:
                with st.spinner('⏳ 正在检索专利数据，请稍候...'):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        start_time = time.time()
                        save_dir = os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), "research_agent/general_analysis_output")
                        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_dir = os.path.join(save_dir, time_str)
                        os.mkdir(save_dir)

                        loop.run_until_complete(
                            general_report_generator.run(save_dir=save_dir, tech_map=st.session_state.tech_genealogy))
                        # 记录获取到的专利总数
                        st.session_state.patent_num = general_report_generator.patent_num
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        st.session_state.step3_time = elapsed_time

                        print(f"Patent statistics report generated and saved to: {save_dir}")
                        print(f"Total time: {elapsed_time:.2f} seconds")

                        general_report_path = os.path.join(save_dir, 'patent_report.md')
                        if os.path.exists(general_report_path):
                            display_markdown_with_images_from_file(
                                general_report_path, save_dir)
                        else:
                            st.error("未找到专利统计文件")

                        st.session_state.patent_data_generated = True

                    except Exception as e:
                        st.error(f"专利统计生成失败: {str(e)}")
                        st.exception(e)
        else:
            st.info("专利数据已检索，可进入下一步。")

    elif st.session_state.current_step == 4:
        st.header("📊 步骤4 - 生成专利报告")
        if st.button("🚀 生成专利报告", type="primary", key="generate_patent_trend_report"):
            report_container = st.container()
            with report_container:
                with st.spinner('⏳ 正在生成专利报告，请稍候...'):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        start_time = time.time()
                        save_dir = os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), "research_agent/detail_analysis_output")
                        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        time_dir = os.path.join(save_dir, time_str)

                        full_report, report_path = loop.run_until_complete(
                            generate_full_report(save_dir=time_dir, map_tech=st.session_state.map_tech))

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        st.session_state.step4_time = elapsed_time

                        print(f"Patent report generated and saved to: {time_dir}")
                        print(f"Total time: {elapsed_time:.2f} seconds")

                        report_path = os.path.join(time_dir, 'patent_analysis_report.md')
                        word_file_path = os.path.join(time_dir, 'patent_analysis_report.docx')
                        if os.path.exists(report_path):
                            display_markdown_with_images_from_file(markdown_file_path=report_path, time_dir=time_dir)
                            with open(word_file_path, "rb") as file:
                                st.download_button(
                                    label="📥 下载专利报告（Word）",
                                    data=file,
                                    file_name="patent_analysis_report.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                        else:
                            st.error("未找到专利报告文件")
                    # ...
                    except Exception as e:
                        st.error(f"专利报告生成失败: {str(e)}")
                        st.exception(e)

    elif st.session_state.current_step == 5:
        st.header("📊 步骤5 - 分析完成")

        # 显示彩色卡片式的专利统计与耗时统计
        st.markdown(
            f"""
            <div style="background: linear-gradient(90deg,#E6F3FF 40%,#f2f7fa 100%);padding:18px 26px;border-radius:12px;border-left:5px solid #339af0;margin-bottom:18px">
                <h3 style="margin-bottom:0.2em;color:#2176ae;">✅ <span style='color:#1b7a5a'>{st.session_state.tech_topic}</span> 的专利分析已完成</h3>
                <p style="font-size:1.13em;color:#2b5876;margin-top:0.3em;">
                    本次分析共使用了 <b>{st.session_state.patent_num}</b> 项专利。
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 获取耗时
        step1_time = st.session_state.get('step1_time', 0) or 0
        step2_time = st.session_state.get('step2_time', 0) or 0
        step3_time = st.session_state.get('step3_time', 0) or 0
        step4_time = st.session_state.get('step4_time', 0) or 0
        # 步骤一不计入总耗时
        total_time = step2_time + step3_time + step4_time
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)

        # 展示四列metric：步骤一为“——”
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("步骤1", "——")
        col2.metric("步骤2", f"{step2_time:.2f}", "秒")
        col3.metric("步骤3", f"{step3_time:.2f}", "秒")
        col4.metric("步骤4", f"{step4_time:.2f}", "秒")

        # 总耗时卡片
        st.markdown(
            f"""
            <div style="margin-top:10px;margin-bottom:8px;">
                <span style="font-weight:bold;font-size:17px;color:#2b5876;">
                    ⏱️ 总耗时：<span style="color:#207567;">{total_minutes} 分 {total_seconds} 秒</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='margin:18px 0 10px 0'>", unsafe_allow_html=True)

        # 两个操作按钮
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("🔄 开启新一轮专利分析", key="continue_analysis"):
                # ...原有重置代码...
                st.session_state.current_step = 1
                st.session_state.uploaded_file = None
                st.session_state.file_details = None
                st.session_state.analysis_started = False
                for i in range(1, 6):
                    st.session_state[f'step{i}_time'] = None
                st.rerun()

        with action_col2:
            if st.button("🏁 结束分析", key="end_analysis"):
                st.success("感谢使用专利分析系统！")

# Navigation buttons, only show when not in step 5
if st.session_state.current_step != 5:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        prev_disabled = st.session_state.current_step <= 1
        prev_btn = st.button(
            "◀ 上一步",
            disabled=prev_disabled,
            use_container_width=True,
            help="返回上一步" if not prev_disabled else "已是第一步"
        )

    with col3:
        next_disabled = st.session_state.current_step >= 5
        next_btn = st.button(
            "下一步 ▶",
            type="primary",
            disabled=next_disabled,
            use_container_width=True,
            help="进入下一步" if not next_disabled else "已是最后一步"
        )

    if 'show_topic_warn' not in st.session_state:
        st.session_state.show_topic_warn = False

    if prev_btn and not prev_disabled:
        st.session_state.current_step -= 1
        st.session_state.show_topic_warn = False
        st.rerun()

    if next_btn and not next_disabled:
        if st.session_state.current_step == 1:
            if not st.session_state.tech_topic.strip():
                st.session_state.show_topic_warn = True
                st.rerun()
            else:
                st.session_state.show_topic_warn = False
                with st.spinner('⏳ 正在生成技术图谱，请稍候...'):
                    try:
                        step2_start_time = time.time()
                        st.session_state.tech_genealogy = asyncio.run(
                            tech_genealogy_generator.generate_tech_genealogy(
                                topic=st.session_state.tech_topic,
                                genealogy_type=st.session_state.data_source_type
                            )
                        )
                        step2_end_time = time.time()
                        st.session_state.step2_time = step2_end_time - step2_start_time
                        st.session_state.last_used_topic = st.session_state.tech_topic
                        st.session_state.last_used_data_source_type = st.session_state.data_source_type
                        st.session_state.current_step += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating technical map: {str(e)}")

        else:
            st.session_state.current_step += 1
            st.session_state.show_topic_warn = False
            st.rerun()

    if st.session_state.current_step == 1 and st.session_state.show_topic_warn:
        st.warning("请输入技术主题")
