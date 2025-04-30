from datetime import datetime
import os
import time
import streamlit as st
import pandas as pd
import asyncio
from research_agent.core.generate_tech_genealogy import Tech_Gene_Generator
from research_agent.core.utils import transform_data_zh, flatten_tech_structure_en
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
    1: "Web",
    2: "Patent",
    3: "Web + Patent"
}

step_name_dict = {1: "Enter technical topic", 2: "Generate technical map", 3: "Retrieve patent data",
                  4: "Generate patent report", 5: "Finish"}

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
                <span class="logo-text">Patent Analysis System</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.header("Analysis Process")

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
            f"<div class='step' style='{style}'>📌 Step {step}: {step_name_dict[step]}</div>",
            unsafe_allow_html=True
        )

# Main interface content
content = st.container()
tech_topic = ""

# Show different content based on the current step
with content:
    if st.session_state.current_step == 1:
        st.header("📤 Step 1 - Enter Technical Topic")
        tech_topic = st.text_input(
            label="Please enter the technical topic (e.g., Artificial Intelligence, Cloud Computing):",
            placeholder="Artificial Intelligence",
            value=st.session_state.tech_topic  # Retain input consistency
        )
        st.session_state.tech_topic = tech_topic

        selected_data_source = st.selectbox(
            "Select data source for generating the technical map:",
            options=list(data_source_options.keys()),
            format_func=lambda x: data_source_options[x],
            index=list(data_source_options.keys()).index(st.session_state.data_source_type)
        )
        st.session_state.data_source_type = selected_data_source

    elif st.session_state.current_step == 2:
        st.header("⚙️ Step 2 - Generate Technical Map")
        st.write("---")
        step2_start_time = time.time()

        st.session_state.map_tech = st.session_state.tech_genealogy
        initial_data = flatten_tech_structure_en(st.session_state.tech_genealogy)

        if "df" not in st.session_state:
            st.session_state.df = pd.DataFrame(initial_data)
        st.write(f"**Technical Map for {st.session_state.tech_topic}**")
        st.dataframe(st.session_state.df, use_container_width=True)

        # 仅首次设置本步耗时
        step2_end_time = time.time()
        if st.session_state.step2_time is None:
            st.session_state.step2_time = step2_end_time - step2_start_time

    elif st.session_state.current_step == 3:
        st.header("🚀 Step 3 - Retrieve Patent Data")
        if not st.session_state.patent_data_generated:  # Check if data has been generated
            report_container = st.container()
            with report_container:
                with st.spinner('⏳ Retrieving patent data, please wait...'):
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
                            st.error("Patent statistics file not found")

                        st.session_state.patent_data_generated = True

                    except Exception as e:
                        st.error(f"Patent statistics generation failed: {str(e)}")
                        st.exception(e)
        else:
            st.info("Patent data has already been retrieved. Proceed to the next step.")

    elif st.session_state.current_step == 4:
        st.header("📊 Step 4 - Generate Patent Report")
        if st.button("🚀 Generate Patent Report", type="primary", key="generate_patent_trend_report"):
            report_container = st.container()
            with report_container:
                with st.spinner('⏳ Generating the patent report, please wait...'):
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
                                    label="📥 Download Patent Report (Word)",
                                    data=file,
                                    file_name="patent_analysis_report.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                        else:
                            st.error("Patent report file not found")

                    except Exception as e:
                        st.error(f"Patent report generation failed: {str(e)}")
                        st.exception(e)

    elif st.session_state.current_step == 5:
        st.header("📊 Step 5 - Analysis Completed")
        # 显示成功信息，并显示使用的专利数量

        st.success(
            f"✅ Patent analysis in {st.session_state.tech_topic} has been completed. A total of {st.session_state.patent_num} patents were used in this analysis.")
        # Show the time taken for each step, in seconds
        step1_time = st.session_state.get('step1_time', 0) or 0
        step2_time = st.session_state.get('step2_time', 0) or 0
        step3_time = st.session_state.get('step3_time', 0) or 0
        step4_time = st.session_state.get('step4_time', 0) or 0
        total_time = step1_time + step2_time + step3_time + step4_time

        st.metric("Step 1 Duration (s)", f"{step1_time:.2f}")
        st.metric("Step 2 Duration (s)", f"{step2_time:.2f}")
        st.metric("Step 3 Duration (s)", f"{step3_time:.2f}")
        st.metric("Step 4 Duration (s)", f"{step4_time:.2f}")
        st.metric("Total Duration (s)", f"{total_time:.2f}")
        # Add options for continuing analysis
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Patent Analysis", key="continue_analysis"):
                # Reset session state
                st.session_state.current_step = 1
                st.session_state.uploaded_file = None
                st.session_state.file_details = None
                st.session_state.analysis_started = False
                # 清空耗时记录
                for i in range(1, 6):
                    st.session_state[f'step{i}_time'] = None
                st.rerun()

        with col2:
            if st.button("🏁 End Analysis", key="end_analysis"):
                st.success("Thank you for using the Patent Analysis System!")

# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    prev_disabled = st.session_state.current_step <= 1
    prev_btn = st.button(
        "◀ Previous Step",
        disabled=prev_disabled,
        use_container_width=True,
        help="Go back to the previous step" if not prev_disabled else "This is the first step"
    )

with col3:
    next_disabled = st.session_state.current_step >= 5
    next_btn = st.button(
        "Next Step ▶",
        type="primary",
        disabled=next_disabled,
        use_container_width=True,
        help="Continue to the next step" if not next_disabled else "This is the last step"
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
            with st.spinner('⏳ Generating technical map, please wait...'):
                try:
                    step1_start_time = time.time()
                    st.session_state.tech_genealogy = asyncio.run(
                        tech_genealogy_generator.generate_tech_genealogy(
                            topic=st.session_state.tech_topic,
                            genealogy_type=st.session_state.data_source_type
                        )
                    )
                    step1_end_time = time.time()
                    st.session_state.step1_time = step1_end_time - step1_start_time
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
    st.warning("Please enter a technical topic")
