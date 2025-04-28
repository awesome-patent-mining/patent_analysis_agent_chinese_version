from datetime import datetime
import os
import time
import streamlit as st
import pandas as pd
import asyncio
from research_agent.core.generate_tech_genealogy import Tech_Gene_Generator
from research_agent.core.utils import transform_data_zh,flatten_tech_structure_zh
from research_agent.core.markdown_display import display_markdown_with_images_from_file
from research_agent.core.applicant_analysis import generate_full_report
from research_agent.core.patent_tech_analysis_1 import PatentTechAnalyzer
# 初始化当前步骤
general_report_generator = PatentTechAnalyzer()
tech_genealogy_generator = Tech_Gene_Generator()
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = 1  # 默认网页
if 'last_used_data_source_type' not in st.session_state:
    st.session_state.last_used_data_source_type = 1
data_source_options = {
    1: "网页",
    2: "专利",
    3: "网页+专利"
}

step_name_dict = {1: "输入技术主题", 2: "生成技术图谱", 3: "获取专利数据", 4: "生成专利报告", 5: "结束"}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
# 初始化技术主题
if 'tech_topic' not in st.session_state:
    st.session_state.tech_topic = ""

if 'tech_genealogy' not in st.session_state:
    st.session_state.tech_genealogy = None

if 'last_used_topic' not in st.session_state:
    st.session_state.last_used_topic = ""  # 新增：记录上一次用于生成技术图谱的主题

# 自定义CSS样式
st.markdown("""
    <style>
    /* Logo样式 */
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

    /* 步骤样式 */
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

# 侧边栏内容
with st.sidebar:
    # 添加Logo和系统名称
    st.markdown("""
        <div class="logo-container">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 28px;">📑</span>
                <span class="logo-text">专利分析报告自动撰写</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.header("分析流程")

    # 生成四个步骤
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
            f"<div class='step' style='{style}'>📌 步骤 {step}：{step_name_dict[step]}</div>",
            unsafe_allow_html=True
        )

# 主界面内容
#st.title("📑 专利分析系统")

# 步骤内容容器
content = st.container()
tech_topic = ""
# 根据当前步骤显示不同内容
with content:
    if st.session_state.current_step == 1:
        st.header("📤 第一步 - 输入技术主题")
        tech_topic = st.text_input(
            label="请输入技术主题（例如：人工智能、云计算等）：",
            placeholder="人工智能",
            value=st.session_state.tech_topic  # 保持输入的一致性
        )
        # 保存输入的技术主题到 session_state 中
        st.session_state.tech_topic = tech_topic
        #print(f"step 1: {st.session_state.tech_topic}")
        # 下拉框，返回值为序号1,2,3
        selected_data_source = st.selectbox(
            "选择生成技术图谱的数据：",
            options=list(data_source_options.keys()),
            format_func=lambda x: data_source_options[x],
            index=list(data_source_options.keys()).index(st.session_state.data_source_type)
        )
        # 实时保存选择
        st.session_state.data_source_type = selected_data_source

        print(f"step 1: {st.session_state.tech_topic}")
    elif st.session_state.current_step == 2:
        st.header("⚙️ 第二步 - 生成技术图谱")
        #st.subheader(f"{st.session_state.tech_topic}的技术图谱")
        # 分割线
        st.write("---")
        # 获取技术图谱
        # 技术图谱定义

        #MAP_TECH = '''
        # # 表格数据
        # initial_data = flatten_tech_structure_zh(st.session_state.tech_genealogy)

        # 表格数据
        st.session_state.map_tech = st.session_state.tech_genealogy

        initial_data = flatten_tech_structure_zh(st.session_state.tech_genealogy)

        # 初始化 session_state 中的表格数据
        if "df" not in st.session_state:
            st.session_state.df = pd.DataFrame(initial_data)
        # **展示表格**
        st.write(f"**{st.session_state.tech_topic}的技术图谱**")
        st.dataframe(st.session_state.df, use_container_width=True)
        #print(st.session_state.df)

    elif st.session_state.current_step == 3:
        st.header("🚀 第三步 - 获取专利数据")
        report_container = st.container()
        with report_container:
            # 添加加载动画
            with st.spinner('⏳ 正在获取专利数据，请稍候...'):
                try:
                    # 异步执行报告生成
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    start_time = time.time()  # 记录开始时间
                    save_dir = os.path.join(os.path.dirname(
                        os.path.abspath(__file__)), "research_agent/general_analysis_output")
                    # 创建基于时间的子目录
                    #print(save_dir)
                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式示例: 20231225_143022
                    # 示例
                    # time_str = "20250425_161638"
                    save_dir = os.path.join(save_dir, time_str)
                    os.mkdir(save_dir)
                    #print(save_dir)
                    # 生成报告内容
                    loop.run_until_complete(
                        general_report_generator.run(save_dir=save_dir, tech_map=st.session_state.tech_genealogy))
                    end_time = time.time()  # 记录结束时间
                    elapsed_time = end_time - start_time  # 计算耗时（秒）
                    print(f"专利统计报告已生成并保存至: {save_dir}")
                    print(f"总耗时: {elapsed_time:.2f} 秒")  # 保留2位小数

                    # 直接加载并展示本地生成的Markdown文件
                    general_report_path = os.path.join(save_dir, 'patent_report.md')
                    #st.write(general_report_path, os.path.exists(general_report_path))
                    if os.path.exists(general_report_path):
                        display_markdown_with_images_from_file(
                            general_report_path, save_dir)
                    else:
                        st.error("专利统计文件未找到")

                except Exception as e:
                    st.error(f"专利统计生成失败: {str(e)}")
                    st.exception(e)  # 显示详细错误信息

    elif st.session_state.current_step == 4:
        st.header("📊 第四步 - 生成专利报告")
        if st.button("🚀 生成专利报告", type="primary", key="generate_patent_trend_report"):
            # 创建报告容器
            report_container = st.container()
            with report_container:
                # 添加加载动画
                with st.spinner('⏳ 正在生成生成专利报告，请稍候...'):
                    try:
                        # 异步执行报告生成
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        start_time = time.time()  # 记录开始时间
                        save_dir = os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), "research_agent\detail_analysis_output")
                        # 创建基于时间的子目录
                        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式示例: 20231225_143022
                        # 示例
                        # time_str = "20250425_161638"
                        time_dir = os.path.join(save_dir, time_str)
                        # 生成报告内容
                        full_report, report_path = loop.run_until_complete(
                            generate_full_report(save_dir=time_dir, map_tech=st.session_state.map_tech))
                        end_time = time.time()  # 记录结束时间
                        elapsed_time = end_time - start_time  # 计算耗时（秒）
                        print(f"生成专利报告已生成并保存至: {time_dir}")
                        print(f"总耗时: {elapsed_time:.2f} 秒")  # 保留2位小数

                        # 直接加载并展示本地生成的Markdown文件
                        report_path = os.path.join(time_dir, '专利分析报告.md')
                        st.write(report_path, os.path.exists(report_path))
                        if os.path.exists(report_path):
                            display_markdown_with_images_from_file(
                                report_path, time_dir)
                        else:
                            st.error("专利报告文件未找到")

                    except Exception as e:
                        st.error(f"专利报告生成失败: {str(e)}")
                        st.exception(e)  # 显示详细错误信息
    elif st.session_state.current_step == 5:
        st.header("📊 第五步 - 结束，是否继续进行专利分析？")
        st.success("✅ 处理结果已就绪！")
        st.metric("处理效率", "98.7%", "1.2%")
        st.progress(80)

# 导航按钮
# 导航按钮
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    prev_disabled = st.session_state.current_step <= 1
    prev_btn = st.button(
        "◀ 上一步",
        disabled=prev_disabled,
        use_container_width=True,
        help="返回上一步骤" if not prev_disabled else "已是第一步"
    )

with col3:
    next_disabled = st.session_state.current_step >= 5
    next_btn = st.button(
        "下一步 ▶",
        type="primary",
        disabled=next_disabled,
        use_container_width=True,
        help="继续下一步骤" if not next_disabled else "已是最后一步"
    )

# 步骤一输入内容为空的提示
if 'show_topic_warn' not in st.session_state:
    st.session_state.show_topic_warn = False

if prev_btn and not prev_disabled:
    st.session_state.current_step -= 1
    st.session_state.show_topic_warn = False  # 回退时清除警告
    st.rerun()

# # 重点：判断步骤一且下一步时判断内容
if next_btn and not next_disabled:
    if st.session_state.current_step == 1:
        if not st.session_state.tech_topic.strip():
            st.session_state.show_topic_warn = True
            st.rerun()
        else:
            st.session_state.show_topic_warn = False
            # 只有在技术主题或数据类型发生变化时才生成新图谱
            if (
                st.session_state.tech_topic != st.session_state.last_used_topic or
                st.session_state.data_source_type != st.session_state.last_used_data_source_type or
                not st.session_state.tech_genealogy
            ):
                # 生成技术图谱
                st.session_state.tech_genealogy = asyncio.run(
                    tech_genealogy_generator.generate_tech_genealogy(
                        topic=st.session_state.tech_topic,
                        genealogy_type=st.session_state.data_source_type
                    )
                )
                # 记录本次用于生成的主题与数据类型
                st.session_state.last_used_topic = st.session_state.tech_topic
                st.session_state.last_used_data_source_type = st.session_state.data_source_type
            # 如果主题和数据类型都没变且已有图谱，直接进入第二步
            st.session_state.current_step += 1
            st.rerun()
    else:
        st.session_state.current_step += 1
        st.session_state.show_topic_warn = False
        st.rerun()

# 展示警告
if st.session_state.current_step == 1 and st.session_state.show_topic_warn:
    st.warning("请输入技术主题内容")