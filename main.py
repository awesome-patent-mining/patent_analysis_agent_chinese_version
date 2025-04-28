import asyncio
from dotenv import load_dotenv
import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from research_agent.core.config import Config
from research_agent.core.run_xgboost import Run_xgb
from pyaml_env import parse_config
load_dotenv()

import argparse
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_output_dir = Path(r"log_file")
log_output_dir.mkdir(exist_ok=True)
console_handler = logging.StreamHandler()
f_handler = logging.FileHandler(str(log_output_dir)+"/"+f"{timestamp}.log",encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler,f_handler])
logger = logging.getLogger(__name__)
import hashlib
def calculate_file_hash(file_path):
    hash_func= getattr(hashlib,"md5")()
    with open(file_path,"rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def paras_args():
    parser = argparse.ArgumentParser(description='generate a research survey based on input prompt')
    # 添加参数 --topic，这是一个可选参数
    parser.add_argument('--topic', type=str, help='Input the topic of the survey', required=True)
    parser.add_argument('--language', type=str, help='Input the language type',default="English")
    parser.add_argument('--token', type=str, help='Input api_key',required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    args = paras_args()

    async def main(args):

        draft_iteration_output_dir = Path(r"draft_iteration_output")
        draft_iteration_output_dir.mkdir(exist_ok=True)

        final_survey_output_dir = Path(r"survey_output")
        final_survey_output_dir.mkdir(exist_ok=True)


        xgb_model_output_dir = Path(r"xgb_model_output")
        xgb_model_output_dir.mkdir(exist_ok=True)

        #topic = "What does the technology development roadmap for multi-modal large models look like?"
        from research_agent.core.pipeline import Pipeline


        # 读出yaml文件，将api_key写入

        absolute_path = os.path.abspath(Config.YAML_CONFIG)
        configs = parse_config(absolute_path)

        configs['glm-4']['API_KEY'] = args.token
        # 读取YAML文件
        with open(absolute_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(configs, file)

        pipeline = Pipeline(draft_iteration_output_dir=draft_iteration_output_dir,xgb_model_output_dir = xgb_model_output_dir)
        survey_markdown = await pipeline.iteration(topic=args.topic,language=args.language)

        with open('review.md', 'w', encoding='utf-8') as file:
            file.write(survey_markdown)

        hash_value = calculate_file_hash('review.md')
        print(f"当前文章对应的MD5码如下：{hash_value}")
    asyncio.run(main(args))
