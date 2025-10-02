import re
import json
from tqdm import tqdm
import os
import argparse
# from PathRAG import PathRAG, QueryParam
# from PathRAG.llm import gpt_4o_mini_complete
# from LightRAG import LightRAG, QueryParam
# from LightRAG.llm import gpt_4o_mini_complete
from QPathRAG import QPathRAG, QueryParam
from QPathRAG.llm import gpt_4o_mini_complete

api_key = ""
os.environ["OPENAI_API_KEY"] = api_key

'''
让模型基于step1中已经提取的知识图谱对step2中的问题进行回答并保存
'''

def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"- Question \d+: (.+)", data)

    return queries


def process_query(query_text, rag_instance, query_param):
    try:
        result = rag_instance.query(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    all_results = []
    all_errors = []

    for query_text in tqdm(queries):
        result, error = process_query(query_text, rag_instance, query_param)

        if result:
            all_results.append(result)
        elif error:
            all_errors.append(error)

    # 写入结果
    if all_results:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    if all_errors:
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(all_errors, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="step3")

    # 添加三个位置参数
    parser.add_argument("--cls", type=str, help="类别，一个类别只能同时运行一个实例")
    parser.add_argument("--mode", type=str, help="第一个模式")
    parser.add_argument("--base", type=str, help="工作路径的基础路径")

    args = parser.parse_args()

    cls = args.cls
    mode = args.mode
    base = args.base

    print("cls =", cls)
    print("mode =", mode)
    print("base =", base)

    WORKING_DIR = f"../{base}/dickens_{cls}"

    rag = QPathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
    )
    query_param = QueryParam(mode=mode)

    base_dir = "../datasets/questions"
    queries = extract_queries(f"{base_dir}/{cls}_questions.txt")
    run_queries_and_save_to_json(
        queries, rag, query_param, f"{base_dir}/{cls}_{mode}_result.json", f"{base_dir}/{cls}_{mode}_errors.json"
    )
