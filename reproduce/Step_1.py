import os
import json
import time

from QPathRAG import QPathRAG
from QPathRAG.llm import gpt_4o_mini_complete

'''
利用模型提取知识图谱的能力，将step0中的文档提取成知识图谱
'''

def insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


cls = "mix"
WORKING_DIR = f"../{cls}"

api_key = ""
os.environ["OPENAI_API_KEY"] = api_key

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


def initialize_rag():
    rag = QPathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
    )
    return rag


def main():
    # Initialize RAG instance
    rag = initialize_rag()
    insert_text(rag, f"../datasets/unique_contexts/{cls}_unique_contexts.json")


if __name__ == "__main__":
    main()
