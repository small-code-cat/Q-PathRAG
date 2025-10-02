import os
import json
import time

from PathRAG import PathRAG
from GoGRAG.utils import EmbeddingFunc
from GoGRAG.hf import hf_model_complete, hf_embed
from transformers import AutoModel, AutoTokenizer


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

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

def initialize_rag():
    rag = PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name="/mnt/data/xkj/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        embedding_func=EmbeddingFunc(
            embedding_dim=384, max_token_size=8192, func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "/mnt/data/xkj/sentence-transformers/all-MiniLM-L6-v2"
                ),
                embed_model=AutoModel.from_pretrained(
                    "/mnt/data/xkj/sentence-transformers/all-MiniLM-L6-v2"
                ),
            ),
        ),
    )

    return rag


def main():
    # Initialize RAG instance
    rag = initialize_rag()
    insert_text(rag, f"../datasets/unique_contexts/{cls}_unique_contexts.json")


if __name__ == "__main__":
    main()
