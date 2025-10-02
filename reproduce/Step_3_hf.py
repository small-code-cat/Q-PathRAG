import re
import json
from GoGRAG import GoGRAG, QueryParam
from GoGRAG.utils import EmbeddingFunc
from GoGRAG.hf import hf_model_complete, hf_embed
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

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
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    cls = "mix"
    mode = "hybrid"
    WORKING_DIR = f"../{cls}"

    rag = GoGRAG(
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
    query_param = QueryParam(mode=mode)

    base_dir = "../datasets/questions"
    queries = extract_queries(f"{base_dir}/{cls}_questions.txt")
    run_queries_and_save_to_json(
        queries, rag, query_param, f"{base_dir}/{cls}_{mode}_result.json", f"{base_dir}/{cls}_{mode}_errors.json"
    )
