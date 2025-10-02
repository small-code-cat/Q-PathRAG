import re
import json
from tqdm import tqdm
import argparse
from openai import OpenAI

def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"- Question \d+: (.+)", data)

    return queries

def batch_eval(query_file, result1_file, result2_file, output_file_path):
    client = OpenAI(
        # 将这里换成你在便携AI聚合API后台生成的令牌
        api_key="",
        # 这里将官方的接口访问地址替换成便携AI聚合API的入口地址
        base_url=""
    )

    queries = extract_queries(query_file)

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    query2answer1 = {item['query']:item['result'] for item in answers1}
    answers1 = [query2answer1[q] for q in queries]

    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    query2answer2 = {item['query']: item['result'] for item in answers2}
    answers2 = [query2answer2[q] for q in queries]

    results = []
    for i, (query, answer1, answer2) in tqdm(enumerate(zip(queries, answers1, answers2))):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on five criteria: **Comprehensiveness**, **Diversity**, **Logicality**, **Relevance**, and **Coherence**.
        """
# '- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?'
#         '''"Empowerment": {{
#                 "Winner": "[Answer 1 or Answer 2]",
#                 "Explanation": "[Provide explanation here]"
#             }},,
        #             "Overall": {{
        #                 "Winner": "[Answer 1 or Answer 2]",
        #                 "Explanation": "[Summarize why this answer is the overall winner based on the five criteria]"
        #             }}
#         '''
        prompt = f"""
        You will evaluate two answers to the same question based on five criteria: **Comprehensiveness**, **Diversity**, **Logicality**, **Relevance**, and **Coherence**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Logicality**: How logically does the answer respond to all parts of the question?
        - **Relevance**: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
        - **Coherence**: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the five criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Logicality": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Relevance": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Coherence": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }}
        }}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            result_data = {
                "custom_id": f"request-{i + 1}",
                "query": prompt,
                "response": response.choices[0].message.content
            }
            results.append(result_data)
            print(f"[{i + 1}] Success")
        except Exception as e:
            print(f"[{i + 1}] Failed: {e}")
            results.append({
                "custom_id": f"request-{i + 1}",
                "query": prompt,
                "error": str(e)
            })
    # 写入结果到本地文件
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"All results saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 mix、mode1、mode2 参数")

    # 添加三个位置参数
    parser.add_argument("--cls", type=str, help="类别")
    parser.add_argument("--mode1", type=str, help="第一个模式")
    parser.add_argument("--mode2", type=str, help="第二个模式")

    args = parser.parse_args()

    cls = args.cls
    mode1 = args.mode1
    mode2 = args.mode2

    print("cls =", cls)
    print("mode1 =", mode1)
    print("mode2 =", mode2)

    base_dir = "../datasets/questions"
    result_dir = '../result'
    batch_eval(
        f"{base_dir}/{cls}_questions.txt",
        f"{base_dir}/{cls}_{mode1}_result.json",
        f"{base_dir}/{cls}_{mode2}_result.json",
        f"{result_dir}/{cls}_{mode1}_{mode2}_result.json"
    )
