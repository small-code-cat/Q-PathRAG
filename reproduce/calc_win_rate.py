import json
import re

def balance_json_braces(json_str: str) -> str:
    left = json_str.count("{")
    right = json_str.count("}")
    if left == right:
        return json_str  # ✅ 已配对
    elif left > right:
        # 缺右括号，补足
        json_str += "}" * (left - right)
    elif right > left:
        # 缺左括号，前面补左括号
        json_str = "{" * (right - left) + json_str
    return json_str.replace('”', '"').replace('”', '"')

def compute_win_rates(data, mode_dict, dimensions):
    answer_1, answer_2 = list(mode_dict.values())
    win_counts = {dim: {answer_1: 0, answer_2: 0} for dim in dimensions}

    for item in data:
        for dim in dimensions:
            winner = item['response'].get(dim, {}).get("Winner", "").strip()
            win_counts[dim][mode_dict[winner]] += 1

    for dim in dimensions:
        total = win_counts[dim][answer_1] + win_counts[dim][answer_2]
        if total == 0:
            print(f"- {dim}: 无有效对比数据")
            continue
        win_counts[dim][answer_1] /= total
        win_counts[dim][answer_2] /= total

    return win_counts

def main(cls, mode1, mode2):
    dimensions = ['Comprehensiveness', 'Diversity', 'Logicality', 'Relevance', 'Coherence']

    compare_file1 = f'../result/{cls}_{mode1}_{mode2}_result.json'
    mode_dict1 = {'Answer 1': mode1, 'Answer 2': mode2}

    compare_file2 = f'../result/{cls}_{mode2}_{mode1}_result.json'
    mode_dict2 = {'Answer 1': mode2, 'Answer 2': mode1}

    with open(compare_file1, "r") as f:
        result1 = json.load(f)
    for i in result1:
        response = balance_json_braces(re.search(r"{.*}", i['response'], re.DOTALL).group(0))
        i['response'] = json.loads(response)
    win_counts1 = compute_win_rates(
        result1,
        mode_dict1,
        dimensions
    )

    with open(compare_file2, "r") as f:
        result2 = json.load(f)
    for i in result2:
        response = balance_json_braces(re.search(r"{.*}", i['response'], re.DOTALL).group(0))
        i['response'] = json.loads(response)
    win_counts2 = compute_win_rates(
        result2,
        mode_dict2,
        dimensions
    )

    avg_dict = {
        d: {
            sub_key: (win_counts1[d][sub_key] + win_counts2[d][sub_key]) / 2
            for sub_key in win_counts1[d]
        }
        for d in dimensions
    }

    print(f"{cls} 胜率统计 | {mode1} vs {mode2}")
    for dim in dimensions:
        print(f"- {dim}:")
        print(f"  - {mode1} 胜率: {avg_dict[dim][mode1]:.2%}")
        print(f"  - {mode2} 胜率: {avg_dict[dim][mode2]:.2%}")

if __name__ == '__main__':
    cls = 'mix'
    mode1 = "path"
    mode2 = 'qpath-ktn-25'

    main(cls, mode1, mode2)