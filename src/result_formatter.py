#返回结果结构化，返回结构化字段
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
from src.data_loader import FIELD_MAPPING

def format_result(results: List[Dict]) -> str:
    """格式化输出：仅显示你的4个字段，清晰易读"""
    if not results:
        return "未找到匹配词条，我还需更新，麻烦换种问法~"
    
    output = ["="*60, "检索结果（按匹配度排序）", "="*60]
    for idx, res in enumerate(results, 1):
        # 提取4个核心字段
        dialect_word = res[FIELD_MAPPING["dialect_word"]]
        simple_pron = res[FIELD_MAPPING["simple_pron"]]
        standard_pron = res[FIELD_MAPPING["standard_pron"]]
        definition = res[FIELD_MAPPING["definition"]]
        # 组装输出（含相似度，语义检索才有）
        output.append(f"\n【结果{idx}】")
        output.append(f"方言词：{dialect_word}")
        output.append(f"简易发音：{simple_pron if simple_pron else '无'}")
        output.append(f"标准发音：{standard_pron if standard_pron else '无'}")
        output.append(f"释义：{definition}")
        if "相似度" in res:
            output.append(f"匹配度：{res['相似度']}")
        output.append("-"*60)
    return "\n".join(output)

# 测试代码
if __name__ == "__main__":
    test_res = [{"方言词": "漉", "简易发音": "lorh6", "标准发音": "lɒʔ1", "释义注释": "踩、涉", "相似度": 0.98}]
    print(format_result(test_res))