#主程序链接加载 + 嵌入 + 向量 + 查询分析重写 + 结构化返回
# 注意：需先按方案A或B，替换src/query_rewriter.py
from src.query_rewriter import parse_query  # 查询重写（方案A/B共用）
from src.vector_db import core_search      # 核心检索
from src.result_formatter import format_result  # 结果格式化

def main():
    print("="*60)
    print("        莆仙方言精准检索系统")
    print("="*60)
    print("支持查询：")
    print("1. 方言查方言（如：漉、𢫫裤）")
    print("2. 普通话/释义查方言（如：爸爸、踩水）")
    print("输入 q/quit 退出\n")

    while True:
        user_input = input("请输入查询：").strip()
        if user_input.lower() in ["q", "quit"]:
            print("再见！")
            break
        if not user_input:
            print("查询不能为空，请重新输入！\n")
            continue
        
        try:
            # 1. 解析查询（LLM重写核心）
            parsed_query = parse_query(user_input)
            # 2. 执行检索
            results = core_search(parsed_query)
            # 3. 格式化输出
            print("\n" + format_result(results) + "\n")
        except Exception as e:
            print(f"出错：{e}\n")
            continue

if __name__ == "__main__":
    main()