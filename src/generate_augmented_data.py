import pandas as pd
import time
from tqdm import tqdm
from openai import OpenAI

# ====================== 配置区域 ======================
# 你的Ollama模型名称（注意：要和你在Ollama里拉取的名字完全一致）
# 通常是 deepseek-r1:8b 或 deepseek-r1:latest
OLLAMA_MODEL_NAME = "deepseek-r1:8b"
# 原始方言词典Excel路径
INPUT_EXCEL = "data/dialect_dict.xlsx"
# 生成的增强版Excel保存路径
OUTPUT_EXCEL = "data/dialect_dict_augmented.xlsx"
# Ollama API 初始化（Ollama默认运行在 http://localhost:11434）
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama不需要真实API Key，随便填一个即可
)
# ====================== 配置结束 ======================

def generate_augmented_text(dialect_word: str, definition: str) -> str:
    """
    用本地Ollama的DeepSeek R1模型生成检索增强文本
    :param dialect_word: 兴化方言词
    :param definition: 词条对应的普通话释义
    :return: 空格分隔的增强文本
    """
    # 约束性Prompt，保证输出格式符合要求
    prompt = f"""
你是兴化方言专家，请为以下兴化方言词条生成检索增强文本。
严格遵守以下要求：
1. 生成3-5个该词条的普通话同义词/近义词；
2. 生成2-3个用户可能用普通话提问的句式；
3. 所有内容用空格分隔，不要标点、不要换行、不要额外解释；
4. 必须包含原方言词和原释义。

示例：
方言词：郎罢
释义：父亲，爸爸
输出：郎罢 父亲 爸爸 老爸 爹 父亲用方言怎么说 爸爸用方言怎么说 父亲，爸爸

现在处理：
方言词：{dialect_word}
释义：{definition}
输出：
    """

    # 调用本地Ollama API
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # 低温保证输出稳定
            max_tokens=150    # 限制输出长度
        )
        # 提取并返回生成的增强文本
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成出错（词条：{dialect_word}）：{e}")
        # 调用出错时，返回原始内容保底
        return f"{dialect_word} {definition}"

def main():
    """批量生成增强文本的主函数"""
    # 1. 读取原始方言词典
    print(f"正在读取原始Excel：{INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, engine="openpyxl")
    
    # 2. 批量生成增强文本
    augmented_texts = []
    print(f"开始生成增强文本，共 {len(df)} 条词条...")
    print(f"使用模型：{OLLAMA_MODEL_NAME}（本地Ollama）")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        dialect_word = str(row["方言词"])
        definition = str(row["释义注释"])
        # 生成单条增强文本
        augmented = generate_augmented_text(dialect_word, definition)
        augmented_texts.append(augmented)
        # 加轻微延迟，避免Ollama过载
        time.sleep(0.1)
    
    # 3. 将增强文本加入数据集
    df["检索增强文本"] = augmented_texts
    
    # 4. 保存为增强版Excel
    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"增强文本生成完成！已保存到：{OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()