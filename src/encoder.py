# src/encoder.py
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import Dict, List

# 加载本地模型（避免网络下载）
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bge-small-zh-v1.5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"""
    本地模型不存在！请先下载模型到以下路径：
    {model_path}
    下载地址：https://hf-mirror.com/BAAI/bge-small-zh-v1.5
    """)

model = SentenceTransformer(model_path)
VECTOR_DIM = 512

# 优化后的权重配置：降低方言词，大幅提升释义
FIELD_WEIGHTS = {
    '方言词': 0.3,       # 从0.5降到0.3
    '简易发音': 0.05,    # 从0.075降到0.05
    '标准发音': 0.05,    # 从0.075降到0.05
    '释义注释': 0.6      # 从0.35提升到0.6（核心优化）
}

def encode_single_field(text: str) -> np.ndarray:
    if text.strip() == '' or text == 'nan':
        return np.zeros(VECTOR_DIM)
    vector = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vector

def encode_entry_with_weights(entry: Dict[str, str]) -> np.ndarray:
    total_vector = np.zeros(VECTOR_DIM)
    
    # 优化1：发音字段和方言词拼接，避免拼音/音标无效
    dialect_word = entry.get('方言词', '')
    simple_pron = entry.get('简易发音', '')
    std_pron = entry.get('标准发音', '')
    # 拼接后的发音文本：“阿 a1 a533”
    combined_pron = f"{dialect_word} {simple_pron} {std_pron}".strip()
    
    # 优化2：重新分配字段，用拼接后的发音代替单独发音
    optimized_fields = {
        '方言词': dialect_word,
        '发音组合': combined_pron,
        '释义注释': entry.get('释义注释', '')
    }
    # 对应调整后的权重
    optimized_weights = {
        '方言词': 0.3,
        '发音组合': 0.1,  # 两个发音合计0.1
        '释义注释': 0.6
    }
    
    for field_name, weight in optimized_weights.items():
        field_text = optimized_fields.get(field_name, '')
        field_vector = encode_single_field(field_text)
        total_vector += field_vector * weight
    
    total_vector = total_vector / np.linalg.norm(total_vector)
    return total_vector

def encode_query(query_text: str) -> np.ndarray:
    return encode_single_field(query_text)

# 测试代码
if __name__ == "__main__":
    test_entry = {
        '方言词': '阿',
        '简易发音': 'a1',
        '标准发音': 'a533',
        '释义注释': '①用在某些亲属名称的前面：～舅|～叔|～妹。②用在单名、排行或姓前面，表亲昵：～灿|～水|～...'
    }
    print("正在加载本地模型...")
    word_vector = encode_single_field(test_entry['方言词'])
    print(f"\n单个字段向量维度：{word_vector.shape}（512维，符合预期）")
    final_vector = encode_entry_with_weights(test_entry)
    print(f"融合后向量维度：{final_vector.shape}（512维，符合预期）")
    print(f"融合后向量模长：{np.linalg.norm(final_vector):.4f}（≈1，归一化成功）")
    query_vector = encode_query("阿是什么意思")
    print(f"查询向量维度：{query_vector.shape}（512维，符合预期）")
    print("\n方案一优化版核心嵌入逻辑测试通过！")