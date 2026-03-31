# """
# 语义嵌入的作用，是把文本转化为向量，让计算机能够理解词语之间的语义相似性，
# 从而实现同义词匹配、普通话查方言、自然语言提问、描述性查询，
# """
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import Dict

# ====================== 配置 ======================
# bge模型路径
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-small-zh-v1.5")
VECTOR_DIM = 512  # bge-small固定512维
# 字段权重：释义权重最高
FIELD_WEIGHTS = {
    "definition": 0.6,    # 释义：核心权重，适配普通话查方言
    "dialect_word": 0.3,  # 方言词：辅助匹配
    "simple_pron": 0.05,  # 简易发音：轻微辅助
    "standard_pron": 0.05 # 标准发音：轻微辅助
}
# 字段映射
FIELD_MAPPING = {
    "dialect_word": "方言词",
    "simple_pron": "简易发音",
    "standard_pron": "标准发音",
    "definition": "释义注释"
}
# ===================================================

# 全局单例模型
_model = None

def load_embedding_model():
    """加载嵌入模型，仅加载一次"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"bge模型不存在：{MODEL_PATH}")
        print("加载bge嵌入模型...")
        _model = SentenceTransformer(MODEL_PATH)
    return _model

def encode_single_text(text: str) -> np.ndarray:
    """单文本生成归一化向量（空文本返回全0）"""
    if text.strip() == "" or text == "nan":
        return np.zeros(VECTOR_DIM)
    model = load_embedding_model()
    return model.encode(text, normalize_embeddings=True, show_progress_bar=False)

def encode_entry(entry: Dict[str, str]) -> np.ndarray:
    """词条向量：按权重融合4个字段"""
    total_vec = np.zeros(VECTOR_DIM)
    for field_key, weight in FIELD_WEIGHTS.items():
        field_name = FIELD_MAPPING[field_key]
        field_text = entry.get(field_name, "")
        total_vec += encode_single_text(field_text) * weight
    # 归一化（保证余弦相似度准确）
    norm = np.linalg.norm(total_vec)
    return total_vec / norm if norm > 1e-6 else total_vec

def encode_query(query_text: str) -> np.ndarray:
    """查询向量：用户输入生成向量"""
    return encode_single_text(query_text)

# 测试代码
if __name__ == "__main__":
    test_entry = {"方言词": "伓", "释义注释": "相当于普通话“不”"}
    vec = encode_entry(test_entry)
    print(f"向量维度：{vec.shape}，模长：{np.linalg.norm(vec):.3f}")  # 模长应为1
