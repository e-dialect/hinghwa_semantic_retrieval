#加载excel数据 + 精准倒排索引
import pandas as pd
import os
from typing import Dict, List, Tuple

# ====================== 核心配置：你的Excel信息 ======================
EXCEL_PATH = "data/dialect_dict.xlsx"  

FIELD_MAPPING = {
    "dialect_word": "方言词",
    "simple_pron": "简易发音",
    "standard_pron": "标准发音",
    "definition": "释义注释"
}
# ====================================================================

# 全局单例：方言词精准倒排索引（方言词→全字段数据）
INVERTED_INDEX: Dict[str, dict] = {}
# 全局单例：全量数据集
FULL_DF: pd.DataFrame = None

def load_excel_data() -> Tuple[pd.DataFrame, List[str]]:
    """加载Excel数据，构建精准倒排索引（方言查方言核心）"""
    global FULL_DF
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel文件不存在：{EXCEL_PATH}，请放在data文件夹下")
    
    # 读取并清洗数据（空值填充、去空格）
    print("正在加载方言Excel数据...")
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    df = df.dropna(how="all").copy()  # 删除空行
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().fillna("")  # 去空格+空值处理
    
    # 生成唯一词条ID
    df["entry_id"] = [f"entry_{i:04d}" for i in range(len(df))]
    FULL_DF = df
    
    # 构建倒排索引（支持方言词精准匹配）
    build_inverted_index(df)
    print(f"数据加载完成！共{len(df)}条词条，精准索引构建成功")
    return df, df["entry_id"].tolist()

def build_inverted_index(df: pd.DataFrame):
    """构建方言词倒排索引，确保方言查方言100%精准"""
    global INVERTED_INDEX
    INVERTED_INDEX.clear()
    for _, row in df.iterrows():
        dialect_word = row[FIELD_MAPPING["dialect_word"]]
        index_key = dialect_word.lower()  # 支持大小写不敏感匹配
        INVERTED_INDEX[index_key] = row.to_dict()

def exact_match_search(query: str) -> List[dict]:
    """方言词精准查询：输入方言词，返回全字段数据"""
    if not INVERTED_INDEX:
        load_excel_data()  # 未加载则自动加载
    query_key = query.strip().lower()
    # 精准匹配优先
    if query_key in INVERTED_INDEX:
        return [INVERTED_INDEX[query_key]]
    # 子串匹配兜底（如查“𢫫裤”匹配“输会𢫫裤”）
    return [data for word, data in INVERTED_INDEX.items() if query_key in word]

def get_full_df() -> pd.DataFrame:
    """获取全量数据集（供语义检索用）"""
    if FULL_DF is None:
        load_excel_data()
    return FULL_DF

# 测试代码：运行此文件可验证数据加载
if __name__ == "__main__":
    load_excel_data()
    test_res = exact_match_search("郎罢")  # 替换为你的方言词测试
    print(f"精准查询测试结果：{test_res}")