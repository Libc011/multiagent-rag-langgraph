import os
import csv
import numpy as np
import redis
from openai import OpenAI
from dotenv import load_dotenv

from config.load_key import load_key

load_dotenv()

# Redis 连接
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=False,
    timeout=60,      # 默认太短就容易炸
    max_retries=3,   # 自动重试
)
client = OpenAI(
    api_key=load_key("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

EMBED_MODEL = "embedding-3"   # 智谱向量模型

def embed_text(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _auto_delimiter(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
    return "\t" if "\t" in sample else ","


def init_from_file_once(file_path: str = "data/couplets.csv"):
    """
    首次自动从你的两列表(text1,text2)导入 Redis。
    只在 Redis 里没有数据时导入一次。
    """
    has_data = any(r.scan_iter("couplet:*"))
    if has_data:
        return

    if not os.path.exists(file_path):
        print(f"[rag_redis] 未找到语料文件: {file_path}")
        return

    delimiter = _auto_delimiter(file_path)
    count = 0

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            text1 = (row.get("text1") or "").strip()
            text2 = (row.get("text2") or "").strip()
            if not text1 or not text2:
                continue

            vec = embed_text(f"{text1} {text2}")
            key = f"couplet:{i}"
            r.hset(
                key,
                mapping={
                    b"text1": text1.encode("utf-8"),
                    b"text2": text2.encode("utf-8"),
                    b"embedding": vec.tobytes(),
                },
            )
            count += 1

    print(f"[rag_redis] 导入完成: {count} 条")


def search_couplets(query: str, k: int = 5):
    """
    给 director.py 的 song_node 调用：
    返回 [(score, text1, text2), ...]
    """
    init_from_file_once()  # 自动懒加载一次

    qvec = embed_text(query)
    results = []

    for key in r.scan_iter("couplet:*"):
        item = r.hgetall(key)
        emb = item.get(b"embedding")
        t1 = item.get(b"text1")
        t2 = item.get(b"text2")
        if not emb or not t1 or not t2:
            continue

        vec = np.frombuffer(emb, dtype=np.float32)
        score = _cos(qvec, vec)
        results.append((score, t1.decode("utf-8"), t2.decode("utf-8")))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]
