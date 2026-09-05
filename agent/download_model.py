"""预下载 Embedding 模型（带进度显示）。"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "shibing624/text2vec-base-chinese"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "models_cache")

print(f"Downloading {MODEL_NAME} ...")
print(f"Cache: {CACHE_DIR}")
print(f"Mirror: {os.environ['HF_ENDPOINT']}")
print()

# Snapshot download with progress bars
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

snapshot_download(
    repo_id=MODEL_NAME,
    cache_dir=CACHE_DIR,
    resume_download=True,
    tqdm_class=tqdm,
)
print()
print("[OK] Model download complete!")
