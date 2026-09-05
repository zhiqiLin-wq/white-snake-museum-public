"""诊断：用 safetensors 从本地路径加载 bge-large，torch 直接 encode。"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

LOCAL = "models_cache/bge-large-zh-v1.5-onnx"

print("loading tokenizer + model (safetensors) ...", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(LOCAL)
model = AutoModel.from_pretrained(LOCAL, use_safetensors=True)
model.eval()
print(f"loaded in {time.time()-t0:.1f}s", flush=True)


def encode(texts, instruction=None):
    if instruction:
        texts = [instruction + t for t in texts]
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :]
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.numpy()


print("single encode ...", flush=True)
t0 = time.time()
e1 = encode(["白娘子的丈夫是谁"], instruction="为这个句子生成表示以用于检索相关文章：")
print(f"single: {time.time()-t0:.2f}s dim={e1.shape[1]}", flush=True)

print("batch 32 encode ...", flush=True)
texts = ["白蛇传故事白娘子许仙法海雷峰塔西湖断桥" * 20] * 32
t0 = time.time()
e32 = encode(texts)
print(f"batch32: {time.time()-t0:.2f}s dim={e32.shape[1]}", flush=True)

print("DONE", flush=True)
