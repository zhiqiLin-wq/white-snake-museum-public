"""中文 Embedding 模型管理 — PyTorch 推理。

使用 BAAI/bge-large-zh-v1.5 (1024 维 CLS 池化) + PyTorch 推理。
模型权重从本地 safetensors 加载。

背景：本机 onnxruntime 在 Windows 上存在版本冲突（1.19.2 内存 arena 泄漏、
1.15.1 无法加载 IR v10 模型），且 torch.load 被 transformers 因 CVE-2025-32434
限制，故改用 safetensors + PyTorch 直接推理。torch 2.5.1 + safetensors 是
troubleshooting 文档验证过的稳定组合。
"""

import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_os.environ.setdefault("OMP_NUM_THREADS", "8")
_os.environ.setdefault("MKL_NUM_THREADS", "8")
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# BGE v1.5 检索查询指令前缀。
# 仅加在 query 侧 (embed_query)，语料侧 (embed) 不加。
# BAAI 官方要求 bge-*-v1.5 检索时 query 需加此指令，缺失会降低 dense 召回。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 模型缓存目录 (项目根目录下的 models_cache)，含 model.safetensors + config + tokenizer
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_CACHE_DIR = _PROJECT_ROOT / "models_cache" / "bge-large-zh-v1.5-onnx"


class Embedder:
    """中文文本向量化 (bge-large-zh-v1.5, 1024 维, PyTorch)。

    BGE 模型使用 CLS Token 池化 + L2 归一化。
    模型在后台线程加载，不阻塞 FastAPI 启动。
    """

    def __init__(self, model_name: str, local_path: str | None = None):
        self.model_name = model_name
        self.local_path = local_path
        self._model = None          # transformers.AutoModel (torch)
        self._tokenizer = None      # transformers.AutoTokenizer
        self._load_progress = 0.0
        self._load_error: str | None = None
        self._loading = False
        self._lock = threading.Lock()

    def start_loading(self):
        """在后台线程启动模型加载。

        所有重型 import 必须在主线程完成，避免 Windows 后台线程 import 死锁。
        """
        if self._loading:
            return

        # 主线程预加载重型模块
        import torch as _torch_preload  # noqa: F401
        from transformers import AutoModel as _am_preload  # noqa: F401
        from transformers import AutoTokenizer as _tok_preload  # noqa: F401

        self._loading = True
        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()

    def _load_model(self):
        """加载模型 (在后台线程中执行)。用 safetensors 从本地缓存加载。

        已知坑（本项目实测）：新版 transformers 在 safetensors + accelerate 可用时
        强制走 meta-device 加载路径（即使显式传 low_cpu_mem_usage=False 也会被忽略）。
        持久权重会被 assign 成真实 CPU 存储，但 `embeddings.position_ids` 是
        non-persistent buffer、不在 safetensors 权重文件里（已实测文件头 391 键无此键），
        会残留在 meta 设备上 —— 构造成功、首次 forward 报
        "Tensor on device meta is not on the expected device cpu"。
        因此加载后需扫描并重建 meta 残留 buffer，再做 warmup forward 自检。
        """
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer

            model_dir = self.local_path or str(MODEL_CACHE_DIR)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            # 用 safetensors 加载，绕开 transformers 对 torch.load 的 CVE 限制
            model = AutoModel.from_pretrained(model_dir, use_safetensors=True)
            model.eval()
            # 注意：不要 model.to("cpu") —— 若存在 meta 残留 buffer，
            # .to() 会抛 "Cannot copy out of meta tensor"。应重建 buffer（见下）。

            # ---- 重建 meta 残留 buffer ----
            fixed_buffers: list[str] = []
            for name, buf in list(model.named_buffers()):
                if buf.device.type == "meta":
                    if name == "embeddings.position_ids":
                        new_buf = torch.arange(model.config.max_position_embeddings).expand((1, -1))
                        model.embeddings.register_buffer("position_ids", new_buf, persistent=False)
                        fixed_buffers.append(name)
                    else:
                        # 未知的 meta buffer 无法安全重建，fail fast 暴露到 _load_error
                        raise RuntimeError(f"模型存在未知 meta 设备 buffer: {name}，无法自动重建")
            if fixed_buffers:
                logger.info(f"Rebuilt meta buffers on CPU: {fixed_buffers}")

            # 参数也应全部在 CPU 上（assign 路径下 safetensors 是 CPU mmap 存储）；
            # 若仍有参数留在 meta 上说明权重文件缺键，让 warmup 或此处 fail fast
            meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
            if meta_params:
                raise RuntimeError(f"模型存在 meta 设备参数（权重文件缺键？）: {meta_params[:5]}")

            # ---- warmup forward 自检（用局部变量，不依赖 self._model）----
            warmup_enc = tokenizer(
                ["模型加载自检"],
                padding=True,
                truncation=True,
                max_length=16,
                return_tensors="pt",
            )
            with torch.no_grad():
                warmup_out = model(**warmup_enc)
            dim = warmup_out.last_hidden_state.shape[-1]
            logger.info(f"Embedding model warmup OK: hidden_dim={dim}")

            with self._lock:
                self._model = model
                self._tokenizer = tokenizer
                self._load_progress = 1.0

            logger.info(f"Embedding model loaded (torch): {self.model_name}")

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            with self._lock:
                self._load_error = str(e)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def progress(self) -> float:
        with self._lock:
            return self._load_progress

    @property
    def load_error(self) -> str | None:
        with self._lock:
            return self._load_error

    @property
    def status_info(self) -> dict:
        with self._lock:
            if self._model is not None:
                return {"status": "ready", "progress": 1.0, "model": self.model_name}
            elif self._load_error:
                return {"status": "error", "progress": self._load_progress, "model": self.model_name, "error": self._load_error}
            else:
                return {"status": "downloading", "progress": self._load_progress, "model": self.model_name}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """对文本列表做向量化。

        PyTorch 推理流程:
          1. Tokenize -> input_ids / attention_mask / token_type_ids
          2. model forward -> last_hidden_state (batch, seq, 1024)
          3. CLS pooling: hidden[:, 0, :] -> (batch, 1024)
          4. L2 normalization: emb / ||emb||
        """
        if not self._model or not self._tokenizer:
            raise RuntimeError("Embedding model not loaded yet")

        if not texts:
            return []

        import torch

        enc = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            out = self._model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :]  # CLS token pooling
        cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
        return cls_emb.numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        # 加 BGE 检索指令前缀 (仅 query 侧)。缺失时 bge-*-v1.5 的 dense 检索会掉点。
        return self.embed([BGE_QUERY_INSTRUCTION + text])[0]
