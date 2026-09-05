"""Cross-Encoder 重排序器 —— 支持 ONNX / PyTorch 后端，默认 ONNX 加速。

CU-29~CU-31: Reranker 完整实现。
"""

# PyTorch 2.13.0 在 Windows 上默认12线程加载大模型会 segfault (C 层崩溃)
# 必须在任何 torch/sentence_transformers import 之前设置，否则环境变量不生效
import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_os.environ.setdefault("OMP_NUM_THREADS", "4")
_os.environ.setdefault("MKL_NUM_THREADS", "4")

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder 精排器 (CU-29)。

    使用 sentence_transformers.CrossEncoder 作为后端，
    模型在后台线程加载，不阻塞服务启动 (CU-30)。

    支持 ONNX / PyTorch 两种后端:
    - backend="onnx"  (默认): CPU 上约 2x 加速，需安装 optimum
    - backend="torch": PyTorch 原生推理，GPU 场景更优
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
                 backend: str = "onnx"):
        self.model_name = model_name
        self.backend = backend  # "onnx" | "torch"
        self._model = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._is_ready: bool = False
        self._load_error: str | None = None
        self._loading: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # CU-30: 后台模型加载
    # ------------------------------------------------------------------
    def start_loading(self):
        """在后台线程加载 CrossEncoder 模型。

        所有重型 import 必须在主线程完成：Windows 上 Python 的全局 import 锁
        与 DLL 加载交互时，后台线程 import torch/transformers 极易死锁。
        """
        if self._loading:
            return

        # ---- 主线程预加载：必须在 spawn 线程之前完成所有重型 import ----
        from sentence_transformers import CrossEncoder as _CE_preload  # noqa: F401

        self._loading = True
        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()
        logger.info(f"Reranker: 后台加载模型 {self.model_name} ...")

    def _load_model(self):
        """实际加载模型（后台线程）。

        策略: 先用 huggingface_hub 解析本地缓存路径（零网络），
        再将本地路径传给 CrossEncoder，彻底避免 list_repo_files() 网络请求。

        根据 self.backend 选择加载路径:
        - "onnx": 使用 sentence_transformers ONNX 后端 (CPU 上约 2x 加速)
        - "torch": 使用 PyTorch 原生后端
        """
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import CrossEncoder

            backend = getattr(self, "backend", "onnx")

            # 先解析本地缓存路径（local_files_only=True 确保零网络）
            local_path = snapshot_download(
                self.model_name,
                local_files_only=True,
            )
            logger.info(
                f"Reranker: 模型缓存路径 {local_path} (backend={backend})"
            )

            # 用本地路径加载，CrossEncoder 直接读磁盘
            if backend == "onnx":
                model = CrossEncoder(local_path, backend="onnx")
            else:
                model = CrossEncoder(local_path)

            with self._lock:
                self._model = model
                self._is_ready = True
            logger.info(f"Reranker: 模型 {self.model_name} 加载完成 (backend={backend})")
        except Exception as e:
            logger.error(f"Reranker: 模型加载失败 {e}")
            with self._lock:
                self._load_error = str(e)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._is_ready

    @property
    def status_info(self) -> dict:
        with self._lock:
            if self._is_ready:
                return {"status": "ready", "model": self.model_name}
            elif self._load_error:
                return {"status": "error", "model": self.model_name, "error": self._load_error}
            else:
                return {"status": "loading", "model": self.model_name}

    # ------------------------------------------------------------------
    # CU-31: rerank() 方法
    # ------------------------------------------------------------------
    def rerank(
        self, query: str, documents: list[str],
        top_k: Optional[int] = None, batch_size: int = 32,
    ) -> list[tuple[int, float]]:
        """对候选文档重排序。

        Args:
            query: 查询文本
            documents: 候选文档文本列表
            top_k: 返回前 top_k 个结果，None 表示返回全部
            batch_size: 批量预测大小

        Returns:
            [(doc_index_in_input_list, relevance_score), ...] 按分数降序
        """
        if not self._is_ready or self._model is None:
            raise RuntimeError("Reranker: 模型未就绪")

        if not documents:
            return []

        # 构建 (query, doc) 对
        pairs = [(query, doc) for doc in documents]

        # 批量预测
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        # 转为 (index, score) 列表
        scored = [(i, float(scores[i])) for i in range(len(scores))]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored
