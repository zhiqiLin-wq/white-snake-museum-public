"""标签数据存储：加载 chunk 标签 + 词表，提供检索增强所需的数据结构。

- chunk 标签归一化（原始值 -> canonical/raw 双名）
- IDF 表（canonical 实体区分度加权）
- 标签增强文本（原文 + 标签词，供 BM25）
- 标签向量库（标签文本 embed，供语义召回）

数据来源：rag_eval/results/ 下的 chunk_labels_full.json 与 merged_vocab_full.json。
"""
import json
import math
from collections import defaultdict
from pathlib import Path

logger = None
try:
    import logging
    logger = logging.getLogger(__name__)
except Exception:
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TAG_DIR = _PROJECT_ROOT / "rag_eval" / "results"
CHUNK_LABELS_PATH = TAG_DIR / "chunk_labels_full.json"
VOCAB_PATH = TAG_DIR / "merged_vocab_full.json"
EMB_CACHE_DIR = TAG_DIR / "emb_cache"
CORPUS_PATH = _PROJECT_ROOT / "rag_eval" / "ground_truth" / "corpus.jsonl"


def _load_vecs(name):
    """从磁盘缓存加载向量 dict（{id: vec}），无缓存返回 None。"""
    import numpy as np
    p = EMB_CACHE_DIR / f"{name}.npz"
    if not p.exists():
        return None
    try:
        data = np.load(p, allow_pickle=True)
        return {str(cid): vec.tolist() for cid, vec in zip(data["cids"], data["vecs"])}
    except Exception:
        return None


def _save_vecs(name, vec_map):
    """把向量 dict 落盘缓存。"""
    import numpy as np
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cids = list(vec_map.keys())
    vecs = np.asarray([vec_map[cid] for cid in cids], dtype=float)
    np.savez(EMB_CACHE_DIR / f"{name}.npz", cids=np.array(cids, dtype=str), vecs=vecs)


def load_corpus(path=CORPUS_PATH):
    """从 corpus.jsonl 加载语料，返回 {chunk_id: {"text", "metadata"}}。

    生产问答检索用，与 rag_eval 评测语料完全对齐（doc_id == chunk_id）。
    不走 rag_eval 包导入，因为生产以 `python -m server.main` 启动时 rag_eval 不在 sys.path。
    """
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id") or doc.get("_id")
            if not doc_id:
                continue
            corpus[doc_id] = {
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            }
    return corpus

IDF_MIN = 0.3
IDF_MAX = 1.5
IDF_THRESHOLD = 0.8  # 标签语义路过滤高频实体的 IDF 阈值（低于此值视为高频、区分度低）


def _cset(items):
    return {x["canonical"] for x in items if isinstance(x, dict) and x.get("canonical")}


def _wait_embedder_ready(embedder, timeout: int = 300) -> bool:
    """等待后台加载的 embedding 模型就绪。

    首次启动（三个向量缓存为空）时 build_*_vectors 需要 embedder，
    而模型在后台线程加载中——必须等待，否则 lifespan 直接 RuntimeError。
    """
    import time
    waited = 0
    while not embedder.is_ready and waited < timeout:
        if embedder.load_error:
            return False
        if waited % 30 == 0 and logger:
            logger.info(f"等待 embedding 模型加载... ({waited}s)")
        time.sleep(3)
        waited += 3
    return embedder.is_ready


def _embed_batched(embedder, texts, batch: int = 4):
    """小批量 embed。

    本机空闲内存有限时，单批数百条文本会触发 transformers/torch 原生崩溃
    （0xC0000005，实测与 rebuild 阶段批 16 崩溃同因），批 4 全流程稳定。
    """
    out: list = []
    total = len(texts)
    for i in range(0, total, batch):
        out.extend(embedder.embed(texts[i:i + batch]))
        if logger and (i // batch) % 10 == 0:
            logger.info(f"embed 进度: {min(i + batch, total)}/{total}")
    return out


def build_raw_to_canonical(vocab):
    m = {}
    for kind in ["persons", "locations", "plot_units", "topics"]:
        for e in vocab.get(kind, []):
            canonical = e.get("canonical", "")
            if canonical == "待删除":
                continue
            for raw in e.get("raw", []):
                m.setdefault(raw, canonical)
    return m


def normalize_list(values, r2c):
    out = []
    for v in values:
        out.append({"canonical": r2c.get(v, v), "raw": v})
    return out


def normalize_chunk_label(label, genre, r2c):
    out = {}
    if genre == "research_literature":
        ent = label.get("引用实体", {}) or {}
        out["涉及版本"] = label.get("涉及版本", []) or []
        out["分析主题"] = normalize_list(label.get("分析主题", []) or [], r2c)
        out["引用实体"] = {
            "persons": normalize_list(ent.get("persons", []) or [], r2c),
            "locations": normalize_list(ent.get("locations", []) or [], r2c),
        }
        out["引用情节"] = normalize_list(label.get("引用情节", []) or [], r2c)
        out["summary"] = label.get("summary", "") or ""
    else:
        out["persons"] = normalize_list(label.get("persons", []) or [], r2c)
        out["locations"] = normalize_list(label.get("locations", []) or [], r2c)
        out["plot_unit"] = normalize_list(label.get("plot_unit", []) or [], r2c)
        out["plot_detail"] = label.get("plot_detail", "") or ""
        out["summary"] = label.get("summary", "") or ""
    return out


def compute_idf(norm_chunk):
    df = defaultdict(int)
    for nl in norm_chunk.values():
        if nl is None:
            continue
        persons = _cset(nl.get("persons", []))
        locs = _cset(nl.get("locations", []))
        ent = nl.get("引用实体") or {}
        persons |= _cset(ent.get("persons", []))
        locs |= _cset(ent.get("locations", []))
        for p in persons:
            df[p] += 1
        for l in locs:
            df[l] += 1
    N = len(norm_chunk)
    idf = {e: math.log((N + 1) / (d + 1)) for e, d in df.items()}
    if idf:
        mean_idf = sum(idf.values()) / len(idf)
        return {e: max(IDF_MIN, min(IDF_MAX, idf[e] / mean_idf)) for e in idf}
    return {}


def build_chunk_tags(nl, genre):
    """提取 chunk 的标签词（canonical 实体 + plot_unit + 分析主题 + plot_detail）。"""
    tags = []
    if genre == "research_literature":
        tags += [t["canonical"] for t in nl.get("分析主题", []) if isinstance(t, dict)]
        ent = nl.get("引用实体") or {}
        tags += [p["canonical"] for p in ent.get("persons", []) if isinstance(p, dict)]
        tags += [l["canonical"] for l in ent.get("locations", []) if isinstance(l, dict)]
        tags += [p["canonical"] for p in nl.get("引用情节", []) if isinstance(p, dict)]
    else:
        tags += [p["canonical"] for p in nl.get("persons", []) if isinstance(p, dict)]
        tags += [l["canonical"] for l in nl.get("locations", []) if isinstance(l, dict)]
        tags += [p["canonical"] for p in nl.get("plot_unit", []) if isinstance(p, dict)]
        pd = nl.get("plot_detail", "")
        if pd:
            tags.append(pd)
    return [t for t in tags if t and t != "待删除"]


def build_chunk_tags_disc(nl, genre, idf, threshold=IDF_THRESHOLD):
    """提取 chunk 的判别性标签词（过滤高频 persons/locations）。

    标签语义路用（余弦无 IDF，需过滤高频实体降低噪声）。
    """
    tags = []
    if genre == "research_literature":
        tags += [t["canonical"] for t in nl.get("分析主题", []) if isinstance(t, dict)]
        ent = nl.get("引用实体") or {}
        tags += [p["canonical"] for p in ent.get("persons", []) if isinstance(p, dict) and idf(p["canonical"]) >= threshold]
        tags += [l["canonical"] for l in ent.get("locations", []) if isinstance(l, dict) and idf(l["canonical"]) >= threshold]
        tags += [p["canonical"] for p in nl.get("引用情节", []) if isinstance(p, dict)]
    else:
        tags += [p["canonical"] for p in nl.get("persons", []) if isinstance(p, dict) and idf(p["canonical"]) >= threshold]
        tags += [l["canonical"] for l in nl.get("locations", []) if isinstance(l, dict) and idf(l["canonical"]) >= threshold]
        tags += [p["canonical"] for p in nl.get("plot_unit", []) if isinstance(p, dict)]
        pd = nl.get("plot_detail", "")
        if pd:
            tags.append(pd)
    return [t for t in tags if t and t != "待删除"]


class TagStore:
    """标签数据存储。"""

    def __init__(self, embedder=None):
        self._labels = {}
        self._norm_chunk = {}
        self._r2c = {}
        self._idf = {}
        self._tag_text = {}      # chunk_id -> 标签文本（全标签，sparse 用）
        self._tag_text_disc = {} # chunk_id -> 判别性标签文本（去高频，标签语义用）
        self._aug_text = {}      # chunk_id -> 原文+标签文本
        self._tag_vectors = {}   # chunk_id -> 标签向量
        self._summary_vectors = {}  # chunk_id -> summary 向量
        self._plot_unit_vecs = {}   # plot_unit canonical -> 向量
        self._embedder = embedder
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        vocab = json.load(open(VOCAB_PATH, encoding="utf-8"))
        data = json.load(open(CHUNK_LABELS_PATH, encoding="utf-8"))
        self._labels = data["chunk_labels"]
        self._r2c = build_raw_to_canonical(vocab)

        for cid, v in self._labels.items():
            lab = v.get("label") or {}
            if "_error" in lab:
                self._norm_chunk[cid] = None
                continue
            self._norm_chunk[cid] = normalize_chunk_label(lab, v["genre"], self._r2c)

        self._idf = compute_idf(self._norm_chunk)

        for cid, nl in self._norm_chunk.items():
            if nl is None:
                continue
            genre = self._labels[cid]["genre"]
            tags = build_chunk_tags(nl, genre)
            summary = nl.get("summary", "")
            if summary:
                tags.append(summary)
            self._tag_text[cid] = " ".join(tags)

            disc_tags = build_chunk_tags_disc(nl, genre, self.idf)
            if summary:
                disc_tags.append(summary)
            self._tag_text_disc[cid] = " ".join(disc_tags)
        self._loaded = True
        if logger:
            logger.info(f"TagStore loaded: {len(self._norm_chunk)} chunks, "
                        f"{len(self._idf)} entities with IDF")

    def build_tag_vectors(self):
        """用 embedder 构建标签向量库（启动时一次，带缓存）。"""
        cached = _load_vecs("tag_vectors")
        if cached is not None:
            self._tag_vectors = cached
            return
        if not self._embedder:
            raise RuntimeError("TagStore 需要 embedder 才能构建标签向量库")
        if not _wait_embedder_ready(self._embedder):
            raise RuntimeError(f"embedding 模型未就绪/加载失败: {self._embedder.load_error}")
        cids = list(self._tag_text_disc.keys())
        vecs = _embed_batched(self._embedder, [self._tag_text_disc[cid] for cid in cids])
        self._tag_vectors = {cid: v for cid, v in zip(cids, vecs)}
        _save_vecs("tag_vectors", self._tag_vectors)
        if logger:
            logger.info(f"TagStore tag vectors built: {len(self._tag_vectors)}")

    def build_summary_vectors(self):
        """预计算所有 chunk 的 summary 向量（启动时一次，带缓存）。"""
        cached = _load_vecs("summary_vectors")
        if cached is not None:
            self._summary_vectors = cached
            return
        if not self._embedder:
            raise RuntimeError("TagStore 需要 embedder 才能构建 summary 向量")
        if not _wait_embedder_ready(self._embedder):
            raise RuntimeError(f"embedding 模型未就绪/加载失败: {self._embedder.load_error}")
        cids, texts = [], []
        for cid, nl in self._norm_chunk.items():
            if nl is None:
                continue
            s = nl.get("summary", "")
            if s:
                cids.append(cid)
                texts.append(s)
        vecs = _embed_batched(self._embedder, texts)
        self._summary_vectors = {cid: v for cid, v in zip(cids, vecs)}
        _save_vecs("summary_vectors", self._summary_vectors)
        if logger:
            logger.info(f"TagStore summary vectors built: {len(self._summary_vectors)}")

    # ---- 查询接口 ----
    def norm_label(self, cid):
        return self._norm_chunk.get(cid)

    def idf(self, entity):
        return self._idf.get(entity, 1.0)

    def tag_text(self, cid):
        return self._tag_text.get(cid, "")

    def aug_text(self, cid, original):
        return original + " " + self._tag_text.get(cid, "")

    def tag_vector(self, cid):
        return self._tag_vectors.get(cid)

    def summary_vector(self, cid):
        return self._summary_vectors.get(cid)

    def summary_text(self, cid):
        nl = self._norm_chunk.get(cid)
        return nl.get("summary", "") if nl else ""

    def build_tag_augmented_text(self, cid, original):
        """标签词拼在原文开头（不重复），用于标签增强 dense。"""
        nl = self._norm_chunk.get(cid)
        if nl is None:
            return original
        genre = self._labels[cid]["genre"]
        tags = build_chunk_tags(nl, genre)
        summary = nl.get("summary", "")
        if summary:
            tags.append(summary)
        seen = set()
        dedup = []
        for t in tags:
            if t and t not in seen:
                seen.add(t)
                dedup.append(t)
        return " ".join(dedup) + " " + original

    def build_plot_unit_vectors(self):
        """预计算词表 plot_unit canonical 的 embedding（语义匹配用，带缓存）。"""
        cached = _load_vecs("plot_unit_vectors")
        if cached is not None:
            self._plot_unit_vecs = cached
            return
        if not self._embedder:
            raise RuntimeError("TagStore 需要 embedder 才能构建 plot_unit 向量")
        if not _wait_embedder_ready(self._embedder):
            raise RuntimeError(f"embedding 模型未就绪/加载失败: {self._embedder.load_error}")
        vocab = json.load(open(VOCAB_PATH, encoding="utf-8"))
        units = [e["canonical"] for e in vocab.get("plot_units", []) if e.get("canonical") != "待删除"]
        vecs = _embed_batched(self._embedder, units)
        self._plot_unit_vecs = {u: v for u, v in zip(units, vecs)}
        _save_vecs("plot_unit_vectors", self._plot_unit_vecs)
        if logger:
            logger.info(f"TagStore plot_unit vectors built: {len(self._plot_unit_vecs)}")

    def plot_unit_sim(self, a, b):
        """两个 plot_unit canonical 的 embedding 余弦相似度。"""
        import numpy as np
        va = self._plot_unit_vecs.get(a)
        vb = self._plot_unit_vecs.get(b)
        if va is None or vb is None:
            return 0.0
        va = np.asarray(va, dtype=float)
        vb = np.asarray(vb, dtype=float)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
        return float(va @ vb / denom)

    @property
    def chunk_ids(self):
        return list(self._tag_text.keys())

    @property
    def loaded(self):
        return self._loaded
