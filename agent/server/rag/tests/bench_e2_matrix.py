"""E2 矩阵扫描端到端计时（真实装配：embedder + chroma + BM25 + TagPipeline）。

按 main.py 生产顺序构建检索管线（无 LLM：母题预填 _search_keywords，
E2 关键词阶段自动跳过 LLM 调用），直接跑 EvolutionWorkbench._scan_motif_matrix，
输出 6 母题 × 4 朝代 = 24 格的真实扫描墙钟时间与每格命中数。

运行（建议先停掉后端服务，避免双进程各载一份模型导致内存崩溃）:
  python server/rag/tests/bench_e2_matrix.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from server.rag.config import rag_config
from server.rag.embedder import Embedder
from server.rag.vector_store import VectorStore
from server.rag.bm25_retriever import BM25Retriever, BM25Config
from server.rag.tag_store import TagStore
from server.rag.tag_retriever import TagRetriever
from server.rag.tag_pipeline import TagPipeline
from server.langgraph.evolution_workbench import EvolutionWorkbench


async def main():
    t_start = time.perf_counter()

    embedder = Embedder(
        model_name=rag_config.embedding_model,
        local_path=rag_config.embedding_local_path or None,
    )
    embedder.start_loading()
    for _ in range(300):
        if embedder.is_ready or embedder.load_error:
            break
        await asyncio.sleep(1)
    if not embedder.is_ready:
        print(f"模型加载失败: {embedder.load_error}")
        sys.exit(1)
    print(f"[setup] embedder ready ({time.perf_counter() - t_start:.1f}s)")

    bm25 = BM25Retriever(config=BM25Config(
        k1=rag_config.bm25_k1, b=rag_config.bm25_b,
        tokenizer=rag_config.bm25_tokenizer,
        top_k_multiplier=rag_config.bm25_top_k_multiplier,
    ))
    vs = VectorStore(
        persist_dir=rag_config.chroma_persist_path,
        embedder=embedder, bm25_retriever=bm25,
    )
    live_docs = vs.get_all_documents()
    tag_corpus = {
        d["chunk_id"]: {"text": d.get("content", ""), "metadata": d.get("metadata", {})}
        for d in live_docs
    }
    print(f"[setup] corpus: {len(tag_corpus)} chunks")

    bm25_docs = [
        {"content": doc["text"], "metadata": {**doc.get("metadata", {}), "chunk_id": cid},
         "chunk_id": cid}
        for cid, doc in tag_corpus.items()
    ]
    bm25.index(bm25_docs)

    tag_store = TagStore(embedder=embedder)
    tag_store.load()
    tag_store.build_tag_vectors()
    tag_store.build_summary_vectors()
    tag_store.build_plot_unit_vectors()

    tr = TagRetriever(vs, embedder, tag_store)
    tag_meta_lookup = {cid: doc.get("metadata", {}) for cid, doc in tag_corpus.items()}
    tr.build_tag_bm25(tag_corpus)
    tr.build_tag_aug_dense(tag_corpus)
    pipeline = TagPipeline(tr, None, tag_corpus, tag_meta_lookup)
    print(f"[setup] pipeline ready ({time.perf_counter() - t_start:.1f}s)")

    wf = EvolutionWorkbench(llm=None, retriever=pipeline, prompt_registry=None)

    motifs = [
        {"id": "M1", "name": "镇压与囚禁", "core": "白娘子被法海镇压于雷峰塔下",
         "_search_keywords": "雷峰塔 镇压 永镇 塔下 祭塔 法海 金钵"},
        {"id": "M2", "name": "断桥相遇", "core": "许仙白娘子西湖断桥初遇借伞",
         "_search_keywords": "断桥 西湖 借伞 相遇 初遇 游船 雨伞"},
        {"id": "M3", "name": "水漫金山", "core": "白娘子水漫金山寺斗法法海",
         "_search_keywords": "金山寺 水漫 斗法 海水 法海 虾兵蟹将"},
        {"id": "M4", "name": "端午现形", "core": "端午节雄黄酒后白蛇现原形吓死许仙",
         "_search_keywords": "端午 雄黄 现形 原形 吓死 白蛇 药酒"},
        {"id": "M5", "name": "盗仙草", "core": "白素贞盗灵芝仙草救活许仙",
         "_search_keywords": "盗草 灵芝 仙草 南极仙翁 救活 还阳"},
        {"id": "M6", "name": "儿子祭塔", "core": "许梦蛟中状元祭塔救母白蛇出世",
         "_search_keywords": "祭塔 状元 许梦蛟 救母 出世 团圆 拜塔"},
    ]
    state = {"motifs": motifs, "dynasties": ["唐", "宋", "明", "清"], "thread_id": "bench"}

    # 单次检索基线（新代码单格单级调用耗时，对照日志旧代码 3457-3596ms）
    t0 = time.perf_counter()
    probe = await pipeline.retrieve(
        "雷峰塔 镇压 永镇 塔下 祭塔", top_k=40, skip_tagging=True)
    print(f"[probe] 单次 skip_tagging 检索: {time.perf_counter() - t0:.2f}s, {len(probe)} docs")

    t0 = time.perf_counter()
    result = await wf._scan_motif_matrix(state)
    dt = time.perf_counter() - t0

    matrix = result["motif_matrix"]
    nonempty = sum(1 for c in matrix if c["occurrence_count"] > 0)
    print(f"\n==== E2 扫描: {len(matrix)} cells, 非空 {nonempty}, "
          f"扫描墙钟 {dt:.1f}s ====")
    for c in matrix:
        print(f"  {c['motif_id']}@{c['dynasty']}: {c['occurrence_count']} 条")


if __name__ == "__main__":
    asyncio.run(main())
