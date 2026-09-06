# 白蛇传数字博物馆（White Snake Museum）

> 以《白蛇传》历代文献为语料的数字人文研究平台：RAG 智能对话、文献地图、原文标注、母题演化分析与报告库，三位一体。

[![Node](https://img.shields.io/badge/Node.js-22%2B-339933)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![Vue](https://img.shields.io/badge/Vue-3.5-42B883)](https://vuejs.org/)
[![License](https://img.shields.io/badge/license-学术研究用途-lightgrey)]()

## 项目简介

本项目是一个面向古典文学研究的**数字人文（Digital Humanities）平台**，以《白蛇传》历代文献（话本、小说、戏曲、宝卷、地方志等）为研究语料，将传统文献研读与 LLM（大语言模型）能力结合：

- **读者视角**：与"白蛇传博物馆 Agent"对话，问答、讲故事、规划游览路线，回答可溯源到原文段落；
- **研究者视角**：在原文上做结构化标注（人物 / 地点 / 情节 / 母题），跨版本对比同一母题的演化，生成研究报告；
- **工程视角**：一套完整的、带评测体系的 RAG 系统——混合检索、标签增强、引用溯源、检索质量量化评估。

系统采用三端架构：Vue 3 前端 + Fastify（Node.js）业务后端 + FastAPI（Python）AI Agent，全部使用 TypeScript / Python 编写。

## 核心功能

| 模块 | 说明 |
|------|------|
| 智能对话 | 基于 LangGraph 多工作流的 Agent，SSE 流式输出，思考步骤可视化，回答附原文引用并可跳转高亮 |
| 文献地图 | Leaflet 地图呈现故事地点（杭州西湖、镇江金山寺、峨眉山等），支持朝代图层切换、地点文学资料卡 |
| 文本阅读器 | 分章阅读原始文献，划词标注（人物 / 地点 / 情节 / 母题），批注气泡，标注与 chunk 精确双向定位 |
| 母题演化工作台 | 跨文献追踪同一母题（如"水漫金山""断桥相会"）的形态演变，假设 → 取证 → 综合的研究流程，关键节点人工介入 |
| 跨版本对比 | 任意两个版本 / 段落并排对比，差异元素与共享段落自动识别 |
| 朝代对比 | 唐 / 宋 / 明 / 清等时期人物、地点、情节元素的统计对比与图表可视化 |
| 报告库 | Agent 生成的长报告（研究快照、地点综述、人物关系等）集中保存与回看 |
| 统计图表 | ECharts 图谱 / 桑基图 / 旭日图 / 热力图等，呈现共现关系与标注分布 |
| 用户系统 | 注册登录（bcrypt 密码哈希 + 会话令牌），每用户独立数据库与标注数据 |

## 系统架构

```text
浏览器 (Vue 3 + Vite)
   │  HTTP / SSE
   ▼
Node 业务后端 (Fastify, :3000)  ──────  用户认证 / 文献 / 标注 / 会话 / 静态资源
   │  内部 HTTP (x-user-id 头)
   ▼
Python AI Agent (FastAPI, :8000)
   ├── LangGraph 工作流（supervisor 调度）
   │     ├── literature_qa    文献问答
   │     ├── location_story   地点叙事
   │     ├── comparison       跨版本对比
   │     ├── itinerary        游览路线规划
   │     ├── memory           长期记忆
   │     └── evolution_workbench  母题演化研究
   ├── MCP 工具层（60+ 工具：检索 / 标注 / 统计 / 地图 / 记忆 / 报告）
   ├── RAG 检索（ChromaDB 向量 + BM25 + 标签向量，混合融合 + 重排）
   └── LLM Provider（DeepSeek / Anthropic Claude / 通义千问 / 文心，OpenAI 兼容协议）
```

开发环境下 Vite 将 `/api` 请求代理到 Node 后端；生产环境由 Node 后端（或 Nginx）统一托管前端静态资源，同源部署无跨域问题。

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Vue 3.5、Vite 6、TypeScript、Pinia、Vue Router、Leaflet、ECharts、marked |
| 业务后端 | Node.js 22+（内置 `node:sqlite`）、Fastify 5、bcryptjs、nodejieba、xlsx、mammoth |
| AI Agent | Python 3.11+、FastAPI、LangGraph、LangChain、ChromaDB、sentence-transformers、jieba、MCP |
| Embedding | `shibing624/text2vec-base-chinese`（本地 CPU 推理，首次运行自动下载） |
| 数据存储 | SQLite（用户 / 会话 / 记忆）、ChromaDB（向量索引）、JSON（词表 / 配置） |

## 目录结构

```text
.
├── client/                # Vue 3 前端
│   └── src/
│       ├── views/         # 15 个页面（博物馆首页 / 地图 / 阅读器 / 演化工作台 / 报告库…）
│       ├── components/    # 对话、地图、标注、图表、布局等组件
│       ├── stores/        # Pinia 状态（agent / 标注 / 地图 / 演化 / 源树…）
│       └── services/      # API 封装与 SSE 事件总线
├── server/                # Node.js 业务后端（Fastify + TypeScript）
│   └── src/
│       ├── routes/        # 认证 / 文献 / 地点 / Agent 代理 / 标注 / 会话
│       ├── services/      # 认证、Excel 解析、文献切分、用户数据迁移
│       └── middleware/    # 会话鉴权
├── agent/                 # Python AI Agent（FastAPI + LangGraph）
│   └── server/
│       ├── langgraph/     # 工作流节点与 supervisor 调度
│       ├── mcp/tools/     # 60+ MCP 工具
│       ├── rag/           # 切分 / 向量库 / BM25 / 标签管线 / 重排 / 引用溯源
│       ├── context/       # 上下文管理、记忆提取与评分、FTS 索引
│       ├── skills/        # 角色技能定义（YAML）
│       ├── prompts/       # 提示词模板（YAML）
│       ├── llm/           # 多 Provider 适配（DeepSeek / Anthropic / OpenAI 兼容）
│       └── security/      # 限流、令牌预算、内部鉴权
├── rag_eval/              # RAG 检索评测体系（查询集 / qrels 真值 / 多模型对比 / boost 调参）
├── tests/                 # 标注质量评测（L1-L5）与 D1-D7 维度评测
├── scripts/               # 辅助脚本（研究文档抽取、溯源数据生成、一键启动）
└── shared/                # 跨端测试夹具
```

## RAG 与评测体系

检索不是"向量库一把梭"，而是一条可量化、可调优的流水线：

- **混合检索**：稠密向量（ChromaDB）+ BM25 关键词 + 标签向量（人物 / 地点 / 情节 / 母题标签）三路召回，加权融合后重排；
- **标签管线**：LLM 为每个 chunk 生成结构化标签并建立标签向量库，查询侧同步打标，显著提升实体类查询召回；
- **引用溯源**：每个 chunk 携带文献、章节、段落偏移信息，前端可精确跳转到原文对应段落并临时高亮；
- **评测驱动**：`rag_eval/` 内含人工校验的查询集与 qrels 真值，支持 Recall@k、多 Provider 对比、融合权重 / boost 系数扫描等实验，检索质量用数据说话。

## 语料数据与格式要求（重要）

本仓库**不包含语料文本**，且系统的文献解析与分块是围绕特定语料结构设计的——随便放入一个文本文件无法正常工作。以下内容说明自备语料时必须遵守的格式约定。

### 不包含在仓库中的内容

| 路径 | 内容 | 不包含原因 |
|------|------|-----------|
| `excel_data/白蛇传文献选集.txt` / `.docx` | 主文献：历代白蛇传文本汇编 | **文献版权**，私有研究资料 |
| `excel_data/文本景观建构研究文献.txt` / `.docx` | 研究文献 | 同上 |
| `excel_data/*.xlsx` | 各地点文献的 jieba 分词结果 | 由语料派生 |
| `docs/`、`CLAUDE.md` | 内部设计文档与开发约定 | 私有 |
| `.env`、`agent/.env` | API Key 等密钥 | 安全（仓库提供 `.env.example` 模板） |
| `chroma_db/`、`agent/models_cache/`、`*.db`、`logs/` | 向量索引、模型缓存、用户数据库、日志 | 运行时产物，可重建 |

### 主文献的结构约定

系统在两处硬编码了主文献文件名（Node 后端与 Python Agent 一致）：

- `excel_data/白蛇传文献选集.txt` —— 主文献（UTF-8 文本，docx 可用 mammoth 抽取为 txt）
- `excel_data/文本景观建构研究文献.txt` —— 研究文献

主文献必须是**严格七章**结构：以行首 `一、` ~ `七、` 开头的行被识别为章节标题（正则 `^\s*(一|二|三|四|五|六|七)\s*[、，,.\s]\s*(.+)$`），章节之间按标题行切分；匹配不到任何章节时整篇按单章处理（分块质量会显著下降）。

章节标题需包含两类信息：

1. **朝代**：标题中含 `唐 / 宋 / 元 / 明 / 清 / 现当代` 字样，用于朝代图层与统计；
2. **体裁后缀**：标题末尾的全角括号标记，分块器据此选择分块策略：

| 标题后缀 | 体裁 | 分块策略 | 章内结构识别 |
|---------|------|---------|-------------|
| （文言小说） | 文言小说 | 句子级语义合并 | — |
| （话本小说） | 话本 | 段落语义合并 | — |
| （拟话本小说） | 拟话本 | 语义段落合并 | — |
| （章回体白话小说） | 章回体白话 | 结构章节合并 | `第X回/章/节`、"诗曰/话说/且说"等叙事标记 |
| （戏曲） | 戏曲 | 折/出结构分块 | `第X出/折`、`【曲牌名】`、`〔角色〕` |
| （剧本） | 剧本 | 场次结构分块 | `第X场`、`（唱/白/念）` |
| （弹词） | 弹词 | 场景窗口滑动 | "不题/再说"场景切换、"曰/唱/白"标记 |

标题示例：`三、发展期 宋：话本小说的成熟（话本小说）`。若标题缺体裁后缀，系统按章节序号回退（一→文言、二→话本……七→剧本），但显式标注更可靠。

### 分块（chunking）逻辑

分块器（`agent/server/rag/chunker.py`）是**体裁感知**的：8 种体裁各有独立策略，按"结构单元切分 → 句/段边界对齐 → embedding 语义相似度合并（低于阈值的相邻块合并）→ 最小长度约束"的流程产出 chunk。每个 chunk 携带完整溯源元数据：文献、章节号、章节标题、朝代、体裁、段落索引、字符偏移、chunk_id（格式 `{来源}_{体裁}_{章节号}_{序号}`，如 `primary_huaben_三_00042`）。前端"跳转原文"即依赖这些偏移信息定位。

### 地点数据（xlsx）

文献地图与地点统计依赖 `excel_data/*.xlsx`：每个地点一个文件（文件名即地点名，如"杭州西湖分析词结果.xlsx"），内容为该地点相关文献段落的分词结果表（含摘要、匹配词数、文本字数等列）。后端用 nodejieba 做分词、xlsx 库解析。

### 自备语料的完整步骤

1. 按上述七章 + 朝代 + 体裁后缀格式整理你的文献文本，命名为 `白蛇传文献选集.txt` 放入 `excel_data/`（研究文献同理）；
2. 在 DeepSeek / Anthropic / 阿里云百炼等平台申请 LLM API Key，填入 `.env` 与 `agent/.env`（模板见 `.env.example`）；
3. 启动 Agent 后运行分块索引构建（`agent/run_chunking.py` 调试 / `agent/rebuild_index.py` 全量重建），生成 ChromaDB 向量库与 chunk 快照；
4. 地点分词 xlsx 可由 `server` 的分词服务生成，或自行用 jieba 处理。

> 一言以蔽之：**本仓库是一套面向特定语料形态（历代白蛇传文献汇编）的完整系统实现与研究方法展示**，不是开箱即用的通用产品；代码可自由阅读、安装、在符合格式约定的语料上运行，但受版权保护的原始语料不随仓库发布。

## 环境要求

- **Node.js 22+**（使用内置 `node:sqlite`，需 22.5 以上）
- **Python 3.11+**（建议 3.11 / 3.12）
- Windows / macOS / Linux 均可；生产部署建议 Ubuntu 22.04+
- 无需 GPU：embedding 模型 CPU 推理即可（首次加载约 1-2 分钟，占用内存约 2GB）

## 安装与配置

```bash
# 1. 前端
cd client
npm install

# 2. 业务后端
cd ../server
npm install

# 3. AI Agent
cd ../agent
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

配置（从模板复制后填入自己的 Key）：

```bash
cp .env.example .env              # 业务后端配置
cp agent/.env.example agent/.env  # Agent 配置（LLM Provider、embedding、数据目录等）
```

`agent/.env` 关键项：

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的key
EMBEDDING_MODEL=shibing624/text2vec-base-chinese
SENTENCE_TRANSFORMERS_HOME=./models_cache
CHROMA_PERSIST_DIR=./chroma_db
DATA_DIR=../excel_data
```

## 启动

三个终端分别启动（开发模式）：

```bash
# 终端 1：AI Agent（:8000）
cd agent
uvicorn server.main:app --reload --port 8000

# 终端 2：业务后端（:3000）
cd server
npm run dev

# 终端 3：前端（:5173）
cd client
npm run dev
```

浏览器打开 http://localhost:5173 即可。Windows 下也可直接运行 `scripts/start-dev.bat` 一键拉起。

首次启动 Agent 会自动下载 embedding 模型（约 400MB，走 HF 镜像）；放入语料后按 `agent/run_chunking.py` 说明构建索引。

## 测试

```bash
# 前端单元测试（Vitest）
cd client && npx vitest run

# RAG 评测系统自测
python -m pytest rag_eval/tests/ -v

# 标注质量评测（L1-L5：span 校验 / 标注 / 管线 / 覆盖率 / 切分）
cd tests/annotation_eval && python run_eval.py

# 记忆系统 D1-D7 维度评测（脚本与数据集见 tests/eval/）
cd tests/eval && python evaluate_d1.py
```

> 评测子系统的完整说明见 [rag_eval/README.md](./rag_eval/README.md)。开发过程中的一次性诊断 / 调参脚本不随仓库发布，仅保留评测框架与当前数据集。

## 项目状态与说明

- 本项目为个人**学术研究 / 数字人文练习**项目，持续迭代中；
- 文献语料版权归原出版方 / 作者所有，本仓库不包含任何受版权保护的原文；
- 所有 LLM 调用需使用你自己的 API Key，相关费用自理；
- 仓库中的评测脚本（`rag_eval/scripts/`）从环境变量读取密钥，请勿在代码中硬编码任何 Key。

## 许可

仅供学习与学术研究使用。
