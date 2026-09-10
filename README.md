# 白蛇传数字博物馆

以《白蛇传》历代文献为语料的数字人文平台：结合大语言模型，提供文献阅读、智能问答、原文标注、母题演化分析、跨版本对比与研究报告生成等功能。

## 功能

- **文献阅读**：按章节阅读《白蛇传文献选集》，段落级展示，支持目录跳转
- **全文搜索**：在主文献中关键词检索，按相关度排序，高亮命中位置
- **AI 智能对话**：基于文献语料的问答，回答附原文引用，可点击跳转高亮
- **智能标注**：AI 自动识别人物、地点、情节、母题等实体并高亮，支持用户手动增删与旁注
- **母题演化**：跨版本追踪同一母题在不同文献中的形态演变
- **跨版本对比**：两个版本 / 段落并排对照，自动识别差异
- **朝代对比**：各时期人物、地点、情节元素的统计与对比
- **文献地图**：故事地点地图可视化
- **可视化图表**：堆叠柱状图、桑基图、旭日图、热力图、力导向关系图等
- **报告库**：AI 生成的研究报告集中保存、回看与导出
- **用户系统**：注册登录，数据按用户隔离

## 界面演示

部分功能截图，完整录屏（4 分 11 秒）见 [demo.mp4](assets/demo/demo.mp4)。

**主界面**

![主界面](assets/demo/01-overview.png)

**全文搜索**

![全文搜索](assets/demo/02-search.png)

**AI 问答**

![AI 问答](assets/demo/03-agent-thinking.png)

**工具调用**

![工具调用](assets/demo/04-agent-tools.png)

**智能标注**

![智能标注](assets/demo/05-annotation-progress.png)

![标注结果](assets/demo/06-annotation-result.png)

**跨版本对比**

![跨版本对比](assets/demo/07-compare.png)

**文献地图**

![文献地图](assets/demo/08-map.png)

**报告库**

![报告库](assets/demo/09-report.png)

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Vue 3、Vite、TypeScript、Pinia、Leaflet、ECharts |
| 业务后端 | Node.js、Fastify、SQLite |
| AI Agent | Python、FastAPI、LangGraph、ChromaDB、sentence-transformers |

## 快速开始

```bash
# 前端
cd client && npm install

# 业务后端
cd ../server && npm install

# AI Agent
cd ../agent
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

配置 LLM API Key（从模板复制后填入）：

```bash
cp .env.example .env
cp agent/.env.example agent/.env
```

启动（三个终端）：

```bash
# AI Agent
cd agent && uvicorn server.main:app --reload --port 8000

# 业务后端
cd server && npm run dev

# 前端
cd client && npm run dev
```

浏览器打开 http://localhost:5173 。首次启动 Agent 会自动下载 embedding 模型，并需运行索引构建生成向量库。

## 语料数据

仓库已包含主文献《白蛇传文献选集》（公版古籍汇编），可直接用于阅读、搜索、标注与 AI 问答。

以下数据未随仓库发布：

| 数据 | 用途 | 缺失影响 |
|------|------|----------|
| 文本景观建构研究文献 | 研究文献阅读与 RAG 增强 | 研究文献页面不可用，AI 问答仅基于主文献 |
| 各地点分词结果（.xlsx） | 地图地点数据与可视化图表 | 地图无地点标记，相关图表无数据 |

即：克隆仓库后，文献阅读、搜索、AI 问答、标注、演化分析、对比、报告等核心功能可直接运行；地图与部分图表需补充地点分词数据后才有内容。

## 说明

- 项目持续迭代中；
- 主文献为公版古籍；未包含的研究文献版权归原作者所有；
- AI 相关功能需自备 LLM API Key，费用自理。

## 许可

仅供学习与学术研究使用。
