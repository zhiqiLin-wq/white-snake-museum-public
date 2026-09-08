# 白蛇传数字博物馆

以《白蛇传》历代文献为语料的数字人文研究平台：LLM 智能对话、文献地图、原文标注、母题演化分析。

## 简介

面向古典文学研究的数字人文平台，将《白蛇传》历代文献（话本、小说、戏曲、宝卷等）与大语言模型结合，支持可溯源的问答、原文结构化标注、跨版本母题演化追踪与研究报告生成。

三端架构：Vue 3 前端 + Node.js 业务后端 + Python AI Agent。

## 功能

- **智能对话**：Agent 流式问答，回答附原文引用，可跳转高亮
- **文献地图**：故事地点地图与朝代图层
- **文本阅读器**：分章阅读，划词标注（人物 / 地点 / 情节 / 母题），批注气泡
- **母题演化**：跨文献追踪同一母题的形态演变
- **跨版本对比**：两个版本 / 段落并排对比，自动识别差异
- **朝代对比**：各时期人物、地点、情节元素的统计图表
- **报告库**：Agent 生成的研究报告集中保存与回看
- **用户系统**：注册登录，每用户独立数据

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Vue 3、Vite、TypeScript、Pinia、Leaflet、ECharts |
| 业务后端 | Node.js 22+、Fastify、SQLite |
| AI Agent | Python 3.11+、FastAPI、LangGraph、ChromaDB、sentence-transformers |
| 数据 | SQLite、ChromaDB 向量索引 |

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

配置（从模板复制后填入自己的 LLM API Key）：

```bash
cp .env.example .env
cp agent/.env.example agent/.env
```

启动（三个终端）：

```bash
# Agent
cd agent && uvicorn server.main:app --reload --port 8000

# 业务后端
cd server && npm run dev

# 前端
cd client && npm run dev
```

浏览器打开 http://localhost:5173 。首次启动 Agent 会自动下载 embedding 模型。

## 语料

- 主文献《白蛇传文献选集》已随仓库提供，开箱即用；
- 研究文献与派生分词数据不随仓库发布，需自备。

## 说明

- 个人学术研究 / 数字人文练习项目，持续迭代；
- 主文献为公版古籍汇编；研究文献版权归原作者所有，不随仓库发布；
- LLM 调用需使用自备的 API Key，费用自理。

## 许可

仅供学习与学术研究使用。
