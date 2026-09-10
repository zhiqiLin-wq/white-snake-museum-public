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

完整操作流程录屏（4 分 11 秒，约 40 MB）：[`assets/demo/demo.mp4`](assets/demo/demo.mp4)，点击下方图片或链接均可在浏览器中播放。

[<img src="assets/demo/01-overview.png" width="760" alt="主界面：文献目录、原文阅读与 AI Agent 三栏协同">](assets/demo/demo.mp4)

<p align="center"><b>▲ 主界面：文献目录 / 原文阅读 / AI Agent 三栏协同（点击播放完整演示视频）</b></p>

### 文献检索与智能问答

<table>
  <tr>
    <td width="50%" valign="top"><img src="assets/demo/02-search.png" width="100%" alt="全文检索与 AI 归纳回答"><br>全文检索「法海」出场轨迹（187 处匹配），AI 归纳历代形象演变，每条结论附原文引用，可一键跳转高亮</td>
    <td width="50%" valign="top"><img src="assets/demo/03-agent-thinking.png" width="100%" alt="Agent 思考过程"><br>Agent 思考链实时可见：意图识别 → 路径选择 → 关键词抽取 → ReAct 深度检索</td>
  </tr>
</table>

### AI Agent 工具调用

<table>
  <tr>
    <td width="50%" valign="top"><img src="assets/demo/04-agent-tools.png" width="100%" alt="Agent 工具调用轨迹"><br>search_literature、get_chapter_full_text 等工具调用逐条展示，含轮次与耗时，过程透明可追溯</td>
    <td width="50%" valign="top"><img src="assets/demo/09-report.png" width="100%" alt="报告库与长报告预览"><br>长报告自动保存至报告库，支持平台内预览、章节跳转与 Markdown 下载</td>
  </tr>
</table>

### 智能标注

<table>
  <tr>
    <td width="50%" valign="top"><img src="assets/demo/05-annotation-progress.png" width="100%" alt="智能标注进行中"><br>标注流水线分步推进：准备 → 发现实体 → 精确定位 → 保存，实时显示实体数与当前进度</td>
    <td width="50%" valign="top"><img src="assets/demo/06-annotation-result.png" width="100%" alt="标注完成"><br>本章自动定位 332 个人物、地点、术语等实体并分类高亮；点击高亮词可查看释义，支持采纳 / 修改后采纳 / 删除 / 追问</td>
  </tr>
</table>

### 跨版本对比与文献地图

<table>
  <tr>
    <td width="50%" valign="top"><img src="assets/demo/07-compare.png" width="100%" alt="跨版本对勘"><br>明 / 清两个版本并排对勘，按人物、地点、母题等维度对齐，AI 分析差异并生成逐字对勘报告</td>
    <td width="50%" valign="top"><img src="assets/demo/08-map.png" width="100%" alt="故事地点地图"><br>故事地点地图可视化，标记类型随图例区分，支持按朝代切换地图图层</td>
  </tr>
</table>

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
