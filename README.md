# 🚀 Knowledge-base-agent - 知识库智能问答系统

基于 LangGraph、FastAPI、Streamlit 和 ChromaDB 构建的企业级知识库问答 Agent。

系统支持文档上传、自动解析、向量化存储、混合检索（Hybrid RAG）、多轮对话记忆以及引用溯源，可用于企业知识库、内部文档问答、课程资料检索、技术文档助手等场景。

---

# ✨ 项目特性

## 📚 知识库管理

支持上传以下格式文档：

* PDF
* DOCX
* TXT
* Markdown

上传后自动完成：

* 文档解析
* 文本清洗
* OCR图片识别
* 表格提取
* 智能切分
* 向量化存储

---

## 🔍 Hybrid RAG 检索

系统采用混合检索方案：

### 向量检索

基于：

* OpenAI Embedding
* Compatible Embedding API

存储于：

* ChromaDB

实现语义相似度搜索。

---

### BM25关键词检索

适用于：

* 专业术语
* 产品型号
* 人名
* 编号
* 专有名词

匹配。

---

### RRF融合排序

使用：

Reciprocal Rank Fusion (RRF)

融合：

* 向量召回结果
* BM25召回结果

提升最终检索准确率。

---

## 🤖 LangGraph Agent

基于 LangGraph 构建 Agent 工作流。

支持：

* Tool Calling
* 多轮对话
* Agent状态管理
* 流式输出
* 长期记忆
* 短期记忆

---

## 🧠 记忆系统

### 短期记忆

自动：

* 保留最近对话
* 历史对话压缩总结
* 上下文继承

降低 Token 消耗。

---

### 长期记忆

支持记录：

* 用户偏好
* 项目背景
* 常用信息

实现个性化问答体验。

---

## 📑 引用溯源

回答知识库问题时自动生成引用：

示例：

[1] 员工手册.pdf ｜ page: 12 ｜ chunk_id: handbook_chunk_18

支持：

* 页码定位
* Chunk定位
* 文档来源追踪

提高答案可信度。

---

## 🛡️ 防幻觉机制

系统强制执行：

* 先检索后回答
* 无证据不回答
* 指定文档检索
* 来源引用校验

当知识库无依据时：

"知识库中没有找到足够依据，无法回答该问题。"

避免模型编造内容。

---

# 🏗️ 系统架构

用户问题

↓

LangGraph Agent

↓

Database_Search Tool

↓

Hybrid Retriever

├── Chroma Vector Search

└── BM25 Search

↓

RRF Rank Fusion

↓

LLM Answer Generation

↓

引用生成

↓

最终回答

---

# 📂 项目结构

```text
My_Agent
│
├── src
│   ├── agents
│   │   ├── rag_assistant.py
│   │   ├── rag_retriever.py
│   │   ├── tools.py
│   │   └── safeguard.py
│   │
│   ├── kb
│   │   ├── document_processor.py
│   │   ├── document_service.py
│   │   └── task_store.py
│   │
│   ├── memory
│   │   ├── long_term_memory.py
│   │   ├── short_term_memory.py
│   │   └── sqlite.py
│   │
│   ├── service
│   │   ├── service.py
│   │   └── kb_routes.py
│   │
│   ├── client
│   ├── schema
│   └── core
│
├── chroma_db
├── data
├── docs
├── media
├── compose.yaml
└── pyproject.toml
```

---

# ⚙️ 环境要求

* Python 3.11+
* ChromaDB
* FastAPI
* Streamlit
* LangGraph
* LangChain
* OpenAI Compatible API

推荐：

```bash
Python 3.11
```

---

# 🚀 安装

## 1. 克隆项目

```bash
git clone <your_repo_url>
cd My_Agent
```

---

## 2. 创建虚拟环境

使用 uv：

```bash
uv sync
```

或 Conda：

```bash
conda create -n my_agent python=3.11
conda activate my_agent

pip install -e .
```

---

## 3. 配置环境变量

创建：

```bash
.env
```

示例：

```env
OPENAI_API_KEY=sk-xxxx

EMBEDDING_MODEL=text-embedding-v1
EMBEDDING_API_KEY=sk-xxxx

EMBEDDING_BASE_URL=https://api.xxx.com/v1

COMPATIBLE_API_KEY=sk-xxxx
COMPATIBLE_BASE_URL=https://api.xxx.com/v1
```

---

# ▶️ 启动项目

## 启动 Agent Service

```bash
python src/run_service.py
```

默认：

```text
http://localhost:8000
```

---

## 启动 Streamlit

```bash
streamlit run src/streamlit_app.py
```

默认：

```text
http://localhost:8501
```

---

# 📤 文档上传

上传接口：

```http
POST /kb/upload
```

支持：

* PDF
* DOCX
* TXT
* MD

返回：

```json
{
  "success": true,
  "task_id": "xxxxx"
}
```

---

# 📄 查看文档列表

```http
GET /kb/documents
```

---

# 🔎 检索调试

仅检索不调用模型：

```http
GET /kb/debug/retrieve
```

参数：

```text
query=什么是绩效考核
```

可用于：

* 检查召回结果
* 调试RAG效果
* 查看Chunk来源

---

# 🗑️ 删除文档

```http
DELETE /kb/documents/{source}
```

---

# 📈 技术栈

## Agent

* LangGraph
* LangChain

## 服务层

* FastAPI
* Uvicorn

## 前端

* Streamlit

## 向量数据库

* ChromaDB

## Embedding

* OpenAI Embedding
* Compatible Embedding API

## 数据处理

* PyMuPDF
* pdfplumber
* python-docx
* pytesseract

## 检索

* BM25
* RRF
* Hybrid RAG

---

# 🎯 适用场景

* 企业知识库
* 内部文档问答
* 产品手册助手
* 员工培训助手
* 科研论文问答
* 医学文献检索
* 教育知识库
* 私有RAG系统

---

# 📜 License

MIT License
