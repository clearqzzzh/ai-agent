# PetHome AI 智能客服系统

基于 LangGraph 多智能体编排 + 混合检索 RAG 的宠物电商智能客服系统。

## 项目简介

PetHome 是面向 B/C 端用户的宠物用品电商平台，本项目为其 AI 智能客服系统，覆盖售前咨询、售中操作、售后退款全流程。

## 核心特性

- **7-Agent 协作架构**：Supervisor + 6 个执行 Agent（商品咨询/下单操作/售后处理/宠物健康/知识检索/兜底）
- **两层意图路由**：关键词硬规则（60% 分流）+ DeepSeek 语义分类，综合准确率 88%
- **混合检索 RAG**：ChromaDB 向量 + BM25 关键词双路召回，来源准确率 95%，关键词召回率 95.1%
- **Function Calling**：22 个工具按业务域分配，DSML 降级解析兼容非标准输出
- **全链路降级**：RAG/LLM/会话存储三级降级 + 熔断机制
- **全链路可观测**：LangSmith 追踪，三级节点埋点

## 技术栈

- **后端**：Python 3.12 + FastAPI + LangGraph + DeepSeek + Qwen-VL
- **RAG**：ChromaDB + text2vec-base-chinese + BM25
- **会话管理**：Redis Hash + ZSet + tiktoken 裁剪
- **前端**：Streamlit + Vue3

## 项目结构

```
agent-service/
├── access/           # 接入层（FastAPI + Streamlit）
├── cognition/        # 认知层（RAG 引擎 + 文档加载）
├── config/           # 配置
├── execution/        # 执行层（业务服务）
├── memory/           # 记忆层（会话管理 + Token 管理）
├── models/           # 模型定义
├── orchestration/    # 编排层（Supervisor Graph + 工具集）
├── tests/            # 测试与评测
└── tools/            # 工具脚本
```

## 快速开始

### 1. 安装依赖

```bash
cd agent-service
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写 API Key：

```env
DEEPSEEK_API_KEY=your_key_here
QWEN_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379/0
```

### 3. 构建知识库

```bash
python -c "from cognition.engine import build_index; build_index()"
```

### 4. 启动服务

```bash
python access/main.py
```

访问 http://localhost:8001

## 知识库

包含 14 篇宠物养护文档，涵盖：
- 猫咪/狗狗喂养指南
- 宠物医疗/疾病/急救
- 宠物行为/营养/疫苗
- 新手养宠指南
- 季节性护理
- 产品选购指南

## 评测结果

| 指标 | 数值 |
|------|------|
| 意图路由准确率 | 88% |
| 无效分流下降 | ~70% |
| RAG 来源准确率 | 95% |
| 关键词召回率 | 95.1% |
| MRR | 0.75 |

## License

MIT
