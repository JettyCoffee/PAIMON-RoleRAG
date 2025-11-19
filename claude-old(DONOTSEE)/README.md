# RoleRAG-PAIMON

基于知识图谱和检索增强生成(RAG)的原神角色扮演对话系统。

## 项目简介

RoleRAG-PAIMON 是一个完整的角色扮演对话系统，能够：

1. **构建双重信息知识图谱**：同时包含"事实"信息和"风格"信息
2. **智能检索**：使用 Agent 进行 Schema-Aware 的查询分解和迭代检索
3. **角色扮演生成**：基于检索到的上下文，以角色的口吻和风格生成回答
4. **多轮记忆管理**：支持对话历史的缓存和回调

## 系统架构

```
数据预处理 → 实体提取 → 关系提取 → 知识图谱构建 → 社区检测
                                              ↓
用户查询 → 查询分解 → 检索Agent → 上下文整合 → LLM生成回答
            ↓
        多轮记忆管理
```

## 安装

### 1. 创建并激活 conda 环境

```bash
conda create -n rolerag python=3.10
conda activate rolerag
```

### 2. 安装依赖

```bash
cd RoleRAG-PAIMON
pip install -r requirements.txt
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env` 并填入你的 Gemini API Key：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 GEMINI_API_KEY
```

## 使用方法

### 方法一：一键构建知识图谱

```bash
conda activate rolerag
python build_kg.py
```

这个脚本会自动执行所有构建步骤：
1. 数据预处理
2. 实体提取
3. 关系提取
4. 知识图谱构建与去重
5. 社区检测与摘要生成

**注意**：这个过程会调用大量 Gemini API，可能需要较长时间（取决于数据量）。

### 方法二：分步构建

如果需要更细粒度的控制，可以分步执行：

```bash
conda activate rolerag

# 1. 数据预处理
python -m src.data_preprocessing

# 2. 实体提取
python -m src.entity_extraction

# 3. 关系提取
python -m src.relationship_extraction

# 4. 知识图谱构建与去重
python -m src.kg_builder

# 5. 社区检测
python -m src.community_detection
```

### 运行对话系统

构建完知识图谱后，运行主程序开始对话：

```bash
conda activate rolerag
python main.py
```

进入交互模式后，输入问题即可与角色对话。输入 `quit` 或 `exit` 退出。

## 示例对话

```
你: 七七是谁？
系统: 我是七七...是个僵尸。在不卜庐采药...帮白先生。记性不太好...但会努力记住你。

你: 她的性格怎么样？
系统: 话很少...也没什么表情。因为是僵尸，很怕热。喜欢凉快的地方...下雪的时候想堆雪人...
```

## 项目结构

```
RoleRAG-PAIMON/
├── config/              # 配置模块
│   ├── __init__.py
│   └── config.py        # 配置管理
├── models/              # 数据模型
│   ├── __init__.py
│   └── schema.py        # Pydantic 数据模型
├── src/                 # 核心源代码
│   ├── data_preprocessing.py     # 数据预处理
│   ├── entity_extraction.py      # 实体提取
│   ├── relationship_extraction.py # 关系提取
│   ├── kg_builder.py             # 知识图谱构建
│   ├── community_detection.py    # 社区检测
│   ├── retrieval_agent.py        # 检索Agent
│   ├── response_generator.py     # 回答生成
│   ├── memory_manager.py         # 记忆管理
│   └── gemini_client.py          # Gemini API客户端
├── datasets/            # 数据集
│   └── avatar_CHS.json  # 角色数据
├── output/              # 输出目录（自动创建）
│   ├── knowledge_graph/ # 知识图谱文件
│   └── cache/           # 对话缓存
├── main.py              # 主程序入口
├── build_kg.py          # 知识图谱构建脚本
├── requirements.txt     # Python依赖
├── .env.example         # 环境变量模板
└── README.md            # 本文件
```

## 技术特性

### 1. 双重信息知识图谱

- **角色实体**：包含 persona（性格）、style_description（说话风格）、style_exemplars（惯用语）
- **非角色实体**：地点、物品、事件等
- **关系**：包含客观描述、主观态度和强度

### 2. 智能检索Agent

- **Schema-Aware 查询分解**：理解图谱结构，将查询分解为角色类和事件类
- **迭代检索与反思**：自动判断信息是否充分，必要时生成新的子查询
- **社区级检索**：在角色社区和事件社区中分别检索

### 3. 多轮记忆管理

- **对话缓存**：保留最近 k 轮的检索结果，避免重复检索
- **回调检测**：自动识别需要引用之前对话的查询
- **摘要生成**：为每轮对话生成简洁摘要

### 4. 风格化生成

- 基于检索到的角色风格信息
- 使用角色的惯用语和语气
- 保持角色性格一致性

## 配置说明

主要配置项在 `config/config.py` 中：

```python
# API配置
GEMINI_API_KEY = "your_api_key"
GEMINI_MODEL = "gemini-pro"

# 数据处理
CHUNK_SIZE = 512  # 文本块大小

# 社区检测
COMMUNITY_ALGORITHM = "louvain"
MIN_COMMUNITY_SIZE = 2

# Agent参数
MAX_ITERATIONS = 5  # 最大检索迭代次数
CONVERSATION_HISTORY_LIMIT = 5  # 保留对话轮数

# 生成参数
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024
```

## 注意事项

1. **API 使用**：知识图谱构建会调用大量 API，请确保有足够的配额
2. **处理时间**：完整构建可能需要较长时间，建议先用小数据集测试
3. **内存需求**：大型知识图谱可能占用较多内存
4. **数据格式**：确保输入数据符合 `datasets/README.md` 中的格式要求

## 依赖项

主要依赖：
- `google-generativeai`: Gemini API客户端
- `networkx`: 图处理
- `scikit-learn`: 文本相似度计算
- `pydantic`: 数据验证
- `python-dotenv`: 环境变量管理

详见 `requirements.txt`

## 开发计划

- [ ] 支持更多角色数据源
- [ ] 优化检索效率
- [ ] 添加Web界面
- [ ] 支持语音对话
- [ ] 多模态信息整合

## 参考文献

本项目参考了以下论文和项目的思路：
- RoleRAG: 角色扮演的检索增强生成
- G2ConS: 基于知识图谱的对话系统
- GraphRAG: 知识图谱增强的RAG

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
