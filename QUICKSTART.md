# RoleRAG-PAIMON 快速启动指南

## 第一步：环境准备

### 1. 创建 conda 环境
```bash
conda create -n rolerag python=3.10
conda activate rolerag
```

### 2. 安装依赖
```bash
cd /root/kg4game/RoleRAG-PAIMON
pip install -r requirements.txt
```

### 3. 配置 Gemini API Key

编辑 `.env` 文件，填入你的 API Key：
```bash
# .env 文件内容
GEMINI_API_KEY=你的API密钥
GEMINI_MODEL=gemini-pro
```

## 第二步：构建知识图谱

**注意**：以下命令都需要在 `rolerag` conda 环境中运行

### 选项 A：一键构建（推荐）
```bash
conda activate rolerag
python build_kg.py
```

这个脚本会自动完成所有步骤，但会消耗较多 API 调用。

### 选项 B：分步构建（更可控）

```bash
# 1. 数据预处理（无API调用）
python -m src.data_preprocessing

# 2. 实体提取（需要大量API调用）
python -m src.entity_extraction

# 3. 关系提取（需要大量API调用）
python -m src.relationship_extraction

# 4. 知识图谱构建与去重（无API调用）
python -m src.kg_builder

# 5. 社区检测（需要API调用）
python -m src.community_detection
```

**提示**：为了节省API成本和时间，你可以修改代码中的数据量限制。例如在 `build_kg.py` 中：
```python
# 第50行和第63行
entities = extractor.process_chunks(chunks[:50])  # 只处理前50个文本块
relationships = rel_extractor.process_chunks(chunks[:50], entities)
```

## 第三步：运行对话系统

```bash
conda activate rolerag
python main.py
```

进入交互模式后：
- 输入问题与角色对话
- 输入 `quit` 或 `exit` 退出

## 常见问题

### 1. API 配额不足
如果遇到 API 配额限制，可以：
- 减少处理的文本块数量
- 增加 API 调用间隔（修改 `gemini_client.py` 中的重试逻辑）

### 2. 内存不足
如果内存不足，可以：
- 减少 `CHUNK_SIZE`（在 `config/config.py` 中）
- 分批处理数据

### 3. 找不到模块
确保：
- 在正确的目录下运行命令（`/root/kg4game/RoleRAG-PAIMON`）
- 已激活 conda 环境（`conda activate rolerag`）

### 4. 知识图谱不存在
如果运行 `main.py` 提示知识图谱不存在，需要先运行第二步构建知识图谱。

## 输出文件说明

构建完成后，会在 `output/knowledge_graph/` 目录下生成：

- `knowledge_graph.graphml`: NetworkX 图谱文件
- `entities_deduplicated.json`: 去重后的实体
- `relationships_deduplicated.json`: 去重后的关系
- `communities.json`: 社区及其摘要
- `graph_statistics.json`: 图谱统计信息

对话过程中会在 `output/cache/` 生成：
- `conversation_memory.json`: 对话历史和缓存

## 系统要求

- Python 3.10+
- 8GB+ RAM（推荐）
- 稳定的网络连接（用于 API 调用）
- Gemini API 访问权限

## 下一步

- 尝试不同的查询问题
- 调整配置参数（`config/config.py`）
- 扩展数据集
- 优化 Prompt 模板

## 获取帮助

如遇到问题，请检查：
1. 是否正确配置了 API Key
2. 是否在正确的 conda 环境中
3. 是否完成了知识图谱构建
4. 查看终端输出的错误信息

祝使用愉快！
