# RoleRAG-PAIMON 使用示例

## 完整使用流程演示

### 准备工作

```bash
# 1. 激活环境
conda activate rolerag

# 2. 进入项目目录
cd /root/kg4game/RoleRAG-PAIMON

# 3. 确认 API Key 已配置
cat .env
# 应该看到：GEMINI_API_KEY=你的密钥
```

### 第一次使用：构建知识图谱

```bash
# 运行一键构建脚本
python build_kg.py
```

预期输出：
```
============================================================
RoleRAG-PAIMON 知识图谱构建
============================================================

初始化 Gemini API 客户端...

============================================================
步骤 1: 数据预处理
============================================================
处理了 XX 个角色
生成了 XXX 个文本块

============================================================
步骤 2: 实体提取
============================================================
警告: 这一步会调用大量 API，可能需要较长时间...

是否继续？(y/n): y

Processing XX chunks...
提取实体中: 100%|████████████| XX/XX [XX:XX<00:00]
Saved XX characters and XX non-characters

...

============================================================
知识图谱构建完成！
============================================================

输出目录: /root/kg4game/RoleRAG-PAIMON/output/knowledge_graph

生成的文件:
  - knowledge_graph.graphml (图谱文件)
  - entities_deduplicated.json (实体)
  - relationships_deduplicated.json (关系)
  - communities.json (社区)
  - graph_statistics.json (统计信息)

现在可以运行主程序了:
  python main.py
```

### 运行对话系统

```bash
python main.py
```

预期输出：
```
Initializing RoleRAG-PAIMON system...
Loading knowledge graph...
Loaded graph with XX nodes and XX edges
Loading communities...
Loaded XX communities
Loading entities...
Initializing components...
System initialized successfully!

============================================================
RoleRAG-PAIMON 交互模式
============================================================
输入问题与角色对话，输入 'quit' 或 'exit' 退出
============================================================

```

### 对话示例 1：询问角色信息

```
你: 七七是谁？

============================================================
用户问题: 七七是谁？
============================================================

正在分解查询...
Generated 2 sub-queries
需要检索 2 个新查询
  - 七七是谁？ (类型: character)
  - 七七的基本信息 (类型: character)

Query: 七七是谁？
Decomposing query...
Generated 2 sub-queries
Retrieving for: 七七是谁？ (type: character)
Retrieving for: 七七的基本信息 (type: character)

Reflection iteration 1/5
Information is sufficient!

Total retrieved chunks: 2

正在生成回答...

============================================================
系统回答: 我是七七...是个僵尸。在璃月的不卜庐采药，帮白先生的忙。
话很少...也没什么表情。因为是僵尸的关系，记性不太好...但是我会
努力记住你的。怕热...喜欢凉快的地方。
============================================================

正在生成对话摘要...
摘要: 介绍了七七是不卜庐的采药僵尸，记性不好但很努力。
```

### 对话示例 2：多轮对话（使用缓存）

```
你: 她的性格怎么样？

============================================================
用户问题: 她的性格怎么样？
============================================================

检测到需要回调之前的对话（轮次: [0]）
正在分解查询...
Generated 1 sub-queries
从缓存获取: 七七的性格特点
所有信息已在缓存中

正在生成回答...

============================================================
系统回答: 嗯...我的话很少，表情也不多。因为是僵尸嘛...但是我会
认真做事的。虽然记不住很多事情，但我会用笔记本记下来。害怕热，
喜欢凉快...下雪的时候想堆雪人。对了，还会做柔软体操...不要偷看。
============================================================

正在生成对话摘要...
摘要: 描述了七七话少、怕热、用笔记本记事的性格特点。
```

### 对话示例 3：询问关系

```
你: 七七和胡桃的关系怎么样？

============================================================
用户问题: 七七和胡桃的关系怎么样？
============================================================

正在分解查询...
Generated 2 sub-queries
需要检索 2 个新查询
  - 七七对胡桃的态度 (类型: character)
  - 七七和胡桃之间的关系 (类型: event)

...

正在生成回答...

============================================================
系统回答: 胡桃...讨厌。她总是想要埋葬我...还有那个假笑，很不舒服。
虽然她说是为了我好，但是...我不想死。我还有很多事情要做，要帮
白先生，还要记住你...所以，离她远一点比较好。
============================================================
```

### 退出系统

```
你: quit

再见！
```

## 常用命令速查

### 环境管理
```bash
# 激活环境
conda activate rolerag

# 查看已安装的包
pip list

# 更新某个包
pip install --upgrade google-generativeai
```

### 检查配置
```bash
# 查看环境变量
cat .env

# 查看配置参数
python -c "from config import config; print(config.GEMINI_MODEL)"
```

### 查看输出
```bash
# 查看知识图谱统计
cat output/knowledge_graph/graph_statistics.json

# 查看对话历史
cat output/cache/conversation_memory.json

# 查看社区数量
python -c "import json; data=json.load(open('output/knowledge_graph/communities.json')); print(f'Communities: {len(data)}')"
```

### 重新构建
```bash
# 删除旧的输出
rm -rf output/

# 重新构建
python build_kg.py
```

## 高级用法

### 单独运行某个模块

```bash
# 只运行数据预处理
python -m src.data_preprocessing

# 只运行实体提取
python -m src.entity_extraction

# 测试检索功能
python -m src.retrieval_agent

# 测试生成功能
python -m src.response_generator
```

### 自定义查询

创建一个测试脚本 `test_query.py`：
```python
from main import RoleRAGSystem
from config import config

# 初始化系统
system = RoleRAGSystem(
    api_key=config.GEMINI_API_KEY,
    kg_dir=config.KG_DIR,
    memory_file=None  # 不保存记忆
)

# 单次查询
response = system.query("你最喜欢什么？")
print(response)
```

运行：
```bash
python test_query.py
```

## 调试技巧

### 1. 增加日志输出
在相关模块的函数中添加 print 语句

### 2. 检查中间结果
```bash
# 查看处理后的角色数据
python -c "import json; data=json.load(open('output/processed_avatars.json')); print(data[0])"

# 查看提取的实体
python -c "import json; data=json.load(open('output/knowledge_graph/entities_deduplicated.json')); print(f'Characters: {len(data[\"characters\"])}')"
```

### 3. 测试 API 连接
```python
from src.gemini_client import GeminiClient
from config import config

client = GeminiClient(config.GEMINI_API_KEY)
response = client.generate("测试：1+1=?")
print(response)
```

## 性能优化建议

### 减少 API 调用
1. 在 `build_kg.py` 中限制处理的块数
2. 使用缓存避免重复调用
3. 批量处理相似查询

### 加速构建
1. 并行处理多个文本块（需修改代码）
2. 使用更快的模型（如 gemini-pro）
3. 减小 `CHUNK_SIZE`

### 节省内存
1. 分批加载数据
2. 及时清理缓存
3. 使用生成器而非列表

## 故障排除

### API 限流
如果遇到 `429 Too Many Requests`：
```python
# 在 gemini_client.py 中增加等待时间
time.sleep(5)  # 每次调用后等待5秒
```

### 编码问题
确保所有文件使用 UTF-8 编码：
```bash
file -i output/knowledge_graph/entities_deduplicated.json
```

### 依赖冲突
重新创建环境：
```bash
conda deactivate
conda remove -n rolerag --all
conda create -n rolerag python=3.10
conda activate rolerag
pip install -r requirements.txt
```

祝使用顺利！
