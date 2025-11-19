"""
Build knowledge graph from scratch
Run all KG construction steps in sequence
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config
from src.gemini_client import GeminiClient
from src.data_preprocessing import DataPreprocessor
from src.entity_extraction import EntityExtractor
from src.relationship_extraction import RelationshipExtractor
from src.kg_builder import KnowledgeGraphBuilder
from src.community_detection import CommunityDetector
import networkx as nx


def build_knowledge_graph():
    """Build complete knowledge graph"""
    
    print("="*60)
    print("RoleRAG-PAIMON 知识图谱构建")
    print("="*60)
    
    # Check API key
    if not config.GEMINI_API_KEY:
        print("\n错误: 请在 .env 文件中设置 GEMINI_API_KEY")
        print("请复制 .env.example 为 .env 并填入你的 API key\n")
        return False
    
    # Ensure directories exist
    config.ensure_dirs()
    
    # Initialize Gemini client
    print("\n初始化 Gemini API 客户端...")
    client = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    
    # Step 1: Data preprocessing
    print("\n" + "="*60)
    print("步骤 1: 数据预处理")
    print("="*60)
    
    if not config.AVATAR_FILE.exists():
        print(f"错误: 数据文件不存在: {config.AVATAR_FILE}")
        return False
    
    preprocessor = DataPreprocessor(config.AVATAR_FILE)
    chunks = preprocessor.process_all(chunk_size=config.CHUNK_SIZE)
    
    print(f"处理了 {len(preprocessor.processed_data)} 个角色")
    print(f"生成了 {len(chunks)} 个文本块")
    
    # Save intermediate results
    preprocessor.save_processed_data(config.OUTPUT_DIR / "processed_avatars.json")
    preprocessor.save_chunks(chunks, config.OUTPUT_DIR / "text_chunks.json")
    
    # Step 2: Entity extraction
    print("\n" + "="*60)
    print("步骤 2: 实体提取")
    print("="*60)
    print("警告: 这一步会调用大量 API，可能需要较长时间...")
    
    # Ask for confirmation
    response = input("\n是否继续？(y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return False
    
    extractor = EntityExtractor(client)
    entities = extractor.process_chunks(chunks[:50])  # Limit to first 50 chunks for demo
    
    entities_file = config.KG_DIR / "entities.json"
    extractor.save_entities(entities_file)
    
    # Step 3: Relationship extraction
    print("\n" + "="*60)
    print("步骤 3: 关系提取")
    print("="*60)
    
    rel_extractor = RelationshipExtractor(client)
    relationships = rel_extractor.process_chunks(chunks[:50], entities)  # Same limit
    
    relationships_file = config.KG_DIR / "relationships.json"
    rel_extractor.save_relationships(relationships_file)
    
    # Step 4: Build knowledge graph with deduplication
    print("\n" + "="*60)
    print("步骤 4: 构建知识图谱并去重")
    print("="*60)
    
    builder = KnowledgeGraphBuilder()
    builder.load_entities(entities_file)
    builder.load_relationships(relationships_file)
    
    # Find and merge duplicates
    duplicates = builder.find_duplicate_entities(similarity_threshold=0.85)
    if duplicates:
        print(f"发现 {len(duplicates)} 组重复实体")
        builder.merge_duplicate_entities(duplicates)
    
    # Build graph
    builder.build_graph()
    builder.save_graph(config.KG_DIR)
    
    # Step 5: Community detection
    print("\n" + "="*60)
    print("步骤 5: 社区检测与摘要生成")
    print("="*60)
    
    graph = builder.graph
    detector = CommunityDetector(graph, client)
    
    communities = detector.detect_communities(algorithm=config.COMMUNITY_ALGORITHM)
    processed_communities = detector.process_communities(
        communities,
        min_size=config.MIN_COMMUNITY_SIZE
    )
    
    communities_file = config.KG_DIR / "communities.json"
    detector.save_communities(communities_file)
    
    # Done
    print("\n" + "="*60)
    print("知识图谱构建完成！")
    print("="*60)
    print(f"\n输出目录: {config.KG_DIR}")
    print("\n生成的文件:")
    print(f"  - knowledge_graph.graphml (图谱文件)")
    print(f"  - entities_deduplicated.json (实体)")
    print(f"  - relationships_deduplicated.json (关系)")
    print(f"  - communities.json (社区)")
    print(f"  - graph_statistics.json (统计信息)")
    
    print("\n现在可以运行主程序了:")
    print("  python main.py")
    
    return True


if __name__ == "__main__":
    try:
        success = build_knowledge_graph()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
