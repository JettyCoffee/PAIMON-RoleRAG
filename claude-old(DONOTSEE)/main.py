"""
RoleRAG-PAIMON: Role-playing dialogue system with RAG
Main application integrating all components
"""
from typing import Optional
from pathlib import Path
import json
import networkx as nx

from config import config
from src.gemini_client import GeminiClient
from src.retrieval_agent import RetrievalAgent
from src.response_generator import ResponseGenerator
from src.memory_manager import MemoryManager


class RoleRAGSystem:
    """Complete RoleRAG system for role-playing dialogue"""
    
    def __init__(
        self, 
        api_key: str,
        kg_dir: Path,
        memory_file: Optional[Path] = None
    ):
        """
        Initialize RoleRAG system
        
        Args:
            api_key: Gemini API key
            kg_dir: Directory containing knowledge graph files
            memory_file: Optional file to save/load conversation memory
        """
        print("Initializing RoleRAG-PAIMON system...")
        
        # Initialize Gemini client
        self.client = GeminiClient(api_key, config.GEMINI_MODEL)
        
        # Load knowledge graph
        print("Loading knowledge graph...")
        graph_file = kg_dir / "knowledge_graph.graphml"
        self.graph = nx.read_graphml(graph_file)
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes "
              f"and {self.graph.number_of_edges()} edges")
        
        # Load communities
        print("Loading communities...")
        communities_file = kg_dir / "communities.json"
        with open(communities_file, 'r', encoding='utf-8') as f:
            self.communities = json.load(f)
        print(f"Loaded {len(self.communities)} communities")
        
        # Load entities
        print("Loading entities...")
        entities_file = kg_dir / "entities_deduplicated.json"
        with open(entities_file, 'r', encoding='utf-8') as f:
            self.entities = json.load(f)
        
        # Initialize components
        print("Initializing components...")
        self.retrieval_agent = RetrievalAgent(
            self.graph, 
            self.communities, 
            self.entities, 
            self.client
        )
        self.response_generator = ResponseGenerator(self.client)
        self.memory_manager = MemoryManager(
            self.client, 
            history_limit=config.CONVERSATION_HISTORY_LIMIT,
            cache_file=memory_file
        )
        
        self.memory_file = memory_file
        
        print("System initialized successfully!\n")
    
    def query(self, user_query: str) -> str:
        """
        Process a user query and generate response
        
        Args:
            user_query: User's question
            
        Returns:
            System response
        """
        print(f"\n{'='*60}")
        print(f"用户问题: {user_query}")
        print(f"{'='*60}\n")
        
        # Step 1: Check for callback to previous conversations
        needs_callback, related_indices = self.memory_manager.detect_callback(user_query)
        
        callback_context = ""
        if needs_callback:
            print(f"检测到需要回调之前的对话（轮次: {related_indices}）")
            callback_context = self.memory_manager.get_callback_context(related_indices)
            # Prepend callback context to query
            enhanced_query = f"{callback_context}\n\n当前问题：{user_query}"
        else:
            enhanced_query = user_query
        
        # Step 2: Decompose query and check cache
        print("正在分解查询...")
        subqueries = self.retrieval_agent.decompose_query(enhanced_query)
        
        # Check cache for each subquery
        retrieved_info = []
        queries_to_retrieve = []
        
        for sq in subqueries:
            is_cached, cached_info = self.memory_manager.check_cache(sq.text)
            if is_cached:
                print(f"从缓存获取: {sq.text}")
                retrieved_info.extend(cached_info)
            else:
                queries_to_retrieve.append(sq)
        
        # Step 3: Retrieve information for uncached queries
        if queries_to_retrieve:
            print(f"需要检索 {len(queries_to_retrieve)} 个新查询")
            for sq in queries_to_retrieve:
                print(f"  - {sq.text} (类型: {sq.type})")
            
            # Retrieve
            new_retrieved = self.retrieval_agent.retrieve(
                enhanced_query, 
                max_iterations=config.MAX_ITERATIONS
            )
            retrieved_info.extend(new_retrieved)
        else:
            print("所有信息已在缓存中")
        
        # Step 4: Generate response
        print("\n正在生成回答...")
        response = self.response_generator.generate_response(
            user_query,
            retrieved_info,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_OUTPUT_TOKENS
        )
        
        print(f"\n{'='*60}")
        print(f"系统回答: {response}")
        print(f"{'='*60}\n")
        
        # Step 5: Generate summary and update memory
        print("正在生成对话摘要...")
        summary = self.response_generator.generate_summary(user_query, response)
        print(f"摘要: {summary}")
        
        self.memory_manager.add_turn(
            user_query,
            response,
            summary,
            retrieved_info
        )
        
        # Clean old cache
        self.memory_manager.clear_old_cache(max_cache_size=20)
        
        # Save memory if file specified
        if self.memory_file:
            self.memory_manager.save_memory(self.memory_file)
        
        return response
    
    def interactive_mode(self):
        """Run interactive dialogue mode"""
        print("\n" + "="*60)
        print("RoleRAG-PAIMON 交互模式")
        print("="*60)
        print("输入问题与角色对话，输入 'quit' 或 'exit' 退出")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n再见！")
                    break
                
                response = self.query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point"""
    import sys
    
    # Check API key
    if not config.GEMINI_API_KEY:
        print("错误: 请在 .env 文件中设置 GEMINI_API_KEY")
        print("请复制 .env.example 为 .env 并填入你的 API key")
        sys.exit(1)
    
    # Ensure output directories exist
    config.ensure_dirs()
    
    # Check if knowledge graph exists
    graph_file = config.KG_DIR / "knowledge_graph.graphml"
    if not graph_file.exists():
        print("错误: 知识图谱未构建")
        print("请先运行以下命令构建知识图谱：")
        print("  1. conda activate rolerag")
        print("  2. python -m src.data_preprocessing")
        print("  3. python -m src.entity_extraction")
        print("  4. python -m src.relationship_extraction")
        print("  5. python -m src.kg_builder")
        print("  6. python -m src.community_detection")
        sys.exit(1)
    
    # Initialize system
    memory_file = config.CACHE_DIR / "conversation_memory.json"
    system = RoleRAGSystem(
        api_key=config.GEMINI_API_KEY,
        kg_dir=config.KG_DIR,
        memory_file=memory_file
    )
    
    # Run interactive mode
    system.interactive_mode()


if __name__ == "__main__":
    main()
