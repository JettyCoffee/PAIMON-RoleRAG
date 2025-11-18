"""
Community detection and summary generation
"""
from typing import Dict, List, Any
from pathlib import Path
import json
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

from .gemini_client import GeminiClient


class CommunityDetector:
    """Detect communities in knowledge graph and generate summaries"""
    
    COMMUNITY_SUMMARY_PROMPT = """
你是一个专业的知识图谱分析助手。请为以下社区生成一个简洁的摘要。

社区类型：{community_type}

社区成员实体：
{entities_info}

社区内的关系：
{relationships_info}

请根据上述信息生成一个简洁的社区摘要（200-300字），包括：
1. 社区的主要主题或内容
2. 关键实体及其作用
3. 实体之间的主要关系

直接返回摘要文本，不要包含任何额外说明。
"""
    
    def __init__(self, graph: nx.Graph, gemini_client: GeminiClient):
        self.graph = graph
        self.client = gemini_client
        self.communities = []
        
    def detect_communities(self, algorithm: str = "louvain") -> List[List[str]]:
        """
        Detect communities using specified algorithm
        
        Args:
            algorithm: Community detection algorithm ("louvain", "label_propagation", etc.)
            
        Returns:
            List of communities (each community is a list of node names)
        """
        if algorithm == "louvain":
            # Louvain algorithm for community detection
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
        elif algorithm == "label_propagation":
            # Label propagation algorithm
            communities_gen = nx.community.label_propagation_communities(self.graph)
            partition = {}
            for i, comm in enumerate(communities_gen):
                for node in comm:
                    partition[node] = i
        else:
            # Default: greedy modularity
            communities_gen = nx.community.greedy_modularity_communities(self.graph)
            partition = {}
            for i, comm in enumerate(communities_gen):
                for node in comm:
                    partition[node] = i
        
        # Convert partition dict to list of communities
        community_dict = defaultdict(list)
        for node, comm_id in partition.items():
            community_dict[comm_id].append(node)
        
        communities = list(community_dict.values())
        
        print(f"Detected {len(communities)} communities using {algorithm}")
        return communities
    
    def classify_community(self, community: List[str]) -> str:
        """
        Classify community as character-focused or event-focused
        
        Args:
            community: List of entity names in the community
            
        Returns:
            "character-focused" or "event-focused"
        """
        character_count = 0
        non_character_count = 0
        
        for node in community:
            node_data = self.graph.nodes[node]
            if node_data.get("type") == "character":
                character_count += 1
            else:
                non_character_count += 1
        
        # If more than 50% are characters, it's character-focused
        if character_count > non_character_count:
            return "character-focused"
        else:
            return "event-focused"
    
    def generate_community_summary(
        self, 
        community_id: str,
        community: List[str], 
        community_type: str
    ) -> str:
        """
        Generate summary for a community using LLM
        
        Args:
            community_id: Community identifier
            community: List of entity names
            community_type: "character-focused" or "event-focused"
            
        Returns:
            Generated summary text
        """
        # Gather entity information
        entities_info = []
        for node in community:
            node_data = self.graph.nodes[node]
            if node_data.get("type") == "character":
                info = f"- {node}（角色）: {node_data.get('persona', '')}"
            else:
                info = f"- {node}（非角色）: {node_data.get('description', '')}"
            entities_info.append(info)
        
        # Gather relationship information
        relationships_info = []
        for source, target in self.graph.edges(community):
            if source in community and target in community:
                edge_data = self.graph.edges[source, target]
                desc = edge_data.get("description", "")
                rel_info = f"- {source} → {target}: {desc}"
                relationships_info.append(rel_info)
        
        # Prepare prompt
        entities_str = "\n".join(entities_info[:20])  # Limit to first 20 entities
        relationships_str = "\n".join(relationships_info[:20])  # Limit to first 20 relationships
        
        prompt = self.COMMUNITY_SUMMARY_PROMPT.format(
            community_type=community_type,
            entities_info=entities_str,
            relationships_info=relationships_str
        )
        
        try:
            summary = self.client.generate(prompt, temperature=0.5, max_tokens=500)
            return summary
        except Exception as e:
            print(f"Error generating summary for community {community_id}: {e}")
            return f"社区包含 {len(community)} 个实体。"
    
    def process_communities(
        self, 
        communities: List[List[str]], 
        min_size: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Process all communities: classify and generate summaries
        
        Args:
            communities: List of communities
            min_size: Minimum community size to process
            
        Returns:
            List of community dictionaries with metadata and summaries
        """
        print(f"Processing {len(communities)} communities...")
        
        processed_communities = []
        
        for i, community in enumerate(tqdm(communities, desc="Generating summaries")):
            if len(community) < min_size:
                continue
            
            community_id = f"community_{i}"
            community_type = self.classify_community(community)
            
            # Generate summary
            summary = self.generate_community_summary(
                community_id, 
                community, 
                community_type
            )
            
            community_data = {
                "id": community_id,
                "members": community,
                "size": len(community),
                "type": community_type,
                "summary": summary
            }
            
            processed_communities.append(community_data)
            self.communities = processed_communities
        
        print(f"Processed {len(processed_communities)} communities (min_size={min_size})")
        
        # Print community type distribution
        character_focused = sum(1 for c in processed_communities if c["type"] == "character-focused")
        event_focused = len(processed_communities) - character_focused
        print(f"Character-focused: {character_focused}, Event-focused: {event_focused}")
        
        return processed_communities
    
    def save_communities(self, output_file: Path):
        """Save communities to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.communities, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.communities)} communities to {output_file}")


if __name__ == "__main__":
    from config import config
    import sys
    
    # Check API key
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in .env file")
        sys.exit(1)
    
    # Load graph
    graph_file = config.KG_DIR / "knowledge_graph.graphml"
    if not graph_file.exists():
        print(f"Error: {graph_file} not found. Run kg_builder.py first.")
        sys.exit(1)
    
    print("Loading knowledge graph...")
    graph = nx.read_graphml(graph_file)
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Initialize detector
    client = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    detector = CommunityDetector(graph, client)
    
    # Detect communities
    communities = detector.detect_communities(algorithm=config.COMMUNITY_ALGORITHM)
    
    # Process and generate summaries
    processed_communities = detector.process_communities(
        communities, 
        min_size=config.MIN_COMMUNITY_SIZE
    )
    
    # Save results
    output_file = config.KG_DIR / "communities.json"
    detector.save_communities(output_file)
