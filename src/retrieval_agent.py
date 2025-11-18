"""
Agent-based retrieval system with query decomposition and iterative search
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .gemini_client import GeminiClient
from models.schema import Query, SubQuery


class RetrievalAgent:
    """Agent for schema-aware query decomposition and iterative retrieval"""
    
    QUERY_DECOMPOSITION_PROMPT = """
你是一个专业的查询分析助手。请分析用户的问题，并将其分解为多个子查询。

用户问题：{query}

知识图谱结构说明：
- 角色（character）节点：包含角色的基本信息、性格、说话风格等
- 非角色（non-character）节点：包含地点、物品、事件、组织等

请将问题分解为多个子查询，每个子查询应该：
1. text: 子查询的具体内容
2. type: "character"（关于角色的查询）或 "event"（关于事件、地点、物品等的查询）
3. priority: 优先级（1-3，1为最重要）

返回JSON格式：
{{
  "subqueries": [
    {{
      "text": "子查询文本",
      "type": "character" 或 "event",
      "priority": 1
    }}
  ]
}}
"""

    REFLECTION_PROMPT = """
你是一个专业的信息充分性评估助手。请评估当前检索到的信息是否足以回答用户的问题。

用户问题：{query}

已检索到的信息：
{retrieved_info}

请评估：
1. 信息是否充分足以回答问题？
2. 如果不充分，还需要什么额外信息？

返回JSON格式：
{{
  "is_sufficient": true/false,
  "missing_info": "如果信息不充分，描述还需要什么信息；否则为空字符串",
  "new_subquery": {{
    "text": "如果需要更多信息，生成新的子查询文本；否则为空字符串",
    "type": "character 或 event",
    "priority": 1
  }}
}}
"""
    
    def __init__(
        self, 
        graph: nx.Graph,
        communities: List[Dict[str, Any]],
        entities: Dict[str, Any],
        gemini_client: GeminiClient
    ):
        self.graph = graph
        self.communities = communities
        self.entities = entities
        self.client = gemini_client
        
        # Create TF-IDF index for entities
        self._build_entity_index()
        
        # Create TF-IDF index for communities
        self._build_community_index()
    
    def _build_entity_index(self):
        """Build TF-IDF index for entity search"""
        self.entity_names = list(self.graph.nodes())
        entity_texts = []
        
        for name in self.entity_names:
            node_data = self.graph.nodes[name]
            if node_data.get("type") == "character":
                text = f"{name} {node_data.get('persona', '')} {node_data.get('avatarDetail', '')}"
            else:
                text = f"{name} {node_data.get('description', '')}"
            entity_texts.append(text)
        
        self.entity_vectorizer = TfidfVectorizer()
        self.entity_tfidf = self.entity_vectorizer.fit_transform(entity_texts)
    
    def _build_community_index(self):
        """Build TF-IDF index for community search"""
        community_texts = []
        for comm in self.communities:
            text = f"{comm['summary']} {' '.join(comm['members'])}"
            community_texts.append(text)
        
        self.community_vectorizer = TfidfVectorizer()
        self.community_tfidf = self.community_vectorizer.fit_transform(community_texts)
    
    def decompose_query(self, query: str) -> List[SubQuery]:
        """
        Decompose user query into sub-queries
        
        Args:
            query: User query text
            
        Returns:
            List of SubQuery objects
        """
        prompt = self.QUERY_DECOMPOSITION_PROMPT.format(query=query)
        
        try:
            result = self.client.generate_json(prompt, temperature=0.3)
            subqueries_data = result.get("subqueries", [])
            
            subqueries = []
            for sq_data in subqueries_data:
                try:
                    subquery = SubQuery(**sq_data)
                    subqueries.append(subquery)
                except Exception as e:
                    print(f"Error creating SubQuery: {e}")
            
            return sorted(subqueries, key=lambda x: x.priority)
            
        except Exception as e:
            print(f"Error decomposing query: {e}")
            # Fallback: treat entire query as a single sub-query
            return [SubQuery(text=query, type="character", priority=1)]
    
    def search_entities(self, query: str, query_type: str, top_k: int = 5) -> List[str]:
        """
        Search for relevant entities
        
        Args:
            query: Search query
            query_type: "character" or "event"
            top_k: Number of top results
            
        Returns:
            List of entity names
        """
        # Vectorize query
        query_vec = self.entity_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.entity_tfidf)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates
        
        # Filter by type
        results = []
        for idx in top_indices:
            entity_name = self.entity_names[idx]
            entity_data = self.graph.nodes[entity_name]
            
            # Type filtering
            if query_type == "character" and entity_data.get("type") == "character":
                results.append(entity_name)
            elif query_type == "event" and entity_data.get("type") != "character":
                results.append(entity_name)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_communities(self, query: str, query_type: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant communities
        
        Args:
            query: Search query
            query_type: "character" or "event"
            top_k: Number of top results
            
        Returns:
            List of community dictionaries
        """
        # Vectorize query
        query_vec = self.community_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.community_tfidf)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        
        # Filter by type
        results = []
        for idx in top_indices:
            community = self.communities[idx]
            
            # Type filtering: match community type with query type
            if query_type == "character" and community["type"] == "character-focused":
                results.append(community)
            elif query_type == "event" and community["type"] == "event-focused":
                results.append(community)
            elif query_type not in ["character", "event"]:  # Accept any type
                results.append(community)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """Get detailed information about an entity"""
        if entity_name not in self.graph.nodes:
            return {}
        
        node_data = dict(self.graph.nodes[entity_name])
        
        # Get neighbors and relationships
        neighbors = []
        for neighbor in self.graph.neighbors(entity_name):
            edge_data = self.graph.edges[entity_name, neighbor]
            neighbors.append({
                "name": neighbor,
                "relationship": edge_data.get("description", ""),
                "attitude": edge_data.get("attitude"),
                "strength": edge_data.get("strength", 0.5)
            })
        
        node_data["neighbors"] = neighbors
        return node_data
    
    def retrieve_subquery(
        self, 
        subquery: SubQuery, 
        max_entities: int = 5,
        max_communities: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve information for a sub-query
        
        Args:
            subquery: SubQuery object
            max_entities: Maximum number of entities to retrieve
            max_communities: Maximum number of communities to retrieve
            
        Returns:
            Retrieved information dictionary
        """
        # Search entities
        entity_names = self.search_entities(subquery.text, subquery.type, top_k=max_entities)
        
        # Get detailed entity info
        entities_info = []
        for name in entity_names:
            info = self.retrieve_entity_info(name)
            if info:
                entities_info.append(info)
        
        # Search communities
        communities_info = self.search_communities(subquery.text, subquery.type, top_k=max_communities)
        
        return {
            "subquery": subquery.text,
            "type": subquery.type,
            "entities": entities_info,
            "communities": communities_info
        }
    
    def reflect_and_iterate(
        self, 
        query: str, 
        retrieved_info: List[Dict[str, Any]],
        max_iterations: int = 3
    ) -> Tuple[bool, Optional[SubQuery]]:
        """
        Reflect on retrieved information and determine if more retrieval is needed
        
        Args:
            query: Original user query
            retrieved_info: Currently retrieved information
            max_iterations: Maximum reflection iterations
            
        Returns:
            (is_sufficient, new_subquery)
        """
        # Format retrieved info for prompt
        info_summary = json.dumps(retrieved_info, ensure_ascii=False, indent=2)[:2000]  # Limit length
        
        prompt = self.REFLECTION_PROMPT.format(
            query=query,
            retrieved_info=info_summary
        )
        
        try:
            result = self.client.generate_json(prompt, temperature=0.3)
            
            is_sufficient = result.get("is_sufficient", True)
            
            if not is_sufficient:
                new_sq_data = result.get("new_subquery", {})
                if new_sq_data.get("text"):
                    try:
                        new_subquery = SubQuery(**new_sq_data)
                        return False, new_subquery
                    except Exception as e:
                        print(f"Error creating new subquery: {e}")
            
            return True, None
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            return True, None  # Assume sufficient on error
    
    def retrieve(
        self, 
        query: str, 
        max_iterations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval pipeline with iterative refinement
        
        Args:
            query: User query
            max_iterations: Maximum retrieval iterations
            
        Returns:
            List of retrieved information chunks
        """
        print(f"Query: {query}")
        
        # Step 1: Decompose query
        print("Decomposing query...")
        subqueries = self.decompose_query(query)
        print(f"Generated {len(subqueries)} sub-queries")
        
        # Step 2: Retrieve for each sub-query
        all_retrieved = []
        
        for sq in subqueries:
            print(f"Retrieving for: {sq.text} (type: {sq.type})")
            retrieved = self.retrieve_subquery(sq)
            all_retrieved.append(retrieved)
        
        # Step 3: Iterative reflection
        iteration = 0
        while iteration < max_iterations:
            print(f"\nReflection iteration {iteration + 1}/{max_iterations}")
            
            is_sufficient, new_subquery = self.reflect_and_iterate(
                query, 
                all_retrieved,
                max_iterations - iteration
            )
            
            if is_sufficient:
                print("Information is sufficient!")
                break
            
            if new_subquery:
                print(f"Retrieving additional info for: {new_subquery.text}")
                retrieved = self.retrieve_subquery(new_subquery)
                all_retrieved.append(retrieved)
            
            iteration += 1
        
        print(f"\nTotal retrieved chunks: {len(all_retrieved)}")
        return all_retrieved


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
        print(f"Error: {graph_file} not found.")
        sys.exit(1)
    
    print("Loading knowledge graph...")
    graph = nx.read_graphml(graph_file)
    
    # Load communities
    communities_file = config.KG_DIR / "communities.json"
    if not communities_file.exists():
        print(f"Error: {communities_file} not found.")
        sys.exit(1)
    
    with open(communities_file, 'r', encoding='utf-8') as f:
        communities = json.load(f)
    
    # Load entities (for reference)
    entities_file = config.KG_DIR / "entities_deduplicated.json"
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    # Initialize agent
    client = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    agent = RetrievalAgent(graph, communities, entities, client)
    
    # Test query
    test_query = "七七是谁？她的性格和说话风格是什么样的？"
    
    retrieved_info = agent.retrieve(test_query, max_iterations=config.MAX_ITERATIONS)
    
    # Save test results
    output_file = config.OUTPUT_DIR / "test_retrieval.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieved_info, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved test results to {output_file}")
