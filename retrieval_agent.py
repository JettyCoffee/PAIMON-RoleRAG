import json
import networkx as nx
from typing import List, Dict, Any, Tuple
from utils import get_gemini_model, parse_json_response

DECOMPOSITION_PROMPT = """
You are an intelligent Agent for a Role-Playing System.
Your goal is to decompose a complex user query into specific sub-queries that can be answered by a Knowledge Graph.
The Knowledge Graph contains:
- **Characters** (name, persona, style)
- **Non-Characters** (events, items, locations)
- **Relations** (source, target, description, attitude)

User Query: "{query}"

Decompose this into a list of sub-queries.
For each sub-query, specify:
1. `type`: "character_info" (for persona/style), "relation_lookup" (for relationships), or "event_lookup" (for facts/events).
2. `target_entity`: The name of the primary entity involved.
3. `question`: The specific question to answer.

Return JSON format:
{{
  "sub_queries": [
    {{ "type": "...", "target_entity": "...", "question": "..." }},
    ...
  ]
}}
"""

REFLECTION_PROMPT = """
You are an intelligent Agent.
You have gathered the following information to answer the user's query: "{query}"

Current Information:
{info}

Is this information sufficient to construct a complete, role-consistent answer?
If YES, return {{ "sufficient": true }}.
If NO, return {{ "sufficient": false, "missing_info": "...", "new_sub_queries": [...] }}.
"""

class RetrievalAgent:
    def __init__(self, graph_path: str):
        self.model = get_gemini_model()
        self.graph = self._load_graph(graph_path)
    
    def _load_graph(self, path: str) -> nx.MultiDiGraph:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return nx.node_link_graph(data)
        except Exception as e:
            print(f"Error loading graph: {e}")
            return nx.MultiDiGraph()

    def decompose_query(self, query: str) -> List[Dict[str, str]]:
        """Decomposes query into sub-queries."""
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        response = self.model.generate_content(prompt)
        data = parse_json_response(response.text)
        return data.get("sub_queries", [])

    def search_graph(self, sub_query: Dict[str, str]) -> str:
        """Searches the graph for a specific sub-query."""
        target = sub_query.get("target_entity")
        q_type = sub_query.get("type")
        
        results = []
        
        # Fuzzy match for entity name (simple implementation)
        # In production, use vector search or exact match with aliases
        matched_node = None
        for node in self.graph.nodes():
            if target.lower() in node.lower() or node.lower() in target.lower():
                matched_node = node
                break
        
        if not matched_node:
            return f"No information found for entity: {target}"
        
        node_data = self.graph.nodes[matched_node]
        
        if q_type == "character_info":
            info = f"Character: {matched_node}\n"
            info += f"Persona: {node_data.get('persona', 'N/A')}\n"
            info += f"Style: {node_data.get('style_description', 'N/A')}\n"
            exemplars = node_data.get('style_exemplars', [])
            if exemplars:
                info += f"Exemplars: {', '.join(exemplars)}\n"
            results.append(info)
            
        elif q_type == "relation_lookup":
            # Get neighbors
            edges = self.graph.out_edges(matched_node, data=True)
            for u, v, data in edges:
                results.append(f"Relation: {u} -> {v}: {data.get('description')} (Attitude: {data.get('attitude', 'N/A')})")
                
        elif q_type == "event_lookup":
             # Return description and related nodes
             info = f"Entity: {matched_node}\nDescription: {node_data.get('description', 'N/A')}\n"
             results.append(info)
             # Also check neighbors for context
             edges = self.graph.out_edges(matched_node, data=True)
             for u, v, data in edges:
                results.append(f"Related to {v}: {data.get('description')}")

        return "\n".join(results)

    def retrieve(self, query: str) -> str:
        """Main retrieval loop with reflection."""
        print(f"Processing Query: {query}")
        sub_queries = self.decompose_query(query)
        print(f"Sub-queries: {sub_queries}")
        
        all_info = []
        for sq in sub_queries:
            info = self.search_graph(sq)
            all_info.append(info)
        
        combined_info = "\n---\n".join(all_info)
        
        # Reflection step (simplified: just check once)
        prompt = REFLECTION_PROMPT.format(query=query, info=combined_info)
        response = self.model.generate_content(prompt)
        reflection = parse_json_response(response.text)
        
        if not reflection.get("sufficient", True):
            print("Info insufficient, retrieving more...")
            new_queries = reflection.get("new_sub_queries", [])
            for sq in new_queries:
                 info = self.search_graph(sq)
                 combined_info += f"\n---\n{info}"
        
        return combined_info

if __name__ == "__main__":
    # Test
    agent = RetrievalAgent("/root/RoleRAG/output/role_rag_graph.json")
    # Assuming we have some data in the graph from the previous step
    # Let's try a query that might be in the first few chunks of Harry Potter
    q = "Who is Harry Potter and what are his relationships?"
    print(agent.retrieve(q))
