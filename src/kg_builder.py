"""
Knowledge graph construction and entity deduplication
"""
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KnowledgeGraphBuilder:
    """Build and manage knowledge graph with deduplication"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entities = {}  # name -> entity data
        self.relationships = []
        
    def load_entities(self, entities_file: Path):
        """Load entities from file"""
        with open(entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process characters
        for char in data.get("characters", []):
            self.entities[char["name"]] = char
        
        # Process non-characters
        for nc in data.get("non_characters", []):
            self.entities[nc["name"]] = nc
        
        print(f"Loaded {len(self.entities)} entities")
    
    def load_relationships(self, relationships_file: Path):
        """Load relationships from file"""
        with open(relationships_file, 'r', encoding='utf-8') as f:
            self.relationships = json.load(f)
        
        print(f"Loaded {len(self.relationships)} relationships")
    
    def find_duplicate_entities(self, similarity_threshold: float = 0.85) -> Dict[str, List[str]]:
        """
        Find duplicate entities using semantic similarity
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicates
            
        Returns:
            Dictionary mapping canonical name to list of duplicate names
        """
        entity_names = list(self.entities.keys())
        
        if len(entity_names) < 2:
            return {}
        
        # Create text representations for each entity
        entity_texts = []
        for name in entity_names:
            entity = self.entities[name]
            if entity["type"] == "character":
                text = f"{name} {entity.get('persona', '')} {entity.get('avatarDetail', '')}"
            else:
                text = f"{name} {entity.get('description', '')}"
            entity_texts.append(text)
        
        # Calculate similarity matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(entity_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicates
        duplicates = {}
        processed = set()
        
        for i in range(len(entity_names)):
            if entity_names[i] in processed:
                continue
            
            similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]
            
            if len(similar_indices) > 1:
                # Group similar entities
                group = [entity_names[j] for j in similar_indices]
                canonical_name = group[0]  # Use first as canonical
                duplicates[canonical_name] = group[1:]
                
                for name in group:
                    processed.add(name)
        
        return duplicates
    
    def merge_duplicate_entities(self, duplicates: Dict[str, List[str]]):
        """
        Merge duplicate entities
        
        Args:
            duplicates: Dictionary mapping canonical name to duplicate names
        """
        # Create mapping from duplicate to canonical
        duplicate_to_canonical = {}
        for canonical, dups in duplicates.items():
            for dup in dups:
                duplicate_to_canonical[dup] = canonical
        
        # Merge entity data
        for canonical, dups in duplicates.items():
            canonical_entity = self.entities[canonical]
            
            for dup in dups:
                dup_entity = self.entities[dup]
                
                # Merge based on entity type
                if canonical_entity["type"] == "character":
                    # Merge persona (prefer longer)
                    if len(dup_entity.get("persona", "")) > len(canonical_entity.get("persona", "")):
                        canonical_entity["persona"] = dup_entity["persona"]
                    
                    # Merge style_description
                    if len(dup_entity.get("style_description", "")) > len(canonical_entity.get("style_description", "")):
                        canonical_entity["style_description"] = dup_entity["style_description"]
                    
                    # Merge style_exemplars
                    exemplars = canonical_entity.get("style_exemplars", [])
                    for ex in dup_entity.get("style_exemplars", []):
                        if ex not in exemplars:
                            exemplars.append(ex)
                    canonical_entity["style_exemplars"] = exemplars
                else:
                    # Merge description
                    if len(dup_entity.get("description", "")) > len(canonical_entity.get("description", "")):
                        canonical_entity["description"] = dup_entity["description"]
                
                # Remove duplicate entity
                del self.entities[dup]
        
        # Update relationships
        updated_relationships = []
        for rel in self.relationships:
            # Map duplicate names to canonical
            source = duplicate_to_canonical.get(rel["source"], rel["source"])
            target = duplicate_to_canonical.get(rel["target"], rel["target"])
            
            # Skip self-loops
            if source == target:
                continue
            
            rel["source"] = source
            rel["target"] = target
            updated_relationships.append(rel)
        
        self.relationships = updated_relationships
        
        print(f"Merged {len(duplicate_to_canonical)} duplicate entities")
        print(f"Final entity count: {len(self.entities)}")
    
    def build_graph(self):
        """Build NetworkX graph from entities and relationships"""
        # Add nodes
        for name, entity in self.entities.items():
            self.graph.add_node(name, **entity)
        
        # Add edges
        for rel in self.relationships:
            self.graph.add_edge(
                rel["source"],
                rel["target"],
                description=rel.get("description", ""),
                attitude=rel.get("attitude"),
                strength=rel.get("strength", 0.5)
            )
        
        print(f"Built graph with {self.graph.number_of_nodes()} nodes "
              f"and {self.graph.number_of_edges()} edges")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        character_count = sum(1 for _, data in self.graph.nodes(data=True) 
                            if data.get("type") == "character")
        non_character_count = self.graph.number_of_nodes() - character_count
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "character_nodes": character_count,
            "non_character_nodes": non_character_count,
            "total_edges": self.graph.number_of_edges(),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "connected_components": nx.number_connected_components(self.graph)
        }
    
    def save_graph(self, output_dir: Path):
        """Save knowledge graph to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save entities
        entities_output = {
            "characters": [e for e in self.entities.values() if e["type"] == "character"],
            "non_characters": [e for e in self.entities.values() if e["type"] == "non-character"]
        }
        
        with open(output_dir / "entities_deduplicated.json", 'w', encoding='utf-8') as f:
            json.dump(entities_output, f, ensure_ascii=False, indent=2)
        
        # Save relationships
        with open(output_dir / "relationships_deduplicated.json", 'w', encoding='utf-8') as f:
            json.dump(self.relationships, f, ensure_ascii=False, indent=2)
        
        # Save graph as GraphML
        nx.write_graphml(self.graph, output_dir / "knowledge_graph.graphml")
        
        # Save statistics
        stats = self.get_graph_statistics()
        with open(output_dir / "graph_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Saved knowledge graph to {output_dir}")
        print("Statistics:", stats)


if __name__ == "__main__":
    from config import config
    import sys
    
    # Load entities
    entities_file = config.KG_DIR / "entities.json"
    if not entities_file.exists():
        print(f"Error: {entities_file} not found. Run entity_extraction.py first.")
        sys.exit(1)
    
    # Load relationships
    relationships_file = config.KG_DIR / "relationships.json"
    if not relationships_file.exists():
        print(f"Error: {relationships_file} not found. Run relationship_extraction.py first.")
        sys.exit(1)
    
    # Build knowledge graph
    builder = KnowledgeGraphBuilder()
    builder.load_entities(entities_file)
    builder.load_relationships(relationships_file)
    
    # Find and merge duplicates
    print("\nFinding duplicate entities...")
    duplicates = builder.find_duplicate_entities(similarity_threshold=0.85)
    print(f"Found {len(duplicates)} groups of duplicates")
    
    if duplicates:
        print("\nMerging duplicates...")
        builder.merge_duplicate_entities(duplicates)
    
    # Build graph
    print("\nBuilding graph...")
    builder.build_graph()
    
    # Save results
    builder.save_graph(config.KG_DIR)
