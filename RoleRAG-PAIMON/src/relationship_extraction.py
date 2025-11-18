"""
Relationship extraction from text chunks using Gemini API
"""
from typing import List, Dict, Any
from pathlib import Path
import json
from tqdm import tqdm

from .gemini_client import GeminiClient
from models.schema import Relationship


class RelationshipExtractor:
    """Extract relationships between entities using LLM"""
    
    RELATIONSHIP_EXTRACTION_PROMPT = """
你是一个专业的关系提取助手。请从以下文本中提取实体之间的关系。

文本：
{text}

已知实体列表：
{entities}

请识别文本中这些实体之间的关系，对于每个关系，提取以下信息：
1. source: 源实体名称（必须在已知实体列表中）
2. target: 目标实体名称（必须在已知实体列表中）
3. description: 客观的关系描述（事实性的关系）
4. attitude: 源实体对目标实体的主观态度或看法（可选，如果文本中没有明确表达态度可为null）
5. strength: 关系强度，0到1之间的浮点数（0.0-0.3为弱关系，0.3-0.7为中等关系，0.7-1.0为强关系）

返回JSON格式：
{{
  "relationships": [
    {{
      "source": "实体A",
      "target": "实体B",
      "description": "关系描述",
      "attitude": "态度描述或null",
      "strength": 0.8
    }}
  ]
}}
"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        self.relationships = []
        
    def extract_relationships_from_chunk(
        self, 
        chunk: Dict[str, Any], 
        entity_names: List[str]
    ) -> List[Relationship]:
        """
        Extract relationships from a text chunk
        
        Args:
            chunk: Text chunk with metadata
            entity_names: List of known entity names
            
        Returns:
            List of Relationship objects
        """
        text = chunk.get("text", "")
        if not text or len(entity_names) < 2:
            return []
        
        # Format entity list
        entities_str = ", ".join(entity_names)
        
        prompt = self.RELATIONSHIP_EXTRACTION_PROMPT.format(
            text=text,
            entities=entities_str
        )
        
        try:
            result = self.client.generate_json(prompt, temperature=0.3)
            relationships = result.get("relationships", [])
            
            rel_objects = []
            for rel_data in relationships:
                try:
                    # Validate that source and target are in entity list
                    source = rel_data.get("source", "")
                    target = rel_data.get("target", "")
                    
                    if source not in entity_names or target not in entity_names:
                        continue
                    
                    # Handle null attitude
                    if rel_data.get("attitude") == "null":
                        rel_data["attitude"] = None
                    
                    relationship = Relationship(**rel_data)
                    rel_objects.append(relationship)
                except Exception as e:
                    print(f"Error creating Relationship: {e}")
                    print(f"Data: {rel_data}")
            
            return rel_objects
            
        except Exception as e:
            print(f"Error extracting relationships from chunk: {e}")
            return []
    
    def process_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        entities: Dict[str, Any]
    ) -> List[Relationship]:
        """
        Process all chunks and extract relationships
        
        Args:
            chunks: List of text chunks
            entities: Dictionary with character and non-character entities
            
        Returns:
            List of Relationship objects
        """
        # Collect all entity names
        entity_names = []
        entity_names.extend(entities.get("characters", {}).keys())
        entity_names.extend(entities.get("non_characters", {}).keys())
        
        print(f"Processing {len(chunks)} chunks with {len(entity_names)} entities...")
        
        for chunk in tqdm(chunks, desc="Extracting relationships"):
            relationships = self.extract_relationships_from_chunk(chunk, entity_names)
            self.relationships.extend(relationships)
        
        # Deduplicate relationships
        self.relationships = self._deduplicate_relationships(self.relationships)
        
        return self.relationships
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Deduplicate and merge similar relationships
        
        Two relationships are considered the same if they have the same source and target
        """
        rel_dict = {}
        
        for rel in relationships:
            key = f"{rel.source}||{rel.target}"
            
            if key in rel_dict:
                # Merge: keep the one with higher strength
                existing = rel_dict[key]
                if rel.strength > existing.strength:
                    rel_dict[key] = rel
                elif rel.strength == existing.strength:
                    # Merge descriptions
                    if rel.description not in existing.description:
                        existing.description += f"; {rel.description}"
                    # Merge attitudes
                    if rel.attitude and existing.attitude:
                        if rel.attitude not in existing.attitude:
                            existing.attitude += f"; {rel.attitude}"
                    elif rel.attitude and not existing.attitude:
                        existing.attitude = rel.attitude
            else:
                rel_dict[key] = rel
        
        return list(rel_dict.values())
    
    def save_relationships(self, output_file: Path):
        """Save extracted relationships to file"""
        output_data = [rel.model_dump() for rel in self.relationships]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(output_data)} relationships to {output_file}")


if __name__ == "__main__":
    from config import config
    import sys
    
    # Check API key
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in .env file")
        sys.exit(1)
    
    # Load chunks
    chunks_file = config.OUTPUT_DIR / "text_chunks.json"
    if not chunks_file.exists():
        print(f"Error: {chunks_file} not found. Run data_preprocessing.py first.")
        sys.exit(1)
    
    # Load entities
    entities_file = config.KG_DIR / "entities.json"
    if not entities_file.exists():
        print(f"Error: {entities_file} not found. Run entity_extraction.py first.")
        sys.exit(1)
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities_data = json.load(f)
        # Convert to dict format
        entities = {
            "characters": {e["name"]: e for e in entities_data["characters"]},
            "non_characters": {e["name"]: e for e in entities_data["non_characters"]}
        }
    
    # Extract relationships
    client = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    extractor = RelationshipExtractor(client)
    
    relationships = extractor.process_chunks(chunks, entities)
    
    # Save results
    output_file = config.KG_DIR / "relationships.json"
    extractor.save_relationships(output_file)
