"""
Entity extraction from text chunks using Gemini API
"""
from typing import List, Dict, Any
from pathlib import Path
import json
from tqdm import tqdm

from .gemini_client import GeminiClient
from models.schema import CharacterEntity, NonCharacterEntity


class EntityExtractor:
    """Extract entities from text chunks using LLM"""
    
    # Prompt templates
    CHARACTER_EXTRACTION_PROMPT = """
你是一个专业的信息提取助手。请从以下文本中提取角色实体信息。

文本：
{text}

请提取文本中出现的所有角色（character），对于每个角色，提取以下信息：
1. name: 角色名字
2. type: "character"（固定值）
3. persona: 角色的基本简介（性格、身份、背景等）
4. style_description: 角色的说话风格和性格特点（如何说话、语气特点等）
5. style_exemplars: 角色的口癖或代表性语句（列表，选择3-5个最有代表性的短语）

返回JSON格式：
{{
  "characters": [
    {{
      "name": "角色名",
      "type": "character",
      "persona": "角色简介",
      "style_description": "说话风格描述",
      "style_exemplars": ["代表性语句1", "代表性语句2", ...]
    }}
  ]
}}
"""

    NON_CHARACTER_EXTRACTION_PROMPT = """
你是一个专业的信息提取助手。请从以下文本中提取非角色实体信息。

文本：
{text}

请提取文本中出现的所有非角色实体（non-character），包括：
- 地点/场所
- 物品/道具
- 事件
- 组织
- 概念

对于每个实体，提取以下信息：
1. name: 实体名称
2. type: "non-character"（固定值）
3. description: 实体的描述

返回JSON格式：
{{
  "non_characters": [
    {{
      "name": "实体名",
      "type": "non-character",
      "description": "实体描述"
    }}
  ]
}}
"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        self.entities = {
            "characters": {},
            "non_characters": {}
        }
        
    def extract_characters_from_chunk(self, chunk: Dict[str, Any]) -> List[CharacterEntity]:
        """
        Extract character entities from a text chunk
        
        Args:
            chunk: Text chunk with metadata
            
        Returns:
            List of CharacterEntity objects
        """
        text = chunk.get("text", "")
        if not text:
            return []
        
        prompt = self.CHARACTER_EXTRACTION_PROMPT.format(text=text)
        
        try:
            result = self.client.generate_json(prompt, temperature=0.3)
            characters = result.get("characters", [])
            
            entities = []
            for char_data in characters:
                try:
                    entity = CharacterEntity(**char_data)
                    entities.append(entity)
                except Exception as e:
                    print(f"Error creating CharacterEntity: {e}")
                    print(f"Data: {char_data}")
            
            return entities
            
        except Exception as e:
            print(f"Error extracting characters from chunk: {e}")
            return []
    
    def extract_non_characters_from_chunk(self, chunk: Dict[str, Any]) -> List[NonCharacterEntity]:
        """
        Extract non-character entities from a text chunk
        
        Args:
            chunk: Text chunk with metadata
            
        Returns:
            List of NonCharacterEntity objects
        """
        text = chunk.get("text", "")
        if not text:
            return []
        
        prompt = self.NON_CHARACTER_EXTRACTION_PROMPT.format(text=text)
        
        try:
            result = self.client.generate_json(prompt, temperature=0.3)
            non_characters = result.get("non_characters", [])
            
            entities = []
            for nc_data in non_characters:
                try:
                    entity = NonCharacterEntity(**nc_data)
                    entities.append(entity)
                except Exception as e:
                    print(f"Error creating NonCharacterEntity: {e}")
                    print(f"Data: {nc_data}")
            
            return entities
            
        except Exception as e:
            print(f"Error extracting non-characters from chunk: {e}")
            return []
    
    def process_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        extract_characters: bool = True,
        extract_non_characters: bool = True
    ) -> Dict[str, Any]:
        """
        Process all chunks and extract entities
        
        Args:
            chunks: List of text chunks
            extract_characters: Whether to extract characters
            extract_non_characters: Whether to extract non-characters
            
        Returns:
            Dictionary with extracted entities
        """
        print(f"Processing {len(chunks)} chunks...")
        
        for chunk in tqdm(chunks, desc="Extracting entities"):
            # Extract characters
            if extract_characters:
                characters = self.extract_characters_from_chunk(chunk)
                for char in characters:
                    # Merge if entity already exists
                    if char.name in self.entities["characters"]:
                        self._merge_character(char)
                    else:
                        self.entities["characters"][char.name] = char
            
            # Extract non-characters
            if extract_non_characters:
                non_chars = self.extract_non_characters_from_chunk(chunk)
                for nc in non_chars:
                    # Merge if entity already exists
                    if nc.name in self.entities["non_characters"]:
                        self._merge_non_character(nc)
                    else:
                        self.entities["non_characters"][nc.name] = nc
        
        return self.entities
    
    def _merge_character(self, new_char: CharacterEntity):
        """Merge new character info with existing"""
        existing = self.entities["characters"][new_char.name]
        
        # Merge persona (prefer longer description)
        if len(new_char.persona) > len(existing.persona):
            existing.persona = new_char.persona
        
        # Merge style_description (prefer longer)
        if len(new_char.style_description) > len(existing.style_description):
            existing.style_description = new_char.style_description
        
        # Merge style_exemplars (avoid duplicates)
        for exemplar in new_char.style_exemplars:
            if exemplar not in existing.style_exemplars:
                existing.style_exemplars.append(exemplar)
    
    def _merge_non_character(self, new_nc: NonCharacterEntity):
        """Merge new non-character info with existing"""
        existing = self.entities["non_characters"][new_nc.name]
        
        # Merge description (prefer longer)
        if len(new_nc.description) > len(existing.description):
            existing.description = new_nc.description
    
    def save_entities(self, output_file: Path):
        """Save extracted entities to file"""
        output_data = {
            "characters": [char.model_dump() for char in self.entities["characters"].values()],
            "non_characters": [nc.model_dump() for nc in self.entities["non_characters"].values()]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(output_data['characters'])} characters and "
              f"{len(output_data['non_characters'])} non-characters to {output_file}")


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
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Extract entities
    client = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    extractor = EntityExtractor(client)
    
    entities = extractor.process_chunks(chunks)
    
    # Save results
    config.ensure_dirs()
    output_file = config.KG_DIR / "entities.json"
    extractor.save_entities(output_file)
