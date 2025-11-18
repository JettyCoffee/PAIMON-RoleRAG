"""
Data preprocessing for avatar data
"""
import json
from typing import Dict, List, Any
from pathlib import Path


class DataPreprocessor:
    """Preprocess avatar data from JSON"""
    
    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.raw_data = None
        self.processed_data = []
        
    def load_data(self) -> Dict[str, Any]:
        """Load raw avatar data"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        return self.raw_data
    
    def extract_key_fields(self, avatar_name: str, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key fields for a single avatar
        
        Focus on: avatar, avatarDetail, avatarNative, avatarTitle, desc,
                 infoBirthDay, infoBirthMonth, sayings, story
        """
        extracted = {
            "name": avatar_name,
            "avatar": avatar_name,
            "avatarDetail": avatar_data.get("avatarDetail", ""),
            "avatarNative": avatar_data.get("avatarNative", ""),
            "avatarTitle": avatar_data.get("avatarTitle", ""),
            "desc": avatar_data.get("desc", ""),
            "infoBirthDay": avatar_data.get("infoBirthDay", ""),
            "infoBirthMonth": avatar_data.get("infoBirthMonth", ""),
            "sayings": avatar_data.get("sayings", []),
            "story": avatar_data.get("story", [])
        }
        return extracted
    
    def chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters
        Tries to break at sentence boundaries
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        current_chunk = ""
        
        # Split by common sentence endings
        sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def create_text_chunks(self, avatar_data: Dict[str, Any], chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Create text chunks from avatar data for entity extraction
        
        Returns list of chunks with metadata
        """
        chunks = []
        
        # Combine basic info
        basic_info = f"{avatar_data['name']}，{avatar_data['avatarTitle']}。{avatar_data['desc']}"
        if avatar_data['avatarNative']:
            basic_info += f"来自{avatar_data['avatarNative']}。"
        if avatar_data['infoBirthMonth'] and avatar_data['infoBirthDay']:
            basic_info += f"生日：{avatar_data['infoBirthMonth']}月{avatar_data['infoBirthDay']}日。"
        
        chunks.append({
            "avatar": avatar_data['name'],
            "type": "basic_info",
            "text": basic_info
        })
        
        # Process sayings
        if avatar_data['sayings']:
            sayings_text = "\n".join(avatar_data['sayings'])
            saying_chunks = self.chunk_text(sayings_text, chunk_size)
            for i, chunk in enumerate(saying_chunks):
                chunks.append({
                    "avatar": avatar_data['name'],
                    "type": "sayings",
                    "chunk_id": i,
                    "text": chunk
                })
        
        # Process story
        if avatar_data['story']:
            story_text = "\n".join(avatar_data['story'])
            story_chunks = self.chunk_text(story_text, chunk_size)
            for i, chunk in enumerate(story_chunks):
                chunks.append({
                    "avatar": avatar_data['name'],
                    "type": "story",
                    "chunk_id": i,
                    "text": chunk
                })
        
        return chunks
    
    def process_all(self, chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Process all avatars
        
        Returns list of all text chunks with metadata
        """
        if self.raw_data is None:
            self.load_data()
        
        all_chunks = []
        
        for avatar_name, avatar_data in self.raw_data.items():
            # Extract key fields
            extracted = self.extract_key_fields(avatar_name, avatar_data)
            self.processed_data.append(extracted)
            
            # Create text chunks
            chunks = self.create_text_chunks(extracted, chunk_size)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def save_processed_data(self, output_file: Path):
        """Save processed data"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: Path):
        """Save chunks"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    from pathlib import Path
    
    # Test preprocessing
    data_file = Path(__file__).parent.parent / "datasets" / "avatar_CHS.json"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    preprocessor = DataPreprocessor(data_file)
    chunks = preprocessor.process_all(chunk_size=512)
    
    print(f"Loaded {len(preprocessor.processed_data)} avatars")
    print(f"Created {len(chunks)} text chunks")
    
    # Save results
    preprocessor.save_processed_data(output_dir / "processed_avatars.json")
    preprocessor.save_chunks(chunks, output_dir / "text_chunks.json")
    
    print(f"Saved to {output_dir}")
