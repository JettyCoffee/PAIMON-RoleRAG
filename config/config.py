"""
Configuration management for RoleRAG-PAIMON
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Global configuration"""
    
    # Project paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "datasets"
    OUTPUT_DIR = ROOT_DIR / "output"
    KG_DIR = OUTPUT_DIR / "knowledge_graph"
    CACHE_DIR = OUTPUT_DIR / "cache"
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # Data processing
    AVATAR_FILE = DATA_DIR / "avatar_CHS.json"
    
    # KG construction parameters
    CHUNK_SIZE = 512  # characters per chunk
    ENTITY_TYPES = ["character", "non-character"]
    
    # Relationship parameters
    MIN_RELATIONSHIP_STRENGTH = 0.3
    
    # Community detection parameters
    COMMUNITY_ALGORITHM = "louvain"  # louvain, leiden, etc.
    MIN_COMMUNITY_SIZE = 2
    
    # Agent parameters
    MAX_ITERATIONS = 5  # max retrieval iterations
    QUERY_TYPES = ["character", "event"]
    
    # Memory parameters
    CONVERSATION_HISTORY_LIMIT = 5  # keep last k conversations
    
    # LLM generation parameters
    TEMPERATURE = 0.7
    MAX_OUTPUT_TOKENS = 1024
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.KG_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)

config = Config()
