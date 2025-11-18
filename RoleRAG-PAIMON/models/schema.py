"""
Data models for RoleRAG-PAIMON
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CharacterEntity(BaseModel):
    """Character entity with style information"""
    name: str
    type: str = "character"
    persona: str  # Basic character introduction
    style_description: str  # Speaking style/personality
    style_exemplars: List[str] = Field(default_factory=list)  # Representative phrases
    
    
class NonCharacterEntity(BaseModel):
    """Non-character entity (events, items, locations, etc.)"""
    name: str
    type: str = "non-character"
    description: str


class Relationship(BaseModel):
    """Relationship between entities"""
    source: str  # Source entity name
    target: str  # Target entity name
    description: str  # Objective relationship description
    attitude: Optional[str] = None  # Source's attitude towards target
    strength: float = Field(ge=0.0, le=1.0)  # Relationship strength [0, 1]
    

class Entity(BaseModel):
    """Generic entity wrapper"""
    name: str
    type: str
    data: Dict[str, Any]
    
    
class Community(BaseModel):
    """Community detected from KG"""
    id: str
    members: List[str]  # Entity names
    summary: str  # Generated summary
    type: str  # "character-focused" or "event-focused"
    

class Query(BaseModel):
    """User query"""
    text: str
    type: Optional[str] = None  # "character" or "event"
    

class SubQuery(BaseModel):
    """Decomposed sub-query"""
    text: str
    type: str  # "character" or "event"
    priority: int = 1
    

class ConversationTurn(BaseModel):
    """Single conversation turn"""
    query: str
    response: str
    summary: str
    retrieved_context: List[Dict[str, Any]] = Field(default_factory=list)
    

class ConversationMemory(BaseModel):
    """Conversation history and cache"""
    turns: List[ConversationTurn] = Field(default_factory=list)
    cache: Dict[str, Any] = Field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn"""
        self.turns.append(turn)
        
    def get_recent_turns(self, k: int) -> List[ConversationTurn]:
        """Get last k turns"""
        return self.turns[-k:] if len(self.turns) >= k else self.turns
