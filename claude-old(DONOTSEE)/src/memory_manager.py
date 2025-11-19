"""
Conversation memory management for multi-turn dialogue
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from models.schema import ConversationMemory, ConversationTurn
from .gemini_client import GeminiClient


class MemoryManager:
    """Manage conversation memory and cache"""
    
    CALLBACK_DETECTION_PROMPT = """
你是一个对话分析助手。请分析用户的新问题是否涉及之前的对话内容。

对话历史摘要：
{history_summaries}

新问题：{query}

请判断新问题是否需要回调之前的对话内容（例如："我们之前讨论的xx"、"刚才说的"、"那个角色"等）。

返回JSON格式：
{{
  "needs_callback": true/false,
  "related_turn_indices": [相关的对话轮次索引，从0开始],
  "reason": "判断理由"
}}
"""

    CACHE_CHECK_PROMPT = """
你是一个信息充分性评估助手。请判断缓存中的信息是否足以回答新的子查询。

子查询：{subquery}

缓存信息：
{cache_info}

请判断缓存信息是否足够回答子查询。

返回JSON格式：
{{
  "is_sufficient": true/false,
  "reason": "判断理由"
}}
"""
    
    def __init__(
        self, 
        gemini_client: GeminiClient,
        history_limit: int = 5,
        cache_file: Optional[Path] = None
    ):
        self.client = gemini_client
        self.history_limit = history_limit
        self.cache_file = cache_file
        self.memory = ConversationMemory()
        
        # Load existing memory if cache file exists
        if cache_file and cache_file.exists():
            self.load_memory(cache_file)
    
    def detect_callback(self, query: str) -> tuple[bool, List[int]]:
        """
        Detect if query needs to callback previous conversations
        
        Args:
            query: New user query
            
        Returns:
            (needs_callback, related_turn_indices)
        """
        if not self.memory.turns:
            return False, []
        
        # Get recent history summaries
        recent_turns = self.memory.get_recent_turns(self.history_limit)
        history_summaries = []
        for i, turn in enumerate(recent_turns):
            history_summaries.append(f"轮次{i}: {turn.summary}")
        
        history_str = "\n".join(history_summaries)
        
        prompt = self.CALLBACK_DETECTION_PROMPT.format(
            history_summaries=history_str,
            query=query
        )
        
        try:
            result = self.client.generate_json(prompt, temperature=0.3)
            needs_callback = result.get("needs_callback", False)
            related_indices = result.get("related_turn_indices", [])
            
            return needs_callback, related_indices
            
        except Exception as e:
            print(f"Error detecting callback: {e}")
            return False, []
    
    def check_cache(self, subquery: str) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Check if cache contains sufficient information for subquery
        
        Args:
            subquery: Sub-query text
            
        Returns:
            (is_sufficient, cached_info)
        """
        if not self.memory.cache:
            return False, []
        
        # Get recent cached info
        recent_turns = self.memory.get_recent_turns(self.history_limit)
        cached_info = []
        for turn in recent_turns:
            cached_info.extend(turn.retrieved_context)
        
        if not cached_info:
            return False, []
        
        # Format cache for prompt
        cache_summary = json.dumps(cached_info, ensure_ascii=False)[:1000]  # Limit length
        
        prompt = self.CACHE_CHECK_PROMPT.format(
            subquery=subquery,
            cache_info=cache_summary
        )
        
        try:
            result = self.client.generate_json(prompt, temperature=0.3)
            is_sufficient = result.get("is_sufficient", False)
            
            if is_sufficient:
                return True, cached_info
            else:
                return False, []
                
        except Exception as e:
            print(f"Error checking cache: {e}")
            return False, []
    
    def add_turn(
        self, 
        query: str, 
        response: str, 
        summary: str,
        retrieved_context: List[Dict[str, Any]]
    ):
        """
        Add a conversation turn to memory
        
        Args:
            query: User query
            response: System response
            summary: Conversation summary
            retrieved_context: Retrieved context information
        """
        turn = ConversationTurn(
            query=query,
            response=response,
            summary=summary,
            retrieved_context=retrieved_context
        )
        
        self.memory.add_turn(turn)
        
        # Update cache with new retrieved context
        for context in retrieved_context:
            # Use subquery text as cache key
            cache_key = context.get("subquery", "")
            if cache_key:
                self.memory.cache[cache_key] = context
    
    def get_callback_context(self, related_indices: List[int]) -> str:
        """
        Get context from related previous turns
        
        Args:
            related_indices: Indices of related turns
            
        Returns:
            Formatted context string
        """
        if not related_indices:
            return ""
        
        context_parts = []
        recent_turns = self.memory.get_recent_turns(self.history_limit)
        
        for idx in related_indices:
            if 0 <= idx < len(recent_turns):
                turn = recent_turns[idx]
                context_parts.append(f"[之前的对话]\n问：{turn.query}\n答：{turn.response}")
        
        return "\n\n".join(context_parts)
    
    def clear_old_cache(self, max_cache_size: int = 20):
        """
        Clear old cache entries to limit memory usage
        
        Args:
            max_cache_size: Maximum number of cache entries
        """
        if len(self.memory.cache) > max_cache_size:
            # Remove oldest entries (FIFO)
            cache_items = list(self.memory.cache.items())
            self.memory.cache = dict(cache_items[-max_cache_size:])
    
    def save_memory(self, output_file: Path):
        """Save conversation memory to file"""
        memory_data = {
            "turns": [turn.model_dump() for turn in self.memory.turns],
            "cache": self.memory.cache
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
    
    def load_memory(self, input_file: Path):
        """Load conversation memory from file"""
        with open(input_file, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        # Load turns
        for turn_data in memory_data.get("turns", []):
            turn = ConversationTurn(**turn_data)
            self.memory.turns.append(turn)
        
        # Load cache
        self.memory.cache = memory_data.get("cache", {})
    
    def get_recent_context(self, k: int = 3) -> str:
        """
        Get recent conversation context
        
        Args:
            k: Number of recent turns
            
        Returns:
            Formatted context string
        """
        recent_turns = self.memory.get_recent_turns(k)
        
        if not recent_turns:
            return ""
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"问：{turn.query}\n答：{turn.response}")
        
        return "\n\n".join(context_parts)
