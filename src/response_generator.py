"""
Response generation using LLM with retrieved context
"""
from typing import List, Dict, Any
from pathlib import Path
import json

from .gemini_client import GeminiClient


class ResponseGenerator:
    """Generate role-playing responses using LLM"""
    
    RESPONSE_GENERATION_PROMPT = """
你现在要扮演原神游戏中的角色来回答问题。

用户问题：{query}

检索到的上下文信息：
{context}

角色信息：
{character_info}

请基于上述信息，以该角色的口吻和风格来回答用户的问题。注意：
1. 保持角色的性格特点和说话风格
2. 使用角色惯用的语气和措辞
3. 回答要基于检索到的事实信息
4. 保持自然流畅，不要机械照搬
5. 如果信息不足以回答，可以诚实地表示不清楚

直接给出角色的回答，不要包含任何元信息或解释。
"""

    SUMMARY_GENERATION_PROMPT = """
请为以下对话生成一个简洁的摘要（50字以内）。

问题：{query}
回答：{response}

直接返回摘要文本。
"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
    
    def format_context(self, retrieved_info: List[Dict[str, Any]]) -> str:
        """
        Format retrieved information into context string
        
        Args:
            retrieved_info: List of retrieved information chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieved_info):
            context_parts.append(f"### 信息块 {i + 1}")
            context_parts.append(f"查询类型：{chunk.get('type', 'unknown')}")
            
            # Add entity information
            entities = chunk.get("entities", [])
            if entities:
                context_parts.append("\n**相关实体：**")
                for entity in entities[:5]:  # Limit to top 5
                    name = entity.get("name", "")
                    if entity.get("type") == "character":
                        persona = entity.get("persona", "")
                        style = entity.get("style_description", "")
                        exemplars = entity.get("style_exemplars", [])
                        context_parts.append(f"- {name}：{persona}")
                        if style:
                            context_parts.append(f"  说话风格：{style}")
                        if exemplars:
                            context_parts.append(f"  惯用语：{', '.join(exemplars[:3])}")
                    else:
                        desc = entity.get("description", "")
                        context_parts.append(f"- {name}：{desc}")
                    
                    # Add relationships
                    neighbors = entity.get("neighbors", [])
                    if neighbors:
                        rel_strs = []
                        for nb in neighbors[:3]:  # Limit to top 3
                            rel_str = f"{nb['name']}({nb.get('relationship', '')})"
                            rel_strs.append(rel_str)
                        context_parts.append(f"  关系：{', '.join(rel_strs)}")
            
            # Add community summaries
            communities = chunk.get("communities", [])
            if communities:
                context_parts.append("\n**相关社区：**")
                for comm in communities[:2]:  # Limit to top 2
                    summary = comm.get("summary", "")
                    context_parts.append(f"- {summary}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def extract_character_info(self, retrieved_info: List[Dict[str, Any]]) -> str:
        """
        Extract main character information from retrieved data
        
        Args:
            retrieved_info: Retrieved information
            
        Returns:
            Character info string
        """
        # Find the main character entity
        main_character = None
        
        for chunk in retrieved_info:
            entities = chunk.get("entities", [])
            for entity in entities:
                if entity.get("type") == "character":
                    # Assume first character entity is the main one
                    main_character = entity
                    break
            if main_character:
                break
        
        if not main_character:
            return "未找到明确的角色信息。"
        
        # Format character info
        name = main_character.get("name", "")
        persona = main_character.get("persona", "")
        style = main_character.get("style_description", "")
        exemplars = main_character.get("style_exemplars", [])
        
        info_parts = [
            f"角色：{name}",
            f"性格：{persona}",
            f"说话风格：{style}"
        ]
        
        if exemplars:
            info_parts.append(f"惯用语：{', '.join(exemplars[:5])}")
        
        return "\n".join(info_parts)
    
    def generate_response(
        self, 
        query: str, 
        retrieved_info: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate response using LLM
        
        Args:
            query: User query
            retrieved_info: Retrieved context information
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            
        Returns:
            Generated response
        """
        # Format context
        context = self.format_context(retrieved_info)
        character_info = self.extract_character_info(retrieved_info)
        
        # Generate prompt
        prompt = self.RESPONSE_GENERATION_PROMPT.format(
            query=query,
            context=context,
            character_info=character_info
        )
        
        # Generate response
        response = self.client.generate(
            prompt, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        return response
    
    def generate_summary(self, query: str, response: str) -> str:
        """
        Generate summary for a conversation turn
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Summary text
        """
        prompt = self.SUMMARY_GENERATION_PROMPT.format(
            query=query,
            response=response
        )
        
        summary = self.client.generate(prompt, temperature=0.3, max_tokens=100)
        return summary


if __name__ == "__main__":
    from config import config
    import sys
    
    # Check API key
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in .env file")
        sys.exit(1)
    
    # Load test retrieval results
    retrieval_file = config.OUTPUT_DIR / "test_retrieval.json"
    if not retrieval_file.exists():
        print(f"Error: {retrieval_file} not found. Run retrieval_agent.py first.")
        sys.exit(1)
    
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        retrieved_info = json.load(f)
    
    # Initialize generator
    client = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    generator = ResponseGenerator(client)
    
    # Test query
    test_query = "七七是谁？她的性格和说话风格是什么样的？"
    
    print(f"Query: {test_query}\n")
    
    # Generate response
    response = generator.generate_response(
        test_query,
        retrieved_info,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS
    )
    
    print(f"Response:\n{response}\n")
    
    # Generate summary
    summary = generator.generate_summary(test_query, response)
    print(f"Summary: {summary}")
    
    # Save test results
    output_file = config.OUTPUT_DIR / "test_response.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "query": test_query,
            "response": response,
            "summary": summary
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved test response to {output_file}")
