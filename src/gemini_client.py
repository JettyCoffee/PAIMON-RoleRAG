"""
Gemini API client wrapper
"""
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import time


class GeminiClient:
    """Wrapper for Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Gemini API key
            model_name: Model name (default: gemini-pro)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chat = None
        
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retry: int = 3
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            retry: Number of retries on failure
            
        Returns:
            Generated text
        """
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
            
        for attempt in range(retry):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
            except Exception as e:
                if attempt < retry - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API call failed (attempt {attempt + 1}/{retry}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
        
        return ""
    
    def generate_json(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retry: int = 3
    ) -> Dict[str, Any]:
        """
        Generate JSON response from prompt
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            retry: Number of retries on failure
            
        Returns:
            Parsed JSON object
        """
        # Add JSON format instruction
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON only, no additional text."
        
        response_text = self.generate(json_prompt, temperature, max_tokens, retry)
        
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response: {response_text[:500]}")
            return {}
    
    def start_chat(self, history: Optional[List[Dict[str, str]]] = None):
        """
        Start a chat session
        
        Args:
            history: Previous conversation history
        """
        self.chat = self.model.start_chat(history=history or [])
    
    def send_message(
        self, 
        message: str, 
        temperature: float = 0.7,
        retry: int = 3
    ) -> str:
        """
        Send message in chat session
        
        Args:
            message: User message
            temperature: Sampling temperature
            retry: Number of retries on failure
            
        Returns:
            Model response
        """
        if self.chat is None:
            self.start_chat()
            
        generation_config = {"temperature": temperature}
        
        for attempt in range(retry):
            try:
                response = self.chat.send_message(
                    message,
                    generation_config=generation_config
                )
                return response.text
            except Exception as e:
                if attempt < retry - 1:
                    wait_time = 2 ** attempt
                    print(f"Chat API call failed (attempt {attempt + 1}/{retry}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
        
        return ""
