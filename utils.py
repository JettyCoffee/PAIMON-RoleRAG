import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Any, Dict, List

# Load environment variables
load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    # Fallback or warning if key is missing, though usually it should be in .env
    print("Warning: GENAI_API_KEY not found in environment variables.")

genai.configure(api_key=GENAI_API_KEY)

def get_gemini_model(model_name="gemini-2.5-pro"):
    """Returns a configured Gemini model instance."""
    generation_config = {
        "temperature": 0.1, # Low temperature for factual extraction
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

def parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parses JSON response from LLM, handling potential markdown code blocks."""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw text: {text}")
        return {}
