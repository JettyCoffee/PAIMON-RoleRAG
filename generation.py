from typing import List, Dict, Any
from utils import get_gemini_model

GENERATION_PROMPT = """
You are playing the role of **{role}**.
Your goal is to answer the user's question based ONLY on the provided context, while strictly adhering to your character's persona and style.

**Context Information**:
{context}

**User Question**:
{query}

**Instructions**:
1. Answer the question accurately using the context.
2. Adopt the persona of {role}. Use their tone, vocabulary, and mannerisms.
3. If the context doesn't have the answer, admit it in character (e.g., "I haven't the faintest idea...").
4. Do NOT mention that you are an AI or that you were given context.

**Response**:
"""

class Generator:
    def __init__(self, role: str = "Harry Potter"):
        self.model = get_gemini_model()
        self.role = role

    def assemble_context(self, retrieved_info: str) -> str:
        """Formats the retrieved info for the prompt."""
        # In a real system, we might re-rank or filter here.
        return retrieved_info

    def generate_response(self, query: str, context: str) -> str:
        """Generates the final response."""
        prompt = GENERATION_PROMPT.format(
            role=self.role,
            context=context,
            query=query
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I... I can't speak right now."

if __name__ == "__main__":
    # Test
    gen = Generator(role="Harry Potter")
    ctx = "Harry Potter is a wizard. He goes to Hogwarts. He is friends with Ron and Hermione."
    q = "Who are your friends?"
    print(gen.generate_response(q, ctx))
