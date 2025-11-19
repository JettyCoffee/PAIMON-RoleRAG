from typing import List, Dict, Any
from utils import get_gemini_model

SUMMARY_PROMPT = """
Summarize the following conversation turn for future reference.
Focus on key facts discussed and any personal details revealed.

User: {user_query}
AI: {ai_response}

Summary:
"""

class MemoryManager:
    def __init__(self):
        self.history: List[Dict[str, str]] = [] # List of {role: ..., content: ...}
        self.summaries: List[str] = []
        self.model = get_gemini_model()

    def add_turn(self, user_query: str, ai_response: str):
        """Adds a turn to history and generates a summary."""
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": ai_response})
        
        # Generate summary
        try:
            prompt = SUMMARY_PROMPT.format(user_query=user_query, ai_response=ai_response)
            response = self.model.generate_content(prompt)
            self.summaries.append(response.text)
        except Exception as e:
            print(f"Error generating summary: {e}")

    def get_recent_context(self, k: int = 3) -> str:
        """Returns the last k turns formatted as text."""
        recent = self.history[-2*k:]
        context = ""
        for msg in recent:
            context += f"{msg['role'].upper()}: {msg['content']}\n"
        return context

    def check_callback(self, query: str) -> str:
        """Checks if the query refers to past context (simplified)."""
        # In a real system, we would use an LLM to classify if the query needs history
        # and which summaries are relevant.
        # Here we just return all summaries as context if the history is short.
        if not self.summaries:
            return ""
        
        return "Previous Context:\n" + "\n".join(self.summaries)

if __name__ == "__main__":
    mem = MemoryManager()
    mem.add_turn("Who are you?", "I am Harry Potter.")
    print(mem.get_recent_context())
    print(mem.check_callback("What did I just ask?"))
