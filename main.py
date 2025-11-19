from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
import networkx as nx
from retrieval_agent import RetrievalAgent
from generation import Generator
from memory import MemoryManager
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PAIMON: RoleRAG API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
GRAPH_PATH = "/root/RoleRAG/output/role_rag_graph.json"
agent = None
generator = None
memory = None

class ChatRequest(BaseModel):
    message: str
    role: str = "Harry Potter"

class ChatResponse(BaseModel):
    response: str
    context: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global agent, generator, memory
    # Initialize components
    # Note: If graph doesn't exist, agent might fail to load. Handle gracefully.
    if os.path.exists(GRAPH_PATH):
        agent = RetrievalAgent(GRAPH_PATH)
    else:
        print("Warning: Graph file not found. Retrieval will be limited.")
        # Create a dummy agent or handle in endpoint
    
    generator = Generator() # Default role, will update per request
    memory = MemoryManager()

@app.get("/graph")
async def get_graph():
    """Returns the Knowledge Graph for visualization."""
    if not os.path.exists(GRAPH_PATH):
        return {"nodes": [], "links": []}
    
    try:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global agent, generator, memory
    
    query = request.message
    role = request.role
    
    # 1. Check Memory (Callback)
    history_context = memory.check_callback(query)
    
    # 2. Retrieval
    retrieved_context = ""
    if agent:
        try:
            retrieved_context = agent.retrieve(query)
        except Exception as e:
            print(f"Retrieval failed: {e}")
            retrieved_context = "Error retrieving info."
    
    # 3. Assembly
    full_context = f"{history_context}\n---\n{retrieved_context}"
    
    # 4. Generation
    generator.role = role
    try:
        response_text = generator.generate_response(query, full_context)
    except Exception as e:
        response_text = "I cannot answer right now."
        print(f"Generation failed: {e}")

    # 5. Update Memory
    memory.add_turn(query, response_text)
    
    return ChatResponse(response=response_text, context=full_context)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
