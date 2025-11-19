import os
import glob
import json
import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Any
from utils import get_gemini_model, parse_json_response
import time

# Configuration
DATASET_DIR = "/root/RoleRAG/datasets/harry-potter"
OUTPUT_DIR = "/root/RoleRAG/output"
CHUNK_SIZE = 1000  # Characters
OVERLAP = 100

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_documents(directory: str) -> List[str]:
    """Loads all .txt files from the directory."""
    files = glob.glob(os.path.join(directory, "*.txt"))
    files.sort() # Ensure order
    documents = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            documents.append(f.read())
    return documents

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

EXTRACTION_PROMPT = """
You are an expert Knowledge Graph builder for a Role-Playing System.
Your task is to extract entities and relations from the provided text chunk of the Harry Potter series.

Extract the following:
1. **Entities**:
    - If the entity is a **Character**:
        - `name`: Name of the character.
        - `type`: "character"
        - `persona`: Brief description of who they are.
        - `style_description`: How they speak (tone, vocabulary, quirks).
        - `style_exemplars`: List of 1-2 short quotes or dialogue examples from the text if available.
    - If the entity is a **Non-Character** (Location, Item, Spell, Event, Organization):
        - `name`: Name of the entity.
        - `type`: "non-character"
        - `description`: Brief description.

2. **Relations**:
    - Identify relationships between entities.
    - `source`: Name of the source entity.
    - `target`: Name of the target entity.
    - `description`: Description of the relationship (objective fact).
    - `attitude`: (Optional, if applicable) The source's attitude towards the target (subjective).
    - `strength`: A score from 1-10 indicating the strength/importance of this relation.

**Output Format**:
Return a JSON object with two keys: "entities" (list) and "relations" (list).

Text Chunk:
{text}
"""

def extract_entities_relations(model, text_chunk: str) -> Dict[str, Any]:
    """Extracts entities and relations using Gemini."""
    prompt = EXTRACTION_PROMPT.format(text=text_chunk)
    try:
        response = model.generate_content(prompt)
        return parse_json_response(response.text)
    except Exception as e:
        print(f"Error extracting from chunk: {e}")
        return {"entities": [], "relations": []}

def build_graph_from_documents(documents: List[str], limit_chunks: int = None):
    """Main function to process docs and build the graph."""
    model = get_gemini_model()
    G = nx.MultiDiGraph() # Use MultiDiGraph to allow multiple relations between nodes
    
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))
    
    if limit_chunks:
        all_chunks = all_chunks[:limit_chunks]
        print(f"Processing limited chunks: {limit_chunks}")
    else:
        print(f"Processing all {len(all_chunks)} chunks")

    for i, chunk in tqdm(enumerate(all_chunks), total=len(all_chunks)):
        data = extract_entities_relations(model, chunk)
        
        # Add Entities
        for entity in data.get("entities", []):
            name = entity.get("name")
            if not name: continue
            
            # Simple merging: if node exists, update attributes (naive approach)
            # Ideally we would do sophisticated disambiguation here
            if not G.has_node(name):
                G.add_node(name, **entity)
            else:
                # Merge list fields like exemplars
                if "style_exemplars" in entity and "style_exemplars" in G.nodes[name]:
                     # Avoid duplicates
                     current_exemplars = set(G.nodes[name]["style_exemplars"])
                     new_exemplars = set(entity["style_exemplars"])
                     G.nodes[name]["style_exemplars"] = list(current_exemplars.union(new_exemplars))
        
        # Add Relations
        for relation in data.get("relations", []):
            src = relation.get("source")
            tgt = relation.get("target")
            if src and tgt:
                # Ensure nodes exist (sometimes relation extracted but entity missed in same chunk)
                if not G.has_node(src): G.add_node(src, type="unknown")
                if not G.has_node(tgt): G.add_node(tgt, type="unknown")
                
                G.add_edge(src, tgt, **relation)
        
        # Rate limit handling (simple sleep)
        time.sleep(1) 

    return G

def save_graph(G, filename="role_rag_graph.json"):
    """Saves the graph to a JSON file (node-link data)."""
    data = nx.node_link_data(G)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Graph saved to {path}")

if __name__ == "__main__":
    # Test run with a small subset
    docs = load_documents(DATASET_DIR)
    # Take only the first book for testing, or just a few chunks
    print("Loaded documents:", len(docs))
    
    # For the initial run, let's just process a small part of the first book to verify
    first_book = docs[0]
    # Process first 5 chunks for verification
    test_chunks = chunk_text(first_book)[:5] 
    
    # We can't pass chunks directly to build_graph_from_documents as it expects docs
    # So we'll just pass a dummy list containing the text of these chunks
    # Or better, modify build_graph to accept chunks? No, let's just pass the text.
    
    print("Starting KG Construction (Test Run)...")
    # Re-chunking inside, so we just pass the first few KB of text
    G = build_graph_from_documents([first_book[:5000]], limit_chunks=5)
    save_graph(G)
