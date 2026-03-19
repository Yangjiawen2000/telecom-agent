import json
import asyncio
import os
import logging
from app.memory.ltm import LongTermMemory

logger = logging.getLogger(__name__)

async def main():
    ltm = LongTermMemory()
    await ltm.init_collections()
    
    kb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../knowledge"))
    graph_path = os.path.join(kb_path, "graph.json")
    
    if not os.path.exists(graph_path):
        logger.error(f"Graph file {graph_path} not found. Run graph_extractor.py first.")
        return
        
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
        
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])
    
    logger.info(f"Indexing {len(entities)} entities and {len(relationships)} relationships...")
    
    await ltm.upsert_graph(entities, relationships)
    
    logger.info("Indexing complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
