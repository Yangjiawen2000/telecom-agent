import os
import asyncio
import glob
from app.memory.ltm import LongTermMemory
from app.llm import embed

def split_text(text: str, chunk_size=500, overlap=50):
    """简单的文本切片"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

async def ingest():
    ltm = LongTermMemory()
    await ltm.init_collections()
    
    files = glob.glob("knowledge/*.md") + glob.glob("knowledge/*.txt")
    
    for file_path in files:
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = split_text(content)
        docs_to_insert = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            print(f"  Vectorizing chunk (len={len(chunk)})...")
            vector = await embed(chunk)
            
            docs_to_insert.append({
                "content": chunk,
                "embedding": vector,
                "source": file_path,
                "doc_type": "knowledge"
            })
            
        if docs_to_insert:
            print(f"  Inserting {len(docs_to_insert)} chunks into knowledge_base...")
            await ltm.upsert_knowledge(docs_to_insert)
            
    print("Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(ingest())
