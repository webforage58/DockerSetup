from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import json
from typing import Optional, List, Dict, Any

app = FastAPI(title="Ollama API Server")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

class Query(BaseModel):
    prompt: str
    model: str = "llama2"  # default model
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str

async def get_ollama_client():
    return httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=30.0)

@app.get("/models")
async def list_models() -> List[ModelInfo]:
    """List all available models"""
    async with await get_ollama_client() as client:
        response = await client.get("/api/tags")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch models")
        return response.json()["models"]

@app.post("/generate")
async def generate_text(query: Query):
    """Generate text using the specified model"""
    payload = {
        "model": query.model,
        "prompt": query.prompt,
        "stream": False
    }
    
    if query.system:
        payload["system"] = query.system
    if query.template:
        payload["template"] = query.template
    if query.context:
        payload["context"] = query.context
    if query.options:
        payload["options"] = query.options

    async with await get_ollama_client() as client:
        try:
            response = await client.post("/api/generate", json=payload)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to generate response"
                )
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is healthy"""
    try:
        async with await get_ollama_client() as client:
            response = await client.get("/api/tags")
            if response.status_code == 200:
                return {"status": "healthy", "ollama_connected": True}
    except Exception as e:
        return {"status": "unhealthy", "ollama_connected": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)