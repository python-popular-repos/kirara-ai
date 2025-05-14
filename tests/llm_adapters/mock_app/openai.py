from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from typing import Literal

from . import REFERENCE_VECTOR

class RequestBase(BaseModel):
    model: Literal["mock_completion"]
    messages: str

class EmbeddingRequest(BaseModel):
    text: list[str]
    model: Literal["mock_embedding"] = Field(...)

router = APIRouter(tags=["openai"])


@router.post("/v1/chat/completions")
async def completions(request: RequestBase = Body(...)):
    pass

@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest = Body(...)):
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": REFERENCE_VECTOR,
                "index": 0
            }
        ],
        "model": "mock_embedding",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }