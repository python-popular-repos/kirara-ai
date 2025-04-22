from pydantic import BaseModel, ConfigDict
from typing import cast, TypedDict, Literal, Optional

import requests
import asyncio

from kirara_ai.llm.adapter import LLMBackendAdapter, LLMEmbeddingProtocol, LLMReRankProtocol
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.llm.format.rerank import LLMReRankRequest, LLMReRankResponse, ReRankerContent
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent
from kirara_ai.llm.format.response import Usage
from kirara_ai.media.manager import MediaManager
from kirara_ai.logger import get_logger

logger = get_logger("VoyageAdapter")

async def resolve_media_base64(inputs: list[LLMChatImageContent|LLMChatTextContent], media_manager: MediaManager) -> list[dict]:
    results = []
    for input in inputs:
        # 因为inputs字段对其中每个元素具有资源显示且较为容易达到
        # 所以将所有输入放到不同的包含键 content 的字典中，而不是放入单个 content所以对应的列表
        if isinstance(input, LLMChatTextContent):
            input = cast(LLMChatTextContent, input) # 类型标注, cast函数不参与运行时，只是方便在复杂情况下进行类型推导
            results.append({
                "content": [{
                    "type": "text",
                    "text": input.text
                }]
            })
        elif isinstance(input, LLMChatImageContent):
            input = cast(LLMChatImageContent, input)
            media = media_manager.get_media(input.media_id)
            if media is None:
                raise ValueError(f"Media {input.media_id} not found")
            results.append({
                "content": [{
                    "type": "image_base64",
                    "image_base64": await media.get_base64()
                }]
            })
    return results

class ReRankData(TypedDict):
    index: int
    relevance_score: float
    document: Optional[str]

class ReRankResponse(TypedDict):
    """给mypy检查用, 顺便给开发者标识返回json的基本结构。"""
    object: Literal["list"]
    data: list[ReRankData]
    model: str
    usage: dict[Literal["total_tokens"], int]

class VoyageConfig(BaseModel):
    api_key: str
    api_base: str = "https://api.voyageai.com"
    model_config = ConfigDict(frozen=True)

class VoyageAdapter(LLMBackendAdapter, LLMEmbeddingProtocol, LLMReRankProtocol):
    media_manager: MediaManager

    def __init__(self, config: VoyageConfig):
        self.config = config
    
    def embed(self, req: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        # voyage 支持多模态嵌入, 但是两个接口参数不同
        if all(isinstance(input, LLMChatTextContent) for input in req.inputs):
            return self._text_embedding(req)
        else:
            return self._multi_modal_embedding(req, self.media_manager)
       
    def _text_embedding(self, req: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        api_url = f"{self.config.api_base}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        inputs = cast(list[LLMChatTextContent], req.inputs)
        data = {
            "model": req.model,
            "input": [input.text for input in inputs],
            "truncation": req.truncate,
            "input_type": req.input_type,
            "output_dimension": req.dimension,
            "output_dtype": req.encoding_format,
            "encoding_format": req.encoding_format,
        }
        data = { k:v for k,v in data.items() if v is not None }

        response = requests.post(api_url, headers=headers, json=data)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        
        return LLMEmbeddingResponse(
            vectors=[data["embedding"] for data in response_data["data"]],
            usage = Usage(
                total_tokens=response_data["usage"].get("total_tokens", 0)
            )
        )
    
    def _multi_modal_embedding(self, req: LLMEmbeddingRequest, media_manager: MediaManager) -> LLMEmbeddingResponse:
        api_url = f"{self.config.api_base}/v1/multimodalembeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": req.model,
            "inputs": asyncio.run(resolve_media_base64(req.inputs, media_manager)),
            "input_type": req.input_type,
            "truncation": req.truncate,
            "output_encoding": req.encoding_format
        }
        data = { k:v for k,v in data.items() if v is not None }

        response = requests.post(api_url, headers=headers, json=data)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        
        return LLMEmbeddingResponse(
            vectors=[data["embedding"] for data in response_data["data"]],
            usage = Usage(
                total_tokens=response_data["usage"].get("total_tokens", 0)
            )
        )

    def rerank(self, req: LLMReRankRequest) -> LLMReRankResponse:
        api_url = f"{self.config.api_base}/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "query": req.query,
            "documents": req.documents,
            "model": req.model,
            "top_k": req.top_k,
            "return_documents": req.return_documents,
            "truncation": req.truncation
        }

        # 去除 None 值
        data = { k:v for k,v in data.items() if v is not None }

        response = requests.post(api_url, headers=headers, json=data)
        try:
            response.raise_for_status()
            response_data: ReRankResponse = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        
        return LLMReRankResponse(
            contents = [ReRankerContent(
                    document = data.get("document", None),
                    relevance_score = data["relevance_score"]
            ) for data in response_data["data"]],
            usage = Usage(
                total_tokens = response_data["usage"].get("total_tokens", 0)
            )
        )
        