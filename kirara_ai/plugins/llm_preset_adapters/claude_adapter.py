import asyncio
import base64

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import LLMChatImageContent, LLMChatMessage, LLMChatTextContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.logger import get_logger
from kirara_ai.media.manager import MediaManager


class ClaudeConfig(BaseModel):
    api_key: str
    api_base: str = "https://api.anthropic.com/v1"
    model_config = ConfigDict(frozen=True)

async def convert_llm_chat_message_to_claude_message(messages: list[LLMChatMessage], media_manager: MediaManager) -> list[dict]:
    content = []
    for msg in [msg for msg in messages if msg.role in ["user", "assistant"]]:
        parts = []
        for part in msg.content:
            if isinstance(part, LLMChatTextContent):
                parts.append({"text": part.text, "type": "text"})
            elif isinstance(part, LLMChatImageContent):
                media = media_manager.get_media(part.media_id)
                parts.append({"source": {"media_type": media.mime_type, "data": await media.get_base64()}, "type": "image"})
        content.append({
            "role": msg.role,
            "content": parts
        })
    return content
class ClaudeAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):
    
    media_manager: MediaManager
    
    def __init__(self, config: ClaudeConfig):
        self.config = config
        self.logger = get_logger("ClaudeAdapter")

    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        # Claude 的系统消息比较特殊
        system_messages = [msg for msg in req.messages if msg.role == "system"]
        if system_messages:
            system_message = system_messages[0].content
        else:
            system_message = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 构建请求数据
        data = {
            "model": req.model,
            "messages": [
                loop.run_until_complete(convert_llm_chat_message_to_claude_message(req.messages, self.media_manager))
            ],
            "max_tokens": req.max_tokens,
            "system": system_message,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": req.stream,
        }

        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            self.logger.error(f"API Response: {response.text}")
            raise e
        
        content = []
        
        for res in response_data["content"]:
            if res["type"] == "text":
                content.append(LLMChatTextContent(text=res["text"]))
            elif res["type"] == "image":
                data = base64.b64decode(res["source"]["data"])
                media = loop.run_until_complete(self.media_manager.register_from_data(data, res["source"]["mime_type"], source="claude response"))
                content.append(LLMChatImageContent(media_id=media))

        return LLMChatResponse(
            model=req.model,
            usage=Usage(
                prompt_tokens=response_data["usage"]["input_tokens"],
                completion_tokens=response_data["usage"]["output_tokens"],
                total_tokens=response_data["usage"]["input_tokens"] + response_data["usage"]["output_tokens"],
            ),
            message=Message(
                content=content,
                role=response_data.get("role"),
                finish_reason=response_data.get("stop_reason", "stop"),
            ),
        )

    async def auto_detect_models(self) -> list[str]:
        # {
        #   "data": [
        #     {
        #       "type": "model",
        #       "id": "claude-3-5-sonnet-20241022",
        #       "display_name": "Claude 3.5 Sonnet (New)",
        #       "created_at": "2024-10-22T00:00:00Z"
        #     }
        #   ],
        #   "has_more": true,
        #   "first_id": "<string>",
        #   "last_id": "<string>"
        # }
        api_url = f"{self.config.api_base}/models"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                api_url, headers={"x-api-key": self.config.api_key}
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [model["id"] for model in response_data["data"]]
