import asyncio
from typing import List

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import LLMChatContentPartType, LLMChatImageContent, LLMChatTextContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse, Message, Usage
from kirara_ai.logger import get_logger
from kirara_ai.media.manager import MediaManager
from kirara_ai.tracing import trace_llm_chat


class OllamaConfig(BaseModel):
    api_base: str = "http://localhost:11434"
    model_config = ConfigDict(frozen=True)

async def resolv_media_ids(media_ids: list[str], media_manager: MediaManager) -> List[str]:
    result = []
    for media_id in media_ids:
        media = media_manager.get_media(media_id)
        if media is not None:
            base64_data = await media.get_base64()
            result.append(base64_data)
    return result

class OllamaAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.logger = get_logger("OllamaAdapter")
        
    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/api/chat"
        headers = {"Content-Type": "application/json"}

        # 将消息转换为 Ollama 格式
        messages = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for msg in req.messages:
            # 收集每条消息中的文本内容和图像
            text_content = ""
            images = []
            
            for part in msg.content:
                if isinstance(part, LLMChatTextContent):
                    text_content += part.text
                elif isinstance(part, LLMChatImageContent):
                    images.append(part.media_id)
            
            # 创建 Ollama 格式的消息
            message = {"role": msg.role, "content": text_content}
            if images:
                message["images"] = loop.run_until_complete(resolv_media_ids(images, self.media_manager)) # type: ignore
            
            messages.append(message)

        data = {
            "model": req.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "num_predict": req.max_tokens,
                "stop": req.stop,
            },
        }

        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}
        if "options" in data:
            data["options"] = {
                k: v for k, v in data["options"].items() if v is not None
            }

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            print(f"API Response: {response.text}")
            raise e
        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
        content: List[LLMChatContentPartType] = [LLMChatTextContent(text=response_data["message"]["content"])]
        return LLMChatResponse(
            model=req.model,
            message=Message(
                content=content,
                role="assistant",
                finish_reason="stop",
            ),
            usage=Usage(
                prompt_tokens=response_data['prompt_eval_count'],
                completion_tokens=response_data['eval_count'],
                total_tokens=response_data['prompt_eval_count'] + response_data['eval_count'],
            )
        )

    async def auto_detect_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/api/tags"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(api_url) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [tag["name"] for tag in response_data["models"]]
