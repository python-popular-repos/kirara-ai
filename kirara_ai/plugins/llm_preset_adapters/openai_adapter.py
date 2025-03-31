import asyncio

import aiohttp
import requests
from pydantic import BaseModel, ConfigDict

from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import LLMChatImageContent, LLMChatMessage, LLMChatTextContent, LLMToolResultContent, LLMToolCallContent
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse, Message, ToolCall, Usage, Function
from kirara_ai.logger import get_logger
from kirara_ai.media import MediaManager
from kirara_ai.tracing import trace_llm_chat

logger = get_logger("OpenAIAdapter")

async def convert_parts_factory(messages: LLMChatMessage, media_manager: MediaManager) -> list[dict] | dict:
    if messages.role == "tool":
        # content字段为 list[LLMToolResultContent]
        return [{"role": "tool", "tool_call_id": result.id, "content": result.content} for result in messages.content]
    else:
        parts = []
        for element in messages.content:
            if isinstance(element, LLMChatTextContent):
                parts.append(element.model_dump(mode="json"))
            elif isinstance(element, LLMChatImageContent):
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": await media_manager.get_media(element.media_id).get_base64_url()
                    }
                })
            elif isinstance(element, LLMToolCallContent):
                # 忽略tool_call_content，openai api不需要
                continue
        return {"role": messages.role, "content": parts}
def convert_llm_chat_message_to_openai_message(messages: list[LLMChatMessage], media_manager: MediaManager, loop: asyncio.AbstractEventLoop) -> list[dict]:
    """
    gather 返回一个有序结果, 结果中遇到list类型对象, 应该原地展开list(注意保持结果顺序)。
    """
    results = loop.run_until_complete(asyncio.gather(*[convert_parts_factory(message, media_manager) for message in messages]))
    temp = []
    for result in results:
        if isinstance(result, list):
            temp.extend(result)
        else:
            temp.append(result)
    return temp
        
class OpenAIConfig(BaseModel):
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    model_config = ConfigDict(frozen=True)


class OpenAIAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):
    media_manager: MediaManager
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
    @trace_llm_chat
    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        data = {
            "messages": convert_llm_chat_message_to_openai_message(req.messages, self.media_manager, loop),
            "model": req.model,
            "frequency_penalty": req.frequency_penalty,
            "max_tokens": req.max_tokens,
            "presence_penalty": req.presence_penalty,
            "response_format": req.response_format,
            "stop": req.stop,
            "stream": req.stream,
            "stream_options": req.stream_options,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "tools": [tool.model_dump() for tool in req.tools] if req.tools else None,
            "tool_choice": "auto",
            "logprobs": req.logprobs,
            "top_logprobs": req.top_logprobs,
        }

        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}
        
        logger.debug(f"Request: {data}")

        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data: dict = response.json()
        except Exception as e:
            logger.error(f"Response: {response.text}")
            raise e
        logger.debug(f"Response: {response_data}")

        choices = response_data.get("choices", [{}])
        first_choice = choices[0] if choices else {}
        message: dict = first_choice.get("message", {})
        
        # content字段为空必然llm请求调用tool
        if message:= message.get("content", None):
             content = [LLMChatTextContent(text=message)]
        else:
            content = [
                LLMToolCallContent(
                    id=call["id"], 
                    name=call["function"]["name"], 
                    parameters=call["function"].get("parameters", None)
                ) for call in message.get("content")
            ]
        
        usage_data = response_data.get("usage", {})
        
        return LLMChatResponse(
            model=req.model,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            message=Message(
                content=content,
                role=message.get("role", "assistant"),
                tool_calls=[
                    ToolCall(
                        id=tool_call["id"], 
                        type=tool_call["type"],
                        function=Function(name = tool_call["function"]["name"], arguments=tool_call["function"].get("arguments", None))
                    ) for tool_call in message.get("tool_calls")
                ] if message.get("tool_calls", None) else None,
                finish_reason=first_choice.get("finish_reason", ""),
            ),
        )

    async def auto_detect_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/models"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                api_url, headers={"Authorization": f"Bearer {self.config.api_key}"}
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [model["id"] for model in response_data["data"]]