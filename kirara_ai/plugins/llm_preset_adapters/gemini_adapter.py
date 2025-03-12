import aiohttp
import requests
import base64
import imghdr
import io
from pydantic import BaseModel, ConfigDict

from kirara_ai.llm.adapter import AutoDetectModelsProtocol, LLMBackendAdapter
from kirara_ai.llm.format.message import LLMChatMessage
from kirara_ai.llm.format.request import LLMChatRequest
from kirara_ai.llm.format.response import LLMChatResponse
from kirara_ai.logger import get_logger


class GeminiConfig(BaseModel):
    api_key: str
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    model_config = ConfigDict(frozen=True)


def convert_llm_chat_message_to_gemini_message(msg: LLMChatMessage) -> dict:
    parts = []

    # 处理内容是列表的情况
    if isinstance(msg.content, list):

        for item in msg.content:

            if item["type"] == "text":
                parts.append({"text": item["text"]})
            elif item["type"] == "image_url":
                # 获取图片的base64编码
                image_url = item["image_url"]["url"]
                try:
                    if image_url.startswith("http"):
                        from curl_cffi import requests as requests1
                        response = requests1.get(image_url)
                        response.raise_for_status()
                        image_data = response.content
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                    else:
                        # 假设输入的是base64字符串
                        base64_data = image_url
                        # 解码base64以检测图片格式
                        image_data = base64.b64decode(base64_data)
                    # 检测图片格式
                    image_format = imghdr.what(None, image_data)
                    if image_format is None:
                        continue

                    parts.append({
                        "inline_data": {
                            "mime_type": f"image/{image_format}",
                            "data": base64_data
                        }
                    })
                except Exception as e:
                    print(e)
                    continue
    else:
        # 处理内容是字符串的情况
        parts.append({"text": msg.content})

    return {
        "role": "model" if msg.role == "assistant" else "user",
        "parts": parts
    }


class GeminiAdapter(LLMBackendAdapter, AutoDetectModelsProtocol):
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.logger = get_logger("GeminiAdapter")

    def chat(self, req: LLMChatRequest) -> LLMChatResponse:
        api_url = f"{self.config.api_base}/models/{req.model}:generateContent"
        headers = {
            "x-goog-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }
        safety_settings = [{
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },{
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },{
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },{
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },{
            "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
            "threshold": "BLOCK_NONE"
        }]
        data = {
            "contents": [
                convert_llm_chat_message_to_gemini_message(msg) for msg in req.messages
            ],
            "generationConfig": {
                "temperature": req.temperature,
                "topP": req.top_p,
                "topK": 40,
                "maxOutputTokens": req.max_tokens,
                "stopSequences": req.stop,
            },
            "safetySettings": safety_settings,
        }

        self.logger.debug(f"Contents: {data['contents']}")
        # Remove None fields
        data = {k: v for k, v in data.items() if v is not None}
        response = requests.post(api_url, json=data, headers=headers)
        try:
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            print(f"API Response: {response.text}")
            raise e
        print(response_data)

        # Transform Gemini response format to match expected LLMChatResponse format
        transformed_response = {
            "id": response_data.get("promptFeedback", {}).get("blockReason", ""),
            "object": "chat.completion",
            "created": 0,  # Gemini doesn't provide creation timestamp
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_data["candidates"][0]["content"]["parts"][
                            0
                        ]["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Gemini doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        return LLMChatResponse(**transformed_response)

    async def auto_detect_models(self) -> list[str]:
        api_url = f"{self.config.api_base}/models"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(
                api_url, headers={"x-goog-api-key": self.config.api_key}
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return [
                    model["name"].removeprefix("models/")
                    for model in response_data["models"]
                    if "generateContent" in model["supportedGenerationMethods"]
                ]
