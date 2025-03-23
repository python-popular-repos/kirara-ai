from typing import List, Literal, Union

from pydantic import BaseModel


class LLMChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class LLMChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    media_id: str

LLMChatContentPartType = Union[LLMChatTextContent, LLMChatImageContent]

class LLMChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: List[LLMChatContentPartType]

