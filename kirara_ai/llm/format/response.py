from typing import List, Optional

from pydantic import BaseModel

from kirara_ai.llm.format.message import LLMChatMessage


class Function(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[Function] = None

class Message(LLMChatMessage):
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

class LLMChatResponse(BaseModel):
    model: Optional[str] = None
    usage: Optional[Usage] = None
    message: Message
