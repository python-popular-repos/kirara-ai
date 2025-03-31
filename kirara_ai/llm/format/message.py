from typing import Literal, List, Optional, Union

from pydantic import BaseModel


class LLMChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class LLMChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    media_id: str

class LLMToolCallContent(BaseModel):
    """
    这是模型请求工具的消息内容,
    模型强相关内容，如果你 message 或者 memory 内包含了这个内容，请保证调用同一个 model
    此部分 role 应该归属于"assistant"
    """
    type: Literal["tool_call"] = "tool_call"
    # 有些model不会返回 call_id ，点名 gemini
    id: Optional[str] = None
    name: str
    # tool可能没有参数。
    parameters: Optional[str] = None

class LLMToolResultContent(BaseModel):
    """
    这是工具回应的消息内容,
    模型强相关内容，如果你 message 或者 memory 内包含了这个内容，请保证调用同一个 model
    此部分 role 应该对应 "tool"
    """
    type: Literal["tool_result"] = "tool_result"
    # 为与 gemini 兼容，此处 id 改为 Optional. 因为 gemini 回应中没有 call_id.
    id: Optional[str] = None
    name: str
    content: str

LLMChatContentPartType = Union[LLMChatTextContent, LLMChatImageContent, LLMToolCallContent]
RoleTypes = Literal["user", "assistant", "system", "tool"]
NormalTypes = Literal["user", "assistant", "system"]

class LLMChatMessage(BaseModel):
    """
    当 role 为 "tool" 时，content 内部只能为 list[LLMToolResultContent]
    """
    content: Union[List[LLMChatContentPartType], List[LLMToolResultContent]]
    role: RoleTypes

