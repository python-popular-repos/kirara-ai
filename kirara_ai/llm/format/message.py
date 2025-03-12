from typing import Any, List, Literal, Optional, Union,Dict
from pydantic import BaseModel


class LLMChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]  # 支持字符串或多模态内容列表
