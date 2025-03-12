from typing import Any, List, Literal, Optional, Union, Dict
from pydantic import BaseModel
import base64
from curl_cffi import requests
import magic  # Ensure you have python-magic installed


class LLMChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]  # 支持字符串或多模态内容列表

    def download_and_encode_base64(self,url: str) -> str:
        """
        Downloads a resource from a URL and returns it as a base64 encoded string.
        """
        response = requests.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')

    def get_format_from_base64(self,base64_data: str) -> str:
        """
        Decodes a base64 string and returns the format of the data using the magic library.
        """
        binary_data = base64.b64decode(base64_data)
        mime = magic.Magic(mime=True)
        return mime.from_buffer(binary_data)
