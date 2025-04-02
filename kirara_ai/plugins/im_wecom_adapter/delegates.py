from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any

from wechatpy.messages import BaseMessage

from kirara_ai.logger import get_logger

if TYPE_CHECKING:
    from .adapter import WecomConfig

class WechatApiDelegate(ABC):
    """微信API代理接口，用于处理不同类型的微信API调用"""

    @abstractmethod
    def setup_api(self, config: "WecomConfig"):
        """设置API相关组件"""

    @abstractmethod
    def check_signature(self, signature: str, timestamp: str, nonce: str, echo_str: str) -> str:
        """验证签名"""

    @abstractmethod
    def decrypt_message(self, message: bytes, signature: str, timestamp: str, nonce: str) -> str:
        """解密消息"""

    @abstractmethod
    def parse_message(self, message: str) -> BaseMessage:
        """解析消息"""

    @abstractmethod
    async def send_text(self, app_id: str, user_id: str, text: str) -> Any:
        """发送文本消息"""

    @abstractmethod
    async def send_media(self, app_id: str, user_id: str, media_type: str, media_bytes: BytesIO) -> Any:
        """发送媒体消息"""


class CorpWechatApiDelegate(WechatApiDelegate):
    """企业微信API代理实现"""

    def setup_api(self, config: "WecomConfig"):
        """设置企业微信API相关组件"""
        from wechatpy.enterprise import parse_message
        from wechatpy.enterprise.client import WeChatClient
        from wechatpy.enterprise.crypto import WeChatCrypto

        self.crypto = WeChatCrypto(
            config.token, config.encoding_aes_key, config.corp_id
        )
        self.client = WeChatClient(config.corp_id, config.secret)
        self.parse_message_func = parse_message
        self.logger = get_logger("CorpWechatApiDelegate")

    def check_signature(self, signature: str, timestamp: str, nonce: str, echo_str: str) -> str:
        """验证企业微信签名"""
        return self.crypto.check_signature(signature, timestamp, nonce, echo_str)

    def decrypt_message(self, message: bytes, signature: str, timestamp: str, nonce: str) -> str:
        """解密企业微信消息"""
        return self.crypto.decrypt_message(message, signature, timestamp, nonce)

    def parse_message(self, message: str) -> BaseMessage:
        """解析企业微信消息"""
        return self.parse_message_func(message) # type: ignore

    async def send_text(self, app_id: str, user_id: str, text: str) -> Any:
        """发送企业微信文本消息"""
        return self.client.message.send_text(app_id, user_id, text)

    async def send_media(self, app_id: str, user_id: str, media_type: str, media_bytes: BytesIO) -> Any:
        """发送企业微信媒体消息"""
        media_id = self.client.media.upload(media_type, media_bytes)["media_id"]
        send_method = getattr(self.client.message, f"send_{media_type}")
        return send_method(app_id, user_id, media_id)


class PublicWechatApiDelegate(WechatApiDelegate):
    """公众号微信API代理实现"""

    def setup_api(self, config: "WecomConfig"):
        """设置公众号API相关组件"""
        from wechatpy import WeChatClient
        from wechatpy.crypto import WeChatCrypto
        from wechatpy.parser import parse_message

        self.crypto = WeChatCrypto(
            config.token, config.encoding_aes_key, config.app_id
        )
        self.client = WeChatClient(config.app_id, config.secret)
        self.parse_message_func = parse_message
        self.logger = get_logger("PublicWechatApiDelegate")

    def check_signature(self, signature: str, timestamp: str, nonce: str, echo_str: str) -> str:
        """验证公众号签名"""
        from wechatpy.utils import check_signature as wechat_check_signature
        wechat_check_signature(self.crypto.token, signature, timestamp, nonce)
        return echo_str

    def decrypt_message(self, message: bytes, signature: str, timestamp: str, nonce: str) -> str:
        """解密公众号消息"""
        return self.crypto.decrypt_message(message, signature, timestamp, nonce)

    def parse_message(self, message: str) -> BaseMessage:
        """解析公众号消息"""
        return self.parse_message_func(message) # type: ignore

    async def send_text(self, app_id: str, user_id: str, text: str) -> Any:
        """发送公众号文本消息"""
        # 公众号API不需要app_id参数
        return self.client.message.send_text(user_id, text)

    async def send_media(self, app_id: str, user_id: str, media_type: str, media_bytes: BytesIO) -> Any:
        """发送公众号媒体消息"""
        media_id = self.client.media.upload(media_type, media_bytes)["media_id"]
        send_method = getattr(self.client.message, f"send_{media_type}")
        # 公众号API不需要app_id参数
        return send_method(user_id, media_id)
