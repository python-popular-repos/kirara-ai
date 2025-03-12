import asyncio
import base64
import os
import uuid
from io import BytesIO
from typing import Any, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from wechatpy.client import BaseWeChatClient
from wechatpy.exceptions import InvalidSignatureException
from wechatpy.replies import create_reply

from kirara_ai.im.adapter import IMAdapter
from kirara_ai.im.message import FileElement, ImageMessage, IMMessage, TextMessage, VideoElement, VoiceMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.logger import HypercornLoggerWrapper, get_logger
from kirara_ai.web.app import WebServer
from kirara_ai.workflow.core.dispatch.dispatcher import WorkflowDispatcher

from .delegates import CorpWechatApiDelegate, PublicWechatApiDelegate

WECOM_TEMP_DIR = os.path.join(os.getcwd(), 'data', 'temp', 'wecom')

WEBHOOK_URL_PREFIX = "/im/webhook/wechat"

def make_webhook_url():
    return f"{WEBHOOK_URL_PREFIX}/{str(uuid.uuid4())[:8]}"


def auto_generate_webhook_url(s: dict):
    s["readOnly"] = True
    s["default"] = make_webhook_url()
    s["textType"] = True


class WecomConfig(BaseModel):
    """企业微信配置
    文档： https://work.weixin.qq.com/api/doc/90000/90136/91770
    """

    app_id: str = Field(title="应用ID", description="见微信侧显示")
    secret: str = Field(title="应用Secret", description="见微信侧显示")
    token: str = Field(title="Token", description="与微信侧填写保持一致")
    encoding_aes_key: str = Field(
        title="EncodingAESKey", description="请通过微信侧随机生成")
    corp_id: Optional[str] = Field(
        title="企业ID", description="企业微信后台显示的企业ID，微信公众号等场景无需填写。", default=None)
    webhook_url: str = Field(
        title="微信端回调地址",
        description="供微信端请求的 Webhook URL，填写在微信端，由系统自动生成，无法修改。",
        default_factory=make_webhook_url,
        json_schema_extra=auto_generate_webhook_url
    )

    host: Optional[str] = Field(title="HTTP 服务地址", description="已过时，请删除并使用 webhook_url 代替。",
                                default=None, json_schema_extra={"hidden_unset": True})
    port: Optional[int] = Field(title="HTTP 服务端口", description="已过时，请删除并使用 webhook_url 代替。",
                                default=None, json_schema_extra={"hidden_unset": True})

    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs: Any):
        # 如果 agent_id 存在，则自动使用 agent_id 作为 app_id
        if "agent_id" in kwargs:
            kwargs["app_id"] = str(kwargs["agent_id"])
        super().__init__(**kwargs)


class WeComUtils:
    """企业微信相关的工具类"""

    def __init__(self, client: BaseWeChatClient):
        self.client = client
        self.logger = get_logger("WeComUtils")
        
    @property
    def access_token(self) -> str:
        return self.client.access_token

    async def download_and_save_media(self, media_id: str, file_name: str) -> Optional[str]:
        """下载并保存媒体文件到本地"""
        file_path = os.path.join(WECOM_TEMP_DIR, file_name)
        try:
            media_data = await self.download_media(media_id)
            if media_data:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(media_data)
                return file_path
        except Exception as e:
            self.logger.error(f"Failed to save media: {str(e)}")
        return None

    async def download_media(self, media_id: str) -> Optional[bytes]:
        """下载企业微信的媒体文件"""
        url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={self.access_token}&media_id={media_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    self.logger.error(
                        f"Failed to download media: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to download media: {str(e)}")
        return None


class WecomAdapter(IMAdapter):
    """企业微信适配器"""

    dispatcher: WorkflowDispatcher
    web_server: WebServer
    
    def __init__(self, config: WecomConfig):
        self.config = config
        if self.config.host:
            self.app = FastAPI()
        else:
            self.app = self.web_server.app
            
        self.logger = get_logger("Wecom-Adapter")
        self.is_running = False
        if not self.config.host:
            self.config.host = None
            self.config.port = None
        elif not self.config.port:
            self.config.port = 15650
        if not self.config.webhook_url:
            self.config.webhook_url = make_webhook_url()
        
        self.reply_tasks = {}
        
        # 根据配置选择合适的API代理
        self.setup_wechat_api()
        
    def setup_wechat_api(self):
        """根据配置设置微信API代理"""
        if self.config.corp_id:
            self.api_delegate = CorpWechatApiDelegate()
        else:
            self.api_delegate = PublicWechatApiDelegate()
        
        self.api_delegate.setup_api(self.config)
        
        # 设置工具类
        self.wecom_utils = WeComUtils(self.api_delegate.client)
            
    def setup_routes(self):
        if self.config.host:
            webhook_url = '/wechat'
        else:
            webhook_url = self.config.webhook_url
        # unregister old route if exists
        for route in self.app.routes:
            if route.path == webhook_url:
                self.app.routes.remove(route)
        
        @self.app.get(webhook_url)
        async def handle_check_request(request: Request):
            """处理 GET 请求"""
            if not self.is_running:
                self.logger.warning("Wecom-Adapter is not running, skipping check request.")
                raise HTTPException(status_code=404)
            
            signature = request.query_params.get("msg_signature", "")
            if not signature:
                signature = request.query_params.get("signature", "")
            timestamp = request.query_params.get("timestamp", "")
            nonce = request.query_params.get("nonce", "")
            echo_str = request.query_params.get("echostr", "")

            try:            
                echo_str = self.api_delegate.check_signature(
                    signature, timestamp, nonce, echo_str
                )
                return Response(content=echo_str, media_type="text/plain")
            except InvalidSignatureException:
                self.logger.error("failed to check signature, please check your settings.")
                raise HTTPException(status_code=403)

        @self.app.post(webhook_url)
        async def handle_message(request: Request):
            """处理 POST 请求"""
            if not self.is_running:
                self.logger.warning("Wecom-Adapter is not running, skipping message request.")
                raise HTTPException(status_code=404)
            signature = request.query_params.get("msg_signature", "")
            if not signature:
                signature = request.query_params.get("signature", "")
            timestamp = request.query_params.get("timestamp", "")
            nonce = request.query_params.get("nonce", "")
            try:
                msg = self.api_delegate.decrypt_message(
                    await request.body(), signature, timestamp, nonce
                )
            except InvalidSignatureException:
                self.logger.error("failed to check signature, please check your settings.")
                raise HTTPException(status_code=403)
            msg = self.api_delegate.parse_message(msg)
            
            if msg.id in self.reply_tasks:
                self.logger.debug(f"skip processing due to duplicate msgid: {msg.id}")
                reply = await self.reply_tasks[msg.id]
                del self.reply_tasks[msg.id]
                return Response(content=create_reply(reply, msg).render(), media_type="text/xml")
            
            # 预处理媒体消息
            media_path = None
            if msg.type in ["voice", "video", "file"]:
                media_id = msg.media_id
                file_name = f"temp_{msg.type}_{media_id}.{msg.type}"
                media_path = await self.wecom_utils.download_and_save_media(media_id, file_name)

            # 转换消息
            message = self.convert_to_message(msg, media_path)
            self.reply_tasks[msg.id] = asyncio.Future()
            message.sender.raw_metadata["reply"] = self.reply_tasks[msg.id]
            # 分发消息
            asyncio.create_task(self.dispatcher.dispatch(self, message))
            reply = await message.sender.raw_metadata["reply"]
            del message.sender.raw_metadata["reply"]
            return Response(content=create_reply(reply, msg).render(), media_type="text/xml")

    def convert_to_message(self, raw_message: Any, media_path: Optional[str] = None) -> IMMessage:
        """将企业微信消息转换为统一消息格式"""
        # 企业微信应用似乎没有群聊的概念，所以这里只能用单聊
        sender = ChatSender.from_c2c_chat(
            raw_message.source, raw_message.source)

        message_elements = []
        raw_message_dict = raw_message.__dict__

        if raw_message.type == "text":
            message_elements.append(TextMessage(text=raw_message.content))
        elif raw_message.type == "image":
            message_elements.append(ImageMessage(url=raw_message.image))
        elif raw_message.type == "voice" and media_path:
            message_elements.append(VoiceMessage(url=media_path))
        elif raw_message.type == "video" and media_path:
            message_elements.append(VideoElement(file=media_path))
        elif raw_message.type == "file" and media_path:
            message_elements.append(FileElement(path=media_path))
        elif raw_message.type == "location":
            location_text = f"[Location] {raw_message.label} (X: {raw_message.location_x}, Y: {raw_message.location_y})"
            message_elements.append(TextMessage(text=location_text))
        elif raw_message.type == "link":
            link_text = f"[Link] {raw_message.title}: {raw_message.description} ({raw_message.url})"
            message_elements.append(TextMessage(text=link_text))
        else:
            message_elements.append(TextMessage(
                text=f"Unsupported message type: {raw_message.type}"))

        return IMMessage(
            sender=sender,
            message_elements=message_elements,
            raw_message=raw_message_dict,
        )

    async def _send_text(self, user_id: str, text: str):
        """发送文本消息"""
        try:
            return await self.api_delegate.send_text(self.config.app_id, user_id, text)
        except Exception as e:
            self.logger.error(f"Failed to send text message: {e}")
            raise e

    async def _send_media(self, user_id: str, media_data: str, media_type: str):
        """发送媒体消息的通用方法"""
        try:
            media_bytes = BytesIO(base64.b64decode(media_data))
            return await self.api_delegate.send_media(self.config.app_id, user_id, media_type, media_bytes)
        except Exception as e:
            self.logger.error(f"Failed to send {media_type} message: {e}")
            raise e

    async def send_message(self, message: IMMessage, recipient: ChatSender):
        """发送消息到企业微信"""
        user_id = recipient.user_id
        res = None
        
        try:
            for element in message.message_elements:
                if isinstance(element, TextMessage) and element.text:
                    res = await self._send_text(user_id, element.text)
                elif isinstance(element, ImageMessage) and element.url:
                    res = await self._send_media(user_id, element.url, "image")
                elif isinstance(element, VoiceMessage) and element.url:
                    res = await self._send_media(user_id, element.url, "voice")
                elif isinstance(element, VideoElement) and element.file:
                    res = await self._send_media(user_id, element.file, "video")
                elif isinstance(element, FileElement) and element.path:
                    res = await self._send_media(user_id, element.path, "file")
            if res:
                print(res)
            recipient.raw_metadata["reply"].set_result(None)
        except Exception as e:
            if 'Error code: 48001' in str(e):
                # 未开通主动回复能力
                self.logger.warning("未开通主动回复能力，将采用被动回复消息 API，此模式下只能回复一条消息。")
                recipient.raw_metadata["reply"].set_result(message.content)

    async def _start_standalone_server(self):
        """启动服务"""
        from hypercorn.asyncio import serve
        from hypercorn.config import Config
        from hypercorn.logging import Logger

        config = Config()
        config.bind = [f"{self.config.host}:{self.config.port}"]
        # config._log = get_logger("Wecom-API")
        # hypercorn 的 logger 需要做转换
        config._log = Logger(config)
        config._log.access_logger = HypercornLoggerWrapper(self.logger)
        config._log.error_logger = HypercornLoggerWrapper(self.logger)

        self.server_task = asyncio.create_task(serve(self.app, config))

    async def _stop_standalone_server(self):
        """停止服务"""
        if hasattr(self, "server_task"):
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error(f"Error during server shutdown: {e}")

    async def start(self):
        self.setup_wechat_api()
        if self.config.host:
            self.logger.warning("正在使用过时的启动模式，请尽快更新为 Webhook 模式。")
            await self._start_standalone_server()
        self.setup_routes()
        self.is_running = True
        self.logger.info("Wecom-Adapter 启动成功")

    async def stop(self):
        if self.config.host:
            await self._stop_standalone_server()
        self.is_running = False
        self.logger.info("Wecom-Adapter 停止成功")
