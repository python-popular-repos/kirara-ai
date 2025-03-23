import re
from datetime import datetime, timedelta
from typing import List, Union

from kirara_ai.im.message import IMMessage, MediaMessage
from kirara_ai.im.sender import ChatSender
from kirara_ai.llm.format.message import LLMChatImageContent, LLMChatMessage, LLMChatTextContent
from kirara_ai.llm.format.response import Message
from kirara_ai.memory.entry import MemoryEntry

from .base import ComposableMessageType, MemoryComposer, MemoryDecomposer


class DefaultMemoryComposer(MemoryComposer):
    def compose(
        self, sender: ChatSender, message: List[ComposableMessageType]
    ) -> MemoryEntry:
        composed_message = ""
        for msg in message:
            if isinstance(msg, IMMessage):
                composed_message += f"{sender.display_name} 说: \n"
                for element in msg.message_elements:
                    if isinstance(element, MediaMessage):
                        desc = element.get_description()
                        composed_message += f"<media_msg id={element.media_id} description=\"{desc}\" />\n"
                    else:
                        composed_message += f"{element.to_plain()}"
            elif isinstance(msg, LLMChatMessage) or isinstance(msg, Message):
                composed_message += f"你回答: {msg.content}\n"

        composed_message = composed_message.strip()
        composed_at = datetime.now()
        return MemoryEntry(
            sender=sender,
            content=composed_message,
            timestamp=composed_at,
        )


class DefaultMemoryDecomposer(MemoryDecomposer):
    def decompose(self, entries: List[MemoryEntry]) -> List[str]:
        if len(entries) == 0:
            return [self.empty_message]

        # 7秒前，<记忆内容>
        memory_texts = []
        for entry in entries[-10:]:
            time_diff = datetime.now() - entry.timestamp
            time_str = self.get_time_str(time_diff)
            memory_texts.append(f"{time_str}，{entry.content}")

        return memory_texts

    def get_time_str(self, time_diff: timedelta) -> str:
        if time_diff.days > 0:
            return f"{time_diff.days}天前"
        elif time_diff.seconds > 3600:
            return f"{time_diff.seconds // 3600}小时前"
        elif time_diff.seconds > 60:
            return f"{time_diff.seconds // 60}分钟前"
        else:
            return "刚刚"


class MultiElementDecomposer(MemoryDecomposer):
    def decompose(self, entries: List[MemoryEntry]) -> List[LLMChatMessage]:
        decomposed_messages = []
        for entry in entries:
            # 首先判断MemoryEntry的内容是否包含"你回答:"
            if "你回答:" in entry.content:
                # 如果包含，则分割MemoryEntry的内容为用户消息和AI回答
                user_content = entry.content.split("你回答:")[0].strip()
                assistant_content = entry.content.split("你回答:")[1].strip()

                # 处理用户消息
                user_message = self.create_llm_chat_message(user_content, "user", entry.sender)
                if user_message:
                    decomposed_messages.append(user_message)

                # 处理AI回答
                assistant_message = self.create_llm_chat_message(assistant_content, "assistant", entry.sender)
                if assistant_message:
                    decomposed_messages.append(assistant_message)
            else:
                # 如果不包含"你回答:"，则全部视为用户消息
                user_message = self.create_llm_chat_message(entry.content, "user", entry.sender)
                if user_message:
                    decomposed_messages.append(user_message)

        return decomposed_messages

    def create_llm_chat_message(self, content: str, role: str, sender: ChatSender) -> Union[LLMChatMessage, None]:
        message_content: List[Union[LLMChatTextContent, LLMChatImageContent]] = []

        if isinstance(sender, ChatSender) and role == "user":
            message_content.append(LLMChatTextContent(text=f"{sender.display_name} 说: \n"))

        # 使用正则表达式提取 <media_msg> 标签
        media_msg_pattern = re.compile(r'<media_msg id=(.*?) description="(.*?)" />')
        matches = media_msg_pattern.findall(content)

        last_index = 0
        for media_id, description in matches:
            start_index = content.find(f'<media_msg id={media_id} description="{description}" />', last_index)
            if start_index > last_index:
                text_content = content[last_index:start_index]
                message_content.append(LLMChatTextContent(text=text_content))

            message_content.append(LLMChatImageContent(media_id=media_id))
            last_index = start_index + len(f'<media_msg id={media_id} description="{description}" />')

        # 处理剩余的文本内容
        if last_index < len(content):
            message_content.append(LLMChatTextContent(text=content[last_index:]))

        if message_content:
            return LLMChatMessage(role=role, content=message_content)
        else:
            return None

