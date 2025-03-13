import asyncio
import base64
import json
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import aiofiles
import aiohttp
import magic

if TYPE_CHECKING:
    from kirara_ai.im.message import MediaMessage

from kirara_ai.logger import get_logger


class MediaType(Enum):
    """媒体类型枚举"""
    IMAGE = "image"
    VOICE = "audio"
    VIDEO = "video"
    FILE = "file"
    
    @classmethod
    def from_mime(cls, mime_type: str) -> 'MediaType':
        """从MIME类型获取媒体类型"""
        main_type = mime_type.split('/')[0]
        if main_type == "image":
            return cls.IMAGE
        elif main_type == "audio":
            return cls.VOICE
        elif main_type == "video":
            return cls.VIDEO
        else:
            return cls.FILE


mime_remapping = {
    "audio/mpeg": "audio/mp3",
    "audio/x-wav": "audio/wav",
    "audio/x-m4a": "audio/m4a",
    "audio/x-flac": "audio/flac",
}
def detect_mime_type(data: bytes = None, path: str = None) -> Tuple[str, MediaType, str]:
    """
    检测文件的MIME类型
    
    Args:
        data: 文件数据
        path: 文件路径
        
    Returns:
        Tuple[str, MediaType, str]: (mime_type, media_type, format)
    """
    if data is not None:
        mime_type = magic.from_buffer(data, mime=True)
    elif path is not None:
        mime_type = magic.from_file(path, mime=True)
    else:
        raise ValueError("Must provide either data or path")
    if mime_type in mime_remapping:
        mime_type = mime_remapping[mime_type]
        
    media_type = MediaType.from_mime(mime_type)
    format = mime_type.split('/')[-1]
    
    return mime_type, media_type, format


class MediaMetadata:
    """媒体元数据类"""
    
    def __init__(
        self,
        media_id: str,
        media_type: Optional[MediaType] = None,
        format: Optional[str] = None,
        size: Optional[int] = None,
        created_at: Optional[datetime] = None,
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        references: Optional[Set[str]] = None,
        url: Optional[str] = None,
        path: Optional[str] = None
    ):
        self.media_id = media_id
        self.media_type = media_type
        self.format = format
        self.size = size
        self.created_at = created_at or datetime.now()
        self.source = source
        self.description = description
        self.tags = tags or []
        self.references = references or set()
        self.url = url
        self.path = path
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "media_id": self.media_id,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "description": self.description,
            "tags": self.tags,
            "references": list(self.references),
        }
        
        # 添加可选字段
        if self.media_type:
            result["media_type"] = self.media_type.value
        if self.format:
            result["format"] = self.format
        if self.size is not None:
            result["size"] = self.size
        if self.url:
            result["url"] = self.url
        if self.path:
            result["path"] = self.path
            
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MediaMetadata':
        """从字典创建元数据"""
        return cls(
            media_id=data["media_id"],
            media_type=MediaType(data["media_type"]) if "media_type" in data else None,
            format=data.get("format"),
            size=data.get("size"),
            created_at=datetime.fromisoformat(data["created_at"]),
            source=data.get("source"),
            description=data.get("description"),
            tags=data.get("tags", []),
            references=set(data.get("references", [])),
            url=data.get("url"),
            path=data.get("path")
        )


class MediaManager:
    """媒体管理器，负责媒体文件的注册、引用计数和生命周期管理"""
    
    def __init__(self, media_dir: str = "data/media"):
        self.media_dir = Path(media_dir)
        self.metadata_dir = self.media_dir / "metadata"
        self.files_dir = self.media_dir / "files"
        self.metadata_cache: Dict[str, MediaMetadata] = {}
        self.logger = get_logger("MediaManager")
        self._pending_tasks = set()
        
        # 确保目录存在
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载所有元数据
        self._load_all_metadata()
        
    def _load_all_metadata(self) -> None:
        """加载所有媒体元数据"""
        self.metadata_cache.clear()
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = MediaMetadata.from_dict(json.load(f))
                    self.metadata_cache[metadata.media_id] = metadata
            except Exception as e:
                self.logger.error(f"Failed to load metadata from {metadata_file}: {e}")
                
    def _save_metadata(self, metadata: MediaMetadata) -> None:
        """保存媒体元数据"""
        metadata_path = self.metadata_dir / f"{metadata.media_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
        self.metadata_cache[metadata.media_id] = metadata
        
    def _get_file_path(self, media_id: str, format: str) -> Path:
        """获取媒体文件路径"""
        return self.files_dir / f"{media_id}.{format}"
    
    def _create_task(self, coro, name=None):
        """创建后台任务并跟踪它"""
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task
    
    async def _save_file_async(self, data: bytes, target_path: Path):
        """异步保存文件"""
        async with aiofiles.open(target_path, "wb") as f:
            await f.write(data)
    
    async def _download_file_async(self, url: str) -> bytes:
        """异步下载文件"""
        # 如果 url 是 file:// 开头，则直接返回文件内容
        if url.startswith("file://"):
            async with aiofiles.open(url[7:], "rb") as f:
                return await f.read()
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise ValueError(f"Failed to download file from {url}, status: {resp.status}")
                return await resp.read()
    
    def register_media(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        data: Optional[bytes] = None,
        format: Optional[str] = None,
        media_type: Optional[MediaType] = None,
        size: Optional[int] = None,
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reference_id: Optional[str] = None
    ) -> str:
        """
        注册媒体（统一方法）
        
        Args:
            url: 媒体URL
            path: 媒体文件路径
            data: 媒体二进制数据
            format: 媒体格式
            media_type: 媒体类型
            size: 媒体大小
            source: 媒体来源
            description: 媒体描述
            tags: 媒体标签
            reference_id: 引用ID
            
        Returns:
            str: 媒体ID
        """
        # 检查参数
        if not any([url, path, data]):
            raise ValueError("Must provide at least one of url, path, or data")
        
        # 生成唯一ID
        media_id = str(uuid.uuid4())
        
        # 如果提供了path，获取文件大小
        if path and not size:
            file_path = Path(path)
            if file_path.exists():
                size = file_path.stat().st_size
        
        # 如果提供了data，获取数据大小
        if data and not size:
            size = len(data)
        
        # 如果提供了data，检测文件类型
        if data and (not media_type or not format):
            mime_type, detected_media_type, detected_format = detect_mime_type(data=data)
            media_type = media_type or detected_media_type
            format = format or detected_format
        
        # 如果提供了path，检测文件类型
        if path and (not media_type or not format):
            mime_type, detected_media_type, detected_format = detect_mime_type(path=path)
            media_type = media_type or detected_media_type
            format = format or detected_format
        
        # 创建元数据
        metadata = MediaMetadata(
            media_id=media_id,
            media_type=media_type,
            format=format,
            size=size,
            created_at=datetime.now(),
            source=source,
            description=description,
            tags=tags,
            references=set([reference_id]) if reference_id else set(),
            url=url,
            path=path
        )
        
        # 保存元数据
        self._save_metadata(metadata)
        
        # 如果提供了path，在后台复制文件
        if path and format:
            target_path = self._get_file_path(media_id, format)
            
            def copy_file():
                shutil.copy2(path, target_path)
            
            # 创建后台任务
            try:
                loop = asyncio.get_event_loop()                
                self._create_task(asyncio.to_thread(copy_file), f"copy_file_{media_id}")
                # 如果没有运行中的事件循环，直接复制
            except Exception as e:
                if "event loop" in str(e):
                    copy_file()
                else:
                    self.logger.error(f"Failed to copy file: {e}")

        # 如果提供了data，在后台保存文件
        elif data and format:
            target_path = self._get_file_path(media_id, format)
            
            async def save_file():
                await self._save_file_async(data, target_path)
            
            # 创建后台任务
            try:
                loop = asyncio.get_event_loop()
                self._create_task(save_file(), f"save_file_{media_id}")
            except Exception as e:
                if "There is no current event loop" in str(e):
                    with open(target_path, "wb") as f:
                        f.write(data)
                else:
                    self.logger.error(f"Failed to save file: {e}")
        
        return media_id
    
    def register_from_path(
        self, 
        path: str, 
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reference_id: Optional[str] = None
    ) -> str:
        """从文件路径注册媒体"""
        # 检查文件是否存在
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        return self.register_media(
            path=path,
            source=source,
            description=description,
            tags=tags,
            reference_id=reference_id
        )
    
    def register_from_url(
        self, 
        url: str, 
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reference_id: Optional[str] = None
    ) -> str:
        """从URL注册媒体"""
        return self.register_media(
            url=url,
            source=source,
            description=description,
            tags=tags,
            reference_id=reference_id
        )
    
    def register_from_data(
        self, 
        data: bytes,
        format: Optional[str] = None,
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        reference_id: Optional[str] = None
    ) -> str:
        """从二进制数据注册媒体"""
        return self.register_media(
            data=data,
            format=format,
            source=source,
            description=description,
            tags=tags,
            reference_id=reference_id
        )
    
    def add_reference(self, media_id: str, reference_id: str) -> None:
        """添加引用"""
        if media_id not in self.metadata_cache:
            raise ValueError(f"Media not found: {media_id}")
        
        metadata = self.metadata_cache[media_id]
        metadata.references.add(reference_id)
        self._save_metadata(metadata)
        
    def remove_reference(self, media_id: str, reference_id: str) -> None:
        """移除引用"""
        if media_id not in self.metadata_cache:
            raise ValueError(f"Media not found: {media_id}")
        
        metadata = self.metadata_cache[media_id]
        if reference_id in metadata.references:
            metadata.references.remove(reference_id)
            self._save_metadata(metadata)
            
            # 如果没有引用了，删除文件
            if not metadata.references:
                self._delete_media(media_id)
    
    def _delete_media(self, media_id: str) -> None:
        """删除媒体文件和元数据"""
        if media_id not in self.metadata_cache:
            return
        
        metadata = self.metadata_cache[media_id]
        
        # 删除文件
        if metadata.format:
            file_path = self._get_file_path(media_id, metadata.format)
            if file_path.exists():
                file_path.unlink()
        
        # 删除元数据
        metadata_path = self.metadata_dir / f"{media_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # 从缓存中移除
        del self.metadata_cache[media_id]
        
        self.logger.info(f"Deleted media: {media_id}")
    
    def update_metadata(
        self,
        media_id: str,
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        url: Optional[str] = None,
        path: Optional[str] = None
    ) -> None:
        """更新媒体元数据"""
        if media_id not in self.metadata_cache:
            raise ValueError(f"Media not found: {media_id}")
        
        metadata = self.metadata_cache[media_id]
        
        if source is not None:
            metadata.source = source
        
        if description is not None:
            metadata.description = description
        
        if tags is not None:
            metadata.tags = tags
            
        if url is not None:
            metadata.url = url
            
        if path is not None:
            metadata.path = path
        
        self._save_metadata(metadata)
    
    def add_tags(self, media_id: str, tags: List[str]) -> None:
        """添加标签"""
        if media_id not in self.metadata_cache:
            raise ValueError(f"Media not found: {media_id}")
        
        metadata = self.metadata_cache[media_id]
        for tag in tags:
            if tag not in metadata.tags:
                metadata.tags.append(tag)
        
        self._save_metadata(metadata)
    
    def remove_tags(self, media_id: str, tags: List[str]) -> None:
        """移除标签"""
        if media_id not in self.metadata_cache:
            raise ValueError(f"Media not found: {media_id}")
        
        metadata = self.metadata_cache[media_id]
        for tag in tags:
            if tag in metadata.tags:
                metadata.tags.remove(tag)
        
        self._save_metadata(metadata)
    
    def get_metadata(self, media_id: str) -> Optional[MediaMetadata]:
        """获取媒体元数据"""
        return self.metadata_cache.get(media_id)
    
    async def ensure_file_exists(self, media_id: str) -> Optional[Path]:
        """确保媒体文件存在，如果不存在则尝试下载或复制"""
        if media_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[media_id]
        
        # 如果没有格式信息，无法确定文件路径
        if not metadata.format:
            # 如果有URL，尝试下载并检测格式
            if metadata.url:
                try:
                    data = await self._download_file_async(metadata.url)
                    _, media_type, format = detect_mime_type(data=data)
                    
                    # 更新元数据
                    metadata.media_type = media_type
                    metadata.format = format
                    metadata.size = len(data)
                    self._save_metadata(metadata)
                    
                    # 保存文件
                    target_path = self._get_file_path(media_id, format)
                    await self._save_file_async(data, target_path)
                    
                    return target_path
                except Exception as e:
                    self.logger.error(f"Failed to download media from URL: {metadata.url}, error: {e}")
                    return None
            
            # 如果有path，尝试复制并检测格式
            elif metadata.path:
                try:
                    file_path = Path(metadata.path)
                    if not file_path.exists():
                        return None
                    
                    _, media_type, format = detect_mime_type(path=str(file_path))
                    
                    # 更新元数据
                    metadata.media_type = media_type
                    metadata.format = format
                    metadata.size = file_path.stat().st_size
                    self._save_metadata(metadata)
                    
                    # 复制文件
                    target_path = self._get_file_path(media_id, format)
                    shutil.copy2(file_path, target_path)
                    
                    return target_path
                except Exception as e:
                    self.logger.error(f"Failed to copy media from path: {metadata.path}, error: {e}")
                    return None
            
            return None
        
        # 检查文件是否存在
        file_path = self._get_file_path(media_id, metadata.format)
        if file_path.exists():
            return file_path
        
        # 如果文件不存在，尝试从URL下载
        if metadata.url:
            try:
                data = await self._download_file_async(metadata.url)
                await self._save_file_async(data, file_path)
                return file_path
            except Exception as e:
                self.logger.error(f"Failed to download media from URL: {metadata.url}, error: {e}")
        
        # 如果文件不存在，尝试从path复制
        if metadata.path:
            try:
                source_path = Path(metadata.path)
                if source_path.exists():
                    shutil.copy2(source_path, file_path)
                    return file_path
            except Exception as e:
                self.logger.error(f"Failed to copy media from path: {metadata.path}, error: {e}")
        
        return None
    
    async def get_file_path(self, media_id: str) -> Optional[Path]:
        """获取媒体文件路径，如果文件不存在则尝试下载或复制"""
        if media_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[media_id]
        
        # 如果有原始路径，直接返回
        if metadata.path and Path(metadata.path).exists():
            return Path(metadata.path)
        
        # 否则确保文件存在并返回
        return await self.ensure_file_exists(media_id)
    
    async def get_data(self, media_id: str) -> Optional[bytes]:
        """获取媒体文件数据"""
        if media_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[media_id]
        
        # 如果有原始数据，直接返回
        if hasattr(metadata, 'data') and metadata.data:
            return metadata.data
        
        # 尝试从文件读取
        file_path = await self.get_file_path(media_id)
        if file_path:
            try:
                async with aiofiles.open(file_path, "rb") as f:
                    return await f.read()
            except Exception as e:
                self.logger.error(f"Failed to read media file: {file_path}, error: {e}")
        
        # 尝试从URL下载
        if metadata.url:
            try:
                return await self._download_file_async(metadata.url)
            except Exception as e:
                self.logger.error(f"Failed to download media from URL: {metadata.url}, error: {e}")
        
        return None
    
    async def get_url(self, media_id: str) -> Optional[str]:
        """获取媒体文件URL"""
        if media_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[media_id]
        
        # 如果有原始URL，直接返回
        if metadata.url:
            return metadata.url
        
        # 尝试生成文件URL
        file_path = await self.get_file_path(media_id)
        if file_path:
            return f"file://{file_path.absolute()}"
        
        # 尝试生成data URL
        data = await self.get_data(media_id)
        if data and metadata.media_type and metadata.format:
            mime_type = f"{metadata.media_type.value}/{metadata.format}"
            return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"
        
        return None
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[str]:
        """根据标签搜索媒体"""
        results = []
        
        for media_id, metadata in self.metadata_cache.items():
            if match_all:
                # 必须匹配所有标签
                if all(tag in metadata.tags for tag in tags):
                    results.append(media_id)
            else:
                # 匹配任一标签
                if any(tag in metadata.tags for tag in tags):
                    results.append(media_id)
        
        return results
    
    def search_by_description(self, query: str) -> List[str]:
        """根据描述搜索媒体"""
        results = []
        
        for media_id, metadata in self.metadata_cache.items():
            if metadata.description and query.lower() in metadata.description.lower():
                results.append(media_id)
        
        return results
    
    def search_by_source(self, source: str) -> List[str]:
        """根据来源搜索媒体"""
        results = []
        
        for media_id, metadata in self.metadata_cache.items():
            if metadata.source and source.lower() in metadata.source.lower():
                results.append(media_id)
        
        return results
    
    def search_by_type(self, media_type: MediaType) -> List[str]:
        """根据媒体类型搜索媒体"""
        results = []
        
        for media_id, metadata in self.metadata_cache.items():
            if metadata.media_type == media_type:
                results.append(media_id)
        
        return results
    
    def get_all_media_ids(self) -> List[str]:
        """获取所有媒体ID"""
        return list(self.metadata_cache.keys())
    
    def cleanup_unreferenced(self) -> int:
        """清理没有引用的媒体文件，返回清理的文件数量"""
        count = 0
        for media_id, metadata in list(self.metadata_cache.items()):
            if not metadata.references:
                self._delete_media(media_id)
                count += 1
        
        return count
    
    async def create_media_message(self, media_id: str) -> Optional["MediaMessage"]:
        """根据媒体ID创建MediaMessage对象"""
        if media_id not in self.metadata_cache:
            return None
        from kirara_ai.im.message import FileElement, ImageMessage, VideoElement, VoiceMessage
        
        metadata = self.metadata_cache[media_id]
        
        # 根据媒体类型创建不同的MediaMessage子类
        if metadata.media_type == MediaType.IMAGE:
            return ImageMessage(media_id=media_id)
        elif metadata.media_type == MediaType.VOICE:
            return VoiceMessage(media_id=media_id)
        elif metadata.media_type == MediaType.VIDEO:
            return VideoElement(media_id=media_id)
        else:
            return FileElement(media_id=media_id)


# 单例模式
_media_manager: Optional[MediaManager] = None

def get_media_manager() -> MediaManager:
    """获取媒体管理器单例"""
    global _media_manager
    if _media_manager is None:
        _media_manager = MediaManager()
    return _media_manager
