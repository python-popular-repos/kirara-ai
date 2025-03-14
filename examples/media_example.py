#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from kirara_ai.im.message import create_image_message
from kirara_ai.media import get_media_manager


async def main():
    """媒体API示例 - 懒加载策略"""
    print("媒体API示例 - 懒加载策略")
    
    # 获取媒体管理器
    media_manager = get_media_manager()
    
    # 示例1：只提供URL注册媒体
    print("\n示例1：只提供URL注册媒体")
    # 使用一个公共图片URL
    image_url = "https://picsum.photos/200/300"
    
    # 注册图片（只提供URL）
    url_media_id = media_manager.register_from_url(
        url=image_url,
        source="url_example",
        description="只提供URL的示例",
        tags=["example", "url"],
        reference_id="url_ref"
    )
    print(f"注册URL媒体成功，ID: {url_media_id}")
    
    # 获取元数据（此时可能没有完整信息）
    metadata = media_manager.get_metadata(url_media_id)
    print(f"初始元数据: {metadata.to_dict()}")
    
    # 获取数据（此时会触发下载）
    print("获取数据（将触发下载）...")
    data = await media_manager.get_data(url_media_id)
    print(f"数据大小: {len(data) if data else 'None'} 字节")
    
    # 再次获取元数据（此时应该有更多信息）
    metadata = media_manager.get_metadata(url_media_id)
    print(f"更新后的元数据: {metadata.to_dict()}")
    
    # 获取文件路径（此时文件应该已经存在）
    file_path = await media_manager.get_file_path(url_media_id)
    print(f"文件路径: {file_path}")
    
    # 示例2：只提供路径注册媒体
    print("\n示例2：只提供路径注册媒体")
    # 创建一个测试图片
    image_path = Path(__file__).parent / "test_image.jpg"
    if not image_path.exists():
        with open(image_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C")
    
    # 注册图片（只提供路径）
    path_media_id = media_manager.register_from_path(
        path=str(image_path),
        source="path_example",
        description="只提供路径的示例",
        tags=["example", "path"],
        reference_id="path_ref"
    )
    print(f"注册路径媒体成功，ID: {path_media_id}")
    
    # 获取元数据
    metadata = media_manager.get_metadata(path_media_id)
    print(f"元数据: {metadata.to_dict()}")
    
    # 获取URL（此时会生成文件URL）
    url = await media_manager.get_url(path_media_id)
    print(f"生成的URL: {url}")
    
    # 示例3：只提供数据注册媒体
    print("\n示例3：只提供数据注册媒体")
    # 创建一些测试数据
    test_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C"
    
    # 注册媒体（只提供数据）
    data_media_id = media_manager.register_from_data(
        data=test_data,
        source="data_example",
        description="只提供数据的示例",
        tags=["example", "data"],
        reference_id="data_ref"
    )
    print(f"注册数据媒体成功，ID: {data_media_id}")
    
    # 获取元数据
    metadata = media_manager.get_metadata(data_media_id)
    print(f"元数据: {metadata.to_dict()}")
    
    # 获取路径（此时会生成文件）
    file_path = await media_manager.get_file_path(data_media_id)
    print(f"生成的文件路径: {file_path}")
    
    # 示例4：使用MediaMessage
    print("\n示例4：使用MediaMessage - 只提供URL")
    # 创建只有URL的媒体消息
    url_message = await create_image_message(
        url=image_url,
        source="message_example",
        description="只提供URL的媒体消息",
        tags=["example", "message", "url"],
        reference_id="message_url_ref"
    )
    print(f"创建媒体消息成功，ID: {url_message.media_id}")
    
    # 获取路径（此时会触发下载和文件创建）
    print("获取路径（将触发下载）...")
    path = await url_message.get_path()
    print(f"生成的路径: {path}")
    
    # 再次获取URL（此时应该直接返回缓存的URL）
    url = await url_message.get_url()
    print(f"原始URL: {url}")
    
    # 示例5：使用MediaMessage
    print("\n示例5：使用MediaMessage - 只提供路径")
    # 创建只有路径的媒体消息
    path_message = await create_image_message(
        path=str(image_path),
        source="message_example",
        description="只提供路径的媒体消息",
        tags=["example", "message", "path"],
        reference_id="message_path_ref"
    )
    print(f"创建媒体消息成功，ID: {path_message.media_id}")
    
    # 获取URL（此时会生成URL）
    url = await path_message.get_url()
    print(f"生成的URL: {url}")
    
    # 获取数据（此时会读取文件）
    data = await path_message.get_data()
    print(f"数据大小: {len(data)} 字节")
    
    # 示例6：使用MediaMessage
    print("\n示例6：使用MediaMessage - 只提供数据")
    # 创建只有数据的媒体消息
    data_message = await create_image_message(
        data=test_data,
        format="jpg",
        source="message_example",
        description="只提供数据的媒体消息",
        tags=["example", "message", "data"],
        reference_id="message_data_ref"
    )
    print(f"创建媒体消息成功，ID: {data_message.media_id}")
    
    # 获取路径（此时会创建文件）
    path = await data_message.get_path()
    print(f"生成的路径: {path}")
    
    # 获取URL（此时会生成URL）
    url = await data_message.get_url()
    print(f"生成的URL: {url}")
    
    # 示例7：清理无引用的媒体
    print("\n示例7：清理无引用的媒体")
    # 创建一个无引用的媒体
    no_ref_id = media_manager.register_from_path(
        str(image_path),
        source="no_ref_example",
        description="无引用的媒体",
        tags=["no_ref"]
    )
    print(f"创建无引用媒体，ID: {no_ref_id}")
    
    # 清理无引用的媒体
    count = media_manager.cleanup_unreferenced()
    print(f"清理了 {count} 个无引用的媒体")
    
    # 验证是否已被清理
    if media_manager.get_metadata(no_ref_id) is None:
        print(f"媒体 {no_ref_id} 已被清理")
    else:
        print(f"媒体 {no_ref_id} 仍然存在")
    
    print("\n示例结束")


if __name__ == "__main__":
    asyncio.run(main()) 