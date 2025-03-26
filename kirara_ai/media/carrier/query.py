from typing import Dict, List, Tuple

from kirara_ai.media.media_object import Media

from .service import MediaCarrierService


class MediaReferenceQuery:
    """媒体引用查询工具"""
    
    def __init__(self, carrier_service: MediaCarrierService):
        self.carrier_service = carrier_service
    
    def get_orphaned_media(self) -> List[Media]:
        """获取没有引用的媒体"""
        media_manager = self.carrier_service.media_manager
        orphaned = []
        
        for media_id, metadata in media_manager.metadata_cache.items():
            if not metadata.references:
                media = media_manager.get_media(media_id)
                if media:
                    orphaned.append(media)
        
        return orphaned
    
    def get_reference_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """获取引用关系图"""
        media_manager = self.carrier_service.media_manager
        graph = {}
        
        for media_id in media_manager.metadata_cache:
            references = self.carrier_service.get_references_by_media(media_id)
            if references:
                graph[media_id] = references
        
        return graph
    
    def find_media_by_provider(self, provider_name: str) -> Dict[str, List[Media]]:
        """查找特定提供者引用的所有媒体"""
        result = {}
        
        # 获取提供者实例
        provider = self.carrier_service.registry.get_provider(provider_name)
        
        # 获取所有引用键
        for reference_key in provider.get_reference_keys():
            # 获取媒体列表
            media_list = self.carrier_service.get_media_by_reference(
                provider_name, reference_key
            )
            if media_list:
                result[reference_key] = media_list
        
        return result
