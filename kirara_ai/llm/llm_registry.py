from typing import Dict, Optional, Type

from pydantic import BaseModel

from kirara_ai.logger import get_logger

from .adapter import LLMBackendAdapter
from .model_types import LLMAbility  # noqa: F401


class LLMBackendRegistry:
    """
    LLM后端注册表
    """

    _adapters: Dict[str, Type[LLMBackendAdapter]]
    _configs: Dict[str, Type[BaseModel]]

    def __init__(self):
        self._adapters = {}
        self._configs = {}
        self.logger = get_logger(__name__)

    def register(
        self,
        adapter_type: str,
        adapter_class: Type[LLMBackendAdapter],
        config_class: Type[BaseModel],
        *args, **kwargs
    ):
        """
        注册一个LLM后端适配器
        :param adapter_type: 适配器类型
        :param adapter_class: 适配器类
        :param config_class: 配置类
        """

        self._adapters[adapter_type] = adapter_class
        self._configs[adapter_type] = config_class
        self.logger.info(
            f"Registered LLM backend adapter: {adapter_type}"
        )

    def get(self, adapter_type: str) -> Optional[Type[LLMBackendAdapter]]:
        """
        获取指定类型的适配器类
        :param adapter_type: 适配器类型
        :return: 适配器类,如果没有找到则返回None
        """
        return next(
            (adapter for key, adapter in self._adapters.items() if key.lower() == adapter_type.lower()),
            None
        )

    def get_config_class(self, adapter_type: str) -> Optional[Type[BaseModel]]:
        """
        获取指定类型的配置类
        :param adapter_type: 适配器类型
        :return: 配置类,如果没有找到则返回None
        """
        return next(
            (config for key, config in self._configs.items() if key.lower() == adapter_type.lower()),
            None
        )

    def get_adapter_types(self) -> list[str]:
        """
        获取所有已注册的适配器类型
        :return: 适配器类型列表
        """
        return list(self._adapters.keys())

    def get_all_adapters(self) -> Dict[str, Type[LLMBackendAdapter]]:
        """
        获取所有已注册的 LLM 适配器。
        :return: 所有已注册的 LLM 适配器字典。
        """
        return self._adapters.copy()
