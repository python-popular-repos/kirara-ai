from llm_preset_adapters.alibabacloud_adapter import AlibabaCloudAdapter, AlibabaCloudConfig
from llm_preset_adapters.claude_adapter import ClaudeAdapter, ClaudeConfig
from llm_preset_adapters.deepseek_adapter import DeepSeekAdapter, DeepSeekConfig
from llm_preset_adapters.gemini_adapter import GeminiAdapter, GeminiConfig
from llm_preset_adapters.minimax_adapter import MinimaxAdapter, MinimaxConfig
from llm_preset_adapters.moonshot_adapter import MoonshotAdapter, MoonshotConfig
from llm_preset_adapters.ollama_adapter import OllamaAdapter, OllamaConfig
from llm_preset_adapters.openai_adapter import OpenAIAdapter, OpenAIConfig
from llm_preset_adapters.openrouter_adapter import OpenRouterAdapter, OpenRouterConfig
from llm_preset_adapters.siliconflow_adapter import SiliconFlowAdapter, SiliconFlowConfig
from llm_preset_adapters.tencentcloud_adapter import TencentCloudAdapter, TencentCloudConfig
from llm_preset_adapters.volcengine_adapter import VolcengineAdapter, VolcengineConfig

from kirara_ai.logger import get_logger
from kirara_ai.plugin_manager.plugin import Plugin
from kirara_ai.plugins.llm_preset_adapters.mistral_adapter import MistralAdapter, MistralConfig

logger = get_logger("LLMPresetAdapters")


class LLMPresetAdaptersPlugin(Plugin):
    def __init__(self):
        pass

    def on_load(self):
        self.llm_registry.register(
            "OpenAI", OpenAIAdapter, OpenAIConfig
        )
        self.llm_registry.register(
            "DeepSeek", DeepSeekAdapter, DeepSeekConfig
        )
        self.llm_registry.register(
            "Gemini", GeminiAdapter, GeminiConfig
        )
        self.llm_registry.register(
            "Ollama", OllamaAdapter, OllamaConfig
        )
        self.llm_registry.register(
            "Claude", ClaudeAdapter, ClaudeConfig
        )
        self.llm_registry.register(
            "SiliconFlow", SiliconFlowAdapter, SiliconFlowConfig
        )
        self.llm_registry.register(
            "TencentCloud", TencentCloudAdapter, TencentCloudConfig
        )
        self.llm_registry.register(
            "AlibabaCloud", AlibabaCloudAdapter, AlibabaCloudConfig
        )
        self.llm_registry.register(
            "Moonshot", MoonshotAdapter, MoonshotConfig
        )
        self.llm_registry.register(
            "OpenRouter", OpenRouterAdapter, OpenRouterConfig
        )
        self.llm_registry.register(
            "Minimax", MinimaxAdapter, MinimaxConfig
        )
        self.llm_registry.register(
            "Volcengine", VolcengineAdapter, VolcengineConfig
        )
        self.llm_registry.register(
            "Mistral", MistralAdapter, MistralConfig
        )
        logger.info("LLMPresetAdaptersPlugin loaded")

    def on_start(self):
        logger.info("LLMPresetAdaptersPlugin started")

    def on_stop(self):
        logger.info("LLMPresetAdaptersPlugin stopped")
