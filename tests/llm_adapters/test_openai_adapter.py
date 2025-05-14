from kirara_ai.plugins.llm_preset_adapters.openai_adapter import OpenAIAdapter, OpenAIConfig
from kirara_ai.llm.format.embedding import LLMEmbeddingRequest, LLMEmbeddingResponse
from kirara_ai.llm.format.message import LLMChatTextContent, LLMChatImageContent
import pytest

from .mock_app import AUTH_KEY, REFERENCE_VECTOR, OPENAI_ENDPOINT
class TestOpenAIAdapter:
    @pytest.fixture(scope="class")
    def openai_adapter(self, mock_media_manager) -> OpenAIAdapter:
        config  = OpenAIConfig(
            api_base=OPENAI_ENDPOINT,
            api_key=AUTH_KEY
        )
        adapter = OpenAIAdapter(config)
        adapter.media_manager = mock_media_manager
        return adapter

    def test_embed(self, openai_adapter: OpenAIAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatTextContent(text="hello world", type="text")],
            model="mock_embedding",
        )

        response = openai_adapter.embed(req)
        assert isinstance(response, LLMEmbeddingResponse)
        assert response.vectors[0] == REFERENCE_VECTOR

    def test_embed_with_image(self, openai_adapter: OpenAIAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatImageContent(media_id="mock_media_id", type="image")],
            model="mock_embedding",
        )

        with pytest.raises(ValueError, match="openai does not support multi-modal embedding"):
            openai_adapter.embed(req)

    def test_embed_with_input_out_of_range(self, openai_adapter: OpenAIAdapter):
        req = LLMEmbeddingRequest(
            inputs=[LLMChatTextContent(text="hello world", type="text") for _ in range(2050)],
            model="mock_embedding"
        )
        
        with pytest.raises(ValueError, match="Text list has too many dimensions, max dimension is 2048"):
            openai_adapter.embed(req)