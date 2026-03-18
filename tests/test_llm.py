import pytest
import asyncio
from app.llm import chat, embed
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_chat_response():
    # Mocking httpx response for non-streaming chat
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, I am your telecom assistant."}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        messages = [{"role": "user", "content": "Hello"}]
        response = await chat(messages)
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_chat_streaming():
    # Mocking streaming response
    class MockStreamResponse:
        def __init__(self):
            self.status_code = 200
            
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
            yield 'data: [DONE]'
        def raise_for_status(self):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStreamResponse()):
        messages = [{"role": "user", "content": "Hello"}]
        tokens = []
        async for token in await chat(messages, stream=True):
            tokens.append(token)
        
        assert len(tokens) == 2
        assert "".join(tokens) == "Hello world"

@pytest.mark.asyncio
async def test_embed_response():
    # Mocking embedding response with 1024 dimensions
    mock_embedding = [0.1] * 1024
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": mock_embedding}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        embedding = await embed("sample text")
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)
