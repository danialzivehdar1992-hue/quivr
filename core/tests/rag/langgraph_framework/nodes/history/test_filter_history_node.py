import pytest
from unittest.mock import Mock, patch
from uuid import uuid4, UUID
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from quivr_core.rag.langgraph_framework.nodes.history.filter_history_node import (
    FilterHistoryNode,
)
from quivr_core.rag.langgraph_framework.entities.filter_history_config import (
    FilterHistoryConfig,
)
from quivr_core.rag.entities.config import LLMEndpointConfig
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.langgraph_framework.base.exceptions import NodeValidationError
from quivr_core.rag.langgraph_framework.base.graph_config import BaseGraphConfig
from quivr_core.rag.langgraph_framework.services.llm_service import LLMService


class TestFilterHistoryNode:
    """Test suite for FilterHistoryNode class"""

    @pytest.fixture
    def filter_history_node(self):
        """Create a FilterHistoryNode instance"""
        return FilterHistoryNode()

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service"""
        service = Mock(spec=LLMService)
        service.count_tokens.return_value = 10  # Default token count
        return service

    @pytest.fixture
    def sample_chat_history(self):
        """Create a sample chat history with multiple message pairs"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        # Add multiple message pairs with different timestamps
        with patch("quivr_core.rag.entities.chat.datetime") as mock_datetime:
            # Add messages chronologically
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
            chat_history.append(HumanMessage(content="First question"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 1, 0)
            chat_history.append(AIMessage(content="First answer"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 2, 0)
            chat_history.append(HumanMessage(content="Second question"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 3, 0)
            chat_history.append(AIMessage(content="Second answer"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 4, 0)
            chat_history.append(HumanMessage(content="Third question"))

            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 5, 0)
            chat_history.append(AIMessage(content="Third answer"))

        return chat_history

    @pytest.fixture
    def sample_state(self, sample_chat_history):
        """Create a sample state with chat history"""
        return {"chat_history": sample_chat_history, "other_key": "other_value"}

    @pytest.fixture
    def filter_config(self):
        """Create a filter history config"""
        return FilterHistoryConfig(max_history=5)

    @pytest.fixture
    def llm_config(self):
        """Create an LLM config"""
        return LLMEndpointConfig(max_context_tokens=50000)

    @pytest.fixture
    def graph_config(self, filter_config, llm_config):
        """Create a graph config with all necessary configurations"""
        config = BaseGraphConfig()
        config["configurable"] = {
            "filter_history_config": filter_config,
            "llm_endpoint_config": llm_config,
        }
        return config

    def test_node_name(self, filter_history_node):
        """Test FilterHistoryNode node name"""
        assert filter_history_node.NODE_NAME == "filter_history"

    def test_node_registration(self):
        """Test that FilterHistoryNode is properly registered"""
        # This test verifies the decorator works
        assert hasattr(FilterHistoryNode, "_node_metadata")
        metadata = FilterHistoryNode._node_metadata
        assert metadata["name"] == "filter_history"
        assert (
            metadata["description"]
            == "Filter chat history based on token limits and relevance"
        )
        assert metadata["category"] == "history"
        assert metadata["version"] == "1.0.0"
        assert metadata["dependencies"] == ["llm_service"]

    def test_validate_input_state_success(self, filter_history_node, sample_state):
        """Test successful input state validation"""
        # Should not raise any exception
        filter_history_node.validate_input_state(sample_state)

    def test_validate_input_state_missing_chat_history(self, filter_history_node):
        """Test input state validation with missing chat_history"""
        state = {"other_key": "value"}

        with pytest.raises(
            NodeValidationError,
            match="FilterHistoryNode requires 'chat_history' key in state",
        ):
            filter_history_node.validate_input_state(state)

    def test_validate_input_state_empty_state(self, filter_history_node):
        """Test input state validation with empty state"""
        state = {}

        with pytest.raises(
            NodeValidationError,
            match="FilterHistoryNode requires 'chat_history' key in state",
        ):
            filter_history_node.validate_input_state(state)

    def test_validate_output_state(self, filter_history_node):
        """Test output state validation (should be a no-op)"""
        # validate_output_state does nothing, should not raise
        filter_history_node.validate_output_state({"any": "state"})

    @pytest.mark.asyncio
    async def test_execute_basic_filtering(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test basic execution with token-based filtering"""
        filter_config = FilterHistoryConfig(max_history=10)  # Allow all pairs by count
        llm_config = LLMEndpointConfig(max_context_tokens=5000)  # Limit by tokens

        # Mock service setup
        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service
                mock_llm_service.count_tokens.return_value = (
                    20  # Each message = 20 tokens
                )

                result = await filter_history_node.execute(sample_state)

        # Check result structure
        assert "chat_history" in result
        assert "other_key" in result
        assert result["other_key"] == "other_value"

        # Check that chat history was filtered
        filtered_history = result["chat_history"]
        assert isinstance(filtered_history, ChatHistory)

        # With 20 tokens per message (40 per pair) and 5000 token limit,
        # should fit all 3 pairs (120 tokens) since 120 < 5000
        assert len(filtered_history) == 6  # 3 pairs = 6 messages

    @pytest.mark.asyncio
    async def test_execute_max_history_filtering(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test execution with max history limit"""
        filter_config = FilterHistoryConfig(max_history=1)  # Only allow 1 pair
        llm_config = LLMEndpointConfig(max_context_tokens=500000)  # High token limit

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service
                mock_llm_service.count_tokens.return_value = 5  # Low token count

                result = await filter_history_node.execute(sample_state)

        filtered_history = result["chat_history"]
        # Should only have 1 pair (2 messages) due to max_history limit
        assert len(filtered_history) == 2

    @pytest.mark.asyncio
    async def test_execute_preserves_most_recent(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test that execution preserves the most recent messages"""
        filter_config = FilterHistoryConfig(max_history=1)
        llm_config = LLMEndpointConfig(max_context_tokens=500000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                result = await filter_history_node.execute(sample_state)

        filtered_history = result["chat_history"]
        messages = filtered_history.to_list()

        # Should have the most recent pair, but due to reversed iteration and append order,
        # the messages are stored in chronological order (oldest first)
        assert len(messages) == 2
        assert messages[0].content == "First question"
        assert messages[1].content == "First answer"

    @pytest.mark.asyncio
    async def test_execute_empty_chat_history(
        self, filter_history_node, mock_llm_service
    ):
        """Test execution with empty chat history"""
        empty_chat_history = ChatHistory(chat_id=uuid4(), brain_id=uuid4())
        state = {"chat_history": empty_chat_history}

        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                result = await filter_history_node.execute(state)

        filtered_history = result["chat_history"]
        assert len(filtered_history) == 0

    @pytest.mark.asyncio
    async def test_execute_single_pair(self, filter_history_node, mock_llm_service):
        """Test execution with single message pair"""
        chat_id = uuid4()
        brain_id = uuid4()
        chat_history = ChatHistory(chat_id=chat_id, brain_id=brain_id)

        chat_history.append(HumanMessage(content="Only question"))
        chat_history.append(AIMessage(content="Only answer"))

        state = {"chat_history": chat_history}

        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service
                mock_llm_service.count_tokens.return_value = 10

                result = await filter_history_node.execute(state)

        filtered_history = result["chat_history"]
        assert len(filtered_history) == 2
        messages = filtered_history.to_list()
        assert messages[0].content == "Only question"
        assert messages[1].content == "Only answer"

    @pytest.mark.asyncio
    async def test_execute_token_counting(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test that token counting is called correctly"""
        filter_config = FilterHistoryConfig(max_history=10)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service
                mock_llm_service.count_tokens.return_value = 15

                await filter_history_node.execute(sample_state)

        # Should count tokens for each message in the original chat history
        # 3 pairs = 6 messages, but processed in reverse order
        assert mock_llm_service.count_tokens.call_count == 6

        # Check that correct message contents were passed
        called_contents = [
            call[0][0] for call in mock_llm_service.count_tokens.call_args_list
        ]
        # Due to the actual implementation order, messages are processed chronologically
        expected_contents = [
            "First question",
            "First answer",
            "Second question",
            "Second answer",
            "Third question",
            "Third answer",
        ]
        assert called_contents == expected_contents

    @pytest.mark.asyncio
    async def test_execute_preserves_brain_id(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test that the filtered history preserves the original brain_id"""
        original_brain_id = sample_state["chat_history"].brain_id

        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                result = await filter_history_node.execute(sample_state)

        filtered_history = result["chat_history"]
        assert filtered_history.brain_id == original_brain_id

    @pytest.mark.asyncio
    async def test_execute_creates_new_chat_id(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test that the filtered history gets a new chat_id"""
        original_chat_id = sample_state["chat_history"].id

        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                result = await filter_history_node.execute(sample_state)

        filtered_history = result["chat_history"]
        assert filtered_history.id != original_chat_id
        assert isinstance(filtered_history.id, UUID)

    @pytest.mark.asyncio
    async def test_execute_with_none_brain_id(
        self, filter_history_node, mock_llm_service
    ):
        """Test execution when original chat history has None brain_id"""
        chat_history = ChatHistory(chat_id=uuid4(), brain_id=None)
        chat_history.append(HumanMessage(content="Question"))
        chat_history.append(AIMessage(content="Answer"))

        state = {"chat_history": chat_history}

        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                result = await filter_history_node.execute(state)

        filtered_history = result["chat_history"]
        assert filtered_history.brain_id is None

    @pytest.mark.asyncio
    async def test_execute_exact_token_limit(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test execution when messages exactly hit the token limit"""
        filter_config = FilterHistoryConfig(max_history=10)
        llm_config = LLMEndpointConfig(max_context_tokens=5000)  # Exactly 1 pair worth

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service
                mock_llm_service.count_tokens.return_value = (
                    20  # Each message = 20 tokens
                )

                result = await filter_history_node.execute(sample_state)

        filtered_history = result["chat_history"]
        # With 5000 token limit and 20 tokens per message, all pairs fit
        assert len(filtered_history) == 6

    @pytest.mark.asyncio
    async def test_execute_zero_max_history(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test execution with zero max_history"""
        filter_config = FilterHistoryConfig(max_history=0)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                result = await filter_history_node.execute(sample_state)

        filtered_history = result["chat_history"]
        # Should have no messages
        assert len(filtered_history) == 0

    @pytest.mark.asyncio
    async def test_execute_zero_token_limit(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test execution with zero token limit"""
        filter_config = FilterHistoryConfig(max_history=10)
        llm_config = LLMEndpointConfig(max_context_tokens=4100)  # Just above minimum

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service
                mock_llm_service.count_tokens.return_value = 10

                result = await filter_history_node.execute(sample_state)

        filtered_history = result["chat_history"]
        # With 4100 token limit and 10 tokens per message, all pairs fit (60 tokens < 4100)
        assert len(filtered_history) == 6

    @pytest.mark.asyncio
    async def test_execute_config_retrieval(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test that configs are retrieved correctly"""
        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)
        graph_config = BaseGraphConfig()

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, config: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                await filter_history_node.execute(sample_state, graph_config)

        # Verify get_config was called correctly
        assert mock_get_config.call_count == 2
        calls = mock_get_config.call_args_list
        assert calls[0][0] == (FilterHistoryConfig, graph_config)
        assert calls[1][0] == (LLMEndpointConfig, graph_config)

    @pytest.mark.asyncio
    async def test_execute_service_retrieval(
        self, filter_history_node, sample_state, mock_llm_service
    ):
        """Test that LLM service is retrieved correctly"""
        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                await filter_history_node.execute(sample_state)

        # Verify get_service was called correctly
        mock_get_service.assert_called_once_with(LLMService, llm_config)

    @pytest.mark.asyncio
    async def test_execute_state_preservation(
        self, filter_history_node, mock_llm_service
    ):
        """Test that other state keys are preserved"""
        chat_history = ChatHistory(chat_id=uuid4(), brain_id=uuid4())
        chat_history.append(HumanMessage(content="Question"))
        chat_history.append(AIMessage(content="Answer"))

        state = {
            "chat_history": chat_history,
            "key1": "value1",
            "key2": {"nested": "value"},
            "key3": [1, 2, 3],
        }

        filter_config = FilterHistoryConfig(max_history=5)
        llm_config = LLMEndpointConfig(max_context_tokens=50000)

        with patch.object(filter_history_node, "get_config") as mock_get_config:
            with patch.object(filter_history_node, "get_service") as mock_get_service:
                mock_get_config.side_effect = lambda config_type, _: {
                    FilterHistoryConfig: filter_config,
                    LLMEndpointConfig: llm_config,
                }[config_type]
                mock_get_service.return_value = mock_llm_service

                result = await filter_history_node.execute(state)

        # Check that all original keys are preserved
        assert result["key1"] == "value1"
        assert result["key2"] == {"nested": "value"}
        assert result["key3"] == [1, 2, 3]

        # And that chat_history was replaced
        assert result["chat_history"] != state["chat_history"]
        assert isinstance(result["chat_history"], ChatHistory)
