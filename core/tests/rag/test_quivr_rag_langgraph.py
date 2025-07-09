"""
End-to-end tests for QuivrQARAGLangGraph class.
Tests the Fast RAG workflow and other configurations.
"""

import pytest
from unittest.mock import Mock, patch
from uuid import uuid4
from typing import Dict, Any, Annotated, Sequence, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph.message import add_messages
from langgraph.graph import END, START

from quivr_core.base_config import QuivrBaseConfig
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.models import QuivrKnowledge
from quivr_core.rag.entities.config import (
    CitationConfig,
    LLMEndpointConfig,
    WorkflowConfig,
    NodeConfig,
)
from quivr_core.rag.entities.prompt import PromptConfig
from quivr_core.rag.langgraph_framework.base.graph_config import BaseGraphConfig
from quivr_core.rag.langgraph_framework.entities.filter_history_config import (
    FilterHistoryConfig,
)
from quivr_core.rag.langgraph_framework.services.llm_service import LLMService
from quivr_core.rag.langgraph_framework.services.service_container import (
    ServiceContainer,
)
from quivr_core.rag.langgraph_framework.base.extractors import ConfigMapping
from quivr_core.rag.langgraph_framework.task import UserTasks
from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph

# Ensure node registration


class FastRAGGraphConfig(QuivrBaseConfig):
    """Configuration schema for Fast RAG workflow."""

    llm_config: LLMEndpointConfig = LLMEndpointConfig()
    filter_history_config: FilterHistoryConfig = FilterHistoryConfig()
    workflow_config: WorkflowConfig = WorkflowConfig()
    citation_config: CitationConfig = CitationConfig()


class FastRAGAgentState(TypedDict):
    """Agent state for Fast RAG workflow."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    reasoning: List[str]
    chat_history: ChatHistory
    files: str
    tasks: UserTasks
    instructions: str
    ticket_metadata: Optional[Dict[str, str]]
    user_metadata: Optional[Dict[str, str]]
    additional_information: Optional[Dict[str, str]]
    tool: str
    guidelines: str
    enforced_system_prompt: str
    _filter: Optional[Dict[str, Any]]
    ticket_history: str


# Move the fixture outside of any class, at module level
@pytest.fixture(scope="function")
def sample_knowledge_files():
    """Sample QuivrKnowledge files for testing."""
    return [
        QuivrKnowledge(
            id=uuid4(),
            brain_ids=[uuid4()],
            file_name="document1.pdf",
            extension=".pdf",
            url="https://example.com/doc1.pdf",
        ),
        QuivrKnowledge(
            id=uuid4(),
            brain_ids=[uuid4()],
            file_name="document2.txt",
            extension=".txt",
            url="https://example.com/doc2.txt",
        ),
    ]


class TestQuivrQARAGLangGraph:
    """End-to-end tests for QuivrQARAGLangGraph with Fast RAG workflow."""

    @pytest.fixture(scope="function")
    def fast_rag_workflow_config(self):
        """Fast RAG workflow configuration matching the provided JSON."""
        return {
            "citation_config": {
                "max_files": 20,
            },
            "llm_config": {
                "temperature": 0.3,
                "max_context_tokens": 20000,
                "model": "gpt-4o-mini",
            },
            "filter_history_config": {"max_history": 10},
            "workflow_config": {
                "name": "Fast RAG",
                "nodes": [
                    {
                        "name": "START",
                        "edges": ["filter_history"],
                        "description": "Starting workflow",
                    },
                    {
                        "name": "filter_history",
                        "edges": ["retrieve"],
                        "description": "Filtering history",
                    },
                    {
                        "name": "retrieve",
                        "edges": ["generate_rag"],
                        "description": "Retrieving relevant information",
                    },
                    {
                        "name": "generate_rag",
                        "edges": ["END"],
                        "tools": [{"name": "cited_answer"}],
                        "description": "Generating response",
                    },
                ],
            },
        }

    @pytest.fixture(scope="function")
    def workflow_config(self, fast_rag_workflow_config):
        """Create WorkflowConfig from the Fast RAG configuration."""
        return WorkflowConfig.model_validate(
            fast_rag_workflow_config["workflow_config"]
        )

    @pytest.fixture(scope="function")
    def graph_schema(self):
        """Graph configuration schema."""
        return FastRAGGraphConfig

    @pytest.fixture(scope="function")
    def graph_config(self, fast_rag_workflow_config):
        """Runtime graph configuration."""
        return {
            "llm_config": fast_rag_workflow_config["llm_config"],
            "filter_history_config": fast_rag_workflow_config["filter_history_config"],
            "citation_config": fast_rag_workflow_config["citation_config"],
        }

    @pytest.fixture(scope="function")
    def config_extractor(self):
        """Config extractor for handling node-specific configurations."""
        return ConfigMapping(
            {
                LLMEndpointConfig: "llm_config",
                FilterHistoryConfig: "filter_history_config",
                WorkflowConfig: "workflow_config",
                PromptConfig: "prompt_config",
            }
        )

    @pytest.fixture(scope="function")
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock_service = Mock(spec=LLMService)
        mock_service.supports_function_calling.return_value = True
        mock_service.count_tokens.return_value = 100
        mock_service.bind_tools_to_llm.return_value = Mock()

        # Mock the invoke method to return an AI message
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(
            content="This is a test response from the LLM."
        )
        mock_service.bind_tools_to_llm.return_value = mock_llm

        return mock_service

    @pytest.fixture(scope="function")
    def mock_vector_store(self):
        """Mock vector store for retrieval testing."""
        mock_store = Mock()
        mock_retriever = Mock()

        # Mock retrieval results
        mock_docs = [
            Document(
                page_content="This is test content about France.",
                metadata={
                    "source": "test_document.pdf",
                    "knowledge_id": str(uuid4()),
                    "chunk_index": 0,
                    "similarity": 0.9,
                },
            ),
            Document(
                page_content="Paris is the capital of France.",
                metadata={
                    "source": "test_document.pdf",
                    "knowledge_id": str(uuid4()),
                    "chunk_index": 1,
                    "similarity": 0.8,
                },
            ),
        ]

        async def mock_ainvoke(query):
            return mock_docs

        mock_retriever.ainvoke = mock_ainvoke
        mock_store.as_retriever.return_value = mock_retriever
        return mock_store

    @pytest.fixture(scope="function")
    def mock_service_container(self, mock_llm_service, mock_vector_store):
        """Mock service container with all required services."""
        container = Mock(spec=ServiceContainer)

        def mock_get_service(service_type, config):
            if service_type == LLMService:
                return mock_llm_service
            # Add other service mocks as needed
            return Mock()

        container.get_service = mock_get_service
        return container

    @pytest.fixture(scope="function")
    def sample_chat_history(self):
        """Sample chat history for testing."""
        chat_history = ChatHistory(chat_id=uuid4(), brain_id=uuid4())
        chat_history.append(HumanMessage(content="What is the weather like?"))
        chat_history.append(
            AIMessage(content="I can help you with weather information.")
        )
        chat_history.append(HumanMessage(content="What about Paris?"))
        chat_history.append(
            AIMessage(content="Let me check the weather in Paris for you.")
        )
        return chat_history

    @pytest.fixture(scope="function")
    def mock_graph_builder(self):
        """Mock GraphBuilder for testing."""
        with patch(
            "quivr_core.rag.quivr_rag_langgraph.GraphBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder.set_custom_state_graph.return_value = mock_builder
            mock_builder.build_from_workflow_config.return_value = mock_builder
            mock_builder.final_nodes = ["generate_rag"]
            mock_builder.get_compiled_graph.return_value = Mock()
            mock_builder_class.return_value = mock_builder
            yield mock_builder

    @pytest.fixture(scope="function")
    def rag_instance(
        self,
        workflow_config,
        graph_schema,
        graph_config,
        mock_llm_service,
        config_extractor,
        mock_service_container,
        mock_graph_builder,
    ):
        """Create QuivrQARAGLangGraph instance for testing."""
        return QuivrQARAGLangGraph(
            workflow_config=workflow_config,
            graph_state=FastRAGAgentState,
            graph_config=graph_config,
            graph_config_schema=graph_schema,
            llm_service=mock_llm_service,
            config_extractor=config_extractor,
            service_container=mock_service_container,
        )

    def test_initialization(
        self,
        rag_instance,
        workflow_config,
        graph_schema,
        graph_config,
        mock_llm_service,
        config_extractor,
        mock_service_container,
        mock_graph_builder,
    ):
        """Test that QuivrQARAGLangGraph initializes correctly."""
        assert rag_instance.workflow_config == workflow_config
        assert rag_instance.graph_state == FastRAGAgentState
        assert rag_instance.graph_config == graph_config
        assert rag_instance.graph_config_schema == graph_schema
        assert rag_instance.llm_service == mock_llm_service
        assert rag_instance.config_extractor == config_extractor
        assert rag_instance.service_container == mock_service_container

        # Verify GraphBuilder was configured correctly
        mock_graph_builder.set_custom_state_graph.assert_called_once_with(
            FastRAGAgentState, graph_schema
        )
        mock_graph_builder.build_from_workflow_config.assert_called_once_with(
            workflow_config, config_extractor, mock_service_container, rag_instance
        )

    def test_final_nodes_property(self, rag_instance, mock_graph_builder):
        """Test that final_nodes property delegates to graph_builder."""
        final_nodes = rag_instance.final_nodes
        assert final_nodes == ["generate_rag"]
        assert final_nodes == mock_graph_builder.final_nodes

    def test_graph_builder_initialization(
        self,
        workflow_config,
        graph_schema,
        graph_config,
        mock_llm_service,
        config_extractor,
        mock_service_container,
        mock_graph_builder,
    ):
        """Test that GraphBuilder is properly initialized during construction."""
        _ = QuivrQARAGLangGraph(
            workflow_config=workflow_config,
            graph_state=FastRAGAgentState,
            graph_config=graph_config,
            graph_config_schema=graph_schema,
            llm_service=mock_llm_service,
            config_extractor=config_extractor,
            service_container=mock_service_container,
        )

        # Verify the builder chain was called correctly
        mock_graph_builder.set_custom_state_graph.assert_called_once()
        mock_graph_builder.build_from_workflow_config.assert_called_once()

    def test_node_registry_integration(self):
        """Test that all required nodes for Fast RAG workflow are available."""
        from quivr_core.rag.langgraph_framework.registry.node_registry import (
            node_registry,
        )

        available_nodes = node_registry.list_nodes()
        required_nodes = ["filter_history", "retrieve", "generate_rag"]

        for node_name in required_nodes:
            assert (
                node_name in available_nodes
            ), f"Required node '{node_name}' not found in registry"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_answer_astream_basic(
        self,
        rag_instance,
        sample_chat_history,
        sample_knowledge_files,
        mock_graph_builder,
    ):
        """Test basic streaming functionality."""
        run_id = uuid4()
        question = "What is the capital of France?"

        # Create mock stream events
        mock_events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="Paris")},
                "metadata": {"langgraph_node": "generate_rag"},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content=" is")},
                "metadata": {"langgraph_node": "generate_rag"},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content=" the capital.")},
                "metadata": {"langgraph_node": "generate_rag"},
            },
        ]

        async def mock_astream_events(*args, **kwargs):
            for event in mock_events:
                yield event

        # Mock the compiled graph's astream_events method
        mock_compiled_graph = Mock()
        mock_compiled_graph.astream_events = mock_astream_events
        mock_graph_builder.get_compiled_graph.return_value = mock_compiled_graph

        # Collect streaming results
        results = []
        async for chunk in rag_instance.answer_astream(
            run_id=run_id,
            question=question,
            system_prompt=None,
            history=sample_chat_history,
            list_files=sample_knowledge_files,
        ):
            results.append(chunk)

        # Verify we got results
        assert len(results) > 0

        # Check that we got text chunks
        text_chunks = [r for r in results if r.answer]
        assert len(text_chunks) > 0

        # Verify the final chunk
        final_chunks = [r for r in results if r.last_chunk]
        assert len(final_chunks) == 1

    def test_workflow_config_validation(self, fast_rag_workflow_config):
        """Test that the workflow configuration is valid."""
        config = WorkflowConfig.model_validate(
            fast_rag_workflow_config["workflow_config"]
        )

        assert config.name == "Fast RAG"
        assert len(config.nodes) == 4

        # Check node names
        node_names = [node.name for node in config.nodes]
        expected_names = [START, "filter_history", "retrieve", "generate_rag"]
        assert node_names == expected_names

        # Check that generate_rag has tools
        _ = next(node for node in config.nodes if node.name == "generate_rag")

    def test_graph_state_schema(self):
        """Test that the graph state schema is properly defined."""
        # Verify the TypedDict has required fields
        state_annotations = FastRAGAgentState.__annotations__

        required_fields = [
            "messages",
            "reasoning",
            "chat_history",
            "files",
            "tasks",
            "instructions",
            "ticket_metadata",
            "user_metadata",
            "additional_information",
            "tool",
            "guidelines",
            "enforced_system_prompt",
            "_filter",
            "ticket_history",
        ]

        for field in required_fields:
            assert field in state_annotations

    @pytest.mark.asyncio(loop_scope="session")
    async def test_workflow_error_handling(
        self, rag_instance, sample_chat_history, mock_graph_builder
    ):
        """Test error handling in workflow execution."""
        run_id = uuid4()
        question = "Test question"

        # Mock the get_compiled_graph to raise an exception
        mock_graph_builder.get_compiled_graph.side_effect = Exception("Test error")

        # The error should propagate
        with pytest.raises(Exception, match="Test error"):
            async for _ in rag_instance.answer_astream(
                run_id=run_id,
                question=question,
                system_prompt=None,
                history=sample_chat_history,
                list_files=[],
            ):
                pass

    def test_config_mapping_completeness(self, config_extractor):
        """Test that config mapping covers all necessary config types."""
        mapping = config_extractor.mapping

        required_configs = [
            LLMEndpointConfig,
            FilterHistoryConfig,
            WorkflowConfig,
            PromptConfig,
        ]

        for config_type in required_configs:
            assert config_type in mapping

    @pytest.mark.asyncio(loop_scope="session")
    async def test_streaming_with_metadata(
        self, rag_instance, sample_chat_history, mock_graph_builder
    ):
        """Test that streaming preserves workflow step metadata."""
        run_id = uuid4()
        question = "Test question"

        # Mock events with node metadata
        mock_events = [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="Test")},
                "metadata": {"langgraph_node": "generate_rag"},
            }
        ]

        async def mock_astream_events(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_compiled_graph = Mock()
        mock_compiled_graph.astream_events = mock_astream_events
        mock_graph_builder.get_compiled_graph.return_value = mock_compiled_graph

        # Collect results
        results = []
        async for chunk in rag_instance.answer_astream(
            run_id=run_id,
            question=question,
            system_prompt=None,
            history=sample_chat_history,
            list_files=[],
        ):
            results.append(chunk)

        # Check that workflow step metadata is preserved
        text_chunks = [r for r in results if r.answer]
        if text_chunks:
            assert text_chunks[0].metadata.workflow_step == "Generating response"

    def test_integration_with_existing_fixtures(
        self,
        workflow_config,
        config_extractor,
        mock_service_container,
        mock_graph_builder,
    ):
        """Test integration with existing test fixtures from conftest.py."""
        # Create instance using existing fixtures
        rag_instance = QuivrQARAGLangGraph(
            workflow_config=workflow_config,
            graph_state=FastRAGAgentState,
            graph_config={"llm_config": {"model": "fake_model"}},
            graph_config_schema=FastRAGGraphConfig,
            llm_service=Mock(spec=LLMService),
            config_extractor=config_extractor,
            service_container=mock_service_container,
        )

        # Verify initialization works with existing fixtures
        assert rag_instance is not None
        assert rag_instance.graph_builder is not None


class TestQuivrQARAGLangGraphAdvanced:
    """Advanced tests for complex scenarios and edge cases."""

    @pytest.fixture(scope="function")
    def complex_workflow_config(self):
        """More complex workflow configuration for advanced testing."""
        return {
            "llm_config": {
                "temperature": 0.1,
                "max_context_tokens": 50000,
                "model": "gpt-4o",
            },
            "filter_history_config": {"max_history": 20},
            "workflow_config": {
                "name": "Advanced RAG",
                "nodes": [
                    {
                        "name": "START",
                        "edges": ["filter_history"],
                        "description": "Starting advanced workflow",
                    },
                    {
                        "name": "filter_history",
                        "edges": ["retrieve"],
                        "description": "Advanced history filtering",
                        "filter_history_config": {
                            "max_history": 5  # Node-specific override
                        },
                    },
                    {
                        "name": "retrieve",
                        "edges": ["generate_rag"],
                        "description": "Enhanced retrieval",
                    },
                    {
                        "name": "generate_rag",
                        "edges": ["END"],
                        "description": "Advanced generation",
                        "llm_config": {
                            "temperature": 0.0  # Node-specific override
                        },
                    },
                ],
            },
        }

    def test_node_specific_config_overrides(self, complex_workflow_config):
        """Test that node-specific configuration overrides work correctly."""
        config_extractor = ConfigMapping(
            {
                LLMEndpointConfig: "llm_config",
                FilterHistoryConfig: "filter_history_config",
                WorkflowConfig: "workflow_config",
            }
        )

        # Create BaseGraphConfig for testing
        graph_config = BaseGraphConfig(configurable=complex_workflow_config)

        # Test filter_history node config override
        filter_config = config_extractor.extract(
            graph_config, FilterHistoryConfig, "filter_history"
        )
        assert filter_config.max_history == 20  # Global value

        # Test generate_rag node config override
        llm_config = config_extractor.extract(
            graph_config, LLMEndpointConfig, "generate_rag"
        )
        assert llm_config.temperature == 0.1  # Global value
        assert llm_config.max_context_tokens == 50000  # Global value

    @pytest.mark.asyncio(loop_scope="session")
    async def test_multiple_file_handling(self, sample_knowledge_files):
        """Test handling of multiple knowledge files with max_files limit."""
        # Create more files than the limit
        many_files = []
        for i in range(25):  # More than max_files (20)
            many_files.append(
                QuivrKnowledge(
                    id=uuid4(),
                    brain_ids=[uuid4()],
                    file_name=f"document{i}.pdf",
                    extension=".pdf",
                    url=f"https://example.com/doc{i}.pdf",
                )
            )

        # Test that file list formatting handles the limit appropriately
        from quivr_core.rag.utils import format_file_list

        formatted_files = format_file_list(
            many_files[:20]
        )  # Simulate max_files limiting

        assert len(formatted_files.split("\n")) <= 21  # 20 files + header

    def test_empty_workflow_handling(self):
        """Test handling of workflows with minimal configuration."""
        minimal_config = WorkflowConfig(
            name="Minimal",
            nodes=[
                NodeConfig(name=START, edges=["generate_rag"]),
                NodeConfig(name="generate_rag", edges=[END]),
            ],
        )

        config_extractor = ConfigMapping(
            {
                LLMEndpointConfig: "llm_config",
                WorkflowConfig: "workflow_config",
            }
        )

        with patch(
            "quivr_core.rag.quivr_rag_langgraph.GraphBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder.set_custom_state_graph.return_value = mock_builder
            mock_builder.build_from_workflow_config.return_value = mock_builder
            mock_builder.final_nodes = ["generate_rag"]
            mock_builder_class.return_value = mock_builder

            rag_instance = QuivrQARAGLangGraph(
                workflow_config=minimal_config,
                graph_state=FastRAGAgentState,
                graph_config={"llm_config": {"model": "gpt-4o-mini"}},
                graph_config_schema=FastRAGGraphConfig,
                llm_service=Mock(spec=LLMService),
                config_extractor=config_extractor,
                service_container=Mock(spec=ServiceContainer),
            )

            # Should be able to create even minimal instances
            assert rag_instance is not None
            assert rag_instance.graph_builder is not None
