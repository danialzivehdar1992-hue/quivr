"""Tests for the GraphBuilder class."""

import pytest
from unittest.mock import Mock, patch
from langgraph.graph import StateGraph, END, START

from quivr_core.rag.langgraph_framework.graph_builder import GraphBuilder
from quivr_core.rag.langgraph_framework.registry.node_registry import NodeRegistry
from quivr_core.rag.entities.config import (
    WorkflowConfig,
    NodeConfig,
    ConditionalEdgeConfig,
)
from quivr_core.rag.langgraph_framework.base.extractors import ConfigMapping
from quivr_core.rag.langgraph_framework.services.service_container import (
    ServiceContainer,
)
from quivr_core.base_config import QuivrBaseConfig

from tests.rag.langgraph_framework.fixtures.mock_nodes import MockNode


class TestGraphBuilderInitialization:
    """Test GraphBuilder initialization."""

    def test_initialization_with_default_registry(self):
        """Test initialization with default registry."""
        builder = GraphBuilder()

        assert builder.registry is not None
        assert isinstance(builder.graph, StateGraph)
        assert builder.nodes == {}
        assert builder.final_nodes == []
        assert builder.compiled_graph is None

    def test_initialization_with_custom_registry(self):
        """Test initialization with custom registry."""
        custom_registry = Mock(spec=NodeRegistry)
        builder = GraphBuilder(registry=custom_registry)

        assert builder.registry is custom_registry
        assert isinstance(builder.graph, StateGraph)
        assert builder.nodes == {}
        assert builder.final_nodes == []
        assert builder.compiled_graph is None


class TestSetCustomStateGraph:
    """Test custom state graph functionality."""

    def test_set_custom_state_graph(self):
        """Test setting custom state graph with config schema."""
        builder = GraphBuilder()

        # Mock graph state and config schema
        mock_graph_state = Mock()
        mock_config_schema = Mock(spec=QuivrBaseConfig)

        with patch(
            "quivr_core.rag.langgraph_framework.graph_builder.StateGraph"
        ) as mock_state_graph:
            mock_graph_instance = Mock()
            mock_state_graph.return_value = mock_graph_instance

            result = builder.set_custom_state_graph(
                mock_graph_state, mock_config_schema
            )

            # Should return self for chaining
            assert result is builder

            # Should create new graph with custom parameters
            mock_state_graph.assert_called_once_with(
                mock_graph_state, config_schema=mock_config_schema
            )
            assert builder.graph is mock_graph_instance


class TestBuildFromWorkflowConfig:
    """Test building workflow from configuration."""

    @pytest.fixture(scope="function")
    def mock_registry(self):
        """Create a mock registry."""
        registry = Mock(spec=NodeRegistry)
        registry.list_nodes.return_value = ["test_node", "another_node"]
        registry.create_node.return_value = MockNode()
        return registry

    @pytest.fixture(scope="function")
    def mock_config_extractor(self):
        """Create a mock config extractor."""
        return Mock(spec=ConfigMapping)

    @pytest.fixture(scope="function")
    def mock_service_container(self):
        """Create a mock service container."""
        return Mock(spec=ServiceContainer)

    @pytest.fixture(scope="function")
    def mock_routing_provider(self):
        """Create a mock routing functions provider."""
        provider = Mock()
        provider.route_to_end = Mock(return_value=END)
        provider.route_to_node = Mock(return_value="next_node")
        return provider

    @pytest.fixture(scope="function")
    def sample_workflow_config(self):
        """Create a sample workflow configuration."""
        nodes = [
            NodeConfig(name=START, description="Starting node", edges=["middle_node"]),
            NodeConfig(
                name="middle_node",
                description="Middle node",
                conditional_edge=ConditionalEdgeConfig(
                    routing_function="route_to_end", conditions={"end": END}
                ),
            ),
        ]
        return WorkflowConfig(name="test_workflow", nodes=nodes)

    def test_build_from_workflow_config_success(
        self,
        mock_registry,
        mock_config_extractor,
        mock_service_container,
        mock_routing_provider,
        sample_workflow_config,
    ):
        """Test successful workflow building from config."""
        builder = GraphBuilder(registry=mock_registry)

        result = builder.build_from_workflow_config(
            sample_workflow_config,
            mock_config_extractor,
            mock_service_container,
            mock_routing_provider,
        )

        # Should return self for chaining
        assert result is builder

        # Should have created nodes (excluding START and END nodes)
        assert (
            len(builder.nodes) == 1
        )  # Only middle_node gets created, START is special
        assert "middle_node" in builder.nodes

        # Should have called registry.create_node for non-START/END nodes
        assert mock_registry.create_node.call_count == 1

    def test_build_from_workflow_config_empty_registry(
        self,
        mock_config_extractor,
        mock_service_container,
        mock_routing_provider,
        sample_workflow_config,
    ):
        """Test error when registry is empty."""
        empty_registry = Mock(spec=NodeRegistry)
        empty_registry.list_nodes.return_value = []

        builder = GraphBuilder(registry=empty_registry)

        with pytest.raises(RuntimeError, match="No nodes found in registry"):
            builder.build_from_workflow_config(
                sample_workflow_config,
                mock_config_extractor,
                mock_service_container,
                mock_routing_provider,
            )

    def test_build_from_workflow_config_node_not_found(
        self,
        mock_registry,
        mock_config_extractor,
        mock_service_container,
        mock_routing_provider,
    ):
        """Test error when node type not found in registry."""
        mock_registry.create_node.side_effect = KeyError("Node not found")
        mock_registry.list_nodes.return_value = ["valid_node"]

        builder = GraphBuilder(registry=mock_registry)

        invalid_config = WorkflowConfig(
            name="test",
            nodes=[
                NodeConfig(name=START, edges=["invalid_node"]),
                NodeConfig(name="invalid_node", edges=[END]),
            ],
        )

        with pytest.raises(
            ValueError, match="Node 'invalid_node' not found in registry"
        ):
            builder.build_from_workflow_config(
                invalid_config,
                mock_config_extractor,
                mock_service_container,
                mock_routing_provider,
            )


class TestNodeEdgeConfiguration:
    """Test node and edge configuration from workflow config."""

    @pytest.fixture(scope="function")
    def builder_with_mocks(self):
        """Create builder with mocked dependencies."""
        mock_registry = Mock(spec=NodeRegistry)
        mock_registry.list_nodes.return_value = ["test_node"]
        mock_registry.create_node.return_value = MockNode()

        builder = GraphBuilder(registry=mock_registry)

        # Mock the graph methods
        builder.graph.add_node = Mock()
        builder.graph.add_edge = Mock()
        builder.graph.add_conditional_edges = Mock()

        return builder, mock_registry

    def test_add_node_edges_simple_edge(self, builder_with_mocks):
        """Test adding simple edges from node config."""
        builder, mock_registry = builder_with_mocks

        mock_config_extractor = Mock(spec=ConfigMapping)
        mock_service_container = Mock(spec=ServiceContainer)
        mock_routing_provider = Mock()

        workflow_config = WorkflowConfig(
            nodes=[
                NodeConfig(name=START, edges=["test_node"]),
                NodeConfig(name="test_node", edges=["next_node"]),
            ]
        )

        builder.build_from_workflow_config(
            workflow_config,
            mock_config_extractor,
            mock_service_container,
            mock_routing_provider,
        )

        # Should add edge to graph
        builder.graph.add_edge.assert_any_call("test_node", "next_node")

    def test_add_node_edges_to_end(self, builder_with_mocks):
        """Test adding edge to END."""
        builder, mock_registry = builder_with_mocks

        mock_config_extractor = Mock(spec=ConfigMapping)
        mock_service_container = Mock(spec=ServiceContainer)
        mock_routing_provider = Mock()

        workflow_config = WorkflowConfig(
            nodes=[
                NodeConfig(name=START, edges=["test_node"]),
                NodeConfig(name="test_node", edges=[END]),
            ]
        )

        builder.build_from_workflow_config(
            workflow_config,
            mock_config_extractor,
            mock_service_container,
            mock_routing_provider,
        )

        # Should add edge to END and mark as final node
        builder.graph.add_edge.assert_any_call("test_node", END)
        assert "test_node" in builder.final_nodes

    def test_add_conditional_edge(self, builder_with_mocks):
        """Test adding conditional edges from node config."""
        builder, mock_registry = builder_with_mocks

        mock_config_extractor = Mock(spec=ConfigMapping)
        mock_service_container = Mock(spec=ServiceContainer)
        mock_routing_provider = Mock()
        mock_routing_provider.test_routing = Mock(return_value="next_node")

        conditional_edge = ConditionalEdgeConfig(
            routing_function="test_routing",
            conditions={"route1": "node1", "route2": "node2"},
        )
        workflow_config = WorkflowConfig(
            nodes=[
                NodeConfig(name=START, edges=["test_node"]),
                NodeConfig(name="test_node", conditional_edge=conditional_edge),
            ]
        )

        builder.build_from_workflow_config(
            workflow_config,
            mock_config_extractor,
            mock_service_container,
            mock_routing_provider,
        )

        # Should add conditional edges to graph
        builder.graph.add_conditional_edges.assert_called_once()
        args = builder.graph.add_conditional_edges.call_args[0]
        assert args[0] == "test_node"
        assert args[1] is mock_routing_provider.test_routing
        assert args[2] == {"route1": "node1", "route2": "node2"}

    def test_node_without_edges_raises_error(self, builder_with_mocks):
        """Test that node without edges or conditional_edge raises error."""
        builder, mock_registry = builder_with_mocks

        mock_config_extractor = Mock(spec=ConfigMapping)
        mock_service_container = Mock(spec=ServiceContainer)
        mock_routing_provider = Mock()

        # Node with neither edges nor conditional_edge
        workflow_config = WorkflowConfig(
            nodes=[
                NodeConfig(name=START, edges=["test_node"]),
                NodeConfig(name="test_node"),  # No edges or conditional_edge
            ]
        )

        with pytest.raises(
            ValueError, match="Node should have at least one edge or conditional_edge"
        ):
            builder.build_from_workflow_config(
                workflow_config,
                mock_config_extractor,
                mock_service_container,
                mock_routing_provider,
            )


class TestGraphCompilation:
    """Test graph compilation functionality."""

    def test_get_compiled_graph_first_time(self):
        """Test compiling graph for the first time."""
        builder = GraphBuilder()

        mock_compiled_graph = Mock()
        with patch.object(builder.graph, "compile", return_value=mock_compiled_graph):
            result = builder.get_compiled_graph()

            assert result is mock_compiled_graph
            assert builder.compiled_graph is mock_compiled_graph
            builder.graph.compile.assert_called_once()

    def test_get_compiled_graph_cached(self):
        """Test that compiled graph is cached."""
        builder = GraphBuilder()

        # Set up cached compiled graph
        mock_cached_graph = Mock()
        builder.compiled_graph = mock_cached_graph

        with patch.object(builder.graph, "compile") as mock_compile:
            result = builder.get_compiled_graph()

            assert result is mock_cached_graph
            mock_compile.assert_not_called()


class TestListAvailableNodes:
    """Test listing available nodes functionality."""

    def test_list_available_nodes(self):
        """Test listing nodes by category."""
        mock_registry = Mock(spec=NodeRegistry)
        mock_registry.list_categories.return_value = ["category1", "category2"]
        mock_registry.list_nodes.side_effect = lambda cat: {
            "category1": ["node1", "node2"],
            "category2": ["node3"],
        }[cat]

        builder = GraphBuilder(registry=mock_registry)
        result = builder.list_available_nodes()

        expected = {"category1": ["node1", "node2"], "category2": ["node3"]}
        assert result == expected

        mock_registry.list_categories.assert_called_once()
        assert mock_registry.list_nodes.call_count == 2


class TestWorkflowBuildingIntegration:
    """Integration tests for complete workflow building."""

    def test_build_complete_workflow(self):
        """Test building a complete workflow end-to-end."""
        # Setup mocks
        mock_registry = Mock(spec=NodeRegistry)
        mock_registry.list_nodes.return_value = ["retrieve", "generate"]
        mock_registry.create_node.return_value = MockNode()

        mock_config_extractor = Mock(spec=ConfigMapping)
        mock_service_container = Mock(spec=ServiceContainer)
        mock_routing_provider = Mock()

        # Create workflow config
        workflow_config = WorkflowConfig(
            name="rag_workflow",
            nodes=[
                NodeConfig(name=START, edges=["retrieve"]),
                NodeConfig(name="retrieve", edges=["generate"]),
                NodeConfig(name="generate", edges=[END]),
            ],
        )

        # Build workflow
        builder = GraphBuilder(registry=mock_registry)

        with patch.object(builder.graph, "add_node") as mock_add_node, patch.object(
            builder.graph, "add_edge"
        ) as mock_add_edge:
            result = builder.build_from_workflow_config(
                workflow_config,
                mock_config_extractor,
                mock_service_container,
                mock_routing_provider,
            )

            # Verify builder state
            assert result is builder
            assert (
                len(builder.nodes) == 2
            )  # retrieve and generate (START/END are special)
            assert "retrieve" in builder.nodes
            assert "generate" in builder.nodes
            assert "generate" in builder.final_nodes

            # Verify graph construction calls
            assert mock_add_node.call_count == 2  # Only non-START/END nodes
            assert (
                mock_add_edge.call_count == 3
            )  # START->retrieve, retrieve->generate, generate->END
