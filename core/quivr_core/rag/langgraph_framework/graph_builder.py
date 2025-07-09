from typing import Dict, Any, List, Optional, Type
from langgraph.graph import StateGraph, END, START
import logging

from quivr_core.rag.langgraph_framework.registry.node_registry import (
    node_registry,
    NodeRegistry,
)
from quivr_core.rag.langgraph_framework.base.node import BaseNode
from quivr_core.rag.langgraph_framework.state import AgentState
from quivr_core.rag.entities.config import WorkflowConfig, NodeConfig
from quivr_core.rag.langgraph_framework.base.extractors import ConfigMapping
from quivr_core.rag.langgraph_framework.services.service_container import (
    ServiceContainer,
)
from quivr_core.base_config import QuivrBaseConfig

logger = logging.getLogger("quivr_core")


class GraphBuilder:
    """Builder for creating LangGraph workflows using registered nodes."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or node_registry
        self.graph = StateGraph(AgentState)
        self.nodes: Dict[str, BaseNode] = {}
        self.final_nodes: List[str] = []
        self.compiled_graph = None

    def set_custom_state_graph(self, graph_state, config_schema: Type[QuivrBaseConfig]):
        """Override the default graph with custom state and config schema."""
        self.graph = StateGraph(graph_state, config_schema=config_schema)
        return self

    def build_from_workflow_config(
        self,
        workflow_config: WorkflowConfig,
        config_extractor: ConfigMapping,
        service_container: ServiceContainer,
        routing_functions_provider: Any,  # Object that provides routing functions via getattr
    ) -> "GraphBuilder":
        """Build the entire workflow from a WorkflowConfig."""
        # Log available nodes
        available_nodes = self.registry.list_nodes()
        logger.info(f"Available nodes: {available_nodes}")

        if not available_nodes:
            raise RuntimeError(
                "No nodes found in registry. Make sure to import "
                "quivr_core.rag.langgraph_framework.nodes before using this class."
            )

        # Add all nodes from the workflow config
        for node in workflow_config.nodes:
            if node.name not in [START, END]:
                self._add_node_from_config(node, config_extractor, service_container)

        # Add edges from the workflow config
        for node in workflow_config.nodes:
            self._add_node_edges_from_config(node, routing_functions_provider)

        return self

    def _add_node_from_config(
        self,
        node: NodeConfig,
        config_extractor: ConfigMapping,
        service_container: ServiceContainer,
    ):
        """Add a node from workflow configuration."""
        try:
            node_instance = self.registry.create_node(
                node.name,
                config_extractor=config_extractor,
                service_container=service_container,
            )
            self.graph.add_node(node.name, node_instance)
            self.nodes[node.name] = node_instance
            logger.info(f"Added node '{node.name}' from registry")
        except KeyError:
            available_nodes = self.registry.list_nodes()
            raise ValueError(
                f"Node '{node.name}' not found in registry. Available nodes: {available_nodes}"
            )

    def _add_node_edges_from_config(
        self, node: NodeConfig, routing_functions_provider: Any
    ):
        """Add node edges from workflow configuration."""
        if node.edges:
            for edge in node.edges:
                self.graph.add_edge(node.name, edge)
                if edge == END:
                    self.final_nodes.append(node.name)
        elif node.conditional_edge:
            routing_function = getattr(
                routing_functions_provider, node.conditional_edge.routing_function
            )
            self.graph.add_conditional_edges(
                node.name, routing_function, node.conditional_edge.conditions
            )
            # Check if END is in conditions (handles both dict and list formats)
            conditions = node.conditional_edge.conditions
            if isinstance(conditions, dict):
                if END in conditions.values():
                    self.final_nodes.append(node.name)
            elif isinstance(conditions, list):
                if END in conditions:
                    self.final_nodes.append(node.name)
        else:
            raise ValueError("Node should have at least one edge or conditional_edge")

    def get_compiled_graph(self):
        """Build and return the compiled graph."""
        if not self.compiled_graph:
            self.compiled_graph = self.graph.compile()
        return self.compiled_graph

    def list_available_nodes(self) -> Dict[str, List[str]]:
        """List all available node types by category."""
        result = {}
        for category in self.registry.list_categories():
            result[category] = self.registry.list_nodes(category)
        return result
