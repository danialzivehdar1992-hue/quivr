from uuid import UUID
from typing import AsyncGenerator, Any, Type


from quivr_core.base_config import QuivrBaseConfig
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.models import ParsedRAGChunkResponse, QuivrKnowledge
from langchain.schema.messages import AIMessageChunk
from langchain_core.documents import Document

from quivr_core.rag.langgraph_framework.services.llm_service import LLMService
from quivr_core.rag.entities.models import (
    RAGResponseMetadata,
)
from quivr_core.rag.entities.config import (
    WorkflowConfig,
)

import quivr_core.rag.langgraph_framework.nodes as _  # noqa: F401

from quivr_core.rag.langgraph_framework.graph_builder import GraphBuilder
from quivr_core.rag.langgraph_framework.registry.node_registry import node_registry
from quivr_core.rag.langgraph_framework.services.service_container import (
    ServiceContainer,
)
from quivr_core.rag.utils import (
    LangfuseService,
    format_file_list,
    get_chunk_metadata,
    parse_chunk_response,
    is_final_node_with_docs,
    is_final_node_and_chat_model_stream,
    extract_node_name,
)
from quivr_core.rag.langgraph_framework.base.extractors import ConfigMapping

import logging

logger = logging.getLogger("quivr_core")

langfuse_service = LangfuseService()
langfuse_handler = langfuse_service.get_handler()


class QuivrQARAGLangGraph:
    def __init__(
        self,
        workflow_config: WorkflowConfig,
        graph_state,
        graph_config: dict[str, Any],
        graph_config_schema: Type[QuivrBaseConfig],
        llm_service: LLMService,
        config_extractor: ConfigMapping,
        service_container: ServiceContainer,
    ):
        self.workflow_config = workflow_config
        self.graph_state = graph_state
        self.graph_config = graph_config
        self.graph_config_schema = graph_config_schema
        self.llm_service = llm_service
        self.config_extractor = config_extractor
        self.service_container = service_container

        # Initialize GraphBuilder and build the workflow
        self.graph_builder = (
            GraphBuilder(registry=node_registry)
            .set_custom_state_graph(self.graph_state, self.graph_config_schema)
            .build_from_workflow_config(
                workflow_config, config_extractor, service_container, self
            )
        )

    @property
    def final_nodes(self) -> list[str]:
        """Get final nodes from the graph builder."""
        return self.graph_builder.final_nodes

    async def answer_astream(
        self,
        run_id: UUID,
        question: str,
        system_prompt: str | None,
        history: ChatHistory,
        list_files: list[QuivrKnowledge],
        metadata: dict[str, str] = {},
        **input_kwargs,
    ) -> AsyncGenerator[ParsedRAGChunkResponse, ParsedRAGChunkResponse]:
        """
        Answer a question using the langgraph chain and yield each chunk of the answer separately.
        """
        concat_list_files = format_file_list(list_files)
        conversational_qa_chain = self.graph_builder.get_compiled_graph()

        rolling_message = AIMessageChunk(content="")
        docs: list[Document] | None = None
        previous_content = ""
        system_prompt = system_prompt
        messages = [("system", system_prompt)] if system_prompt else []
        messages.append(("user", question))

        async for event in conversational_qa_chain.astream_events(
            {
                "messages": messages,
                "chat_history": history,
                "files": concat_list_files,
                **input_kwargs,
            },
            version="v1",
            config={
                "configurable": self.graph_config,
                "run_id": run_id,
                "metadata": metadata,
                "callbacks": [langfuse_handler],
            },
        ):
            node_name = extract_node_name(self.workflow_config.nodes, event)

            if is_final_node_with_docs(self.final_nodes, event):
                event_data = event.get("data", {})
                if "output" in event_data and event_data["output"]:
                    tasks = event_data["output"].get("tasks")
                    docs = tasks.docs if tasks else []

            if is_final_node_and_chat_model_stream(self.final_nodes, event):
                event_data = event.get("data", {})
                if "chunk" in event_data:
                    chunk = event_data["chunk"]
                    rolling_message, new_content, previous_content = (
                        parse_chunk_response(
                            rolling_message,
                            chunk,
                            self.llm_service.supports_function_calling(),
                            previous_content,
                        )
                    )

                    if new_content:
                        chunk_metadata = get_chunk_metadata(rolling_message, docs)
                        if node_name:
                            chunk_metadata.workflow_step = node_name
                        yield ParsedRAGChunkResponse(
                            answer=new_content, metadata=chunk_metadata
                        )
            else:
                if node_name:
                    yield ParsedRAGChunkResponse(
                        answer="",
                        metadata=RAGResponseMetadata(workflow_step=node_name),
                    )

        # Yield final metadata chunk
        yield ParsedRAGChunkResponse(
            answer="",
            metadata=get_chunk_metadata(rolling_message, docs),
            last_chunk=True,
        )
