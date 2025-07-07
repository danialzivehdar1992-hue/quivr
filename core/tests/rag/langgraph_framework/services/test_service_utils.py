import pytest
from unittest.mock import Mock, patch
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_compressors import JinaRerank

from quivr_core.rag.langgraph_framework.services.utils import (
    get_retriever,
    get_reranker,
    get_compression_retriever,
    filter_chunks_by_relevance,
    sort_docs_by_relevance,
)
from quivr_core.rag.entities.retriever import RetrieverConfig
from quivr_core.rag.entities.reranker import RerankerConfig, DefaultRerankers


class TestGetRetriever:
    """Test suite for get_retriever function"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        mock_store = Mock(spec=VectorStore)
        mock_retriever = Mock(spec=VectorStoreRetriever)
        mock_store.as_retriever.return_value = mock_retriever
        return mock_store

    @pytest.fixture
    def retriever_config(self):
        """Create a basic retriever config"""
        return RetrieverConfig(k=5, filter={"source": "test.pdf"}, max_chunk_sum=1000)

    def test_get_retriever_basic(self, mock_vector_store, retriever_config):
        """Test basic retriever creation"""
        result = get_retriever(mock_vector_store, retriever_config)

        assert result == mock_vector_store.as_retriever.return_value

        # Check that as_retriever was called with correct config
        expected_config = {
            "search_kwargs": {
                "k": 5,
                "filter": {"source": "test.pdf"},
                "max_chunk_sum": 1000,
            }
        }
        mock_vector_store.as_retriever.assert_called_once_with(**expected_config)

    def test_get_retriever_excludes_extra_config(self, mock_vector_store):
        """Test that extra_config is excluded from search_kwargs"""
        config = RetrieverConfig(
            k=10,
            filter={"type": "pdf"},
            max_chunk_sum=2000,
            extra_config={"top_n_knowledge": 3, "dynamic_retrieval_max_iterations": 5},
        )

        get_retriever(mock_vector_store, config)

        # Verify extra_config is not included in search_kwargs
        call_args = mock_vector_store.as_retriever.call_args
        search_kwargs = call_args[1]["search_kwargs"]

        assert "extra_config" not in search_kwargs
        assert search_kwargs["k"] == 10
        assert search_kwargs["filter"] == {"type": "pdf"}
        assert search_kwargs["max_chunk_sum"] == 2000

    def test_get_retriever_empty_config(self, mock_vector_store):
        """Test retriever creation with default config"""
        config = RetrieverConfig()  # Uses default values

        get_retriever(mock_vector_store, config)

        call_args = mock_vector_store.as_retriever.call_args
        search_kwargs = call_args[1]["search_kwargs"]

        assert search_kwargs["k"] == 40  # Default value
        assert search_kwargs["filter"] is None  # Default value
        assert search_kwargs["max_chunk_sum"] == 10000  # Default value

    def test_get_retriever_with_none_filter(self, mock_vector_store):
        """Test retriever creation with None filter"""
        config = RetrieverConfig(k=5, filter=None, max_chunk_sum=1000)

        get_retriever(mock_vector_store, config)

        call_args = mock_vector_store.as_retriever.call_args
        search_kwargs = call_args[1]["search_kwargs"]

        assert search_kwargs["filter"] is None


class TestGetReranker:
    """Test suite for get_reranker function"""

    def test_get_reranker_cohere(self):
        """Test creating Cohere reranker"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE,
            model="rerank-v3.5",
            top_n=5,
            cohere_api_key="test_cohere_key",
        )

        with patch(
            "quivr_core.rag.langgraph_framework.services.utils.CohereRerank"
        ) as mock_cohere:
            mock_instance = Mock(spec=CohereRerank)
            mock_cohere.return_value = mock_instance

            result = get_reranker(config)

            assert result == mock_instance
            mock_cohere.assert_called_once_with(
                model="rerank-v3.5", top_n=5, cohere_api_key="test_cohere_key"
            )

    def test_get_reranker_jina(self):
        """Test creating Jina reranker"""
        config = RerankerConfig(
            supplier=DefaultRerankers.JINA,
            model="jina-reranker-v2-base-multilingual",
            top_n=3,
            jina_api_key="test_jina_key",
        )

        with patch(
            "quivr_core.rag.langgraph_framework.services.utils.JinaRerank"
        ) as mock_jina:
            mock_instance = Mock(spec=JinaRerank)
            mock_jina.return_value = mock_instance

            result = get_reranker(config)

            assert result == mock_instance
            mock_jina.assert_called_once_with(
                model="jina-reranker-v2-base-multilingual",
                top_n=3,
                jina_api_key="test_jina_key",
            )

    def test_get_reranker_invalid_supplier(self):
        """Test get_reranker with invalid supplier"""
        config = RerankerConfig(
            supplier="invalid_supplier",  # This should raise ValidationError in practice
            model="test-model",
            top_n=5,
        )

        # Mock the config to bypass validation for testing
        config.supplier = "invalid_supplier"

        with pytest.raises(
            ValueError, match="Invalid reranker supplier: invalid_supplier"
        ):
            get_reranker(config)

    def test_get_reranker_cohere_with_api_key_property(self):
        """Test Cohere reranker using api_key property"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE,
            model="rerank-v3.5",
            top_n=5,
            cohere_api_key="property_key",
        )

        with patch(
            "quivr_core.rag.langgraph_framework.services.utils.CohereRerank"
        ) as mock_cohere:
            mock_instance = Mock(spec=CohereRerank)
            mock_cohere.return_value = mock_instance

            _ = get_reranker(config)

            # Verify api_key property is used
            mock_cohere.assert_called_once_with(
                model="rerank-v3.5", top_n=5, cohere_api_key=config.api_key
            )

    def test_get_reranker_jina_with_api_key_property(self):
        """Test Jina reranker using api_key property"""
        config = RerankerConfig(
            supplier=DefaultRerankers.JINA,
            model="jina-reranker-v2-base-multilingual",
            top_n=3,
            jina_api_key="property_key",
        )

        with patch(
            "quivr_core.rag.langgraph_framework.services.utils.JinaRerank"
        ) as mock_jina:
            mock_instance = Mock(spec=JinaRerank)
            mock_jina.return_value = mock_instance

            _ = get_reranker(config)

            # Verify api_key property is used
            mock_jina.assert_called_once_with(
                model="jina-reranker-v2-base-multilingual",
                top_n=3,
                jina_api_key=config.api_key,
            )


class TestGetCompressionRetriever:
    """Test suite for get_compression_retriever function"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        return Mock(spec=VectorStore)

    @pytest.fixture
    def retriever_config(self):
        """Create a retriever config"""
        return RetrieverConfig(k=5, filter={"source": "test.pdf"}, max_chunk_sum=1000)

    @pytest.fixture
    def reranker_config(self):
        """Create a reranker config"""
        return RerankerConfig(
            supplier=DefaultRerankers.COHERE,
            model="rerank-v3.5",
            top_n=3,
            cohere_api_key="test_key",
        )

    @patch("quivr_core.rag.langgraph_framework.services.utils.get_reranker")
    @patch("quivr_core.rag.langgraph_framework.services.utils.get_retriever")
    @patch(
        "quivr_core.rag.langgraph_framework.services.utils.ContextualCompressionRetriever"
    )
    def test_get_compression_retriever(
        self,
        mock_compression_class,
        mock_get_retriever,
        mock_get_reranker,
        mock_vector_store,
        retriever_config,
        reranker_config,
    ):
        """Test creating compression retriever"""
        mock_base_retriever = Mock()
        mock_reranker = Mock()
        mock_compression_retriever = Mock(spec=ContextualCompressionRetriever)

        mock_get_retriever.return_value = mock_base_retriever
        mock_get_reranker.return_value = mock_reranker
        mock_compression_class.return_value = mock_compression_retriever

        result = get_compression_retriever(
            mock_vector_store, retriever_config, reranker_config
        )

        assert result == mock_compression_retriever

        # Verify function calls
        mock_get_retriever.assert_called_once_with(mock_vector_store, retriever_config)
        mock_get_reranker.assert_called_once_with(reranker_config)
        mock_compression_class.assert_called_once_with(
            base_compressor=mock_reranker, base_retriever=mock_base_retriever
        )

    @patch("quivr_core.rag.langgraph_framework.services.utils.get_reranker")
    @patch("quivr_core.rag.langgraph_framework.services.utils.get_retriever")
    def test_get_compression_retriever_error_propagation(
        self,
        mock_get_retriever,
        mock_get_reranker,
        mock_vector_store,
        retriever_config,
        reranker_config,
    ):
        """Test error propagation in get_compression_retriever"""
        # Test error from get_retriever
        mock_get_retriever.side_effect = Exception("Retriever error")

        with pytest.raises(Exception, match="Retriever error"):
            get_compression_retriever(
                mock_vector_store, retriever_config, reranker_config
            )

        # Test error from get_reranker
        mock_get_retriever.side_effect = None
        mock_get_retriever.return_value = Mock()
        mock_get_reranker.side_effect = Exception("Reranker error")

        with pytest.raises(Exception, match="Reranker error"):
            get_compression_retriever(
                mock_vector_store, retriever_config, reranker_config
            )


class TestFilterChunksByRelevance:
    """Test suite for filter_chunks_by_relevance function"""

    def test_filter_chunks_no_threshold(self):
        """Test filtering with no threshold returns all chunks"""
        chunks = [
            Document(page_content="doc1", metadata={"relevance_score": 0.8}),
            Document(page_content="doc2", metadata={"relevance_score": 0.3}),
            Document(page_content="doc3", metadata={"relevance_score": 0.9}),
        ]

        result = filter_chunks_by_relevance(chunks, "relevance_score", None)

        assert result == chunks

    def test_filter_chunks_with_threshold(self):
        """Test filtering chunks by relevance threshold"""
        chunks = [
            Document(page_content="doc1", metadata={"relevance_score": 0.8}),
            Document(page_content="doc2", metadata={"relevance_score": 0.3}),
            Document(page_content="doc3", metadata={"relevance_score": 0.9}),
            Document(page_content="doc4", metadata={"relevance_score": 0.5}),
        ]

        result = filter_chunks_by_relevance(chunks, "relevance_score", 0.6)

        # Should only include docs with score >= 0.6
        assert len(result) == 2
        assert result[0].page_content == "doc1"
        assert result[1].page_content == "doc3"

    def test_filter_chunks_missing_relevance_score(self):
        """Test filtering when some chunks are missing relevance score"""
        chunks = [
            Document(page_content="doc1", metadata={"relevance_score": 0.8}),
            Document(page_content="doc2", metadata={}),  # Missing relevance score
            Document(page_content="doc3", metadata={"relevance_score": 0.9}),
        ]

        with patch(
            "quivr_core.rag.langgraph_framework.services.utils.logger"
        ) as mock_logger:
            result = filter_chunks_by_relevance(chunks, "relevance_score", 0.6)

            # Should include all docs (missing score gets included with warning)
            assert len(result) == 3
            mock_logger.warning.assert_called_once_with(
                "Relevance score key relevance_score not found in metadata, cannot filter chunks by relevance"
            )

    def test_filter_chunks_all_below_threshold(self):
        """Test filtering when all chunks are below threshold"""
        chunks = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.2}),
            Document(page_content="doc3", metadata={"relevance_score": 0.1}),
        ]

        result = filter_chunks_by_relevance(chunks, "relevance_score", 0.5)

        assert len(result) == 0

    def test_filter_chunks_all_above_threshold(self):
        """Test filtering when all chunks are above threshold"""
        chunks = [
            Document(page_content="doc1", metadata={"relevance_score": 0.8}),
            Document(page_content="doc2", metadata={"relevance_score": 0.7}),
            Document(page_content="doc3", metadata={"relevance_score": 0.9}),
        ]

        result = filter_chunks_by_relevance(chunks, "relevance_score", 0.5)

        assert len(result) == 3
        assert result == chunks

    def test_filter_chunks_exact_threshold_match(self):
        """Test filtering when chunk score exactly matches threshold"""
        chunks = [
            Document(page_content="doc1", metadata={"relevance_score": 0.5}),
            Document(page_content="doc2", metadata={"relevance_score": 0.4}),
        ]

        result = filter_chunks_by_relevance(chunks, "relevance_score", 0.5)

        # Score == threshold should be included (>=)
        assert len(result) == 1
        assert result[0].page_content == "doc1"

    def test_filter_chunks_empty_list(self):
        """Test filtering empty chunk list"""
        result = filter_chunks_by_relevance([], "relevance_score", 0.5)
        assert result == []

    def test_filter_chunks_custom_score_key(self):
        """Test filtering with custom relevance score key"""
        chunks = [
            Document(page_content="doc1", metadata={"custom_score": 0.8}),
            Document(page_content="doc2", metadata={"custom_score": 0.3}),
        ]

        result = filter_chunks_by_relevance(chunks, "custom_score", 0.5)

        assert len(result) == 1
        assert result[0].page_content == "doc1"

    def test_filter_chunks_zero_threshold(self):
        """Test filtering with zero threshold"""
        chunks = [
            Document(page_content="doc1", metadata={"relevance_score": 0.1}),
            Document(page_content="doc2", metadata={"relevance_score": -0.1}),
            Document(page_content="doc3", metadata={"relevance_score": 0.0}),
        ]

        result = filter_chunks_by_relevance(chunks, "relevance_score", 0.0)

        # Should include scores >= 0.0
        assert len(result) == 2
        assert result[0].page_content == "doc1"
        assert result[1].page_content == "doc3"


class TestSortDocsByRelevance:
    """Test suite for sort_docs_by_relevance function"""

    def test_sort_docs_descending_order(self):
        """Test sorting documents in descending order by relevance"""
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.9}),
            Document(page_content="doc3", metadata={"relevance_score": 0.6}),
            Document(page_content="doc4", metadata={"relevance_score": 0.1}),
        ]

        result = sort_docs_by_relevance(docs, "relevance_score")

        # Should be sorted highest to lowest
        assert len(result) == 4
        assert result[0].page_content == "doc2"  # 0.9
        assert result[1].page_content == "doc3"  # 0.6
        assert result[2].page_content == "doc1"  # 0.3
        assert result[3].page_content == "doc4"  # 0.1

    def test_sort_docs_already_sorted(self):
        """Test sorting already sorted documents"""
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.9}),
            Document(page_content="doc2", metadata={"relevance_score": 0.6}),
            Document(page_content="doc3", metadata={"relevance_score": 0.3}),
        ]

        result = sort_docs_by_relevance(docs, "relevance_score")

        # Should remain in same order
        assert result == docs

    def test_sort_docs_equal_scores(self):
        """Test sorting documents with equal relevance scores"""
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.5}),
            Document(page_content="doc2", metadata={"relevance_score": 0.5}),
            Document(page_content="doc3", metadata={"relevance_score": 0.5}),
        ]

        result = sort_docs_by_relevance(docs, "relevance_score")

        # Order should be stable for equal scores
        assert len(result) == 3
        # All scores should still be 0.5
        for doc in result:
            assert doc.metadata["relevance_score"] == 0.5

    def test_sort_docs_single_document(self):
        """Test sorting single document"""
        docs = [Document(page_content="doc1", metadata={"relevance_score": 0.5})]

        result = sort_docs_by_relevance(docs, "relevance_score")

        assert result == docs

    def test_sort_docs_empty_list(self):
        """Test sorting empty document list"""
        result = sort_docs_by_relevance([], "relevance_score")
        assert result == []

    def test_sort_docs_custom_score_key(self):
        """Test sorting with custom relevance score key"""
        docs = [
            Document(page_content="doc1", metadata={"custom_score": 0.2}),
            Document(page_content="doc2", metadata={"custom_score": 0.8}),
            Document(page_content="doc3", metadata={"custom_score": 0.5}),
        ]

        result = sort_docs_by_relevance(docs, "custom_score")

        assert result[0].page_content == "doc2"  # 0.8
        assert result[1].page_content == "doc3"  # 0.5
        assert result[2].page_content == "doc1"  # 0.2

    def test_sort_docs_negative_scores(self):
        """Test sorting with negative relevance scores"""
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": -0.1}),
            Document(page_content="doc2", metadata={"relevance_score": 0.5}),
            Document(page_content="doc3", metadata={"relevance_score": -0.5}),
        ]

        result = sort_docs_by_relevance(docs, "relevance_score")

        assert result[0].page_content == "doc2"  # 0.5
        assert result[1].page_content == "doc1"  # -0.1
        assert result[2].page_content == "doc3"  # -0.5

    def test_sort_docs_missing_score_key(self):
        """Test sorting when relevance score key is missing"""
        docs = [
            Document(page_content="doc1", metadata={"other_key": "value"}),
        ]

        with pytest.raises(KeyError):
            sort_docs_by_relevance(docs, "relevance_score")

    def test_sort_docs_mixed_missing_scores(self):
        """Test sorting when some docs are missing relevance scores"""
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.8}),
            Document(page_content="doc2", metadata={}),  # Missing score
        ]

        with pytest.raises(KeyError):
            sort_docs_by_relevance(docs, "relevance_score")

    def test_sort_docs_float_precision(self):
        """Test sorting with float precision edge cases"""
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.123456789}),
            Document(page_content="doc2", metadata={"relevance_score": 0.123456788}),
            Document(page_content="doc3", metadata={"relevance_score": 0.123456790}),
        ]

        result = sort_docs_by_relevance(docs, "relevance_score")

        # Should handle float precision correctly
        assert result[0].page_content == "doc3"  # Highest
        assert result[1].page_content == "doc1"  # Middle
        assert result[2].page_content == "doc2"  # Lowest

    def test_sort_docs_preserves_original_list(self):
        """Test that sorting doesn't modify the original list"""
        docs = [
            Document(page_content="doc1", metadata={"relevance_score": 0.3}),
            Document(page_content="doc2", metadata={"relevance_score": 0.9}),
        ]
        original_order = docs.copy()

        result = sort_docs_by_relevance(docs, "relevance_score")

        # Original list should be unchanged
        assert docs == original_order
        # Result should be different
        assert result != docs
        assert result[0].page_content == "doc2"
        assert result[1].page_content == "doc1"
