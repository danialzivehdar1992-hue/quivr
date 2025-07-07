import pytest
from unittest.mock import Mock, patch
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever

from quivr_core.rag.langgraph_framework.services.retrieval_service import (
    RetrievalService,
)
from quivr_core.rag.langgraph_framework.entities.retrieval_service_config import (
    RetrievalServiceConfig,
)
from quivr_core.rag.entities.retriever import RetrieverConfig
from quivr_core.rag.entities.reranker import RerankerConfig, DefaultRerankers


class TestRetrievalService:
    """Test suite for RetrievalService class"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        mock_store = Mock(spec=VectorStore)
        mock_retriever = Mock(spec=VectorStoreRetriever)
        mock_store.as_retriever.return_value = mock_retriever
        return mock_store

    @pytest.fixture
    def retrieval_config(self):
        """Create a basic retrieval service config"""
        return RetrievalServiceConfig(
            retriever_config=RetrieverConfig(k=5, max_chunk_sum=1000),
            reranker_config=RerankerConfig(
                supplier=DefaultRerankers.COHERE, top_n=3, cohere_api_key="test_key"
            ),
        )

    @pytest.fixture
    def retrieval_service(self, mock_vector_store, retrieval_config):
        """Create a RetrievalService instance"""
        return RetrievalService(config=retrieval_config, vector_store=mock_vector_store)

    def test_initialization(self, mock_vector_store, retrieval_config):
        """Test RetrievalService initialization"""
        service = RetrievalService(
            config=retrieval_config, vector_store=mock_vector_store
        )

        assert service.config == retrieval_config
        assert service.vector_store == mock_vector_store
        assert service._basic_retriever is None
        assert service._compression_retriever is None
        assert service._basic_retriever_config_hash is None
        assert service._compression_retriever_config_hash is None

    def test_compute_config_hash_with_model_dump(self, retrieval_service):
        """Test _compute_config_hash with objects that have model_dump"""
        config1 = RetrieverConfig(k=5)
        config2 = RetrieverConfig(k=10)

        hash1 = retrieval_service._compute_config_hash(config1)
        hash2 = retrieval_service._compute_config_hash(config2)

        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert hash1 != hash2
        assert len(hash1) == 32  # MD5 hash length
        assert len(hash2) == 32

    def test_compute_config_hash_without_model_dump(self, retrieval_service):
        """Test _compute_config_hash with objects that don't have model_dump"""
        obj1 = "simple_string"
        obj2 = 42

        hash1 = retrieval_service._compute_config_hash(obj1)
        hash2 = retrieval_service._compute_config_hash(obj2)

        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert hash1 != hash2
        assert len(hash1) == 32
        assert len(hash2) == 32

    def test_compute_config_hash_multiple_configs(self, retrieval_service):
        """Test _compute_config_hash with multiple configuration objects"""
        config1 = RetrieverConfig(k=5)
        config2 = RerankerConfig(supplier=DefaultRerankers.COHERE, cohere_api_key="key")

        hash_result = retrieval_service._compute_config_hash(config1, config2)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_compute_config_hash_deterministic(self, retrieval_service):
        """Test that _compute_config_hash is deterministic"""
        config = RetrieverConfig(k=5, max_chunk_sum=1000)

        hash1 = retrieval_service._compute_config_hash(config)
        hash2 = retrieval_service._compute_config_hash(config)

        assert hash1 == hash2

    @patch(
        "quivr_core.rag.langgraph_framework.services.retrieval_service.get_retriever"
    )
    def test_get_basic_retriever_first_call(
        self, mock_get_retriever, retrieval_service
    ):
        """Test get_basic_retriever creates new retriever on first call"""
        mock_retriever = Mock(spec=VectorStoreRetriever)
        mock_get_retriever.return_value = mock_retriever

        result = retrieval_service.get_basic_retriever()

        assert result == mock_retriever
        assert retrieval_service._basic_retriever == mock_retriever
        assert retrieval_service._basic_retriever_config_hash is not None
        mock_get_retriever.assert_called_once_with(
            retrieval_service.vector_store, retrieval_service.config.retriever_config
        )

    @patch(
        "quivr_core.rag.langgraph_framework.services.retrieval_service.get_retriever"
    )
    def test_get_basic_retriever_cached(self, mock_get_retriever, retrieval_service):
        """Test get_basic_retriever uses cached retriever"""
        mock_retriever = Mock(spec=VectorStoreRetriever)
        mock_get_retriever.return_value = mock_retriever

        # First call
        result1 = retrieval_service.get_basic_retriever()
        # Second call
        result2 = retrieval_service.get_basic_retriever()

        assert result1 == result2
        assert result1 == mock_retriever
        # get_retriever should only be called once
        mock_get_retriever.assert_called_once()

    @patch(
        "quivr_core.rag.langgraph_framework.services.retrieval_service.get_retriever"
    )
    def test_get_basic_retriever_config_change(
        self, mock_get_retriever, retrieval_service
    ):
        """Test get_basic_retriever creates new retriever when config changes"""
        mock_retriever1 = Mock(spec=VectorStoreRetriever)
        mock_retriever2 = Mock(spec=VectorStoreRetriever)
        mock_get_retriever.side_effect = [mock_retriever1, mock_retriever2]

        # First call
        result1 = retrieval_service.get_basic_retriever()

        # Change config
        retrieval_service.config.retriever_config.k = 10

        # Second call
        result2 = retrieval_service.get_basic_retriever()

        assert result1 == mock_retriever1
        assert result2 == mock_retriever2
        assert result1 != result2
        # get_retriever should be called twice
        assert mock_get_retriever.call_count == 2

    @patch(
        "quivr_core.rag.langgraph_framework.services.retrieval_service.get_compression_retriever"
    )
    def test_get_compression_retriever_first_call(
        self, mock_get_compression_retriever, retrieval_service
    ):
        """Test get_compression_retriever creates new retriever on first call"""
        mock_retriever = Mock(spec=ContextualCompressionRetriever)
        mock_get_compression_retriever.return_value = mock_retriever

        result = retrieval_service.get_compression_retriever()

        assert result == mock_retriever
        assert retrieval_service._compression_retriever == mock_retriever
        assert retrieval_service._compression_retriever_config_hash is not None
        mock_get_compression_retriever.assert_called_once_with(
            retrieval_service.vector_store,
            retrieval_service.config.retriever_config,
            retrieval_service.config.reranker_config,
        )

    @patch(
        "quivr_core.rag.langgraph_framework.services.retrieval_service.get_compression_retriever"
    )
    def test_get_compression_retriever_cached(
        self, mock_get_compression_retriever, retrieval_service
    ):
        """Test get_compression_retriever uses cached retriever"""
        mock_retriever = Mock(spec=ContextualCompressionRetriever)
        mock_get_compression_retriever.return_value = mock_retriever

        # First call
        result1 = retrieval_service.get_compression_retriever()
        # Second call
        result2 = retrieval_service.get_compression_retriever()

        assert result1 == result2
        assert result1 == mock_retriever
        # get_compression_retriever should only be called once
        mock_get_compression_retriever.assert_called_once()

    @patch(
        "quivr_core.rag.langgraph_framework.services.retrieval_service.get_compression_retriever"
    )
    def test_get_compression_retriever_config_change(
        self, mock_get_compression_retriever, retrieval_service
    ):
        """Test get_compression_retriever creates new retriever when config changes"""
        mock_retriever1 = Mock(spec=ContextualCompressionRetriever)
        mock_retriever2 = Mock(spec=ContextualCompressionRetriever)
        mock_get_compression_retriever.side_effect = [mock_retriever1, mock_retriever2]

        # First call
        result1 = retrieval_service.get_compression_retriever()

        # Change config
        retrieval_service.config.reranker_config.top_n = 5

        # Second call
        result2 = retrieval_service.get_compression_retriever()

        assert result1 == mock_retriever1
        assert result2 == mock_retriever2
        assert result1 != result2
        # get_compression_retriever should be called twice
        assert mock_get_compression_retriever.call_count == 2

    def test_get_vector_store(self, retrieval_service, mock_vector_store):
        """Test get_vector_store returns the vector store"""
        result = retrieval_service.get_vector_store()
        assert result == mock_vector_store

    def test_clear_cache(self, retrieval_service):
        """Test clear_cache clears all cached retrievers"""
        # Set some cached data
        retrieval_service._basic_retriever = Mock()
        retrieval_service._compression_retriever = Mock()
        retrieval_service._basic_retriever_config_hash = "hash1"
        retrieval_service._compression_retriever_config_hash = "hash2"

        retrieval_service.clear_cache()

        assert retrieval_service._basic_retriever is None
        assert retrieval_service._compression_retriever is None
        assert retrieval_service._basic_retriever_config_hash is None
        assert retrieval_service._compression_retriever_config_hash is None

    def test_update_config_no_change(self, retrieval_service):
        """Test update_config when config doesn't change"""
        _ = retrieval_service.config

        # Create identical config
        new_config = RetrievalServiceConfig(
            retriever_config=RetrieverConfig(k=5, max_chunk_sum=1000),
            reranker_config=RerankerConfig(
                supplier=DefaultRerankers.COHERE, top_n=3, cohere_api_key="test_key"
            ),
        )

        # Set some cached data
        retrieval_service._basic_retriever = Mock()
        retrieval_service._basic_retriever_config_hash = "hash1"

        retrieval_service.update_config(new_config)

        # Cache should remain intact since config didn't change
        assert retrieval_service._basic_retriever is not None
        assert retrieval_service._basic_retriever_config_hash is not None
        assert retrieval_service.config == new_config

    def test_update_config_with_change(self, retrieval_service):
        """Test update_config when config changes"""
        # Set some cached data
        retrieval_service._basic_retriever = Mock()
        retrieval_service._compression_retriever = Mock()
        retrieval_service._basic_retriever_config_hash = "hash1"
        retrieval_service._compression_retriever_config_hash = "hash2"

        # Create different config
        new_config = RetrievalServiceConfig(
            retriever_config=RetrieverConfig(k=10, max_chunk_sum=2000),
            reranker_config=RerankerConfig(
                supplier=DefaultRerankers.JINA, top_n=5, jina_api_key="different_key"
            ),
        )

        retrieval_service.update_config(new_config)

        # Cache should be cleared since config changed
        assert retrieval_service._basic_retriever is None
        assert retrieval_service._compression_retriever is None
        assert retrieval_service._basic_retriever_config_hash is None
        assert retrieval_service._compression_retriever_config_hash is None
        assert retrieval_service.config == new_config

    def test_update_config_fallback_comparison(self, retrieval_service):
        """Test update_config fallback string comparison"""
        # Mock config without model_dump method
        mock_config = Mock()
        mock_config.model_dump = Mock(side_effect=AttributeError())

        # Set some cached data
        retrieval_service._basic_retriever = Mock()

        with patch.object(
            retrieval_service.config, "model_dump", side_effect=AttributeError()
        ):
            retrieval_service.update_config(mock_config)

        # Should use string comparison and clear cache
        assert retrieval_service._basic_retriever is None
        assert retrieval_service.config == mock_config

    @patch("quivr_core.rag.langgraph_framework.services.retrieval_service.logger")
    def test_logging_basic_retriever(self, mock_logger, retrieval_service):
        """Test logging for basic retriever operations"""
        with patch(
            "quivr_core.rag.langgraph_framework.services.retrieval_service.get_retriever"
        ) as mock_get_retriever:
            mock_get_retriever.return_value = Mock()

            # First call - should log creation
            retrieval_service.get_basic_retriever()
            mock_logger.debug.assert_called_with("Creating new basic retriever")

            # Second call - should log cache usage
            mock_logger.reset_mock()
            retrieval_service.get_basic_retriever()
            mock_logger.debug.assert_called_with("Using cached basic retriever")

    @patch("quivr_core.rag.langgraph_framework.services.retrieval_service.logger")
    def test_logging_compression_retriever(self, mock_logger, retrieval_service):
        """Test logging for compression retriever operations"""
        with patch(
            "quivr_core.rag.langgraph_framework.services.retrieval_service.get_compression_retriever"
        ) as mock_get_compression:
            mock_get_compression.return_value = Mock()

            # First call - should log creation
            retrieval_service.get_compression_retriever()
            mock_logger.debug.assert_called_with("Creating new compression retriever")

            # Second call - should log cache usage
            mock_logger.reset_mock()
            retrieval_service.get_compression_retriever()
            mock_logger.debug.assert_called_with("Using cached compression retriever")

    @patch("quivr_core.rag.langgraph_framework.services.retrieval_service.logger")
    def test_logging_clear_cache(self, mock_logger, retrieval_service):
        """Test logging for clear_cache operation"""
        retrieval_service.clear_cache()
        mock_logger.debug.assert_called_with("Clearing retrieval service cache")

    @patch("quivr_core.rag.langgraph_framework.services.retrieval_service.logger")
    def test_logging_update_config(self, mock_logger, retrieval_service):
        """Test logging for update_config operation"""
        new_config = RetrievalServiceConfig(
            retriever_config=RetrieverConfig(k=10),
            reranker_config=RerankerConfig(
                supplier=DefaultRerankers.JINA, jina_api_key="key"
            ),
        )

        retrieval_service.update_config(new_config)
        mock_logger.debug.assert_called_with("Configuration changed, clearing cache")

    def test_error_handling_in_compute_config_hash(self, retrieval_service):
        """Test error handling in _compute_config_hash"""
        # Create an object that raises an exception in model_dump
        mock_config = Mock()
        mock_config.model_dump.side_effect = Exception("Model dump failed")

        with pytest.raises(Exception, match="Model dump failed"):
            retrieval_service._compute_config_hash(mock_config)

    def test_hash_consistency_across_instances(self, mock_vector_store):
        """Test that hash computation is consistent across different service instances"""
        config = RetrievalServiceConfig(
            retriever_config=RetrieverConfig(k=5),
            reranker_config=RerankerConfig(
                supplier=DefaultRerankers.COHERE, cohere_api_key="key"
            ),
        )

        service1 = RetrievalService(config=config, vector_store=mock_vector_store)
        service2 = RetrievalService(config=config, vector_store=mock_vector_store)

        hash1 = service1._compute_config_hash(config.retriever_config)
        hash2 = service2._compute_config_hash(config.retriever_config)

        assert hash1 == hash2

    def test_memory_efficiency_with_large_configs(self, mock_vector_store):
        """Test memory efficiency with large configuration objects"""
        # Create a config with large data
        large_filter = {f"key_{i}": f"value_{i}" for i in range(1000)}
        config = RetrievalServiceConfig(
            retriever_config=RetrieverConfig(k=5, filter=large_filter),
            reranker_config=RerankerConfig(
                supplier=DefaultRerankers.COHERE, cohere_api_key="key"
            ),
        )

        service = RetrievalService(config=config, vector_store=mock_vector_store)

        # Should handle large configs without issues
        hash_result = service._compute_config_hash(config.retriever_config)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_thread_safety_simulation(self, retrieval_service):
        """Test thread safety simulation by rapid successive calls"""
        import threading

        results = []

        def get_retriever_worker():
            with patch(
                "quivr_core.rag.langgraph_framework.services.retrieval_service.get_retriever"
            ) as mock_get_retriever:
                mock_get_retriever.return_value = Mock()
                result = retrieval_service.get_basic_retriever()
                results.append(result)

        # Simulate multiple threads calling get_basic_retriever
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_retriever_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be the same due to caching
        assert len(set(id(r) for r in results)) <= 1  # At most one unique object
