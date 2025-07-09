import pytest
from pydantic import ValidationError

from quivr_core.rag.entities.retriever import RetrieverConfig, RetrieverExtraConfig


class TestRetrieverExtraConfig:
    """Test suite for RetrieverExtraConfig class"""

    def test_init_with_defaults(self):
        """Test RetrieverExtraConfig initialization with default values"""
        config = RetrieverExtraConfig()

        assert config.top_n_knowledge == 3
        assert config.dynamic_retrieval_max_iterations == 3

    def test_init_with_custom_values(self):
        """Test RetrieverExtraConfig initialization with custom values"""
        config = RetrieverExtraConfig(
            top_n_knowledge=5, dynamic_retrieval_max_iterations=10
        )

        assert config.top_n_knowledge == 5
        assert config.dynamic_retrieval_max_iterations == 10

    def test_init_from_dict(self):
        """Test RetrieverExtraConfig initialization from dictionary"""
        data = {"top_n_knowledge": 7, "dynamic_retrieval_max_iterations": 15}

        config = RetrieverExtraConfig(**data)

        assert config.top_n_knowledge == 7
        assert config.dynamic_retrieval_max_iterations == 15

    def test_validation_positive_integers(self):
        """Test that values must be positive integers"""
        # Valid positive integers
        config = RetrieverExtraConfig(
            top_n_knowledge=1, dynamic_retrieval_max_iterations=1
        )
        assert config.top_n_knowledge == 1
        assert config.dynamic_retrieval_max_iterations == 1

        # Zero should be allowed
        config = RetrieverExtraConfig(
            top_n_knowledge=0, dynamic_retrieval_max_iterations=0
        )
        assert config.top_n_knowledge == 0
        assert config.dynamic_retrieval_max_iterations == 0

    def test_validation_negative_integers(self):
        """Test behavior with negative integers"""
        # Negative integers should still work (no explicit validation)
        config = RetrieverExtraConfig(
            top_n_knowledge=-1, dynamic_retrieval_max_iterations=-1
        )
        assert config.top_n_knowledge == -1
        assert config.dynamic_retrieval_max_iterations == -1

    def test_validation_invalid_types(self):
        """Test validation with invalid types"""
        with pytest.raises(ValidationError):
            RetrieverExtraConfig(top_n_knowledge="invalid")

        with pytest.raises(ValidationError):
            RetrieverExtraConfig(dynamic_retrieval_max_iterations="invalid")

    def test_serialization(self):
        """Test RetrieverExtraConfig serialization"""
        config = RetrieverExtraConfig(
            top_n_knowledge=5, dynamic_retrieval_max_iterations=10
        )

        config_dict = config.model_dump()
        expected_dict = {"top_n_knowledge": 5, "dynamic_retrieval_max_iterations": 10}
        assert config_dict == expected_dict


class TestRetrieverConfig:
    """Test suite for RetrieverConfig class"""

    def test_init_with_defaults(self):
        """Test RetrieverConfig initialization with default values"""
        config = RetrieverConfig()

        assert config.k == 40
        assert config.filter is None
        assert config.max_chunk_sum == 10000
        assert isinstance(config.extra_config, RetrieverExtraConfig)
        assert config.extra_config.top_n_knowledge == 3
        assert config.extra_config.dynamic_retrieval_max_iterations == 3

    def test_init_with_custom_values(self):
        """Test RetrieverConfig initialization with custom values"""
        custom_filter = {"source": "document.pdf", "type": "pdf"}
        extra_config = RetrieverExtraConfig(
            top_n_knowledge=5, dynamic_retrieval_max_iterations=8
        )

        config = RetrieverConfig(
            k=20, filter=custom_filter, max_chunk_sum=5000, extra_config=extra_config
        )

        assert config.k == 20
        assert config.filter == custom_filter
        assert config.max_chunk_sum == 5000
        assert config.extra_config == extra_config
        assert config.extra_config.top_n_knowledge == 5
        assert config.extra_config.dynamic_retrieval_max_iterations == 8

    def test_init_from_dict(self):
        """Test RetrieverConfig initialization from dictionary"""
        data = {
            "k": 50,
            "filter": {"category": "research"},
            "max_chunk_sum": 15000,
            "extra_config": {
                "top_n_knowledge": 7,
                "dynamic_retrieval_max_iterations": 12,
            },
        }

        config = RetrieverConfig(**data)

        assert config.k == 50
        assert config.filter == {"category": "research"}
        assert config.max_chunk_sum == 15000
        assert config.extra_config.top_n_knowledge == 7
        assert config.extra_config.dynamic_retrieval_max_iterations == 12

    def test_filter_validation(self):
        """Test filter validation with different types"""
        # None filter
        config = RetrieverConfig(filter=None)
        assert config.filter is None

        # Empty dict filter
        config = RetrieverConfig(filter={})
        assert config.filter == {}

        # Complex filter
        complex_filter = {
            "metadata": {"source": "document.pdf", "page": 1},
            "content_type": "text",
            "tags": ["important", "research"],
        }
        config = RetrieverConfig(filter=complex_filter)
        assert config.filter == complex_filter

    def test_k_validation(self):
        """Test k parameter validation"""
        # Positive integer
        config = RetrieverConfig(k=100)
        assert config.k == 100

        # Zero
        config = RetrieverConfig(k=0)
        assert config.k == 0

        # Negative (should still work, no explicit validation)
        config = RetrieverConfig(k=-1)
        assert config.k == -1

    def test_k_validation_invalid_type(self):
        """Test k parameter validation with invalid types"""
        with pytest.raises(ValidationError):
            RetrieverConfig(k="invalid")

        with pytest.raises(ValidationError):
            RetrieverConfig(k=3.14)

    def test_max_chunk_sum_validation(self):
        """Test max_chunk_sum parameter validation"""
        # Positive integer
        config = RetrieverConfig(max_chunk_sum=20000)
        assert config.max_chunk_sum == 20000

        # Zero
        config = RetrieverConfig(max_chunk_sum=0)
        assert config.max_chunk_sum == 0

    def test_max_chunk_sum_validation_invalid_type(self):
        """Test max_chunk_sum parameter validation with invalid types"""
        with pytest.raises(ValidationError):
            RetrieverConfig(max_chunk_sum="invalid")

        with pytest.raises(ValidationError):
            RetrieverConfig(max_chunk_sum=3.14)

    def test_extra_config_default_creation(self):
        """Test that extra_config is created with defaults if not provided"""
        config = RetrieverConfig()

        assert isinstance(config.extra_config, RetrieverExtraConfig)
        assert config.extra_config.top_n_knowledge == 3
        assert config.extra_config.dynamic_retrieval_max_iterations == 3

    def test_extra_config_custom_creation(self):
        """Test extra_config with custom values"""
        custom_extra = RetrieverExtraConfig(
            top_n_knowledge=10, dynamic_retrieval_max_iterations=20
        )

        config = RetrieverConfig(extra_config=custom_extra)

        assert config.extra_config == custom_extra
        assert config.extra_config.top_n_knowledge == 10
        assert config.extra_config.dynamic_retrieval_max_iterations == 20

    def test_extra_config_dict_creation(self):
        """Test extra_config creation from dictionary"""
        config = RetrieverConfig(
            extra_config={"top_n_knowledge": 6, "dynamic_retrieval_max_iterations": 9}
        )

        assert isinstance(config.extra_config, RetrieverExtraConfig)
        assert config.extra_config.top_n_knowledge == 6
        assert config.extra_config.dynamic_retrieval_max_iterations == 9

    def test_serialization(self):
        """Test RetrieverConfig serialization"""
        config = RetrieverConfig(
            k=30,
            filter={"type": "pdf"},
            max_chunk_sum=8000,
            extra_config=RetrieverExtraConfig(
                top_n_knowledge=4, dynamic_retrieval_max_iterations=6
            ),
        )

        config_dict = config.model_dump()
        expected_dict = {
            "k": 30,
            "filter": {"type": "pdf"},
            "max_chunk_sum": 8000,
            "extra_config": {
                "top_n_knowledge": 4,
                "dynamic_retrieval_max_iterations": 6,
            },
        }
        assert config_dict == expected_dict

    def test_deserialization(self):
        """Test RetrieverConfig deserialization"""
        config_dict = {
            "k": 25,
            "filter": {"category": "science"},
            "max_chunk_sum": 12000,
            "extra_config": {
                "top_n_knowledge": 8,
                "dynamic_retrieval_max_iterations": 15,
            },
        }

        config = RetrieverConfig.model_validate(config_dict)

        assert config.k == 25
        assert config.filter == {"category": "science"}
        assert config.max_chunk_sum == 12000
        assert config.extra_config.top_n_knowledge == 8
        assert config.extra_config.dynamic_retrieval_max_iterations == 15

    def test_json_serialization(self):
        """Test RetrieverConfig JSON serialization"""
        config = RetrieverConfig(
            k=35, filter={"source": "test.txt"}, max_chunk_sum=7000
        )

        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "35" in json_str
        assert "test.txt" in json_str
        assert "7000" in json_str

    def test_json_deserialization(self):
        """Test RetrieverConfig JSON deserialization"""
        json_str = """
        {
            "k": 45,
            "filter": {"type": "markdown"},
            "max_chunk_sum": 9000,
            "extra_config": {
                "top_n_knowledge": 2,
                "dynamic_retrieval_max_iterations": 4
            }
        }
        """

        config = RetrieverConfig.model_validate_json(json_str)

        assert config.k == 45
        assert config.filter == {"type": "markdown"}
        assert config.max_chunk_sum == 9000
        assert config.extra_config.top_n_knowledge == 2
        assert config.extra_config.dynamic_retrieval_max_iterations == 4

    def test_equality(self):
        """Test RetrieverConfig equality"""
        config1 = RetrieverConfig(k=40, filter={"type": "pdf"}, max_chunk_sum=10000)

        config2 = RetrieverConfig(k=40, filter={"type": "pdf"}, max_chunk_sum=10000)

        config3 = RetrieverConfig(k=50, filter={"type": "pdf"}, max_chunk_sum=10000)

        assert config1 == config2
        assert config1 != config3

    def test_copy(self):
        """Test RetrieverConfig copying"""
        original = RetrieverConfig(k=40, filter={"type": "pdf"}, max_chunk_sum=10000)

        copied = original.model_copy()

        assert copied == original
        assert copied is not original
        # extra_config should be equal but not the same object (deep copy)
        assert copied.extra_config == original.extra_config
        # Note: Due to Pydantic's optimization, identical objects may share references

    def test_update(self):
        """Test RetrieverConfig update"""
        config = RetrieverConfig(k=40)

        updated = config.model_copy(update={"k": 60, "max_chunk_sum": 15000})

        assert updated.k == 60
        assert updated.max_chunk_sum == 15000
        assert config.k == 40  # Original unchanged

    def test_inheritance_from_base_config(self):
        """Test that RetrieverConfig inherits from QuivrBaseConfig"""
        from quivr_core.base_config import QuivrBaseConfig

        config = RetrieverConfig()

        assert isinstance(config, QuivrBaseConfig)

        # Test that it has QuivrBaseConfig methods
        assert hasattr(config, "model_dump")
        assert hasattr(config, "model_validate")
        assert hasattr(config, "model_copy")

    def test_extra_config_inheritance(self):
        """Test that RetrieverExtraConfig inherits from QuivrBaseConfig"""
        from quivr_core.base_config import QuivrBaseConfig

        config = RetrieverExtraConfig()

        assert isinstance(config, QuivrBaseConfig)

        # Test that it has QuivrBaseConfig methods
        assert hasattr(config, "model_dump")
        assert hasattr(config, "model_validate")
        assert hasattr(config, "model_copy")
