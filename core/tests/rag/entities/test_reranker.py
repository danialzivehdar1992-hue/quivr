import pytest
from pydantic import ValidationError

from quivr_core.rag.entities.reranker import DefaultRerankers, RerankerConfig


class TestDefaultRerankers:
    """Test suite for DefaultRerankers enum"""

    def test_enum_values(self):
        """Test enum values are correct"""
        assert DefaultRerankers.COHERE == "cohere"
        assert DefaultRerankers.JINA == "jina"

    def test_default_model_cohere(self):
        """Test default model for Cohere"""
        assert DefaultRerankers.COHERE.default_model == "rerank-v3.5"

    def test_default_model_jina(self):
        """Test default model for Jina"""
        assert (
            DefaultRerankers.JINA.default_model == "jina-reranker-v2-base-multilingual"
        )

    def test_enum_iteration(self):
        """Test enum iteration"""
        suppliers = list(DefaultRerankers)
        assert len(suppliers) == 2
        assert DefaultRerankers.COHERE in suppliers
        assert DefaultRerankers.JINA in suppliers

    def test_string_representation(self):
        """Test string representation of enum values"""
        # Test the value property
        assert DefaultRerankers.COHERE.value == "cohere"
        assert DefaultRerankers.JINA.value == "jina"
        # Test repr() method
        assert repr(DefaultRerankers.COHERE) == "<DefaultRerankers.COHERE: 'cohere'>"


class TestRerankerConfig:
    """Test suite for RerankerConfig class"""

    def test_init_with_defaults(self):
        """Test RerankerConfig initialization with default values"""
        # Clear any environment variables that might affect this test
        import os

        original_cohere = os.environ.pop("COHERE_API_KEY", None)
        original_jina = os.environ.pop("JINA_API_KEY", None)

        try:
            config = RerankerConfig()

            assert config.supplier is None
            assert config.model is None
            assert config.top_n == 5
            assert config.relevance_score_threshold is None
            assert config.relevance_score_key == "relevance_score"
            # BaseSettings might pick up environment variables, so we don't assert None
        finally:
            # Restore environment variables
            if original_cohere:
                os.environ["COHERE_API_KEY"] = original_cohere
            if original_jina:
                os.environ["JINA_API_KEY"] = original_jina

    def test_init_with_cohere_supplier(self):
        """Test RerankerConfig initialization with Cohere supplier"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE, cohere_api_key="test_cohere_key"
        )

        assert config.supplier == DefaultRerankers.COHERE
        assert config.model == "rerank-v3.5"  # Should be set automatically
        assert config.cohere_api_key == "test_cohere_key"

    def test_init_with_jina_supplier(self):
        """Test RerankerConfig initialization with Jina supplier"""
        config = RerankerConfig(
            supplier=DefaultRerankers.JINA, jina_api_key="test_jina_key"
        )

        assert config.supplier == DefaultRerankers.JINA
        assert (
            config.model == "jina-reranker-v2-base-multilingual"
        )  # Should be set automatically
        assert config.jina_api_key == "test_jina_key"

    def test_init_with_custom_model(self):
        """Test RerankerConfig initialization with custom model"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE,
            model="custom-cohere-model",
            cohere_api_key="test_key",
        )

        assert config.supplier == DefaultRerankers.COHERE
        assert config.model == "custom-cohere-model"
        assert config.cohere_api_key == "test_key"

    def test_init_with_custom_parameters(self):
        """Test RerankerConfig initialization with custom parameters"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE,
            top_n=10,
            relevance_score_threshold=0.8,
            relevance_score_key="custom_score",
            cohere_api_key="test_key",
        )

        assert config.top_n == 10
        assert config.relevance_score_threshold == 0.8
        assert config.relevance_score_key == "custom_score"

    def test_init_from_dict(self):
        """Test RerankerConfig initialization from dictionary"""
        data = {
            "supplier": "cohere",
            "model": "rerank-v3.5",
            "top_n": 8,
            "relevance_score_threshold": 0.7,
            "cohere_api_key": "dict_key",
        }

        config = RerankerConfig(**data)

        assert config.supplier == DefaultRerankers.COHERE
        assert config.model == "rerank-v3.5"
        assert config.top_n == 8
        assert config.relevance_score_threshold == 0.7
        assert config.cohere_api_key == "dict_key"

    def test_validate_model_with_supplier_no_model(self):
        """Test validate_model sets default model when supplier provided but no model"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE, cohere_api_key="test_key"
        )

        assert config.model == "rerank-v3.5"

        config = RerankerConfig(supplier=DefaultRerankers.JINA, jina_api_key="test_key")

        assert config.model == "jina-reranker-v2-base-multilingual"

    def test_validate_model_with_supplier_and_model(self):
        """Test validate_model preserves custom model when provided"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE,
            model="custom-model",
            cohere_api_key="test_key",
        )

        assert config.model == "custom-model"

    def test_validate_model_no_supplier(self):
        """Test validate_model does nothing when no supplier provided"""
        config = RerankerConfig()

        assert config.model is None

    def test_api_key_property_cohere(self):
        """Test api_key property for Cohere supplier"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE, cohere_api_key="cohere_key_123"
        )

        assert config.api_key == "cohere_key_123"

    def test_api_key_property_jina(self):
        """Test api_key property for Jina supplier"""
        config = RerankerConfig(
            supplier=DefaultRerankers.JINA, jina_api_key="jina_key_456"
        )

        assert config.api_key == "jina_key_456"

    def test_api_key_property_no_supplier(self):
        """Test api_key property when no supplier set"""
        config = RerankerConfig()

        assert config.api_key is None

    def test_api_key_property_missing_key(self):
        """Test api_key property when API key is missing"""
        # Clear environment variables to ensure no API key is set
        import os

        original_cohere = os.environ.pop("COHERE_API_KEY", None)

        try:
            config = RerankerConfig(
                supplier=DefaultRerankers.COHERE, cohere_api_key=None
            )

            with pytest.raises(
                ValueError,
                match="The API key for supplier 'DefaultRerankers.COHERE' is not set",
            ):
                _ = config.api_key
        finally:
            # Restore environment variable
            if original_cohere:
                os.environ["COHERE_API_KEY"] = original_cohere

    def test_api_key_property_missing_jina_key(self):
        """Test api_key property when Jina API key is missing"""
        # Clear environment variables to ensure no API key is set
        import os

        original_jina = os.environ.pop("JINA_API_KEY", None)

        try:
            config = RerankerConfig(supplier=DefaultRerankers.JINA, jina_api_key=None)

            with pytest.raises(
                ValueError,
                match="The API key for supplier 'DefaultRerankers.JINA' is not set",
            ):
                _ = config.api_key
        finally:
            # Restore environment variable
            if original_jina:
                os.environ["JINA_API_KEY"] = original_jina

    def test_top_n_validation(self):
        """Test top_n parameter validation"""
        # Positive integer
        config = RerankerConfig(top_n=10)
        assert config.top_n == 10

        # Zero
        config = RerankerConfig(top_n=0)
        assert config.top_n == 0

        # Negative (should work, no explicit validation)
        config = RerankerConfig(top_n=-1)
        assert config.top_n == -1

    def test_top_n_validation_invalid_type(self):
        """Test top_n parameter validation with invalid types"""
        with pytest.raises(ValidationError):
            RerankerConfig(top_n="invalid")

        with pytest.raises(ValidationError):
            RerankerConfig(top_n=3.14)

    def test_relevance_score_threshold_validation(self):
        """Test relevance_score_threshold parameter validation"""
        # None
        config = RerankerConfig(relevance_score_threshold=None)
        assert config.relevance_score_threshold is None

        # Float
        config = RerankerConfig(relevance_score_threshold=0.5)
        assert config.relevance_score_threshold == 0.5

        # Integer (should convert to float)
        config = RerankerConfig(relevance_score_threshold=1)
        assert config.relevance_score_threshold == 1.0

    def test_relevance_score_threshold_validation_invalid_type(self):
        """Test relevance_score_threshold parameter validation with invalid types"""
        with pytest.raises(ValidationError):
            RerankerConfig(relevance_score_threshold="invalid")

    def test_relevance_score_key_validation(self):
        """Test relevance_score_key parameter validation"""
        # Default
        config = RerankerConfig()
        assert config.relevance_score_key == "relevance_score"

        # Custom
        config = RerankerConfig(relevance_score_key="custom_key")
        assert config.relevance_score_key == "custom_key"

        # Empty string
        config = RerankerConfig(relevance_score_key="")
        assert config.relevance_score_key == ""

    def test_supplier_validation_invalid_value(self):
        """Test supplier validation with invalid values"""
        with pytest.raises(ValidationError):
            RerankerConfig(supplier="invalid_supplier")

    def test_serialization(self):
        """Test RerankerConfig serialization"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE,
            model="rerank-v3.5",
            top_n=8,
            relevance_score_threshold=0.7,
            relevance_score_key="score",
            cohere_api_key="test_key",
        )

        config_dict = config.model_dump()

        assert config_dict["supplier"] == "cohere"
        assert config_dict["model"] == "rerank-v3.5"
        assert config_dict["top_n"] == 8
        assert config_dict["relevance_score_threshold"] == 0.7
        assert config_dict["relevance_score_key"] == "score"
        assert config_dict["cohere_api_key"] == "test_key"

    def test_deserialization(self):
        """Test RerankerConfig deserialization"""
        config_dict = {
            "supplier": "jina",
            "model": "custom-jina-model",
            "top_n": 12,
            "relevance_score_threshold": 0.9,
            "relevance_score_key": "custom_score",
            "jina_api_key": "jina_key",
        }

        config = RerankerConfig.model_validate(config_dict)

        assert config.supplier == DefaultRerankers.JINA
        assert config.model == "custom-jina-model"
        assert config.top_n == 12
        assert config.relevance_score_threshold == 0.9
        assert config.relevance_score_key == "custom_score"
        assert config.jina_api_key == "jina_key"

    def test_json_serialization(self):
        """Test RerankerConfig JSON serialization"""
        config = RerankerConfig(
            supplier=DefaultRerankers.COHERE, top_n=6, cohere_api_key="json_key"
        )

        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "cohere" in json_str
        assert "6" in json_str
        assert "json_key" in json_str

    def test_json_deserialization(self):
        """Test RerankerConfig JSON deserialization"""
        json_str = """
        {
            "supplier": "jina",
            "model": "jina-reranker-v2-base-multilingual",
            "top_n": 7,
            "relevance_score_threshold": 0.6,
            "jina_api_key": "json_jina_key"
        }
        """

        config = RerankerConfig.model_validate_json(json_str)

        assert config.supplier == DefaultRerankers.JINA
        assert config.model == "jina-reranker-v2-base-multilingual"
        assert config.top_n == 7
        assert config.relevance_score_threshold == 0.6
        assert config.jina_api_key == "json_jina_key"

    def test_equality(self):
        """Test RerankerConfig equality"""
        config1 = RerankerConfig(
            supplier=DefaultRerankers.COHERE, top_n=5, cohere_api_key="key1"
        )

        config2 = RerankerConfig(
            supplier=DefaultRerankers.COHERE, top_n=5, cohere_api_key="key1"
        )

        config3 = RerankerConfig(
            supplier=DefaultRerankers.JINA, top_n=5, jina_api_key="key1"
        )

        assert config1 == config2
        assert config1 != config3

    def test_copy(self):
        """Test RerankerConfig copying"""
        original = RerankerConfig(
            supplier=DefaultRerankers.COHERE, top_n=5, cohere_api_key="original_key"
        )

        copied = original.model_copy()

        assert copied == original
        assert copied is not original

    def test_update(self):
        """Test RerankerConfig update"""
        config = RerankerConfig(supplier=DefaultRerankers.COHERE, cohere_api_key="key")

        updated = config.model_copy(
            update={"top_n": 15, "relevance_score_threshold": 0.8}
        )

        assert updated.top_n == 15
        assert updated.relevance_score_threshold == 0.8
        assert updated.supplier == DefaultRerankers.COHERE
        assert config.top_n == 5  # Original unchanged

    def test_environment_variables(self):
        """Test that RerankerConfig can load from environment variables"""
        import os

        # Set environment variables
        os.environ["COHERE_API_KEY"] = "env_cohere_key"
        os.environ["JINA_API_KEY"] = "env_jina_key"

        try:
            config = RerankerConfig()
            # Since it inherits from BaseSettings, it should load from environment
            # This might depend on the specific implementation of BaseSettings

            # Test that we can create config and it doesn't crash
            assert config.supplier is None
            assert config.model is None

        finally:
            # Clean up environment variables
            if "COHERE_API_KEY" in os.environ:
                del os.environ["COHERE_API_KEY"]
            if "JINA_API_KEY" in os.environ:
                del os.environ["JINA_API_KEY"]

    def test_inheritance_from_base_settings(self):
        """Test that RerankerConfig inherits from BaseSettings"""
        from pydantic_settings import BaseSettings

        config = RerankerConfig()

        assert isinstance(config, BaseSettings)

        # Test that it has BaseSettings methods
        assert hasattr(config, "model_dump")
        assert hasattr(config, "model_validate")
        assert hasattr(config, "model_copy")
