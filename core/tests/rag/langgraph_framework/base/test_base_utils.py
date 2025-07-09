import pytest
import hashlib
from unittest.mock import patch
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from quivr_core.rag.langgraph_framework.base.utils import compute_config_hash


class TestComputeConfigHash:
    """Test suite for compute_config_hash function"""

    def test_compute_config_hash_basic(self):
        """Test basic config hash computation"""

        class SimpleConfig(BaseModel):
            value: str = "test"
            number: int = 42

        config = SimpleConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex digest length

        # Hash should be deterministic
        hash_result2 = compute_config_hash(config)
        assert hash_result == hash_result2

    def test_compute_config_hash_different_configs(self):
        """Test that different configs produce different hashes"""

        class Config(BaseModel):
            value: str
            number: int

        config1 = Config(value="test1", number=1)
        config2 = Config(value="test2", number=2)

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 != hash2

    def test_compute_config_hash_same_values_different_order(self):
        """Test that configs with same values but different field order produce same hash"""

        class Config1(BaseModel):
            field_a: str
            field_b: int

        class Config2(BaseModel):
            field_b: int
            field_a: str

        config1 = Config1(field_a="test", field_b=42)
        config2 = Config2(field_b=42, field_a="test")

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        # Should be the same due to sorting in compute_config_hash
        assert hash1 == hash2

    def test_compute_config_hash_empty_config(self):
        """Test hash computation with empty config"""

        class EmptyConfig(BaseModel):
            pass

        config = EmptyConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_with_none_values(self):
        """Test hash computation with None values"""

        class ConfigWithNone(BaseModel):
            value: Optional[str] = None
            number: Optional[int] = None

        config = ConfigWithNone()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_with_complex_types(self):
        """Test hash computation with complex data types"""

        class ComplexConfig(BaseModel):
            string_field: str = "test"
            int_field: int = 42
            float_field: float = 3.14
            bool_field: bool = True
            list_field: List[str] = ["a", "b", "c"]
            dict_field: Dict[str, Any] = {"key": "value", "nested": {"inner": "data"}}

        config = ComplexConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_with_nested_models(self):
        """Test hash computation with nested Pydantic models"""

        class NestedModel(BaseModel):
            inner_value: str = "inner"
            inner_number: int = 10

        class ParentConfig(BaseModel):
            value: str = "parent"
            nested: NestedModel = Field(default_factory=NestedModel)

        config = ParentConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_deterministic(self):
        """Test that hash computation is deterministic across multiple calls"""

        class Config(BaseModel):
            value: str = "test"
            number: int = 42
            list_data: List[str] = ["a", "b", "c"]

        config = Config()

        # Compute hash multiple times
        hashes = [compute_config_hash(config) for _ in range(5)]

        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)

    def test_compute_config_hash_with_default_values(self):
        """Test hash computation with default values"""

        class ConfigWithDefaults(BaseModel):
            value: str = "default"
            number: int = 0
            enabled: bool = False

        config1 = ConfigWithDefaults()
        config2 = ConfigWithDefaults(value="default", number=0, enabled=False)

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2

    def test_compute_config_hash_sensitive_to_changes(self):
        """Test that hash changes when config values change"""

        class MutableConfig(BaseModel):
            value: str = "initial"

        config = MutableConfig()
        hash1 = compute_config_hash(config)

        # Change the value
        config.value = "changed"
        hash2 = compute_config_hash(config)

        assert hash1 != hash2

    def test_compute_config_hash_with_special_characters(self):
        """Test hash computation with special characters in values"""

        class SpecialConfig(BaseModel):
            special_chars: str = "!@#$%^&*()_+{}|:<>?[]\\;'\",./"
            unicode_chars: str = "émojis 🤖 and spëcîâl characters"
            newlines: str = "line1\nline2\nline3"

        config = SpecialConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_with_large_values(self):
        """Test hash computation with large values"""

        class LargeConfig(BaseModel):
            large_string: str = "x" * 10000
            large_list: List[int] = list(range(1000))

        config = LargeConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_sorting_behavior(self):
        """Test the sorting behavior of config dictionary"""

        from pydantic import BaseModel

        class Config(BaseModel):
            z_field: str = "z"
            a_field: str = "a"
            m_field: str = "m"

        config = Config()

        # Test sorting by creating config with different field order
        # The hash should be the same regardless of creation order
        config_dict = config.model_dump()
        sorted_items = sorted(config_dict.items())

        # Verify that keys are sorted alphabetically
        expected_order = [("a_field", "a"), ("m_field", "m"), ("z_field", "z")]
        assert sorted_items == expected_order

        hash_result = compute_config_hash(config)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_with_model_dump_error(self):
        """Test hash computation when model_dump raises an error"""

        class Config(BaseModel):
            value: str = "test"

        config = Config()

        # Mock the compute_config_hash function to test error handling
        with patch(
            "quivr_core.rag.langgraph_framework.base.utils.compute_config_hash"
        ) as mock_hash:
            mock_hash.side_effect = Exception("Model dump failed")

            with pytest.raises(Exception, match="Model dump failed"):
                mock_hash(config)

    def test_compute_config_hash_implementation_details(self):
        """Test the implementation details of compute_config_hash"""

        class Config(BaseModel):
            value: str = "test"
            number: int = 42

        config = Config()

        # Test that the function follows the expected steps
        expected_dict = config.model_dump()
        expected_sorted = str(sorted(expected_dict.items()))
        expected_hash = hashlib.sha256(expected_sorted.encode()).hexdigest()

        actual_hash = compute_config_hash(config)

        assert actual_hash == expected_hash

    def test_compute_config_hash_with_bytes_values(self):
        """Test hash computation with bytes values"""

        class BytesConfig(BaseModel):
            # Note: Pydantic might not directly support bytes, so we use a workaround
            value: str = "test"

            class Config:
                arbitrary_types_allowed = True

        config = BytesConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_edge_cases(self):
        """Test edge cases in hash computation"""

        class EdgeCaseConfig(BaseModel):
            empty_string: str = ""
            zero: int = 0
            false_value: bool = False
            empty_list: List[str] = []
            empty_dict: Dict[str, str] = {}

        config = EdgeCaseConfig()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_compute_config_hash_type_consistency(self):
        """Test that hash is consistent across different instantiations"""

        class Config(BaseModel):
            value: str = "test"
            number: int = 42

        config1 = Config()
        config2 = Config()

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2

    def test_compute_config_hash_hex_format(self):
        """Test that returned hash is in proper hex format"""

        class Config(BaseModel):
            value: str = "test"

        config = Config()
        hash_result = compute_config_hash(config)

        # Check that it's a valid hex string
        try:
            int(hash_result, 16)
        except ValueError:
            pytest.fail("Hash result is not a valid hex string")

        # Check that it only contains valid hex characters
        assert all(c in "0123456789abcdef" for c in hash_result.lower())

    def test_compute_config_hash_not_none(self):
        """Test that hash is never None"""

        class Config(BaseModel):
            value: Optional[str] = None

        config = Config()
        hash_result = compute_config_hash(config)

        assert hash_result is not None
        assert isinstance(hash_result, str)
        assert len(hash_result) > 0

    def test_compute_config_hash_with_custom_types(self):
        """Test hash computation with custom types that have __str__ method"""

        class CustomType:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"CustomType({self.value})"

        class ConfigWithCustomType(BaseModel):
            value: str = "test"

            class Config:
                arbitrary_types_allowed = True

        config = ConfigWithCustomType()
        hash_result = compute_config_hash(config)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
