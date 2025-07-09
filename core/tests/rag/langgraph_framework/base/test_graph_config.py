import pytest
from unittest.mock import Mock
from typing import Any, Dict

from quivr_core.rag.langgraph_framework.base.graph_config import BaseGraphConfig


class TestBaseGraphConfig:
    """Test suite for BaseGraphConfig class"""

    def test_base_graph_config_inheritance(self):
        """Test that BaseGraphConfig inherits from RunnableConfig"""
        # RunnableConfig might be a TypedDict, check behavior instead
        config = BaseGraphConfig()
        assert isinstance(config, dict)  # RunnableConfig is dict-like
        # Check that it has the expected structure
        assert hasattr(BaseGraphConfig, "__mro__")

    def test_base_graph_config_creation_empty(self):
        """Test BaseGraphConfig creation with no parameters"""
        config = BaseGraphConfig()
        assert isinstance(config, dict)  # BaseGraphConfig is dict-like
        # Test that it can be used as expected
        assert len(config) == 0

    def test_base_graph_config_creation_with_params(self):
        """Test BaseGraphConfig creation with parameters"""
        # Test with common RunnableConfig parameters
        config = BaseGraphConfig(
            tags=["test", "graph"],
            metadata={"version": "1.0", "author": "test"},
            run_name="test_run",
            max_concurrency=5,
        )

        assert config.get("tags") == ["test", "graph"]
        assert config.get("metadata") == {"version": "1.0", "author": "test"}
        assert config.get("run_name") == "test_run"
        assert config.get("max_concurrency") == 5

    def test_base_graph_config_dict_behavior(self):
        """Test BaseGraphConfig behaves like a dictionary"""
        config = BaseGraphConfig()

        # Test item assignment
        config["custom_key"] = "custom_value"
        assert config["custom_key"] == "custom_value"

        # Test get method
        assert config.get("custom_key") == "custom_value"
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"

    def test_base_graph_config_with_callbacks(self):
        """Test BaseGraphConfig with callbacks"""
        mock_callback = Mock()

        config = BaseGraphConfig(callbacks=[mock_callback])

        assert config.get("callbacks") == [mock_callback]

    def test_base_graph_config_with_configurable(self):
        """Test BaseGraphConfig with configurable parameters"""
        configurable = {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 1000}

        config = BaseGraphConfig(configurable=configurable)

        assert config.get("configurable") == configurable

    def test_base_graph_config_update(self):
        """Test BaseGraphConfig update functionality"""
        config = BaseGraphConfig(tags=["initial"])

        config.update({"tags": ["updated"], "metadata": {"updated": True}})

        assert config.get("tags") == ["updated"]
        assert config.get("metadata") == {"updated": True}

    def test_base_graph_config_copy(self):
        """Test BaseGraphConfig copy functionality"""
        original_config = BaseGraphConfig(
            tags=["original"], metadata={"version": "1.0"}, run_name="original_run"
        )

        copied_config = original_config.copy()

        assert copied_config == original_config
        assert copied_config is not original_config

        # Test that changes to copy don't affect original
        copied_config["new_key"] = "new_value"
        assert "new_key" not in original_config

    def test_base_graph_config_keys_values_items(self):
        """Test BaseGraphConfig keys, values, and items methods"""
        config = BaseGraphConfig(
            tags=["test"], metadata={"key": "value"}, run_name="test_run"
        )

        keys = list(config.keys())
        values = list(config.values())
        items = list(config.items())

        assert "tags" in keys
        assert "metadata" in keys
        assert "run_name" in keys

        assert ["test"] in values
        assert {"key": "value"} in values
        assert "test_run" in values

        assert ("tags", ["test"]) in items
        assert ("metadata", {"key": "value"}) in items
        assert ("run_name", "test_run") in items

    def test_base_graph_config_len(self):
        """Test BaseGraphConfig length"""
        config = BaseGraphConfig()
        initial_len = len(config)

        config["new_key"] = "new_value"
        assert len(config) == initial_len + 1

    def test_base_graph_config_contains(self):
        """Test BaseGraphConfig contains operation"""
        config = BaseGraphConfig(tags=["test"])

        assert "tags" in config
        assert "nonexistent" not in config

    def test_base_graph_config_iteration(self):
        """Test BaseGraphConfig iteration"""
        config = BaseGraphConfig(tags=["test"], metadata={"key": "value"})

        keys = list(config)
        assert "tags" in keys
        assert "metadata" in keys

    def test_base_graph_config_clear(self):
        """Test BaseGraphConfig clear"""
        config = BaseGraphConfig(tags=["test"], metadata={"key": "value"})

        initial_len = len(config)
        assert initial_len > 0

        config.clear()
        assert len(config) == 0

    def test_base_graph_config_pop(self):
        """Test BaseGraphConfig pop operations"""
        config = BaseGraphConfig(tags=["test"], metadata={"key": "value"})

        # Test pop existing key
        tags = config.pop("tags")
        assert tags == ["test"]
        assert "tags" not in config

        # Test pop with default
        result = config.pop("nonexistent", "default")
        assert result == "default"

        # Test pop without default (should raise KeyError)
        with pytest.raises(KeyError):
            config.pop("nonexistent")

    def test_base_graph_config_setdefault(self):
        """Test BaseGraphConfig setdefault"""
        config = BaseGraphConfig()

        # Test setdefault with new key
        result = config.setdefault("new_key", "default_value")
        assert result == "default_value"
        assert config["new_key"] == "default_value"

        # Test setdefault with existing key
        result = config.setdefault("new_key", "different_value")
        assert result == "default_value"  # Should return existing value
        assert config["new_key"] == "default_value"  # Should not change

    def test_base_graph_config_equality(self):
        """Test BaseGraphConfig equality"""
        config1 = BaseGraphConfig(tags=["test"], metadata={"key": "value"})
        config2 = BaseGraphConfig(tags=["test"], metadata={"key": "value"})
        config3 = BaseGraphConfig(tags=["different"], metadata={"key": "value"})

        assert config1 == config2
        assert config1 != config3

    def test_base_graph_config_str_representation(self):
        """Test BaseGraphConfig string representation"""
        config = BaseGraphConfig(tags=["test"], run_name="test_run")

        str_repr = str(config)
        assert isinstance(str_repr, str)
        # Should contain key information
        assert "tags" in str_repr or "test" in str_repr

    def test_base_graph_config_repr(self):
        """Test BaseGraphConfig repr"""
        config = BaseGraphConfig(tags=["test"])

        repr_str = repr(config)
        assert isinstance(repr_str, str)
        # Check that repr contains some useful information
        assert len(repr_str) > 0

    def test_base_graph_config_with_complex_types(self):
        """Test BaseGraphConfig with complex data types"""
        complex_data = {
            "callbacks": [Mock(), Mock()],
            "metadata": {
                "nested": {"deep": "value"},
                "list": [1, 2, 3],
                "boolean": True,
            },
            "configurable": {"model_params": {"temperature": 0.7, "max_tokens": 1000}},
        }

        config = BaseGraphConfig(**complex_data)

        assert len(config.get("callbacks", [])) == 2
        assert config.get("metadata", {}).get("nested", {}).get("deep") == "value"
        assert config.get("metadata", {}).get("list") == [1, 2, 3]
        assert config.get("metadata", {}).get("boolean") is True
        assert (
            config.get("configurable", {}).get("model_params", {}).get("temperature")
            == 0.7
        )

    def test_base_graph_config_inheritance_methods(self):
        """Test that BaseGraphConfig has all expected methods from RunnableConfig"""
        config = BaseGraphConfig()

        # Check that it has dictionary methods
        assert hasattr(config, "get")
        assert hasattr(config, "update")
        assert hasattr(config, "copy")
        assert hasattr(config, "keys")
        assert hasattr(config, "values")
        assert hasattr(config, "items")
        assert hasattr(config, "pop")
        assert hasattr(config, "setdefault")
        assert hasattr(config, "clear")

    def test_base_graph_config_type_annotations(self):
        """Test BaseGraphConfig type annotations"""

        # This test ensures type annotations work correctly
        def process_config(config: BaseGraphConfig) -> Dict[str, Any]:
            return dict(config)

        config = BaseGraphConfig(tags=["test"])
        result = process_config(config)

        assert isinstance(result, dict)
        assert "tags" in result

    def test_base_graph_config_serialization(self):
        """Test BaseGraphConfig serialization"""
        import json

        # Create config with JSON-serializable data
        config = BaseGraphConfig(
            tags=["test"],
            metadata={"version": "1.0", "count": 42, "active": True},
            run_name="test_run",
        )

        # Convert to dict and serialize
        config_dict = dict(config)
        json_str = json.dumps(config_dict, default=str)

        assert isinstance(json_str, str)
        assert "test" in json_str
        assert "1.0" in json_str

    def test_base_graph_config_with_none_values(self):
        """Test BaseGraphConfig with None values"""
        config = BaseGraphConfig(tags=None, metadata=None, run_name=None)

        assert config.get("tags") is None
        assert config.get("metadata") is None
        assert config.get("run_name") is None

    def test_base_graph_config_empty_collections(self):
        """Test BaseGraphConfig with empty collections"""
        config = BaseGraphConfig(tags=[], metadata={}, callbacks=[])

        assert config.get("tags") == []
        assert config.get("metadata") == {}
        assert config.get("callbacks") == []

    def test_base_graph_config_merge_configs(self):
        """Test merging multiple BaseGraphConfig instances"""
        config1 = BaseGraphConfig(
            tags=["tag1"], metadata={"key1": "value1"}, run_name="run1"
        )

        config2 = BaseGraphConfig(
            tags=["tag2"], metadata={"key2": "value2"}, max_concurrency=10
        )

        # Merge configs
        merged = BaseGraphConfig()
        merged.update(config1)
        merged.update(config2)

        # config2 should override config1 where keys conflict
        assert merged.get("tags") == ["tag2"]
        assert merged.get("metadata") == {"key2": "value2"}
        assert merged.get("run_name") == "run1"  # Only in config1
        assert merged.get("max_concurrency") == 10  # Only in config2

    def test_base_graph_config_immutability_check(self):
        """Test that BaseGraphConfig doesn't enforce immutability"""
        config = BaseGraphConfig(tags=["original"])

        # Should be able to modify
        config["tags"] = ["modified"]
        assert config["tags"] == ["modified"]

        # Should be able to add new keys
        config["new_key"] = "new_value"
        assert config["new_key"] == "new_value"
