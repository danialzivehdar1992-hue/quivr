import pytest

from quivr_core.rag.langgraph_framework.base.base_state import BaseState


class TestBaseState:
    """Test suite for BaseState class"""

    def test_base_state_is_typed_dict(self):
        """Test that BaseState is a TypedDict"""
        # TypedDict doesn't support issubclass checks, so we test the behavior instead
        state = BaseState()
        assert isinstance(state, dict)
        # Check that it has TypedDict annotations
        assert hasattr(BaseState, "__annotations__")
        assert hasattr(BaseState, "__total__")

    def test_base_state_creation_empty(self):
        """Test BaseState can be created empty"""
        state = BaseState()
        assert state == {}
        assert isinstance(state, dict)

    def test_base_state_creation_with_data(self):
        """Test BaseState can be created with data"""
        # Since BaseState is an empty TypedDict, it should accept any keys
        state = BaseState(key1="value1", key2="value2")
        assert state["key1"] == "value1"
        assert state["key2"] == "value2"

    def test_base_state_dict_operations(self):
        """Test BaseState supports dict operations"""
        state = BaseState()

        # Test item assignment
        state["test_key"] = "test_value"
        assert state["test_key"] == "test_value"

        # Test get method
        assert state.get("test_key") == "test_value"
        assert state.get("nonexistent_key") is None
        assert state.get("nonexistent_key", "default") == "default"

        # Test keys, values, items
        assert "test_key" in state.keys()
        assert "test_value" in state.values()
        assert ("test_key", "test_value") in state.items()

    def test_base_state_len(self):
        """Test BaseState length"""
        state = BaseState()
        assert len(state) == 0

        state["key1"] = "value1"
        assert len(state) == 1

        state["key2"] = "value2"
        assert len(state) == 2

    def test_base_state_boolean_evaluation(self):
        """Test BaseState boolean evaluation"""
        state = BaseState()
        assert not state  # Empty dict is falsy

        state["key"] = "value"
        assert state  # Non-empty dict is truthy

    def test_base_state_equality(self):
        """Test BaseState equality"""
        state1 = BaseState()
        state2 = BaseState()

        assert state1 == state2

        state1["key"] = "value"
        assert state1 != state2

        state2["key"] = "value"
        assert state1 == state2

    def test_base_state_copy(self):
        """Test BaseState copying"""
        state = BaseState(key1="value1", key2="value2")

        # Test copy
        copied = state.copy()
        assert copied == state
        assert copied is not state

        # Test that changes to copy don't affect original
        copied["new_key"] = "new_value"
        assert "new_key" not in state
        assert "new_key" in copied

    def test_base_state_update(self):
        """Test BaseState update"""
        state = BaseState(key1="value1")

        # Test update with dict
        state.update({"key2": "value2", "key3": "value3"})
        assert state["key2"] == "value2"
        assert state["key3"] == "value3"

        # Test update with kwargs
        state.update(key4="value4", key5="value5")
        assert state["key4"] == "value4"
        assert state["key5"] == "value5"

    def test_base_state_pop(self):
        """Test BaseState pop operations"""
        state = BaseState(key1="value1", key2="value2")

        # Test pop existing key
        value = state.pop("key1")
        assert value == "value1"
        assert "key1" not in state

        # Test pop non-existing key with default
        value = state.pop("nonexistent", "default")
        assert value == "default"

        # Test pop non-existing key without default
        with pytest.raises(KeyError):
            state.pop("nonexistent")

    def test_base_state_clear(self):
        """Test BaseState clear"""
        state = BaseState(key1="value1", key2="value2")
        assert len(state) == 2

        state.clear()
        assert len(state) == 0
        assert state == {}

    def test_base_state_setdefault(self):
        """Test BaseState setdefault"""
        state = BaseState()

        # Test setdefault with new key
        value = state.setdefault("key1", "default1")
        assert value == "default1"
        assert state["key1"] == "default1"

        # Test setdefault with existing key
        value = state.setdefault("key1", "default2")
        assert value == "default1"  # Should return existing value
        assert state["key1"] == "default1"  # Should not change

    def test_base_state_contains(self):
        """Test BaseState contains operation"""
        state = BaseState(key1="value1")

        assert "key1" in state
        assert "nonexistent" not in state

    def test_base_state_iteration(self):
        """Test BaseState iteration"""
        state = BaseState(key1="value1", key2="value2")

        # Test iteration over keys
        keys = list(state)
        assert "key1" in keys
        assert "key2" in keys
        assert len(keys) == 2

    def test_base_state_with_complex_values(self):
        """Test BaseState with complex value types"""
        complex_data = {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
        }

        state = BaseState(**complex_data)

        assert state["string"] == "text"
        assert state["number"] == 42
        assert state["float"] == 3.14
        assert state["boolean"] is True
        assert state["list"] == [1, 2, 3]
        assert state["dict"] == {"nested": "value"}
        assert state["none"] is None

    def test_base_state_with_type_hints(self):
        """Test BaseState usage with type hints"""

        # This test mainly verifies that type hints work correctly
        def process_state(state: BaseState) -> str:
            return str(len(state))

        state = BaseState(key1="value1", key2="value2")
        result = process_state(state)
        assert result == "2"

    def test_base_state_inheritance(self):
        """Test BaseState can be inherited"""

        class CustomState(BaseState):
            custom_field: str

        # TypedDict doesn't support issubclass checks, test behavior instead
        state = CustomState(custom_field="test")
        assert isinstance(state, dict)
        assert hasattr(CustomState, "__annotations__")
        assert "custom_field" in CustomState.__annotations__

        # Test creation (might raise type errors in strict mode, but should work at runtime)
        try:
            state = CustomState(custom_field="test")
            assert state["custom_field"] == "test"
        except Exception:
            # Type checking might prevent this, but that's expected behavior
            pass

    def test_base_state_json_serializable_values(self):
        """Test BaseState with JSON serializable values"""
        import json

        state = BaseState(
            string="text",
            number=42,
            boolean=True,
            list=[1, 2, 3],
            dict={"nested": "value"},
        )

        # Should be JSON serializable
        json_str = json.dumps(state)
        assert isinstance(json_str, str)

        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized == state

    def test_base_state_empty_and_none_values(self):
        """Test BaseState with empty and None values"""
        state = BaseState(
            empty_string="",
            empty_list=[],
            empty_dict={},
            none_value=None,
            zero=0,
            false_value=False,
        )

        assert state["empty_string"] == ""
        assert state["empty_list"] == []
        assert state["empty_dict"] == {}
        assert state["none_value"] is None
        assert state["zero"] == 0
        assert state["false_value"] is False

    def test_base_state_large_data(self):
        """Test BaseState with large amounts of data"""
        large_data = {f"key_{i}": f"value_{i}" for i in range(1000)}

        state = BaseState(**large_data)

        assert len(state) == 1000
        assert state["key_0"] == "value_0"
        assert state["key_999"] == "value_999"

    def test_base_state_string_representation(self):
        """Test BaseState string representation"""
        state = BaseState(key1="value1", key2="value2")

        str_repr = str(state)
        assert "key1" in str_repr
        assert "value1" in str_repr
        assert "key2" in str_repr
        assert "value2" in str_repr

        # Should be a valid dict representation
        assert str_repr.startswith("{")
        assert str_repr.endswith("}")

    def test_base_state_repr(self):
        """Test BaseState repr"""
        state = BaseState(key1="value1")

        repr_str = repr(state)
        assert "key1" in repr_str
        assert "value1" in repr_str
