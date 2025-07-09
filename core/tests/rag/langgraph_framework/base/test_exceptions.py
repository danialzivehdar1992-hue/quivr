import pytest
from quivr_core.rag.langgraph_framework.base.exceptions import (
    NodeValidationError,
    NodeExecutionError,
    ConfigExtractionError,
)


class TestNodeValidationError:
    """Test suite for NodeValidationError exception"""

    def test_node_validation_error_inheritance(self):
        """Test that NodeValidationError inherits from Exception"""
        assert issubclass(NodeValidationError, Exception)

    def test_node_validation_error_creation_no_message(self):
        """Test NodeValidationError creation without message"""
        error = NodeValidationError()
        assert isinstance(error, NodeValidationError)
        assert isinstance(error, Exception)

    def test_node_validation_error_creation_with_message(self):
        """Test NodeValidationError creation with message"""
        message = "Node validation failed"
        error = NodeValidationError(message)
        assert str(error) == message

    def test_node_validation_error_raising(self):
        """Test raising NodeValidationError"""
        with pytest.raises(NodeValidationError):
            raise NodeValidationError("Test validation error")

    def test_node_validation_error_catching(self):
        """Test catching NodeValidationError"""
        try:
            raise NodeValidationError("Test error")
        except NodeValidationError as e:
            assert str(e) == "Test error"
        else:
            pytest.fail("NodeValidationError was not raised")

    def test_node_validation_error_with_multiple_args(self):
        """Test NodeValidationError with multiple arguments"""
        error = NodeValidationError("Error message", 123, {"key": "value"})
        assert error.args == ("Error message", 123, {"key": "value"})

    def test_node_validation_error_docstring(self):
        """Test NodeValidationError has correct docstring"""
        assert NodeValidationError.__doc__ == "Raised when node state validation fails."

    def test_node_validation_error_str_representation(self):
        """Test string representation of NodeValidationError"""
        error = NodeValidationError("Validation failed for node X")
        assert str(error) == "Validation failed for node X"

    def test_node_validation_error_repr(self):
        """Test repr of NodeValidationError"""
        error = NodeValidationError("Test error")
        repr_str = repr(error)
        assert "NodeValidationError" in repr_str
        assert "Test error" in repr_str


class TestNodeExecutionError:
    """Test suite for NodeExecutionError exception"""

    def test_node_execution_error_inheritance(self):
        """Test that NodeExecutionError inherits from Exception"""
        assert issubclass(NodeExecutionError, Exception)

    def test_node_execution_error_creation_no_message(self):
        """Test NodeExecutionError creation without message"""
        error = NodeExecutionError()
        assert isinstance(error, NodeExecutionError)
        assert isinstance(error, Exception)

    def test_node_execution_error_creation_with_message(self):
        """Test NodeExecutionError creation with message"""
        message = "Node execution failed"
        error = NodeExecutionError(message)
        assert str(error) == message

    def test_node_execution_error_raising(self):
        """Test raising NodeExecutionError"""
        with pytest.raises(NodeExecutionError):
            raise NodeExecutionError("Test execution error")

    def test_node_execution_error_catching(self):
        """Test catching NodeExecutionError"""
        try:
            raise NodeExecutionError("Test error")
        except NodeExecutionError as e:
            assert str(e) == "Test error"
        else:
            pytest.fail("NodeExecutionError was not raised")

    def test_node_execution_error_with_multiple_args(self):
        """Test NodeExecutionError with multiple arguments"""
        error = NodeExecutionError("Error message", 456, ["item1", "item2"])
        assert error.args == ("Error message", 456, ["item1", "item2"])

    def test_node_execution_error_docstring(self):
        """Test NodeExecutionError has correct docstring"""
        assert NodeExecutionError.__doc__ == "Raised when node execution fails."

    def test_node_execution_error_str_representation(self):
        """Test string representation of NodeExecutionError"""
        error = NodeExecutionError("Execution failed for node Y")
        assert str(error) == "Execution failed for node Y"

    def test_node_execution_error_repr(self):
        """Test repr of NodeExecutionError"""
        error = NodeExecutionError("Test error")
        repr_str = repr(error)
        assert "NodeExecutionError" in repr_str
        assert "Test error" in repr_str


class TestConfigExtractionError:
    """Test suite for ConfigExtractionError exception"""

    def test_config_extraction_error_inheritance(self):
        """Test that ConfigExtractionError inherits from Exception"""
        assert issubclass(ConfigExtractionError, Exception)

    def test_config_extraction_error_creation_no_message(self):
        """Test ConfigExtractionError creation without message"""
        error = ConfigExtractionError()
        assert isinstance(error, ConfigExtractionError)
        assert isinstance(error, Exception)

    def test_config_extraction_error_creation_with_message(self):
        """Test ConfigExtractionError creation with message"""
        message = "Config extraction failed"
        error = ConfigExtractionError(message)
        assert str(error) == message

    def test_config_extraction_error_raising(self):
        """Test raising ConfigExtractionError"""
        with pytest.raises(ConfigExtractionError):
            raise ConfigExtractionError("Test config extraction error")

    def test_config_extraction_error_catching(self):
        """Test catching ConfigExtractionError"""
        try:
            raise ConfigExtractionError("Test error")
        except ConfigExtractionError as e:
            assert str(e) == "Test error"
        else:
            pytest.fail("ConfigExtractionError was not raised")

    def test_config_extraction_error_with_multiple_args(self):
        """Test ConfigExtractionError with multiple arguments"""
        error = ConfigExtractionError("Error message", {"config": "invalid"}, 789)
        assert error.args == ("Error message", {"config": "invalid"}, 789)

    def test_config_extraction_error_docstring(self):
        """Test ConfigExtractionError has correct docstring"""
        assert ConfigExtractionError.__doc__ == "Raised when config extraction fails."

    def test_config_extraction_error_str_representation(self):
        """Test string representation of ConfigExtractionError"""
        error = ConfigExtractionError("Config extraction failed for component Z")
        assert str(error) == "Config extraction failed for component Z"

    def test_config_extraction_error_repr(self):
        """Test repr of ConfigExtractionError"""
        error = ConfigExtractionError("Test error")
        repr_str = repr(error)
        assert "ConfigExtractionError" in repr_str
        assert "Test error" in repr_str


class TestExceptionHierarchy:
    """Test suite for exception hierarchy and relationships"""

    def test_all_exceptions_inherit_from_base_exception(self):
        """Test that all custom exceptions inherit from base Exception"""
        exceptions = [NodeValidationError, NodeExecutionError, ConfigExtractionError]

        for exc_class in exceptions:
            assert issubclass(exc_class, Exception)

    def test_exceptions_are_distinct_classes(self):
        """Test that all exceptions are distinct classes"""
        assert NodeValidationError != NodeExecutionError
        assert NodeValidationError != ConfigExtractionError
        assert NodeExecutionError != ConfigExtractionError

    def test_exception_type_checking(self):
        """Test exception type checking"""
        validation_error = NodeValidationError("validation error")
        execution_error = NodeExecutionError("execution error")
        config_error = ConfigExtractionError("config error")

        assert isinstance(validation_error, NodeValidationError)
        assert not isinstance(validation_error, NodeExecutionError)
        assert not isinstance(validation_error, ConfigExtractionError)

        assert isinstance(execution_error, NodeExecutionError)
        assert not isinstance(execution_error, NodeValidationError)
        assert not isinstance(execution_error, ConfigExtractionError)

        assert isinstance(config_error, ConfigExtractionError)
        assert not isinstance(config_error, NodeValidationError)
        assert not isinstance(config_error, NodeExecutionError)

    def test_exception_catching_hierarchy(self):
        """Test exception catching with base Exception"""
        # All should be catchable as base Exception
        with pytest.raises(Exception):
            raise NodeValidationError("test")

        with pytest.raises(Exception):
            raise NodeExecutionError("test")

        with pytest.raises(Exception):
            raise ConfigExtractionError("test")

    def test_exception_chaining(self):
        """Test exception chaining"""
        original_error = ValueError("Original error")

        try:
            raise original_error
        except ValueError as e:
            # Chain with NodeValidationError
            with pytest.raises(NodeValidationError) as exc_info:
                raise NodeValidationError("Validation failed") from e

            assert exc_info.value.__cause__ is original_error

    def test_exception_context_manager(self):
        """Test exceptions in context managers"""

        class TestContextManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is NodeValidationError:
                    return True  # Suppress the exception
                return False

        # Test suppression
        with TestContextManager():
            raise NodeValidationError("This should be suppressed")

        # Test non-suppression
        with pytest.raises(NodeExecutionError):
            with TestContextManager():
                raise NodeExecutionError("This should not be suppressed")

    def test_exception_equality(self):
        """Test exception equality"""
        error1 = NodeValidationError("same message")
        error2 = NodeValidationError("same message")
        error3 = NodeValidationError("different message")

        # Exception instances are not equal even with same message
        assert error1 != error2
        assert error1 != error3

        # But they have the same type
        assert type(error1) is type(error2)
        assert type(error1) is not type(NodeExecutionError("test"))

    def test_exception_with_traceback(self):
        """Test exceptions preserve traceback information"""
        import traceback

        try:
            raise NodeValidationError("Test error with traceback")
        except NodeValidationError:
            tb = traceback.format_exc()
            assert "NodeValidationError" in tb
            assert "Test error with traceback" in tb
            assert "test_exception_with_traceback" in tb

    def test_exception_attributes(self):
        """Test exception attributes"""
        error = NodeValidationError("test message")

        assert hasattr(error, "args")
        assert hasattr(error, "__str__")
        assert hasattr(error, "__repr__")
        assert hasattr(error, "__class__")

        assert error.args == ("test message",)

    def test_exception_with_custom_attributes(self):
        """Test exceptions with custom attributes"""
        error = NodeValidationError("test")
        error.custom_attr = "custom_value"
        error.error_code = 123

        assert error.custom_attr == "custom_value"
        assert error.error_code == 123

    def test_exception_pickling(self):
        """Test exception pickling/unpickling"""
        import pickle

        error = NodeValidationError("test error")

        # Pickle and unpickle
        pickled = pickle.dumps(error)
        unpickled = pickle.loads(pickled)

        assert isinstance(unpickled, NodeValidationError)
        assert str(unpickled) == str(error)
        assert unpickled.args == error.args
