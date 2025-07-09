import pytest
from quivr_core.rag.entities.utils import normalize_to_env_variable_name


class TestNormalizeToEnvVariableName:
    """Test suite for normalize_to_env_variable_name function"""

    def test_basic_string(self):
        """Test normalization of basic string"""
        result = normalize_to_env_variable_name("test_string")
        assert result == "TEST_STRING"

    def test_lowercase_to_uppercase(self):
        """Test conversion from lowercase to uppercase"""
        result = normalize_to_env_variable_name("lowercase")
        assert result == "LOWERCASE"

    def test_mixed_case_to_uppercase(self):
        """Test conversion from mixed case to uppercase"""
        result = normalize_to_env_variable_name("MixedCase")
        assert result == "MIXEDCASE"

    def test_spaces_to_underscores(self):
        """Test conversion of spaces to underscores"""
        result = normalize_to_env_variable_name("test with spaces")
        assert result == "TEST_WITH_SPACES"

    def test_hyphens_to_underscores(self):
        """Test conversion of hyphens to underscores"""
        result = normalize_to_env_variable_name("test-with-hyphens")
        assert result == "TEST_WITH_HYPHENS"

    def test_dots_to_underscores(self):
        """Test conversion of dots to underscores"""
        result = normalize_to_env_variable_name("test.with.dots")
        assert result == "TEST_WITH_DOTS"

    def test_special_characters_to_underscores(self):
        """Test conversion of special characters to underscores"""
        result = normalize_to_env_variable_name("test@#$%^&*()+=[]{}|;:,.<>?/")
        assert result == "TEST________________________"

    def test_multiple_special_characters(self):
        """Test conversion of multiple consecutive special characters"""
        result = normalize_to_env_variable_name("test---with___multiple")
        assert result == "TEST___WITH___MULTIPLE"

    def test_numbers_preserved(self):
        """Test that numbers are preserved in the result"""
        result = normalize_to_env_variable_name("test123with456numbers")
        assert result == "TEST123WITH456NUMBERS"

    def test_underscores_preserved(self):
        """Test that underscores are preserved"""
        result = normalize_to_env_variable_name("test_with_underscores")
        assert result == "TEST_WITH_UNDERSCORES"

    def test_alphanumeric_and_underscores_only(self):
        """Test string with only valid characters"""
        result = normalize_to_env_variable_name("Valid_String_123")
        assert result == "VALID_STRING_123"

    def test_empty_string(self):
        """Test empty string handling"""
        with pytest.raises(IndexError):
            # Empty string will cause IndexError when checking first character
            normalize_to_env_variable_name("")

    def test_single_character(self):
        """Test single character strings"""
        result = normalize_to_env_variable_name("a")
        assert result == "A"

        with pytest.raises(
            ValueError,
            match="Invalid environment variable name '1': Cannot start with a digit",
        ):
            normalize_to_env_variable_name("1")

        result = normalize_to_env_variable_name("_")
        assert result == "_"

    def test_starts_with_digit_raises_error(self):
        """Test that starting with a digit raises ValueError"""
        with pytest.raises(
            ValueError,
            match="Invalid environment variable name '123_TEST': Cannot start with a digit",
        ):
            normalize_to_env_variable_name("123_test")

    def test_starts_with_digit_after_normalization_raises_error(self):
        """Test that starting with a digit after normalization raises ValueError"""
        with pytest.raises(
            ValueError,
            match="Invalid environment variable name '1TEST': Cannot start with a digit",
        ):
            normalize_to_env_variable_name("1test")

    def test_unicode_characters_to_underscores(self):
        """Test conversion of unicode characters to underscores"""
        result = normalize_to_env_variable_name("test_with_émojis_🤖")
        assert result == "TEST_WITH__MOJIS__"

    def test_complex_mixed_string(self):
        """Test complex string with mixed characters"""
        result = normalize_to_env_variable_name(
            "Complex-String.With@Various#Characters_123"
        )
        assert result == "COMPLEX_STRING_WITH_VARIOUS_CHARACTERS_123"

    def test_sql_injection_like_string(self):
        """Test string that might look like SQL injection"""
        result = normalize_to_env_variable_name("'; DROP TABLE users; --")
        assert result == "___DROP_TABLE_USERS____"

    def test_path_like_string(self):
        """Test path-like strings"""
        result = normalize_to_env_variable_name("/path/to/file.txt")
        assert result == "_PATH_TO_FILE_TXT"

    def test_url_like_string(self):
        """Test URL-like strings"""
        result = normalize_to_env_variable_name("https://example.com/api/v1")
        assert result == "HTTPS___EXAMPLE_COM_API_V1"

    def test_email_like_string(self):
        """Test email-like strings"""
        result = normalize_to_env_variable_name("user@example.com")
        assert result == "USER_EXAMPLE_COM"

    def test_json_key_like_string(self):
        """Test JSON key-like strings"""
        result = normalize_to_env_variable_name("data.nested.key")
        assert result == "DATA_NESTED_KEY"

    def test_camel_case_string(self):
        """Test camelCase strings"""
        result = normalize_to_env_variable_name("camelCaseString")
        assert result == "CAMELCASESTRING"

    def test_pascal_case_string(self):
        """Test PascalCase strings"""
        result = normalize_to_env_variable_name("PascalCaseString")
        assert result == "PASCALCASESTRING"

    def test_kebab_case_string(self):
        """Test kebab-case strings"""
        result = normalize_to_env_variable_name("kebab-case-string")
        assert result == "KEBAB_CASE_STRING"

    def test_snake_case_string(self):
        """Test snake_case strings"""
        result = normalize_to_env_variable_name("snake_case_string")
        assert result == "SNAKE_CASE_STRING"

    def test_screaming_snake_case_string(self):
        """Test SCREAMING_SNAKE_CASE strings"""
        result = normalize_to_env_variable_name("SCREAMING_SNAKE_CASE")
        assert result == "SCREAMING_SNAKE_CASE"

    def test_multiple_consecutive_spaces(self):
        """Test multiple consecutive spaces"""
        result = normalize_to_env_variable_name("test    with    spaces")
        assert result == "TEST____WITH____SPACES"

    def test_leading_and_trailing_spaces(self):
        """Test leading and trailing spaces"""
        result = normalize_to_env_variable_name("  test  ")
        assert result == "__TEST__"

    def test_only_special_characters(self):
        """Test string with only special characters"""
        result = normalize_to_env_variable_name("@#$%^&*()")
        assert result == "_________"

    def test_only_numbers(self):
        """Test string with only numbers"""
        with pytest.raises(
            ValueError,
            match="Invalid environment variable name '123456': Cannot start with a digit",
        ):
            normalize_to_env_variable_name("123456")

    def test_only_underscores(self):
        """Test string with only underscores"""
        result = normalize_to_env_variable_name("___")
        assert result == "___"

    def test_regex_pattern_validation(self):
        """Test that the regex pattern correctly identifies non-alphanumeric/underscore characters"""
        # Test various character types
        test_cases = [
            ("abc123_", "ABC123_"),  # Valid characters only
            ("abc@123", "ABC_123"),  # Special character in middle
            ("abc 123", "ABC_123"),  # Space
            ("abc\t123", "ABC_123"),  # Tab
            ("abc\n123", "ABC_123"),  # Newline
            ("abc-123", "ABC_123"),  # Hyphen
            ("abc+123", "ABC_123"),  # Plus
            ("abc=123", "ABC_123"),  # Equals
        ]

        for input_str, expected in test_cases:
            result = normalize_to_env_variable_name(input_str)
            assert (
                result == expected
            ), f"Failed for input '{input_str}': expected '{expected}', got '{result}'"

    def test_edge_case_single_digit(self):
        """Test edge case with single digit"""
        with pytest.raises(
            ValueError,
            match="Invalid environment variable name '5': Cannot start with a digit",
        ):
            normalize_to_env_variable_name("5")

    def test_edge_case_underscore_then_digit(self):
        """Test edge case with underscore then digit"""
        result = normalize_to_env_variable_name("_5")
        assert result == "_5"

    def test_very_long_string(self):
        """Test very long string"""
        long_string = "a" * 1000
        result = normalize_to_env_variable_name(long_string)
        assert result == "A" * 1000
        assert len(result) == 1000
