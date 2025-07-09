import pytest
from pydantic import ValidationError

from quivr_core.rag.entities.prompt import PromptConfig
from quivr_core.rag.prompts import TemplatePromptName


class TestPromptConfig:
    """Test suite for PromptConfig class"""

    def test_init_with_custom_prompt(self):
        """Test PromptConfig initialization with custom prompt"""
        custom_prompt = "You are a helpful assistant. Answer the question: {question}"

        config = PromptConfig(prompt=custom_prompt)

        assert config.prompt == custom_prompt
        assert config.template_name is None

    def test_init_with_template_name(self):
        """Test PromptConfig initialization with template name"""
        template_name = TemplatePromptName.RAG_ANSWER_PROMPT

        config = PromptConfig(template_name=template_name)

        assert config.template_name == template_name
        assert config.prompt is None

    def test_init_with_both_prompt_and_template(self):
        """Test PromptConfig initialization with both prompt and template"""
        custom_prompt = "You are a helpful assistant."
        template_name = TemplatePromptName.RAG_ANSWER_PROMPT

        config = PromptConfig(prompt=custom_prompt, template_name=template_name)

        assert config.prompt == custom_prompt
        assert config.template_name == template_name

    def test_init_with_none_values(self):
        """Test PromptConfig initialization with None values"""
        config = PromptConfig(prompt=None, template_name=None)

        assert config.prompt is None
        assert config.template_name is None

    def test_init_empty(self):
        """Test PromptConfig initialization with no parameters"""
        config = PromptConfig()

        assert config.prompt is None
        assert config.template_name is None

    def test_init_from_dict(self):
        """Test PromptConfig initialization from dictionary"""
        data = {
            "prompt": "Custom prompt text",
            "template_name": TemplatePromptName.RAG_ANSWER_PROMPT,
        }

        config = PromptConfig(**data)

        assert config.prompt == "Custom prompt text"
        assert config.template_name == TemplatePromptName.RAG_ANSWER_PROMPT

    def test_template_name_enum_validation(self):
        """Test that template_name accepts valid enum values"""
        # Test all valid enum values
        for template_name in TemplatePromptName:
            config = PromptConfig(template_name=template_name)
            assert config.template_name == template_name

    def test_template_name_invalid_value(self):
        """Test that template_name rejects invalid values"""
        with pytest.raises(ValidationError):
            PromptConfig(template_name="invalid_template_name")

    def test_prompt_string_validation(self):
        """Test prompt string validation"""
        # Valid string
        config = PromptConfig(prompt="Valid prompt")
        assert config.prompt == "Valid prompt"

        # Empty string
        config = PromptConfig(prompt="")
        assert config.prompt == ""

        # Multiline string
        multiline_prompt = """
        You are a helpful assistant.
        Please answer the following question:
        {question}
        """
        config = PromptConfig(prompt=multiline_prompt)
        assert config.prompt == multiline_prompt

    def test_prompt_with_placeholders(self):
        """Test prompt with template placeholders"""
        prompt_with_placeholders = (
            "Answer this question: {question} using context: {context}"
        )

        config = PromptConfig(prompt=prompt_with_placeholders)

        assert config.prompt == prompt_with_placeholders
        assert "{question}" in config.prompt
        assert "{context}" in config.prompt

    def test_serialization(self):
        """Test PromptConfig serialization"""
        config = PromptConfig(
            prompt="Test prompt", template_name=TemplatePromptName.RAG_ANSWER_PROMPT
        )

        # Test dict conversion
        config_dict = config.model_dump()
        expected_dict = {
            "prompt": "Test prompt",
            "template_name": TemplatePromptName.RAG_ANSWER_PROMPT,
        }
        assert config_dict == expected_dict

    def test_deserialization(self):
        """Test PromptConfig deserialization"""
        config_dict = {
            "prompt": "Test prompt",
            "template_name": TemplatePromptName.RAG_ANSWER_PROMPT.value,
        }

        config = PromptConfig.model_validate(config_dict)

        assert config.prompt == "Test prompt"
        assert config.template_name == TemplatePromptName.RAG_ANSWER_PROMPT

    def test_json_serialization(self):
        """Test PromptConfig JSON serialization"""
        config = PromptConfig(
            prompt="Test prompt", template_name=TemplatePromptName.RAG_ANSWER_PROMPT
        )

        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test prompt" in json_str
        assert TemplatePromptName.RAG_ANSWER_PROMPT.value in json_str

    def test_json_deserialization(self):
        """Test PromptConfig JSON deserialization"""
        json_str = f'{{"prompt": "Test prompt", "template_name": "{TemplatePromptName.RAG_ANSWER_PROMPT.value}"}}'

        config = PromptConfig.model_validate_json(json_str)

        assert config.prompt == "Test prompt"
        assert config.template_name == TemplatePromptName.RAG_ANSWER_PROMPT

    def test_equality(self):
        """Test PromptConfig equality"""
        config1 = PromptConfig(
            prompt="Test prompt", template_name=TemplatePromptName.RAG_ANSWER_PROMPT
        )

        config2 = PromptConfig(
            prompt="Test prompt", template_name=TemplatePromptName.RAG_ANSWER_PROMPT
        )

        config3 = PromptConfig(
            prompt="Different prompt",
            template_name=TemplatePromptName.RAG_ANSWER_PROMPT,
        )

        assert config1 == config2
        assert config1 != config3

    def test_copy(self):
        """Test PromptConfig copying"""
        original = PromptConfig(
            prompt="Original prompt", template_name=TemplatePromptName.RAG_ANSWER_PROMPT
        )

        # Test shallow copy
        copied = original.model_copy()

        assert copied == original
        assert copied is not original
        assert copied.prompt == original.prompt
        assert copied.template_name == original.template_name

    def test_update(self):
        """Test PromptConfig update"""
        config = PromptConfig(prompt="Original prompt")

        # Update with new values
        updated = config.model_copy(
            update={"template_name": TemplatePromptName.RAG_ANSWER_PROMPT}
        )

        assert updated.prompt == "Original prompt"
        assert updated.template_name == TemplatePromptName.RAG_ANSWER_PROMPT

        # Original should remain unchanged
        assert config.template_name is None

    def test_special_characters_in_prompt(self):
        """Test prompt with special characters"""
        special_prompt = "Prompt with émojis 🤖 and spëcîâl characters: @#$%^&*()"

        config = PromptConfig(prompt=special_prompt)

        assert config.prompt == special_prompt

    def test_very_long_prompt(self):
        """Test with very long prompt"""
        long_prompt = "This is a very long prompt. " * 1000

        config = PromptConfig(prompt=long_prompt)

        assert config.prompt == long_prompt
        assert len(config.prompt) > 10000

    def test_inheritance_from_base_config(self):
        """Test that PromptConfig inherits from QuivrBaseConfig"""
        from quivr_core.base_config import QuivrBaseConfig

        config = PromptConfig()

        assert isinstance(config, QuivrBaseConfig)

        # Test that it has QuivrBaseConfig methods
        assert hasattr(config, "model_dump")
        assert hasattr(config, "model_validate")
        assert hasattr(config, "model_copy")
