"""
Unit tests for the NLPdtSearch class.

These tests use mocking to avoid making actual API calls to OpenAI,
making tests fast and reliable without requiring API keys or internet access.
"""

import pytest
from unittest.mock import Mock, patch
from openai import OpenAI

from nlp_dtsearch import NLPdtSearch


class TestNLPdtSearchInitialization:
    """Test the initialization and configuration of NLPdtSearch."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"})
    @patch("nlp_dtsearch.load_dotenv")
    @patch("nlp_dtsearch.OpenAI")
    def test_init_with_env_key(self, mock_openai_class, mock_load_dotenv):
        """Test initialization with API key from environment."""
        converter = NLPdtSearch(auto_load_env=False)
        
        assert converter.api_key == "sk-test123"
        assert converter.model == "gpt-4o-mini"
        assert converter.system_prompt == NLPdtSearch.DEFAULT_SYSTEM_PROMPT
        assert converter.user_prompt_prefix == NLPdtSearch.DEFAULT_USER_PROMPT_PREFIX
        mock_openai_class.assert_called_once_with(api_key="sk-test123")
    
    @patch("nlp_dtsearch.load_dotenv")
    @patch("nlp_dtsearch.OpenAI")
    def test_init_with_custom_api_key(self, mock_openai_class, mock_load_dotenv):
        """Test initialization with custom API key."""
        converter = NLPdtSearch(api_key="sk-custom456", auto_load_env=False)
        
        assert converter.api_key == "sk-custom456"
        mock_openai_class.assert_called_once_with(api_key="sk-custom456")
    
    @patch("nlp_dtsearch.load_dotenv")
    @patch("nlp_dtsearch.OpenAI")
    def test_init_with_custom_model(self, mock_openai_class, mock_load_dotenv):
        """Test initialization with custom model."""
        converter = NLPdtSearch(model="gpt-4", api_key="sk-test", auto_load_env=False)
        
        assert converter.model == "gpt-4"
    
    @patch("nlp_dtsearch.load_dotenv")
    @patch("nlp_dtsearch.OpenAI")
    def test_init_with_custom_prompts(self, mock_openai_class, mock_load_dotenv):
        """Test initialization with custom prompts."""
        custom_system = "Custom system prompt"
        custom_user = "Custom user prefix: "
        
        converter = NLPdtSearch(
            system_prompt=custom_system,
            user_prompt_prefix=custom_user,
            api_key="sk-test",
            auto_load_env=False
        )
        
        assert converter.system_prompt == custom_system
        assert converter.user_prompt_prefix == custom_user
    
    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
                NLPdtSearch(auto_load_env=False)
    
    def test_init_with_invalid_api_key_format(self):
        """Test that invalid API key format raises ValueError."""
        with pytest.raises(ValueError, match="not a valid OpenAI API key"):
            NLPdtSearch(api_key="invalid-key", auto_load_env=False)
    
    def test_init_with_api_key_containing_whitespace(self):
        """Test that API key with whitespace raises ValueError."""
        with pytest.raises(ValueError, match="contains whitespace"):
            NLPdtSearch(api_key="sk-test123 ", auto_load_env=False)


class TestNLPdtSearchMethods:
    """Test the methods of NLPdtSearch class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "test dtSearch query result"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def converter(self, mock_client):
        """Create an NLPdtSearch instance with mocked client."""
        with patch("nlp_dtsearch.OpenAI", return_value=mock_client):
            return NLPdtSearch(api_key="sk-test123", auto_load_env=False)
    
    def test_build_messages(self, converter):
        """Test the _build_messages method."""
        query = "test query"
        messages = converter._build_messages(query)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == converter.system_prompt
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == converter.user_prompt_prefix + query
    
    def test_convert_to_dtSearch(self, converter, mock_client):
        """Test the convert_to_dtSearch method."""
        query = "find documents about python"
        result, warning = converter.convert_to_dtSearch(query)
        assert result == "test dtSearch query result"
        assert warning is None

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("max_tokens") == converter.max_completion_tokens
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert len(call_args.kwargs["messages"]) == 2
        assert call_args.kwargs["messages"][0]["role"] == "system"
        assert call_args.kwargs["messages"][1]["role"] == "user"
        assert query in call_args.kwargs["messages"][1]["content"]
    
    def test_update_model(self, converter):
        """Test updating the model at runtime."""
        converter.update_model("gpt-4")
        assert converter.model == "gpt-4"
    
    def test_update_system_prompt(self, converter):
        """Test updating the system prompt at runtime."""
        new_prompt = "New system prompt"
        converter.update_system_prompt(new_prompt)
        assert converter.system_prompt == new_prompt
    
    def test_update_user_prompt_prefix(self, converter):
        """Test updating the user prompt prefix at runtime."""
        new_prefix = "New prefix: "
        converter.update_user_prompt_prefix(new_prefix)
        assert converter.user_prompt_prefix == new_prefix


class TestNLPdtSearchValidation:
    """Test API key validation logic."""
    
    def test_validate_api_key_valid(self):
        """Test validation with a valid API key."""
        # Should not raise an exception
        NLPdtSearch._validate_api_key("sk-test123456789")
    
    def test_validate_api_key_none(self):
        """Test validation with None API key."""
        with pytest.raises(ValueError, match="not set"):
            NLPdtSearch._validate_api_key(None)
    
    def test_validate_api_key_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(ValueError, match="not set"):
            NLPdtSearch._validate_api_key("")
    
    def test_validate_api_key_invalid_format(self):
        """Test validation with invalid format."""
        with pytest.raises(ValueError, match="not a valid OpenAI API key"):
            NLPdtSearch._validate_api_key("invalid-key")
    
    def test_validate_api_key_with_leading_whitespace(self):
        """Test validation with leading whitespace."""
        with pytest.raises(ValueError, match="contains whitespace"):
            NLPdtSearch._validate_api_key(" sk-test123")
    
    def test_validate_api_key_with_trailing_whitespace(self):
        """Test validation with trailing whitespace."""
        with pytest.raises(ValueError, match="contains whitespace"):
            NLPdtSearch._validate_api_key("sk-test123 ")


class TestNLPdtSearchIntegration:
    """Integration tests that verify the full flow (with mocking)."""
    
    @patch("nlp_dtsearch.OpenAI")
    def test_full_conversion_flow(self, mock_openai_class):
        """Test the complete flow from initialization to conversion."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "(apples AND bananas) NEAR/5 grape"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create converter and convert
        converter = NLPdtSearch(api_key="sk-test", auto_load_env=False)
        result, _ = converter.convert_to_dtSearch("apples and bananas near grape")
        assert result == "(apples AND bananas) NEAR/5 grape"
        
        # Verify API call was made
        assert mock_client.chat.completions.create.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
