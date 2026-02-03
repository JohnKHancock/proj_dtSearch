import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict, Tuple


class NLPdtSearch:
    """
    A class that converts natural language queries into dtSearch syntax queries
    using OpenAI's GPT models.
    
    This class encapsulates the OpenAI client, prompt configuration, and
    conversion logic for easy reuse and configuration.
    """
    
    # Default prompts as class constants - can be overridden
    DEFAULT_SYSTEM_PROMPT = """You are a professional dtSearch syntax assistant. You help users convert natural language search queries into valid dtSearch syntax queries.

IMPORTANT CONVERSATION GUIDELINES:
- Be professional, courteous, and helpful in all interactions
- When greeted (e.g., "Hello", "Hi"), respond professionally: "Hello, I am a dtSearch assistant. What can I help you with?"
- Engage in friendly conversation while maintaining professionalism
- Always be ready to help with dtSearch query conversion
- Provide clear explanations when needed

dtSearch SYNTAX OPERATORS:
- AND: Both terms must appear (default between terms)
- OR: Either term can appear
- NOT: Exclude term
- NEAR/n: Terms within n words of each other (e.g., NEAR/5)
- WITHIN/n: Terms within n words in order (e.g., WITHIN/10)
- *: Wildcard (e.g., comput* matches computer, computing, etc.)
- ?: Single character wildcard (e.g., colo?r matches color, colour)
- #: Fuzzy search for typos (e.g., Amoxicillin# finds Amoxicillin, Amoxicilin, etc.)
- ~: Root word search (e.g., run~ matches run, running, runner, etc.)
- ||: Phonetic search for similar-sounding words (e.g., Smith|| matches Smyth, etc.)
- "phrase": Exact phrase match

When users provide search queries, convert them to dtSearch syntax. When they engage in conversation, respond professionally and helpfully.
"""
    
    DEFAULT_USER_PROMPT_PREFIX = """
    The user has sent you a message. If it's a greeting or casual conversation, respond professionally and helpfully. 
    If it's a search query, convert it to dtSearch syntax and provide friendly, helpful response with multiple suggestions. 
    Assume the user is not familiar with dtSearch syntax. Explain the different search strategies and when to use each.

User message: """
    
    DEFAULT_SUGGESTIONS_SYSTEM_PROMPT = """You are a helpful dtSearch syntax specialist. You help users create effective search queries by providing multiple search strategy options.

dtSearch SYNTAX OPERATORS:
- AND: Both terms must appear (default between terms)
- OR: Either term can appear
- NOT: Exclude term
- NEAR/n: Terms within n words of each other (e.g., NEAR/5)
- WITHIN/n: Terms within n words in order (e.g., WITHIN/10)
- *: Wildcard (e.g., comput* matches computer, computing, etc.)
- ?: Single character wildcard (e.g., colo?r matches color, colour)
- #: Fuzzy search for typos (e.g., Amoxicillin# finds Amoxicillin, Amoxicilin, etc.)
- ~: Root word search (e.g., run~ matches run, running, runner, etc.)
- ||: Phonetic search for similar-sounding words (e.g., Smith|| matches Smyth, etc.)
- "phrase": Exact phrase match

When providing suggestions, consider:
1. EXACT: The exact term as specified
2. FUZZY: Using # for typo tolerance
3. WILDCARD: Using * or ? for variations
4. ROOT WORD: Using ~ for word variants
5. PHONETIC: Using || for similar-sounding words
6. COMBINATIONS: Combining multiple strategies

Provide friendly, helpful response. Offer multiple suggestions. Assume the user is not familiar with dtSearch syntax. 
Explain the different search strategies and when to use each
"""
    
    DEFAULT_SUGGESTIONS_USER_PROMPT_PREFIX = """The user wants to search for: "{query}"

Please provide multiple dtSearch query suggestions with brief explanations. Include options for:
- Exact match
- Fuzzy search (for typos)
- Wildcard search (for variations)
- Root word search (for word variants)
- Phonetic search (for similar sounds)
- Any relevant combinations

Format your response as a numbered list with each suggestion showing:
1. The dtSearch query syntax
2. A brief explanation of what it finds

Be friendly and helpful!
"""

    # Token limits (approximate: ~4 chars per token for English)
    DEFAULT_MAX_INPUT_TOKENS = 12_000   # system + history + current message
    DEFAULT_MAX_USER_MESSAGE_TOKENS = 2_000  # current message only; over = warn and reject
    DEFAULT_MAX_COMPLETION_TOKENS = 1_024     # API max_tokens for response

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count (~4 chars per token). For accuracy, use tiktoken."""
        if not text:
            return 0
        return (len(text) + 3) // 4

    @classmethod
    def _count_messages_tokens(cls, messages: List[Dict[str, str]]) -> int:
        """Total estimated tokens for a list of message dicts with 'content'."""
        return sum(cls._estimate_tokens(m.get("content", "")) for m in messages)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        system_prompt: Optional[str] = None,
        user_prompt_prefix: Optional[str] = None,
        suggestions_system_prompt: Optional[str] = None,
        auto_load_env: bool = True
    ):
        """
        Initialize the NLPdtSearch converter.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment variable.
            model: The OpenAI model to use for conversion (default: "gpt-4o-mini").
            system_prompt: Custom system prompt. If None, uses default.
            user_prompt_prefix: Custom user prompt prefix. If None, uses default.
            suggestions_system_prompt: Custom system prompt for suggestions method. If None, uses default.
            auto_load_env: Whether to automatically load .env file (default: True).
            
        Raises:
            ValueError: If API key is not provided or invalid.
        """
        # Get API key from parameter first
        if api_key:
            self.api_key = api_key
        else:
            # Check if API key is already in environment
            self.api_key = os.getenv("OPENAI_API_KEY")
            
            # Only load .env file if key is not found and auto_load_env is True
            if not self.api_key and auto_load_env:
                load_dotenv(override=True)
                self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate API key
        self._validate_api_key(self.api_key)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Set model
        self.model = model
        
        # Set prompts (use provided or defaults)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_prefix = user_prompt_prefix or self.DEFAULT_USER_PROMPT_PREFIX
        self.suggestions_system_prompt = suggestions_system_prompt or self.DEFAULT_SUGGESTIONS_SYSTEM_PROMPT
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.suggestions_conversation_history: List[Dict[str, str]] = []

        # Token limits (env overrides: MAX_INPUT_TOKENS, MAX_USER_MESSAGE_TOKENS, MAX_COMPLETION_TOKENS)
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", self.DEFAULT_MAX_INPUT_TOKENS))
        self.max_user_message_tokens = int(os.getenv("MAX_USER_MESSAGE_TOKENS", self.DEFAULT_MAX_USER_MESSAGE_TOKENS))
        self.max_completion_tokens = int(os.getenv("MAX_COMPLETION_TOKENS", self.DEFAULT_MAX_COMPLETION_TOKENS))
    
    @staticmethod
    def _validate_api_key(api_key: Optional[str]) -> None:
        """
        Validate that the API key is present and properly formatted.
        
        Args:
            api_key: The API key to validate.
            
        Raises:
            ValueError: If API key is missing or invalid.
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        elif api_key.strip() != api_key:
            raise ValueError("OPENAI_API_KEY contains whitespace")
        elif not api_key.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY is not a valid OpenAI API key")
    
    def _build_messages(
        self,
        query: str,
        use_history: bool = True,
        conversation_type: str = "standard",
        history_override: Optional[List[Dict[str, str]]] = None,
    ) -> list[dict[str, str]]:
        """
        Build the message array for the OpenAI API call, including conversation history.

        Args:
            query: The natural language query to convert.
            use_history: Whether to include conversation history (default: True).
            conversation_type: Type of conversation ("standard" or "suggestions").
            history_override: If set, use this instead of internal history (e.g. trimmed for token limit).

        Returns:
            List of message dictionaries for the OpenAI API.
        """
        messages = []
        history = (
            history_override
            if history_override is not None
            else (self.conversation_history if conversation_type == "standard" else self.suggestions_conversation_history)
        )

        if not history or not use_history:
            messages.append({"role": "system", "content": self.system_prompt})

        if use_history and history:
            messages.extend(history)

        user_content = self.user_prompt_prefix + query if conversation_type == "standard" else query
        messages.append({"role": "user", "content": user_content})

        return messages
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and constrain the response to ensure it contains only dtSearch syntax.
        
        Args:
            response: The raw response from the LLM.
            
        Returns:
            Cleaned dtSearch query string.
        """
        if not response:
            return response
        
        # Check if the response is an error message
        if "I'm sorry" in response or "can only help with dtSearch" in response:
            return response.strip()
        
        # Remove common prefixes/suffixes that LLM might add
        prefixes_to_remove = [
            "dtSearch query:",
            "dtSearch:",
            "Here is the query:",
            "Query:",
            "The dtSearch syntax is:",
            "dtSearch syntax:"
        ]
        
        cleaned = response.strip()
        
        # Remove prefixes (case-insensitive)
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```)
            if len(lines) > 1:
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        
        # Remove leading/trailing quotes
        cleaned = cleaned.strip('"\'`')
        
        return cleaned.strip()
    
    def convert_to_dtSearch(self, query: str, use_history: bool = True) -> Tuple[str, Optional[str]]:
        """
        Convert a natural language query into dtSearch syntax.

        Enforces token limits: trims oldest history when over input budget, rejects when
        the current user message is over max_user_message_tokens, and caps response length.

        Args:
            query: The natural language query to convert.
            use_history: Whether to use conversation history (default: True).

        Returns:
            Tuple of (dtSearch response string, optional warning message when limits applied).
        """
        warning: Optional[str] = None
        user_content = self.user_prompt_prefix + query
        user_tokens = self._estimate_tokens(user_content)

        if user_tokens > self.max_user_message_tokens:
            approx_chars = self.max_user_message_tokens * 4
            msg = (
                f"Your message is too long (over {self.max_user_message_tokens} tokens). "
                f"Please shorten it to about {approx_chars} characters or less so it can be processed."
            )
            return msg, "Message length exceeded. Please shorten your message."

        history = self.conversation_history if use_history else []
        history_override: Optional[List[Dict[str, str]]] = None
        system_content = self.system_prompt
        system_tokens = self._estimate_tokens(system_content)
        budget_for_history = self.max_input_tokens - system_tokens - user_tokens

        if use_history and history and budget_for_history < self._count_messages_tokens(history):
            # Trim oldest exchanges (pairs of user/assistant) until under budget
            trimmed = list(history)
            while trimmed and self._count_messages_tokens(trimmed) > budget_for_history:
                if len(trimmed) >= 2:
                    trimmed = trimmed[2:]
                else:
                    trimmed = []
            history_override = trimmed
            warning = (
                "Conversation length exceeded the limit. Older messages were omitted for this reply."
            )

        messages = self._build_messages(
            query,
            use_history=use_history,
            conversation_type="standard",
            history_override=history_override,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
        )
        raw_response = response.choices[0].message.content
        cleaned_response = self._clean_response(raw_response)

        if use_history:
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": cleaned_response})

        return cleaned_response, warning
    
    def convert_to_dtSearch_suggestions(
        self, 
        query: str,
        num_suggestions: int = 5
    ) -> List[Dict[str, str]]:
        """
        Convert a natural language query into multiple dtSearch syntax suggestions
        with different search strategies (exact, fuzzy, wildcard, root word, phonetic, etc.).
        
        Args:
            query: The natural language query to convert.
            num_suggestions: Maximum number of suggestions to return (default: 5).
            
        Returns:
            List of dictionaries, each containing:
            - "query": The dtSearch syntax query
            - "description": Brief explanation of what the query finds
            - "type": Type of search (exact, fuzzy, wildcard, root, phonetic, combination)
            
        Raises:
            Exception: If the OpenAI API call fails.
        """
        messages = [
            {
                "role": "system",
                "content": self.suggestions_system_prompt if hasattr(self, 'suggestions_system_prompt') else self.DEFAULT_SUGGESTIONS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": self.DEFAULT_SUGGESTIONS_USER_PROMPT_PREFIX.format(query=query)
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7  # Slightly higher temperature for more varied suggestions
        )
        
        raw_response = response.choices[0].message.content
        
        # Parse the response to extract suggestions
        return self._parse_suggestions(raw_response, num_suggestions)
    
    def _parse_suggestions(self, response: str, max_suggestions: int) -> List[Dict[str, str]]:
        """
        Parse the LLM response to extract structured suggestions.
        
        Args:
            response: The raw response from the LLM.
            max_suggestions: Maximum number of suggestions to extract.
            
        Returns:
            List of suggestion dictionaries.
        """
        suggestions = []
        
        # Split by numbered list items (1., 2., etc.) or bullet points
        # Pattern matches: "1. ", "2. ", "- ", "* ", etc.
        lines = response.split('\n')
        
        current_suggestion = None
        current_query = None
        current_description = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new numbered/bullet item
            match = re.match(r'^(\d+\.|\-|\*)\s+(.+)', line)
            if match:
                # Save previous suggestion if exists
                if current_query and current_description:
                    suggestions.append({
                        "query": current_query.strip(),
                        "description": " ".join(current_description).strip(),
                        "type": self._infer_search_type(current_query)
                    })
                
                # Start new suggestion
                remaining = match.group(2)
                # Try to extract query and description
                # Format: "dtSearch query - explanation" or similar
                if ":" in remaining:
                    parts = remaining.split(":", 1)
                    current_query = parts[0].strip()
                    current_description = [parts[1].strip()] if len(parts) > 1 else []
                else:
                    current_query = remaining
                    current_description = []
                
                if len(suggestions) >= max_suggestions:
                    break
            else:
                # Continuation of description or query
                if current_query:
                    # Check if line looks like a dtSearch query (contains operators)
                    if any(op in line for op in [" AND ", " OR ", " NOT ", " NEAR", " WITHIN", "*", "?", "#", "~", "||"]):
                        if not current_query or current_query in current_description:
                            current_query = line
                        else:
                            current_description.append(line)
                    else:
                        current_description.append(line)
        
        # Add last suggestion
        if current_query and current_description:
            suggestions.append({
                "query": current_query.strip(),
                "description": " ".join(current_description).strip(),
                "type": self._infer_search_type(current_query)
            })
        elif current_query:
            suggestions.append({
                "query": current_query.strip(),
                "description": "dtSearch query suggestion",
                "type": self._infer_search_type(current_query)
            })
        
        # If parsing failed, try to extract queries from the response
        if not suggestions:
            # Look for dtSearch patterns in the text
            dtsearch_pattern = r'((?:[A-Za-z0-9\s\*\+\?\(\)\"\']+(?:\s+(?:AND|OR|NOT|NEAR|WITHIN)\s+[A-Za-z0-9\s\*\+\?\(\)\"\']+)+)|[A-Za-z0-9\*\+\?\(\)\"\'#~|]+)'
            matches = re.findall(dtsearch_pattern, response)
            for match in matches[:max_suggestions]:
                if match.strip():
                    suggestions.append({
                        "query": match.strip(),
                        "description": "dtSearch query suggestion",
                        "type": self._infer_search_type(match)
                    })
        
        return suggestions[:max_suggestions]
    
    def _infer_search_type(self, query: str) -> str:
        """
        Infer the type of search from the dtSearch query.
        
        Args:
            query: The dtSearch query string.
            
        Returns:
            Search type string (exact, fuzzy, wildcard, root, phonetic, combination).
        """
        query_lower = query.lower()
        
        if "#" in query:
            if "*" in query or "?" in query or "~" in query or "||" in query:
                return "combination"
            return "fuzzy"
        elif "~" in query:
            if "*" in query or "?" in query or "#" in query or "||" in query:
                return "combination"
            return "root"
        elif "||" in query:
            if "*" in query or "?" in query or "#" in query or "~" in query:
                return "combination"
            return "phonetic"
        elif "*" in query or "?" in query:
            if "#" in query or "~" in query or "||" in query:
                return "combination"
            return "wildcard"
        elif " and " in query_lower or " or " in query_lower or " not " in query_lower or " near" in query_lower or " within" in query_lower:
            return "combination"
        else:
            return "exact"
    
    def update_model(self, model: str) -> None:
        """
        Update the OpenAI model to use for conversions.
        
        Args:
            model: The new model name (e.g., "gpt-4o-mini", "gpt-4").
        """
        self.model = model
    
    def update_system_prompt(self, prompt: str) -> None:
        """
        Update the system prompt used for conversions.
        
        Args:
            prompt: The new system prompt.
        """
        self.system_prompt = prompt
    
    def update_user_prompt_prefix(self, prefix: str) -> None:
        """
        Update the user prompt prefix used for conversions.
        
        Args:
            prefix: The new user prompt prefix.
        """
        self.user_prompt_prefix = prefix
    
    def clear_conversation_history(self, conversation_type: str = "both") -> None:
        """
        Clear the conversation history.
        
        Args:
            conversation_type: Which conversation to clear - "standard", "suggestions", or "both" (default: "both").
        """
        if conversation_type in ("standard", "both"):
            self.conversation_history = []
        if conversation_type in ("suggestions", "both"):
            self.suggestions_conversation_history = []
    
    def get_conversation_history(self, conversation_type: str = "standard") -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Args:
            conversation_type: Which conversation to get - "standard" or "suggestions" (default: "standard").
            
        Returns:
            List of message dictionaries representing the conversation history.
        """
        if conversation_type == "suggestions":
            return self.suggestions_conversation_history.copy()
        return self.conversation_history.copy()
    
    def set_conversation_history(
        self,
        history: List[Dict[str, str]],
        conversation_type: str = "standard"
    ) -> None:
        """
        Set the conversation history (e.g. after loading a saved history).
        The LLM will use this context for follow-up messages.
        
        Args:
            history: List of message dicts with "role" and "content" keys.
            conversation_type: Which conversation to set - "standard" or "suggestions" (default: "standard").
        """
        if conversation_type == "suggestions":
            self.suggestions_conversation_history = list(history)
        else:
            self.conversation_history = list(history)
    
    def get_conversation_count(self, conversation_type: str = "standard") -> int:
        """
        Get the number of exchanges in the conversation history.
        
        Args:
            conversation_type: Which conversation to check - "standard" or "suggestions" (default: "standard").
            
        Returns:
            Number of user-assistant exchanges (each exchange is 2 messages).
        """
        history = self.suggestions_conversation_history if conversation_type == "suggestions" else self.conversation_history
        return len(history) // 2