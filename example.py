"""
Example script demonstrating how to use the NLPdtSearch class
to convert natural language queries into dtSearch syntax.
"""

from nlp_dtsearch import NLPdtSearch


def main():
    """Demonstrate basic and advanced usage of NLPdtSearch."""
    
    print("=" * 60)
    print("NLPdtSearch Example Script")
    print("=" * 60)
    print()
    
    # Example 1: Basic usage with default configuration
    print("Example 1: Basic Usage")
    print("-" * 60)
    try:
        # Initialize with defaults (loads API key from .env file)
        converter = NLPdtSearch()
        
        # Convert a natural language query
        natural_query = "Show me a dtSearch query where it's apples and bananas which should be searched first. The results of that search should be within 5 of grape."
        
        print(f"Natural Language Query:")
        print(f"  {natural_query}")
        print()
        
        dt_search_query = converter.convert_to_dtSearch(natural_query)
        
        print(f"dtSearch Query Result:")
        print(f"  {dt_search_query}")
        print()
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file or environment variables.")
        return
    
    # Example 2: Using a different model
    print("Example 2: Using a Different Model")
    print("-" * 60)
    try:
        converter_gpt4 = NLPdtSearch(model="gpt-4o-mini")
        simple_query = "Find documents containing Python or JavaScript"
        
        print(f"Natural Language Query:")
        print(f"  {simple_query}")
        print()
        
        result = converter_gpt4.convert_to_dtSearch(simple_query)
        
        print(f"dtSearch Query Result:")
        print(f"  {result}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 3: Custom prompts
    print("Example 3: Custom System Prompt")
    print("-" * 60)
    try:
        custom_system_prompt = """You are an expert in dtSearch syntax conversion. 
Always provide precise, syntactically correct dtSearch queries. 
Be concise and only return the query without explanations."""
        
        converter_custom = NLPdtSearch(system_prompt=custom_system_prompt)
        
        query = "Search for files with 'invoice' and '2024'"
        print(f"Natural Language Query:")
        print(f"  {query}")
        print()
        
        result = converter_custom.convert_to_dtSearch(query)
        
        print(f"dtSearch Query Result:")
        print(f"  {result}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Example 4: Runtime configuration updates
    print("Example 4: Runtime Configuration Updates")
    print("-" * 60)
    try:
        converter = NLPdtSearch()
        
        # Update model at runtime
        converter.update_model("gpt-4o-mini")
        print(f"Updated model to: {converter.model}")
        
        # Update system prompt
        new_prompt = "Convert to dtSearch syntax. Be very concise."
        converter.update_system_prompt(new_prompt)
        print(f"Updated system prompt.")
        print()
        
        query = "Find documents about machine learning"
        result = converter.convert_to_dtSearch(query)
        
        print(f"Query: {query}")
        print(f"Result: {result}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
