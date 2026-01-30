"""
Gradio UI for NLP to dtSearch application.

This application provides a three-panel interface:
- Left panel: Side panel (placeholder for future features)
- Middle panel: Chat interface for search queries and responses
- Right panel: Conversation history display
"""

import gradio as gr
from nlp_dtsearch import NLPdtSearch
from typing import Tuple, List
from datetime import datetime
import tempfile
import os


# Initialize the converter (will be created when the app loads)
converter = None


def initialize_converter():
    """Initialize the NLPdtSearch converter."""
    global converter
    if converter is None:
        converter = NLPdtSearch()
    # Don't return anything since outputs=[] in app.load()


def format_conversation_history(history: List[dict]) -> str:
    """
    Format conversation history for display.
    
    Args:
        history: List of message dictionaries from conversation history.
        
    Returns:
        Formatted string representation of the conversation.
    """
    if not history:
        return "No conversation history yet. Start chatting to see history here!"
    
    formatted = []
    for i, msg in enumerate(history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            # Remove the prompt prefix for cleaner display
            clean_content = content.replace(
                converter.user_prompt_prefix if converter else "", ""
            ).strip()
            formatted.append(f"**You:** {clean_content}")
        elif role == "assistant":
            formatted.append(f"**Assistant:** {content}")
        
        # Add separator between exchanges
        if i < len(history) - 1:
            formatted.append("---")
    
    return "\n\n".join(formatted)


def normalize_history(history):
    """Normalize history to list of dictionaries format expected by Gradio 6.2.0 Chatbot."""
    if history is None:
        return []
    if not isinstance(history, list):
        return []
    if len(history) == 0:
        return []
    
    # Check first element to determine format
    first = history[0]
    if isinstance(first, dict) and "role" in first:
        # Already in dict format with role/content - this is what Gradio 6.2.0 expects
        return history
    elif isinstance(first, tuple):
        # Convert from tuple format to dict format
        normalized = []
        for h in history:
            if isinstance(h, tuple) and len(h) == 2:
                user_msg, bot_msg = h
                if user_msg:
                    normalized.append({"role": "user", "content": user_msg})
                if bot_msg:
                    normalized.append({"role": "assistant", "content": bot_msg})
        return normalized
    else:
        # Unknown format, return empty
        return []


def chat_with_converter(message: str, history: List) -> Tuple[str, List, str]:
    """
    Handle chat interaction with the converter.
    
    Args:
        message: User's input message.
        history: Current chat history from Gradio Chatbot.
        
    Returns:
        Tuple of (empty string to clear input, updated chat history, updated history display).
    """
    global converter
    
    # Initialize converter if needed
    if converter is None:
        converter = initialize_converter()
    
    # Normalize history to dictionary format expected by Gradio 6.2.0 Chatbot
    normalized_history = normalize_history(history)
    
    if not message or not message.strip():
        conv_history = format_conversation_history(converter.get_conversation_history() if converter else [])
        return "", normalized_history, conv_history
    
    try:
        # Get response from converter
        response = converter.convert_to_dtSearch(message.strip())
        
        # Append as dictionaries - Gradio 6.2.0 Chatbot expects dict format with role/content
        new_history = normalized_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        
        # Get formatted conversation history for the right panel
        conv_history = format_conversation_history(converter.get_conversation_history())
        
        return "", new_history, conv_history
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        new_history = normalized_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        conv_history = format_conversation_history(converter.get_conversation_history() if converter else [])
        return "", new_history, conv_history


def clear_conversation() -> Tuple[List, str]:
    """
    Clear the conversation history.
    
    Returns:
        Tuple of (empty chat history, empty conversation history display).
    """
    global converter
    if converter:
        converter.clear_conversation_history()
    return [], "Conversation cleared. Start a new conversation!"


def download_chat_history() -> str:
    """
    Generate and return a downloadable text file of the chat history.
    
    Returns:
        File path for Gradio File component.
    """
    global converter
    
    if not converter:
        return None
    
    history = converter.get_conversation_history()
    
    if not history:
        return None
    
    # Create a formatted text file
    lines = ["NLP to dtSearch - Chat History\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("=" * 50 + "\n\n")
    
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            # Remove prompt prefix for cleaner display
            clean_content = content.replace(
                converter.user_prompt_prefix if converter else "", ""
            ).strip()
            lines.append(f"User ({i}):\n{clean_content}\n\n")
        elif role == "assistant":
            lines.append(f"Assistant ({i}):\n{content}\n\n")
        
        lines.append("-" * 50 + "\n\n")
    
    # Create temporary file
    content = "".join(lines)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    temp_file.write(content)
    temp_file.close()
    
    return temp_file.name


def create_ui():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for styling
    custom_css = """
    /* Arial font for entire app */
    * {
        font-family: Arial, sans-serif !important;
    }
    
    /* Orange buttons */
    .primary, button.primary, .gr-button-primary {
        background-color: #FF8C00 !important;
        border-color: #FF8C00 !important;
    }
    
    .primary:hover, button.primary:hover, .gr-button-primary:hover {
        background-color: #FF7F00 !important;
        border-color: #FF7F00 !important;
    }
    
    .secondary, button.secondary, .gr-button-secondary {
        background-color: #FF8C00 !important;
        border-color: #FF8C00 !important;
        color: white !important;
    }
    
    .secondary:hover, button.secondary:hover, .gr-button-secondary:hover {
        background-color: #FF7F00 !important;
        border-color: #FF7F00 !important;
    }
    """
    
    with gr.Blocks(title="NLP to dtSearch", css=custom_css) as app:
        # Title
        gr.Markdown("# NLP to dtSearch", elem_classes=["title"])
        gr.Markdown("Convert natural language queries into dtSearch syntax with AI assistance.")
        
        with gr.Row():
            # Left Panel - Side panel (collapsible)
            with gr.Column(scale=1, min_width=200, visible=True) as left_panel:
                with gr.Accordion("Side Panel", open=False):
                    gr.Markdown("_Additional features will be added here later._")
                    placeholder_text = gr.Markdown(
                        "This panel is reserved for future features such as:\n"
                        "- Search options\n"
                        "- Settings\n"
                        "- Advanced filters"
                    )
            
            # Middle Panel - Chat interface
            with gr.Column(scale=3, min_width=400):
                gr.Markdown("### Chat")
                
                # Chatbot interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500
                )
                
                # Input box
                with gr.Row():
                    msg = gr.Textbox(
                        label="Enter your search query",
                        placeholder="Type your natural language search query here...",
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat History", variant="secondary")
                    download_btn = gr.Button("Download History", variant="secondary")
                
                # Download file component
                download_file = gr.File(
                    label="Download Chat History",
                    visible=True,
                    interactive=False
                )
            
            # Right Panel - Conversation history
            with gr.Column(scale=2, min_width=300):
                gr.Markdown("### Conversation History")
                history_display = gr.Markdown(
                    "No conversation history yet. Start chatting to see history here!",
                    label="History"
                )
        
        # Event handlers - chatbot manages its own state
        msg.submit(
            chat_with_converter,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, history_display]
        )
        
        submit_btn.click(
            chat_with_converter,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, history_display]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, history_display]
        )
        
        # Download history button
        download_btn.click(
            download_chat_history,
            outputs=[download_file]
        )
        
        # Initialize converter when app loads
        app.load(
            initialize_converter,
            outputs=[],
            show_progress=False
        )
    
    return app


def main():
    """Main function to launch the Gradio app."""
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        show_error=True,
        theme=gr.themes.Monochrome()  # Dark theme (Monochrome is dark)
    )


if __name__ == "__main__":
    main()
