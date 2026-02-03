"""
Gradio UI for NLQ to dtSearch application.

This application provides a three-panel interface:
- Left panel: Side panel (placeholder for future features)
- Middle panel: Chat interface for search queries and responses
- Right panel: Conversation history display
"""

import gradio as gr
from nlp_dtsearch import NLPdtSearch
from typing import Tuple, List, Optional
from datetime import datetime
import tempfile
import os


def _get_auth_credentials() -> Optional[List[Tuple[str, str]]]:
    """
    Read auth credentials from environment (for Hugging Face Space Secrets).
    Supports both BASIC_AUTH_USERS (HF standard) and AUTH_CREDENTIALS.
    Set as comma-separated "username:password" pairs.
    Example: BASIC_AUTH_USERS="admin:mypass,user1:pass1"
    If not set, returns None (no auth required - useful for local dev).
    """
    # Try BASIC_AUTH_USERS first (HF Spaces standard), then AUTH_CREDENTIALS
    raw = os.environ.get("BASIC_AUTH_USERS", "").strip()
    if not raw:
        raw = os.environ.get("AUTH_CREDENTIALS", "").strip()
    if not raw:
        return None
    creds = []
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            user, pwd = part.split(":", 1)
            creds.append((user.strip(), pwd.strip()))
    return creds if creds else None


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
        # Get response from converter (returns response, optional warning)
        response, warning = converter.convert_to_dtSearch(message.strip())
        if warning:
            response = f"⚠️ {warning}\n\n{response}"
        # Append as dictionaries - Gradio Chatbot expects dict format with role/content
        new_history = normalized_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
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
    lines = ["NLQ to dtSearch - Chat History\n"]
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


def parse_saved_history(content: str) -> List[dict]:
    """
    Parse saved chat history text (same format as download) into list of message dicts.
    
    Args:
        content: Raw file content.
        
    Returns:
        List of {"role": "user"|"assistant", "content": str}. User content is clean (no prefix).
    """
    import re
    out = []
    # Split by separator line (dashes)
    blocks = re.split(r"\n-{50,}\s*\n", content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        if len(lines) < 2:
            continue
        header, body = lines[0].strip(), lines[1].strip()
        if header.lower().startswith("user "):
            out.append({"role": "user", "content": body})
        elif header.lower().startswith("assistant "):
            out.append({"role": "assistant", "content": body})
    return out


def load_chat_history(file_obj) -> Tuple[List[dict], str]:
    """
    Load conversation history from an uploaded file and restore it in the converter.
    The LLM will use this history for follow-up messages.
    
    Args:
        file_obj: Gradio file upload value (file path string or object with .name).
        
    Returns:
        Tuple of (chatbot history for display, history display markdown).
    """
    global converter
    
    if converter is None:
        converter = initialize_converter()
    
    if file_obj is None:
        return [], "No file uploaded. Upload a saved chat history file (.txt)."
    
    if isinstance(file_obj, list) and file_obj:
        file_obj = file_obj[0]
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path or not os.path.isfile(path):
        return [], "Invalid or missing file. Upload a saved chat history file (.txt)."
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return [], f"Could not read file: {e}"
    
    parsed = parse_saved_history(content)
    if not parsed:
        return [], "No conversation found in file. Use a file saved with Download History."
    
    # Build history for converter: user messages must include prompt prefix so LLM context matches
    prefix = converter.user_prompt_prefix
    for_converter = []
    for m in parsed:
        role, text = m["role"], m["content"]
        if role == "user":
            for_converter.append({"role": "user", "content": prefix + text})
        else:
            for_converter.append({"role": "assistant", "content": text})
    
    converter.set_conversation_history(for_converter)
    
    # Chatbot display: show clean user messages (no prefix)
    chatbot_history = parsed
    
    history_display = format_conversation_history(for_converter)
    
    return chatbot_history, history_display


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
    
    with gr.Blocks(title="NLQ to dtSearch", css=custom_css, theme=gr.themes.Monochrome()) as app:
        # Title
        gr.Markdown("# NLQ to dtSearch", elem_classes=["title"])
        gr.Markdown("Convert natural language queries into dtSearch syntax with AI assistance.")
        
        with gr.Row():
            # Left Panel - About panel (collapsible)
            with gr.Column(scale=1, min_width=200, visible=True) as left_panel:
                with gr.Accordion("About NLQ to dtSearch", open=False):
                    gr.Markdown(
                        "This application is a prototype eDiscovery tool designed to translate plain-English questions into precise, "
                        "well-formed dtSearch queries. It enables eDiscovery practitioners to move faster from intent to execution, "
                        "reducing the trial-and-error often involved in building complex search syntax.\n\n"
                        "Beyond its immediate utility, this project showcases how quickly and effectively eDiscovery professionals can design, "
                        "build, and deploy impactful in-house tools using modern AI technologies. Large Language Models are reshaping "
                        "the way practitioners work—augmenting expertise, improving consistency, and accelerating insight across the discovery lifecycle.\n\n"
                        "This prototype is intended to demonstrate both practical value and future potential: empowering practitioners to focus less on "
                        "syntax and more on strategy.\n\n"
                        "For questions, feedback, or collaboration opportunities, please contact the developer, John K. Hancock, at jkhancock@gmail.com"
                    )
            
            # Middle Panel - Chat interface
            with gr.Column(scale=3, min_width=400):
                gr.Markdown("### Chat")
                
                # Chatbot interface (type='messages' = openai-style role/content dicts; required in Gradio 5.x+)
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
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
                    load_btn = gr.Button("Load History", variant="secondary")
                
                # Download file component
                download_file = gr.File(
                    label="Download Chat History",
                    visible=True,
                    interactive=False
                )
                
                # Load history: upload saved file
                load_file = gr.File(
                    label="Upload saved history (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                    visible=True
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
        
        # Load history: restore from file and update converter so LLM has context
        load_btn.click(
            load_chat_history,
            inputs=[load_file],
            outputs=[chatbot, history_display]
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
    auth = _get_auth_credentials()
    # ssr_mode=False: avoid SSR Node client calling API before auth (fixes "Login credentials required").
    # share=True: satisfy Gradio's localhost check in containers (HF Spaces serves via its own URL).
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        auth=auth,
        auth_message="Access restricted to approved users. Contact the administrator for credentials."
        if auth
        else None,
        ssr_mode=False,
        share=True,
    )



if __name__ == "__main__":
    main()
