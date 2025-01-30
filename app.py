import re
import base64
import streamlit as st
from ollama import chat

# Set Streamlit page configuration
st.set_page_config(page_title="Ollama Streaming Chat", layout="wide")

def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return re.sub(r"<think>|</think>", "", thinking_content)

def display_message(message):
    """Display a single message in the chat interface."""
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            display_assistant_message(message["content"])
        else:
            st.markdown(message["content"])

def display_assistant_message(content):
    """Display assistant message with thinking content if present."""
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        think_content = format_reasoning_response(match.group(0))
        response_content = content.replace(match.group(0), "")
        with st.expander("Thinking complete! 💡"):
            st.markdown(think_content)
        st.markdown(response_content)
    else:
        st.markdown(content)

def display_chat_history():
    """Display all previous messages in the chat history."""
    for message in st.session_state["messages"]:
        if message["role"] != "system":
            display_message(message)

def process_thinking_phase(stream):
    """Process the thinking phase of the assistant's response."""
    thinking_content = ""
    think_placeholder = st.empty()
    status = st.status("Thinking...", expanded=True)
    
    for chunk in stream:
        content = chunk["message"].get("content", "")
        thinking_content += content
        
        if "<think>" in content:
            continue
        if "</think>" in content:
            content = content.replace("</think>", "")
            status.update(label="Thinking complete!", state="complete", expanded=False)
            break
        think_placeholder.markdown(format_reasoning_response(thinking_content))
    
    return thinking_content

def process_response_phase(stream):
    """Process the response phase of the assistant's response."""
    response_placeholder = st.empty()
    response_content = ""
    for chunk in stream:
        content = chunk["message"].get("content", "")
        response_content += content
        response_placeholder.markdown(response_content)
    return response_content

@st.cache_resource
def get_chat_model():
    """Get a cached instance of the chat model."""
    return lambda messages: chat(model="deepseek-r1:1.5b", messages=messages, stream=True)

def handle_user_input():
    """Handle new user input and generate assistant response."""
    if user_input := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)
        
        with st.chat_message("assistant"):
            chat_model = get_chat_model()
            stream = chat_model(st.session_state["messages"])
            
            thinking_content = process_thinking_phase(stream)
            response_content = process_response_phase(stream)
            
            st.session_state["messages"].append({"role": "assistant", "content": thinking_content + response_content})

def main():
    """Main function to handle the chat interface and streaming responses."""
    logo_path = "assets/deep-seek.png"
    try:
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="50" style="vertical-align: -3px;">'
    except FileNotFoundError:
        logo_html = ""
    
    with st.sidebar:
        st.title("Chat History")
        if st.button("+ Create New Chat"):
            st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]
        for i, chat in enumerate(st.session_state.get("chat_history", [])):
            if st.button(f"Chat {i+1}"):
                st.session_state["messages"] = chat
    
    st.markdown(f"""
    <h1 style='text-align: center;'>SeekMind {logo_html}</h1>
    <h4 style='text-align: center;'>With thinking UI! 💡</h4>
    """, unsafe_allow_html=True)
    
    display_chat_history()
    handle_user_input()
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if st.session_state["messages"] not in st.session_state["chat_history"]:
        st.session_state["chat_history"].append(st.session_state["messages"])

if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]
    main()
