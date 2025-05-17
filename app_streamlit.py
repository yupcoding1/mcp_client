import streamlit as st
import asyncio

import sys

from main import Configuration, Server, LLMClient, ChatSession

st.set_page_config(page_title="MCP Agentic Chatbot", layout="wide")
st.title("🤖 MCP Agentic Chatbot")

# Python/Windows compatibility check
if sys.platform == "win32" and sys.version_info >= (3, 13):
    st.error("This application requires Python 3.11 or 3.10 on Windows for full functionality. Python 3.13+ on Windows does not support async subprocesses, which are required for the SQLite server. Please use Python 3.11 or 3.10.")
    st.stop()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = None
if "loop" not in st.session_state:
    st.session_state["loop"] = asyncio.new_event_loop()

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User input
user_input = st.chat_input("Type your message and press Enter...")

async def get_or_create_chat_session():
    if st.session_state["chat_session"] is None:
        config = Configuration()
        server_config = config.load_config("servers_config.json")
        servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]
        llm_client = LLMClient(config.llm_api_key)
        chat_session = ChatSession(servers, llm_client)
        # Only initialize servers, do not call chat_session.start()
        for server in chat_session.servers:
            await server.initialize()
        st.session_state["chat_session"] = chat_session
    return st.session_state["chat_session"]

async def process_user_message(user_input):
    chat_session = await get_or_create_chat_session()
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    # Compose messages for LLM
    messages = st.session_state["messages"]
    llm_response = chat_session.llm_client.get_response(messages)
    st.session_state["messages"].append({"role": "assistant", "content": llm_response})
    # Process tool calls if any
    result = await chat_session.process_llm_response(llm_response)
    if result != llm_response:
        st.session_state["messages"].append({"role": "system", "content": result})
        final_response = chat_session.llm_client.get_response(st.session_state["messages"])
        st.session_state["messages"].append({"role": "assistant", "content": final_response})
        return final_response
    return llm_response

if user_input:
    with st.spinner("Processing..."):
        try:
            loop = st.session_state["loop"]
            response = loop.run_until_complete(process_user_message(user_input))
            st.chat_message("assistant").write(response)
        except Exception as e:
            st.error(f"Error: {e}")

# Optionally, display logs or tool outputs
with st.expander("Show Log Output"):
    try:
        with open("app.log", "r", encoding="utf-8") as f:
            st.text(f.read())
    except Exception:
        st.info("No log file found yet.")
