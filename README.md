# MCP Client: Agentic Chatbot with Tool-Use and Streamlit UI

## Overview
This project is an agentic chatbot client that can use external tools (such as SQLite and Puppeteer servers) to answer user queries, automate workflows, and perform multi-step reasoning. It features both a command-line interface and a modern Streamlit web UI. The chatbot is powered by an LLM (via Groq API) and can chain tool calls, passing outputs from one tool as inputs to the next.

---

## Features
- **Agentic LLM Chatbot**: Uses an LLM to interpret user queries and decide when/how to use tools.
- **Tool Chaining**: Supports multi-step workflows where the output of one tool is used as input for another.
- **SQLite Tooling**: Query, update, and manage a SQLite database via an MCP server.
- **Puppeteer Tooling**: Automate browser actions (navigate, click, fill, screenshot, evaluate JS, etc.) via Puppeteer MCP server.
- **Streamlit Web UI**: Chat with the agent, view logs, and see tool outputs in a modern web interface.
- **Command-Line Mode**: Interact with the agent via the terminal.
- **Logging**: All actions and errors are logged to `app.log` for transparency and debugging.

---

## Folder Structure
```
app_streamlit.py      # Streamlit web app for chat and tool use
main.py               # Main agentic chatbot logic (CLI entrypoint)
requirements.txt      # Python dependencies
servers_config.json   # Configuration for MCP tool servers
app.log               # Log file (auto-generated)
test.db               # SQLite database (auto-generated/used by sqlite server)
.env                  # Environment variables (API keys, etc.)
```

---

## Setup Instructions

### 1. **Python Version**
- **Windows users:** You must use Python 3.10 or 3.11. Python 3.13+ is not supported for async subprocesses (required for tool servers).
- [Download Python 3.11](https://www.python.org/downloads/release/python-3110/)

### 2. **Clone and Install**
```powershell
# Clone the repo (if not already)
cd path\to\your\projects
# Create and activate a virtual environment
py -3.11 -m venv venv
.\venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

### 3. **Environment Variables**
- Create a `.env` file in the project root with your LLM API key:
  ```env
  LLM_API_KEY=your_groq_api_key_here
  ```

### 4. **Configure Tool Servers**
- Edit `servers_config.json` to define which MCP servers to use. Example:
  ```json
  {
    "mcpServers": {
      "sqlite": {
        "command": "uvx",
        "args": ["mcp-server-sqlite", "--db-path", "./test.db"]
      },
      "puppeteer": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
      }
    }
  }
  ```
- Ensure you have Node.js and the required MCP servers installed globally or accessible via `npx`/`uvx`.

---

## Usage

### **Command-Line Chatbot**
```powershell
python main.py
```
- Type your message and press Enter.
- Type `exit` or `quit` to stop.
- The agent will use tools as needed and log all actions to `app.log`.

### **Streamlit Web App**
```powershell
streamlit run app_streamlit.py
```
- Open the provided local URL in your browser.
- Chat with the agent, view logs, and see tool outputs.
- If you are on Windows with Python 3.13+, you will see an error message (use Python 3.11).

---

## Tool Chaining (Agentic Mode)
- The agent can execute multiple tools in sequence, passing results between them.
- Example LLM tool call (handled automatically):
  ```json
  [
    {"tool": "puppeteer_navigate", "arguments": {"url": "https://news.google.com/"}, "result_var": "page_content"},
    {"tool": "summarize_text", "arguments": {"text": "$page_content"}, "result_var": "summary"},
    {"tool": "write_query", "arguments": {"query": "INSERT INTO summaries (text) VALUES ('$summary')"}}
  ]
  ```
- The agent will resolve `$page_content` and `$summary` automatically.

---

## Logs
- All actions, tool calls, and errors are logged to `app.log`.
- In the Streamlit app, you can view logs in the "Show Log Output" expander.

---

## Troubleshooting
- **Python 3.13+ on Windows:** Not supported for async subprocesses. Use Python 3.11.
- **Tool server fails to start:** Check your `servers_config.json` and ensure `uvx`, `npx`, and the MCP servers are installed and on your PATH.
- **No tools available:** Make sure servers are running and initialized (see logs for errors).
- **API errors:** Check your `.env` and API key.

---

## Credits
- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)
- [Streamlit](https://streamlit.io/)
- [Groq API](https://console.groq.com/)

---

## License
This project is for educational and research purposes. See individual tool/server repos for their licenses.
