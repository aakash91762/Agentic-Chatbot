# Agentic Chatbot with LangGraph, RAG, and Human-in-the-Loop

## 1. Project overview

This project is a Streamlit-based agentic chatbot built with LangGraph and Google Gemini. It can answer general questions, call tools, search uploaded PDF documents using Retrieval-Augmented Generation (RAG), preserve conversations across sessions, and pause for human approval before a simulated stock purchase.

The application has two main Python files:

- `agentic_chatbot_app_frontend.py` — Streamlit user interface, PDF upload, chat threads, tool activity display, and human-approval controls.
- `agentic_chatbot_backend.py` — Gemini models, tools, PDF ingestion and retrieval, LangGraph nodes, human-in-the-loop interrupt, and SQLite checkpointer.

## 2. Key features

### 2.1 Conversational AI

- Uses `gemini-2.5-flash` through `ChatGoogleGenerativeAI`.
- Uses LangGraph to route requests between the LLM node and tool node.
- Streams the assistant's response into the Streamlit chat interface.

### 2.2 Tool calling

The agent has access to the following tools:

| Tool | Purpose | External service |
|---|---|---|
| `search_tool` | Search the web for current or recent information | Tavily |
| `calculator` | Evaluate supported mathematical expressions | Local Python function |
| `get_stock_price` | Retrieve a stock quote by ticker symbol | Alpha Vantage |
| `get_current_weather` | Retrieve current weather for a location | OpenWeatherMap |
| `rag_tool` | Search indexed PDF content and return relevant passages | Local FAISS vector store + Google embeddings |
| `purchase_stock` | Demonstrate a stock-purchase approval workflow | Simulated; no broker integration |

The LLM is configured with these tools using `llm.bind_tools(tools)`. LangGraph's `ToolNode` executes tool calls and the graph returns to the LLM node to prepare a response.

### 2.3 PDF upload and RAG

The Streamlit sidebar allows users to upload one or more PDF files and index them into the knowledge base.

The ingestion pipeline:

1. Receives a PDF upload and validates that it is a non-empty PDF file.
2. Loads PDF pages using `PyPDFLoader`.
3. Splits extracted text into chunks of 1,000 characters with 200 characters of overlap.
4. Generates embeddings using `gemini-embedding-001`.
5. Stores the vectors and document metadata in a local FAISS index under `faiss_db/`.
6. Adds new PDFs to the existing FAISS index instead of replacing the index.
7. Uses a SHA-256 hash to avoid indexing the exact same PDF content more than once.

At query time, `rag_tool` searches the FAISS index using similarity search and retrieves up to four relevant chunks. The LLM uses the retrieved passages to answer document-based questions.

**PDF limitation:** The current loader extracts text from PDFs. Image-only or scanned PDFs may return little or no text and may require an OCR step before they can be indexed.

### 2.4 Human-in-the-loop stock purchase approval

The `purchase_stock` tool demonstrates a human-approval workflow:

1. The user asks to buy a specified number of shares of a stock.
2. The LLM may call `purchase_stock` when the ticker and quantity are known.
3. The tool calls LangGraph's `interrupt()` and pauses the graph.
4. The Streamlit interface displays the approval request with **Approve simulated purchase** and **Decline** buttons.
5. The user's decision is passed back to LangGraph using `Command(resume="yes")` or `Command(resume="no")`.
6. The graph resumes and returns a success or cancelled result.

**Important:** This is a simulation only. It does not connect to a brokerage account and does not place a real stock order. Do not describe it as live trading functionality.

### 2.5 Conversation persistence and threads

- Each conversation has a unique `thread_id`.
- LangGraph uses `SqliteSaver` to persist graph state in `chatbot.db`.
- The Streamlit UI lists saved threads and lets users create or switch conversations.
- When a thread is opened, the latest saved graph state is loaded using its `thread_id`.
- Pending approval interrupts are read from the saved graph state, allowing the approval UI to be restored after a Streamlit rerun or application restart, provided the same SQLite database is retained.

### 2.6 Tool activity display

The frontend listens to LangGraph `updates` and `messages` stream modes. It displays tool activity in an expandable Streamlit status panel while rendering the final assistant answer separately in the chat.

## 3. Technology stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM orchestration | LangGraph |
| LLM | Google Gemini `gemini-2.5-flash` |
| Embeddings | Google `gemini-embedding-001` |
| LLM/tool integration | LangChain |
| PDF loading | PyPDFLoader (`pypdf`) |
| Text splitting | RecursiveCharacterTextSplitter |
| Vector database | FAISS (local) |
| Conversation persistence | SQLite + LangGraph `SqliteSaver` |
| Web search | Tavily |
| Stock quotes | Alpha Vantage |
| Weather | OpenWeatherMap |

## 4. Project structure

```text
Agentic-Chatbot/
├── agentic_chatbot_app_frontend.py
├── agentic_chatbot_backend.py
├── .env                     # API keys; do not commit
├── .gitignore
├── requirements.txt         # Python dependencies
├── chatbot.db               # Created/used by the SQLite checkpointer
└── faiss_db/  
├── Learnings.               # Created for learning purposes
```

The backend stores `chatbot.db` and `faiss_db/` relative to the backend Python file's directory.

## 5. Environment variables

Create a `.env` file in the project root. Add the required credentials:

```env
# Google Gemini (used by the chat model and embeddings)
GOOGLE_API_KEY=your_google_api_key

# Web search
TAVILY_API_KEY=your_tavily_api_key

# Stock quote lookup
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Current weather lookup
OPENWEATHER_API_KEY=your_openweathermap_api_key

# Optional: LangSmith tracing
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your_langsmith_api_key
# LANGSMITH_PROJECT=agentic-chatbot
```

The Google integration checks `GOOGLE_API_KEY` first and also supports `GEMINI_API_KEY`. Do not paste real credentials into documentation, source code, screenshots, or Git commits.

Add the following entries to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
chatbot.db
faiss_db/
```

The database and vector index may contain conversation data and content extracted from uploaded documents. Keep them private and back them up securely if they are important.

## 6. Installation

Use Python 3.11 or another Python version supported by the installed dependencies. Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install streamlit langgraph langgraph-checkpoint-sqlite langchain-core langchain-google-genai langchain-community langchain-text-splitters langchain-tavily faiss-cpu pypdf python-dotenv requests
```

For reproducible installs, record the tested versions from your working environment:

```bash
pip freeze > requirements.txt
```

Then future installations can use:

```bash
pip install -r requirements.txt
```

## 7. Run the application

From the project directory, run:

```bash
streamlit run agentic_chatbot_app_frontend.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 8. How to use the application

### Ask a general question

Enter a question in the chat input. The LLM answers directly when no tool is needed.

### Use a tool

Ask a question that requires one of the available tools. Examples:

- `What is the weather in Jaipur?`
- `What is 250 multiplied by 40?`
- `What is the current stock quote for AAPL?`
- `Search for recent developments in agentic AI.`

### Upload and query a PDF

1. Open the **Knowledge Base** section in the sidebar.
2. Upload one or more PDF files.
3. Select **Index uploaded PDF(s)**.
4. Wait for each indexing result.
5. Ask a question about the indexed documents, such as `What are the key skills listed in the resume?`.

### Test human approval

Use a clearly stated simulation request, for example:

`Buy 10 shares of AAPL.`

Review the approval request and select **Approve simulated purchase** or **Decline**. The graph resumes with the selected decision. This does not place a real order.

## 9. Data storage and persistence

| Data | Location | Purpose |
|---|---|---|
| Conversation checkpoints | `chatbot.db` | Stores graph state and thread history |
| PDF vector index | `faiss_db/` | Stores embedded document chunks and metadata |
| API credentials | `.env` | Stores local secrets |

Keep `chatbot.db` and `faiss_db/` if you want to preserve conversations and indexed documents between application restarts. Deleting either can remove the corresponding saved data.

## 10. Troubleshooting

| Problem | Checks |
|---|---|
| Gemini authentication error | Check that `GOOGLE_API_KEY` is set correctly in `.env`; restart Streamlit after changing it. |
| Tavily search fails | Check `TAVILY_API_KEY` and Tavily account access. |
| Stock quote lookup fails | Check `ALPHA_VANTAGE_API_KEY`, ticker symbol, and API response/rate limits. |
| Weather lookup fails | Check `OPENWEATHER_API_KEY` and the location name. |
| RAG says no documents are indexed | Upload a PDF and run **Index uploaded PDF(s)**; confirm `faiss_db/` exists. |
| PDF produces no chunks | The PDF may be scanned/image-only or contain no extractable text; add OCR support if required. |
| Existing threads are missing | Confirm the application is using the original `chatbot.db` in the project directory. |
| Approval prompt does not resume | Keep the same `thread_id` and `chatbot.db`; check the LangGraph interrupt/resume flow and installed package versions. |

## 11. Security and operational notes

- Never commit `.env` or expose API keys in screenshots or logs.
- The backend loads FAISS with `allow_dangerous_deserialization=True`. Only load a FAISS folder generated by and trusted within this application; do not load an index received from an untrusted source.
- Treat uploaded PDFs, the FAISS index, and `chatbot.db` as potentially sensitive data.
- The current stock purchase function is a demonstration and does not perform a real brokerage transaction.
- Before connecting a real broker, implement authentication, authorization, order validation, explicit approval of the exact order details, duplicate-order/idempotency protection, audit logging, and error handling. Do not reuse the simulation as a live trading implementation without those controls.
- Add OCR if the application needs to index scanned PDFs.

## 12. Future enhancements

Potential next improvements include:

- OCR for scanned PDF documents.
- Document deletion and re-indexing controls.
- Source citations and page references in RAG answers.
- RAG retrieval quality evaluation and configurable `k`/similarity thresholds.
- Authentication and per-user separation of threads and documents.
- Automated tests for tools, RAG ingestion, graph interrupts, and resume behaviour.
- Production-grade broker integration only after appropriate security and compliance review.
