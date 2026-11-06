from __future__ import annotations

import ast
import hashlib
import math
import os
import sqlite3
from pathlib import Path
from typing import Annotated, Any, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command



# --------------------------------------------------
# Paths and environment
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FAISS_DB_PATH = BASE_DIR / "faiss_db"
SQLITE_DB_PATH = BASE_DIR / "chatbot.db"

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# Models
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)


# --------------------------------------------------
# RAG document ingestion and retrieval
# --------------------------------------------------
def ingest_rag_document(
    file_path: str | Path,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Load a PDF and add its chunks to the persistent FAISS index.

    Re-uploading an identical PDF is detected by its SHA-256 hash and will
    not add duplicate chunks. Different PDFs are appended to the same index.
    """
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be indexed.")

    file_bytes = pdf_path.read_bytes()
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    document_id = hashlib.sha256(file_bytes).hexdigest()
    display_name = source_name or pdf_path.name

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    if not documents:
        raise ValueError("No readable pages were found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may contain scanned images and require OCR."
        )

    for chunk in chunks:
        chunk.metadata["source"] = display_name
        chunk.metadata["document_id"] = document_id

    if FAISS_DB_PATH.exists():
        vector_store = FAISS.load_local(
            folder_path=str(FAISS_DB_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        # Avoid duplicate indexing when the exact same PDF is uploaded again.
        for stored_id in vector_store.index_to_docstore_id.values():
            stored_document = vector_store.docstore.search(stored_id)
            if (
                isinstance(stored_document, Document)
                and stored_document.metadata.get("document_id") == document_id
            ):
                return {
                    "status": "already_indexed",
                    "source": display_name,
                    "chunks": len(chunks),
                }

        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(str(FAISS_DB_PATH))

    return {
        "status": "indexed",
        "source": display_name,
        "chunks": len(chunks),
    }


def get_retriever():
    """Load the persisted FAISS index and return a retriever."""
    if not FAISS_DB_PATH.exists():
        raise FileNotFoundError(
            "No PDF documents have been indexed yet. Upload a PDF first."
        )

    vector_store = FAISS.load_local(
        folder_path=str(FAISS_DB_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


def get_indexed_documents() -> list[str]:
    """Return unique source filenames currently represented in FAISS."""
    if not FAISS_DB_PATH.exists():
        return []

    vector_store = FAISS.load_local(
        folder_path=str(FAISS_DB_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    sources: set[str] = set()
    for stored_id in vector_store.index_to_docstore_id.values():
        stored_document = vector_store.docstore.search(stored_id)
        if isinstance(stored_document, Document):
            source = stored_document.metadata.get("source")
            if source:
                sources.add(str(source))

    return sorted(sources)


@tool(
    "rag_tool",
    description="""
    Search the uploaded and indexed PDF documents for relevant information.

    ALWAYS use this tool when users ask questions about information
    that may be available in uploaded documents.

    This includes:
    - A person's name, profile, or resume.
    - Technical skills and professional skills.
    - Work experience and previous projects.
    - Education and certifications.
    - Responsibilities and achievements.
    - Any factual information contained in uploaded PDFs.

    Example:
    If the user asks "What are the top skills of Aakash?",
    search the uploaded documents for Aakash's skills.

    Do not ask the user to upload a document before searching
    the existing knowledge base.
    """
)
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the uploaded PDF documents.

    Use this tool for factual or conceptual questions that may be answered
    from PDFs uploaded to this chatbot's knowledge base.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    try:
        retriever = get_retriever()
        documents = retriever.invoke(query)
    except FileNotFoundError:
        return "No PDF documents have been uploaded and indexed yet."
    except Exception as error:
        return f"Document retrieval failed: {error}"

    if not documents:
        return "No relevant information was found in the uploaded PDFs."

    formatted_documents = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# --------------------------------------------------
# Other tools
# --------------------------------------------------
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression such as 2 + 2 or math.sqrt(16)."""
    allowed_names = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }

    try:
        # Restrict expression syntax to basic arithmetic and approved functions.
        parsed = ast.parse(expression, mode="eval")
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Call,
            ast.Name,
            ast.Attribute,
            ast.Load,
            ast.Tuple,
            ast.List,
        )
        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                raise ValueError("Unsupported operation in expression.")
            if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValueError("Unsupported function call.")

        result = eval(compile(parsed, "<calculator>", "eval"), {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def get_stock_price(symbol: str) -> dict[str, Any]:
    """Fetch the latest stock quote for a ticker such as AAPL or TSLA."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is missing from .env."}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper().strip(),
                "apikey": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        return {"error": f"Stock API request failed: {error}"}


@tool
def get_current_weather(location: str) -> str:
    """Get current weather for a city or location, for example Jaipur or London."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return (
            "Weather API key is missing. Set OPENWEATHER_API_KEY in the .env file."
        )

    try:
        geocoding_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geocoding_response.raise_for_status()
        locations = geocoding_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")
        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]
        if state:
            location_parts.append(state)
        if country:
            location_parts.append(country)

        return (
            f"Current weather in {', '.join(location_parts)}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )
    except requests.Timeout:
        return "The weather service request timed out. Please try again."
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"
        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status_code}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"

##purchase tool
@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }




# --------------------------------------------------
# LangGraph agent
# --------------------------------------------------
tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool,purchase_stock]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that can answer directly or call tools."""

    system_message = SystemMessage(
        content="""
You are a helpful Agentic AI assistant with access to multiple tools.

You have access to an indexed PDF knowledge base through rag_tool.

IMPORTANT KNOWLEDGE BASE INSTRUCTIONS:

1. Always search the uploaded documents using rag_tool
   when the user's question may be answered from those documents.

2. Do not require the user to explicitly mention PDF, document,
   resume, or knowledge base.

3. If the user asks about a person's:
   - Name or profile
   - Technical skills
   - Professional skills
   - Work experience
   - Education
   - Certifications
   - Projects
   - Achievements

   You MUST search rag_tool before answering.

4. For example:
   User: "What are the top skills of Aakash?"

   Action: Call rag_tool with a query such as:
   "Find Aakash's technical skills, professional skills,
   technologies, tools, and competencies from the uploaded resume."

5. Do not ask the user to upload a document if documents
   are already indexed in the knowledge base.

6. Base document-related answers on retrieved information.
   Do not invent information that is not present in the documents.

7. If no relevant information is found, clearly tell the user
   that the indexed documents do not contain the requested information.

OTHER TOOL INSTRUCTIONS:

- Use search_tool for current events and recent internet information.
- Use calculator for mathematical calculations.
- Use get_stock_price for current stock prices.
- Use get_current_weather for current weather information.
- Answer general questions directly when no tool is needed.
- Use purchase_stock when the user explicitly requests
  to buy a stock and both stock symbol and quantity are known.
- If stock symbol or quantity is missing, ask the user
  before calling the tool.
- Do not claim that a purchase is completed before
  receiving the tool result.
- Always wait for the human approval interrupt
  before proceeding with the purchase operation.
- The current purchase_stock tool is a simulation,
  not a real brokerage transaction.

After receiving tool results, provide a clear and helpful final answer.
"""
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)

# The paths are anchored to this Python file so they do not change depending
# on the directory from which Streamlit is launched.
conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads() -> list[str]:
    """Return unique thread IDs saved in the SQLite checkpointer."""
    all_threads: set[str] = set()
    for checkpoint_tuple in checkpoint.list(None):
        configurable = checkpoint_tuple.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if thread_id:
            all_threads.add(str(thread_id))
    return list(all_threads)
