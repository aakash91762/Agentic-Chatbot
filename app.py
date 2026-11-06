import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from backend import (
    chatbot,
    get_all_threads,
    get_indexed_documents,
    ingest_rag_document,
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Agentic Chatbot")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def content_to_text(content: Any) -> str:
    """Convert LangChain message content into displayable text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (
                "text",
                "output_text",
            ):
                parts.append(block.get("text", ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def make_thread_title(messages: list[dict[str, str]]) -> str:
    """Create a sidebar title from the first user message."""
    first_message = next(
        (
            message["content"]
            for message in messages
            if message.get("role") == "user"
        ),
        "New Chat",
    )
    title = " ".join(first_message.split())
    if len(title) > 40:
        title = title[:37].rstrip() + "..."
    return title or "New Chat"


def thread_config(thread_id: str) -> dict[str, Any]:
    """Build a consistent LangGraph config for this conversation."""
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
    }


def load_thread(thread_id: str) -> dict[str, Any]:
    """Load a conversation from the LangGraph checkpointer."""
    snapshot = chatbot.get_state(thread_config(thread_id))
    saved_messages = (
        snapshot.values.get("messages", [])
        if snapshot is not None and snapshot.values
        else []
    )

    ui_messages: list[dict[str, str]] = []
    for message in saved_messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            # Tool messages are shown in the activity panel while running.
            continue

        text = content_to_text(message.content)
        if text.strip():
            ui_messages.append({"role": role, "content": text})

    return {
        "title": make_thread_title(ui_messages),
        "messages": ui_messages,
    }


def get_pending_interrupt(thread_id: str) -> dict[str, Any] | None:
    """Return the first pending LangGraph interrupt for a thread, if any.

    LangGraph stores interrupt information on the pending tasks in the
    checkpoint snapshot. This lets the UI recover approvals after a rerun
    or application restart.
    """
    snapshot = chatbot.get_state(thread_config(thread_id))
    if snapshot is None:
        return None

    # StateSnapshot.interrupts contains interrupts still waiting for a response.
    for interrupt_item in getattr(snapshot, "interrupts", ()) or ():
        return {
            "id": getattr(interrupt_item, "id", None),
            "value": getattr(interrupt_item, "value", interrupt_item),
        }

    # Compatibility fallback: inspect pending task interrupts as well.
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt_item in getattr(task, "interrupts", ()) or ():
            return {
                "id": getattr(interrupt_item, "id", None),
                "value": getattr(interrupt_item, "value", interrupt_item),
            }

    return None


def index_uploaded_pdf(uploaded_file) -> tuple[str, str, int]:
    """Temporarily save an uploaded PDF and send it to the backend indexer."""
    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")
    if not pdf_bytes.lstrip().startswith(b"%PDF-"):
        raise ValueError("This file does not appear to be a valid PDF.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            prefix="rag_upload_",
            delete=False,
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        result = ingest_rag_document(
            temp_path,
            source_name=Path(uploaded_file.name).name,
        )
        return (
            result["status"],
            result["source"],
            result["chunks"],
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# Stream LangGraph events and display tool activity
# --------------------------------------------------
def stream_graph_response(graph_input: Any, thread_id: str, status):
    """Stream assistant text and show tool/graph events in the status panel.

    graph_input can be either a normal state update or Command(resume=...).
    """
    config = thread_config(thread_id)

    tool_labels = {
        "rag_tool": "your uploaded documents (RAG)",
        "calculator": "calculator",
        "get_stock_price": "stock price service",
        "get_current_weather": "weather service",
        "tavily_search": "web search",
        "search_tool": "web search",
        "purchase_stock": "stock purchase approval",
    }

    for mode, chunk in chatbot.stream(
        graph_input,
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            if not isinstance(chunk, dict):
                continue

            # An interrupt may be surfaced in updates as __interrupt__.
            if "__interrupt__" in chunk:
                for interrupt_item in chunk.get("__interrupt__", ()) or ():
                    interrupt_value = getattr(
                        interrupt_item, "value", interrupt_item
                    )
                    status.update(
                        label="⏸️ Waiting for your approval",
                        state="running",
                        expanded=True,
                    )
                    status.write(str(interrupt_value))

            for node_name, node_update in chunk.items():
                if not isinstance(node_update, dict):
                    continue

                node_messages = node_update.get("messages", []) or []

                if node_name == "chat_node":
                    for message in node_messages:
                        tool_calls = getattr(message, "tool_calls", None) or []
                        for call in tool_calls:
                            tool_name = call.get("name", "unknown tool")
                            friendly_name = tool_labels.get(tool_name, tool_name)

                            if tool_name == "rag_tool":
                                label = "🔎 Searching your uploaded documents..."
                            elif tool_name == "purchase_stock":
                                label = "🛒 Preparing stock purchase approval..."
                            else:
                                label = f"🛠️ Using {friendly_name}..."

                            status.update(
                                label=label,
                                state="running",
                                expanded=True,
                            )
                            status.write(f"Requesting {friendly_name}...")

                elif node_name == "tools":
                    for message in node_messages:
                        if getattr(message, "type", "") == "tool":
                            tool_name = getattr(message, "name", "unknown tool")
                            friendly_name = tool_labels.get(tool_name, tool_name)

                            if tool_name == "rag_tool":
                                status.write(
                                    "✅ Retrieved relevant document information."
                                )
                            elif tool_name == "purchase_stock":
                                status.write(
                                    "✅ Stock purchase decision processed."
                                )
                            else:
                                status.write(
                                    f"✅ Received results from {friendly_name}."
                                )

        elif mode == "messages":
            try:
                message_chunk, metadata = chunk
            except (TypeError, ValueError):
                continue

            if not isinstance(metadata, dict):
                continue
            if metadata.get("langgraph_node") != "chat_node":
                continue

            content = getattr(message_chunk, "content", "")
            if isinstance(content, str):
                if content:
                    yield content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        if block:
                            yield block
                    elif isinstance(block, dict) and block.get("type") in (
                        "text",
                        "output_text",
                    ):
                        text = block.get("text", "")
                        if text:
                            yield text


def render_graph_run(graph_input: Any, thread_id: str, status_label: str):
    """Run/resume graph, render streamed text, and return output and interrupt."""
    with st.chat_message("assistant"):
        status = st.status(status_label, expanded=True)
        try:
            answer = st.write_stream(
                stream_graph_response(graph_input, thread_id, status),
                cursor="▌",
            )
            if not isinstance(answer, str):
                answer = str(answer or "")

            pending = get_pending_interrupt(thread_id)
            if pending:
                status.update(
                    label="⏸️ Waiting for your approval",
                    state="running",
                    expanded=True,
                )
            else:
                status.update(
                    label="✅ Response complete",
                    state="complete",
                    expanded=False,
                )

            return answer, pending

        except Exception as error:
            status.update(
                label="⚠️ Something went wrong",
                state="error",
                expanded=True,
            )
            st.error(f"Sorry, an error occurred: {error}")
            return "", get_pending_interrupt(thread_id)


# --------------------------------------------------
# Initialize saved conversations once per session
# --------------------------------------------------
if "threads" not in st.session_state:
    st.session_state["threads"] = {}
    try:
        for saved_thread_id in get_all_threads():
            try:
                st.session_state["threads"][saved_thread_id] = load_thread(
                    saved_thread_id
                )
            except Exception as error:
                st.warning(
                    f"Could not load conversation {saved_thread_id}: {error}"
                )
    except Exception as error:
        st.warning(f"Could not retrieve saved conversations: {error}")

    if not st.session_state["threads"]:
        new_thread_id = str(uuid.uuid4())
        st.session_state["threads"][new_thread_id] = {
            "title": "New Chat",
            "messages": [],
        }

if (
    "active_thread_id" not in st.session_state
    or st.session_state["active_thread_id"] not in st.session_state["threads"]
):
    st.session_state["active_thread_id"] = next(
        iter(st.session_state["threads"])
    )


# --------------------------------------------------
# Sidebar: PDF upload and chat threads
# --------------------------------------------------
with st.sidebar:
    st.subheader("📚 Knowledge Base")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="rag_pdf_uploader",
        help="Upload one or more PDFs and index them for RAG questions.",
    )

    if st.button(
        "📥 Index uploaded PDF(s)",
        key="index_pdf_button",
        width="stretch",
        disabled=not uploaded_pdfs,
    ):
        successful_uploads = 0
        failed_uploads = 0

        for uploaded_pdf in uploaded_pdfs or []:
            try:
                with st.spinner(f"Indexing {uploaded_pdf.name}..."):
                    result_status, source_name, chunk_count = index_uploaded_pdf(
                        uploaded_pdf
                    )

                if result_status == "already_indexed":
                    st.info(f"Already indexed: {source_name}")
                else:
                    st.success(
                        f"Indexed {source_name} ({chunk_count} text chunks)."
                    )
                successful_uploads += 1

            except Exception as error:
                failed_uploads += 1
                st.error(f"Could not index {uploaded_pdf.name}: {error}")

        if successful_uploads and not failed_uploads:
            st.success("Your PDF knowledge base is ready to use in chat.")
        elif successful_uploads and failed_uploads:
            st.warning(
                f"Finished with {successful_uploads} successful and "
                f"{failed_uploads} failed upload(s)."
            )

    try:
        indexed_documents = get_indexed_documents()
        if indexed_documents:
            with st.expander(
                f"Indexed documents ({len(indexed_documents)})",
                expanded=False,
            ):
                for document_name in indexed_documents:
                    st.caption(f"📄 {document_name}")
        else:
            st.caption("No PDF documents indexed yet.")
    except Exception as error:
        st.caption(f"Could not list indexed documents: {error}")

    st.divider()
    st.subheader("💬 Chat Threads")

    if st.button(
        "＋ New Chat",
        key="new_chat_button",
        width="stretch",
    ):
        new_thread_id = str(uuid.uuid4())
        st.session_state["threads"][new_thread_id] = {
            "title": "New Chat",
            "messages": [],
        }
        st.session_state["active_thread_id"] = new_thread_id
        st.rerun()

    st.divider()

    for thread_id, thread in reversed(
        list(st.session_state["threads"].items())
    ):
        thread_title = thread.get("title", "New Chat")
        is_active = thread_id == st.session_state["active_thread_id"]

        if st.button(
            f"💬 {thread_title}",
            key=f"thread_{thread_id}",
            type="primary" if is_active else "secondary",
            width="stretch",
            help=thread_title,
        ):
            st.session_state["active_thread_id"] = thread_id
            st.rerun()


# --------------------------------------------------
# Active conversation and history
# --------------------------------------------------
active_thread_id = st.session_state["active_thread_id"]
active_thread = st.session_state["threads"][active_thread_id]
st.caption(f"Thread: {active_thread['title']}")

for message in active_thread["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Human-in-the-loop approval UI
# --------------------------------------------------
try:
    pending_interrupt = get_pending_interrupt(active_thread_id)
except Exception as error:
    pending_interrupt = None
    st.error(f"Could not check pending approval: {error}")

if pending_interrupt:
    interrupt_value = pending_interrupt.get("value")

    with st.container(border=True):
        st.warning("⏸️ Human approval required")
        st.markdown("**The agent is waiting for your decision:**")
        if isinstance(interrupt_value, (dict, list)):
            st.json(interrupt_value)
        else:
            st.write(str(interrupt_value))

        st.caption(
            "This project currently simulates a stock purchase. "
            "It does not submit a real order to a brokerage."
        )

        approve_col, reject_col = st.columns(2)
        with approve_col:
            approve_clicked = st.button(
                "✅ Approve simulated purchase",
                key=f"approve_purchase_{active_thread_id}",
                type="primary",
                width="stretch",
            )
        with reject_col:
            reject_clicked = st.button(
                "❌ Decline",
                key=f"reject_purchase_{active_thread_id}",
                width="stretch",
            )

        if approve_clicked or reject_clicked:
            decision = "yes" if approve_clicked else "no"
            decision_label = (
                "Resuming after approval..."
                if approve_clicked
                else "Resuming after rejection..."
            )

            answer, still_pending = render_graph_run(
                Command(resume=decision),
                active_thread_id,
                decision_label,
            )

            if answer.strip():
                active_thread["messages"].append(
                    {"role": "assistant", "content": answer}
                )

            # Refresh to show the resumed graph result or any new interrupt.
            st.rerun()


# --------------------------------------------------
# Chat input
# --------------------------------------------------
user_input = st.chat_input(
    "Ask a question about your PDFs or anything else...",
    disabled=bool(pending_interrupt),
)

if isinstance(user_input, str) and user_input.strip():
    user_input = user_input.strip()
    is_first_message = len(active_thread["messages"]) == 0

    if is_first_message:
        active_thread["title"] = make_thread_title(
            [{"role": "user", "content": user_input}]
        )

    active_thread["messages"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    answer, pending = render_graph_run(
        {"messages": [HumanMessage(content=user_input)]},
        active_thread_id,
        "🤔 Thinking...",
    )

    # If the graph paused for human approval, do not show an empty-answer
    # placeholder. The approval UI appears on the following rerun.
    if answer.strip():
        active_thread["messages"].append(
            {"role": "assistant", "content": answer}
        )
    elif not pending:
        fallback = "I couldn't generate a text response."
        active_thread["messages"].append(
            {"role": "assistant", "content": fallback}
        )

    # Rerun after every turn so pending interrupts are rendered immediately.
    st.rerun()
