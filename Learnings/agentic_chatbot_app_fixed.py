import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from Learnings.agentic_chatbot_tool_backend import chatbot, get_all_threads


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Agentic Chatbot with LangGraph")


# --------------------------------------------------
# Helpers for displaying saved messages
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


def load_thread(thread_id: str) -> dict[str, Any]:
    """Load a conversation from the LangGraph checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = chatbot.get_state(config)

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
            # Tool messages are not displayed as ordinary chat messages.
            continue

        text = content_to_text(message.content)
        if text.strip():
            ui_messages.append({"role": role, "content": text})

    return {
        "title": make_thread_title(ui_messages),
        "messages": ui_messages,
    }


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
# Sidebar: conversation list
# --------------------------------------------------
with st.sidebar:
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
# Active conversation
# --------------------------------------------------
active_thread_id = st.session_state["active_thread_id"]
active_thread = st.session_state["threads"][active_thread_id]
st.caption(f"Thread: {active_thread['title']}")


# --------------------------------------------------
# Stream LangGraph events and show tool activity
# --------------------------------------------------
def stream_response(user_input: str, thread_id: str, status):
    """Stream assistant text while displaying real graph/tool events."""
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("The chat message must be a non-empty string.")

    user_input = user_input.strip()
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
    }

    for mode, chunk in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        # Graph node updates tell us when a tool call is requested and when
        # the ToolNode returns its results.
        if mode == "updates":
            if not isinstance(chunk, dict):
                continue

            for node_name, node_update in chunk.items():
                if not isinstance(node_update, dict):
                    continue

                node_messages = node_update.get("messages", []) or []

                if node_name == "chat_node":
                    for message in node_messages:
                        tool_calls = getattr(message, "tool_calls", None) or []
                        if tool_calls:
                            tool_names = [
                                call.get("name", "unknown tool")
                                for call in tool_calls
                            ]
                            names = ", ".join(tool_names)
                            status.update(
                                label=f"🛠️ Calling {names}",
                                state="running",
                                expanded=True,
                            )
                            for name in tool_names:
                                status.write(f"🔎 Requesting {name}...")

                elif node_name == "tools":
                    for message in node_messages:
                        if getattr(message, "type", "") == "tool":
                            tool_name = getattr(message, "name", "tool")
                            status.write(
                                f"✅ Received results from {tool_name}"
                            )

        # Message mode streams token/message chunks from the LLM.
        elif mode == "messages":
            try:
                message_chunk, metadata = chunk
            except (TypeError, ValueError):
                continue

            if not isinstance(metadata, dict):
                continue

            # Only show user-facing text generated by chat_node, not tool output.
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


# Show existing history.
for message in active_thread["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input and response
# IMPORTANT: all response generation stays inside this input check.
# --------------------------------------------------
user_input = st.chat_input("Type your message...")

if isinstance(user_input, str) and user_input.strip():
    user_input = user_input.strip()
    is_first_message = len(active_thread["messages"]) == 0

    # Set the conversation title on its first user message.
    if is_first_message:
        active_thread["title"] = make_thread_title(
            [{"role": "user", "content": user_input}]
        )

    # Save and display the user message.
    active_thread["messages"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the graph only when a valid user message was submitted.
    with st.chat_message("assistant"):
        # Create the status block as a sibling of the answer, rather than
        # using `with st.status(...)`, which would nest the answer inside it.
        status = st.status("🤔 Thinking...", expanded=True)

        try:
            # The status updates stay in the expandable activity panel.
            # Streamed answer text is rendered separately as a normal reply.
            answer = st.write_stream(
                stream_response(user_input, active_thread_id, status),
                cursor="▌",
            )

            if not isinstance(answer, str):
                answer = str(answer or "")

            if not answer.strip():
                answer = "I couldn't generate a text response."

            status.update(
                label="✅ Response complete",
                state="complete",
                expanded=False,
            )

        except Exception as error:
            answer = f"Sorry, an error occurred: {error}"
            try:
                status.update(
                    label="⚠️ Something went wrong",
                    state="error",
                    expanded=True,
                )
            except Exception:
                pass
            st.error(answer)

    # Save the assistant response in this thread's UI history.
    active_thread["messages"].append(
        {"role": "assistant", "content": answer}
    )

    # Refresh the sidebar immediately after the first message.
    if is_first_message:
        st.rerun()
