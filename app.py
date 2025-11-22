import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import streamlit as st
from rag_option import build_index, retrieve_relevant_chunks, generate_answer

def extract_json_object(raw: str):
    """
    Try to extract a JSON object from the model's raw text.
    Handles cases where the model wraps JSON in extra text or code fences.
    Returns a dict or None.
    """
    if not raw:
        return None

    stripped = raw.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        stripped = "".join(parts[1:-1]).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None

# ---------- DATABASE HELPERS ----------

DB_PATH = "reservations.db"

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """
    Create the reservations table if it doesn't exist,
    and ensure rows for table 1..8 exist.
    """
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS reservations (
                table_num INTEGER PRIMARY KEY,
                name TEXT,
                time TEXT
            )
            """
        )
        for i in range(1, 9):
            c.execute(
                "INSERT OR IGNORE INTO reservations(table_num, name, time) "
                "VALUES (?, NULL, NULL)",
                (i,),
            )
        conn.commit()

def get_all_reservations():
    """
    Return dict {table_num: {"name": str|None, "time": str|None} or None}
    """
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT table_num, name, time FROM reservations ORDER BY table_num")
        rows = c.fetchall()

    reservations = {}
    for table_num, name, time in rows:
        if name is None:
            reservations[table_num] = None
        else:
            reservations[table_num] = {"name": name, "time": time}
    return reservations

def reserve_table(table_num: int, user_name: str) -> str:
    """
    Reserve a table if available. Returns a user-facing message string.
    """
    if table_num < 1 or table_num > 8:
        return "In Options only tables 1 to 8 are available. Please select a number in that range."

    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT name, time FROM reservations WHERE table_num = ?",
            (table_num,),
        )
        row = c.fetchone()
        if not row:
            return "An error occurred: table not found in the database."

        current_name, current_time = row
        if current_name is not None:
            info = f" (Booked at {current_time})" if current_time else ""
            return (
                f"❌ Sorry, table {table_num} already booked by **{current_name}**{info}.\n\n"
                "Please select another table (1–8) or ask the Option staff directly."
            )

        now_str = datetime.now().isoformat(timespec="minutes")
        c.execute(
            "UPDATE reservations SET name = ?, time = ? WHERE table_num = ?",
            (user_name or "Guest", now_str, table_num),
        )
        conn.commit()

    return (
        f"✅ Meja {table_num} sudah **dibooking** atas nama **{user_name or 'Guest'}**.\n\n"
        "Silakan tunjukkan konfirmasi ini kepada staf Option saat datang ya."
    )

def cancel_reservation(table_num: int, user_name: str) -> str:
    """
    Cancel a reservation for a given table.

    Policy:
    - Only the person whose name is on the booking can cancel that booking.
    - Name comparison is case-insensitive and trimmed.
    """
    if table_num < 1 or table_num > 8:
        return "Option only has tables 1 to 8. Please choose a table number in that range."

    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT name, time FROM reservations WHERE table_num = ?",
            (table_num,),
        )
        row = c.fetchone()
        if not row:
            return "An error occurred: table not found in the database."

        current_name, current_time = row
        if current_name is None:
            return f"Table {table_num} currently has no active booking."

        stored = current_name.strip().lower()
        requester = (user_name or "").strip().lower()

        if not requester:
            return (
                f"The booking for table {table_num} is under the name **{current_name}**.\n\n"
                "To cancel this booking, please enter the same name in the **Your name** box "
                "and then ask to cancel again, or contact the Option staff directly."
            )

        if requester != stored:
            return (
                f"Only the person who booked this table can cancel it.\n\n"
                f"The booking for table {table_num} is under **{current_name}**, "
                f"but you entered **{user_name}**.\n\n"
                "If this is a mistake, please update the name field or contact the Option staff directly."
            )

        c.execute(
            "UPDATE reservations SET name = NULL, time = NULL WHERE table_num = ?",
            (table_num,),
        )
        conn.commit()

    return (
        f"✅ The booking for table {table_num} under the name **{current_name}** has been cancelled.\n\n"
        "If this is not correct, please confirm with the Option staff."
    )


# ---------- RAG INDEX CACHE ----------

@st.cache_resource
def load_rag_index():
    chunks, embeddings = build_index("option_knowledge.txt")
    return chunks, embeddings

# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="Option Bistro & Billiard Chatbot", page_icon="🎱")
    st.title("🎱 Option Bistro & Billiard Chatbot")

    st.write(
        "Ask me about **billiard prices**, **menu prices**, **opening hours** or **tutorial playing billiard**"
        " at Option.\n\n"
        "You could also ask me to help you reserve or cancel a billiard table."
    )

    init_db()

    user_name = st.text_input(
        "Your name (will be used for confirmation on your reservation table)", key="user_name"
    )

    chunks, embeddings = load_rag_index()

    with st.sidebar:
        st.subheader("Availability Billiard Table (1–8)")
        reservations = get_all_reservations()
        for i in range(1, 9):
            res = reservations.get(i)
            if res is None:
                st.write(f"Table {i}: 🟢 **Available**")
            else:
                name = res.get("name", "Booked")
                st.write(f"Table {i}: 🔴 Booked ({name})")

    # 5) chat history
    if "history" not in st.session_state:
        st.session_state.history = []  

    for role, msg in st.session_state.history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg)

    # 6) chat input
    user_input = st.chat_input(
        "Write your questions about Option bistro or billiard here..."
    )

    if user_input:
        st.session_state.history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        retrieved_chunks, sims = retrieve_relevant_chunks(
            user_input, chunks, embeddings, k=4
        )

        raw_answer = generate_answer(
            question=user_input,
            retrieved_chunks=retrieved_chunks,
            chat_history=st.session_state.history,
            user_name=user_name,   
        )

        answer_text = raw_answer

        obj = extract_json_object(raw_answer)
        action = "chat"
        if obj is not None and isinstance(obj, dict):
            action = obj.get("action", "chat")

        if obj is None or action == "chat":
            if obj is not None:
                answer_text = obj.get("reply", raw_answer)
            else:
                answer_text = raw_answer

        elif action == "reserve_table":
            table_number = obj.get("table_number")
            if table_number is None:
                answer_text = obj.get(
                    "reply",
                    "Which table you want to book (1–8)",
                )
            else:
                answer_text = reserve_table(
                    int(table_number),
                    user_name or obj.get("customer_name_from_text") or "Guest",
                )

        elif action == "cancel_reservation":
            table_number = obj.get("table_number")
            if table_number is None:
                answer_text = obj.get(
                    "reply",
                    "Table cancellation for which number? (1–8)",
                )
            else:
                answer_text = cancel_reservation(
                    int(table_number),
                    user_name or obj.get("customer_name_from_text") or "Guest",
                )

        else:
            answer_text = obj.get("reply", raw_answer) if obj else raw_answer

        st.session_state.history.append(("assistant", answer_text))
        with st.chat_message("assistant"):
            st.markdown(answer_text)

if __name__ == "__main__":
    main()
