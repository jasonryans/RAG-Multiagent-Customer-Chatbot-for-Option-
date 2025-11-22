import os
import textwrap
import json
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import cohere

# ---------- CONFIG ----------

load_dotenv()  
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

# Cohere client for coach agent
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# ---------- DATA LOADING & CHUNKING ----------

def load_knowledge_file(path: str = "option_knowledge.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_chunks(text: str, max_chars: int = 700):
    """Split text into paragraph-based chunks."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current += ("\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

# ---------- EMBEDDINGS & RETRIEVAL ----------

def embed_texts(texts):
    """Compute embeddings for a list of texts using Gemini embeddings."""
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    all_embs = []
    for t in texts:
        r = genai.embed_content(
            model=EMBED_MODEL,
            content=t,
            task_type="retrieval_document",
        )
        all_embs.append(r["embedding"])
    return np.array(all_embs, dtype="float32")

def embed_query(query: str):
    r = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return np.array(r["embedding"], dtype="float32")

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.dot(a, b.T)

def retrieve_relevant_chunks(query, chunks, chunk_embeddings, k=4):
    if len(chunk_embeddings) == 0:
        return [], []
    q_emb = embed_query(query)
    sims = cosine_similarity(q_emb.reshape(1, -1), chunk_embeddings)[0]
    k = min(k, len(chunks))
    top_idx = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_idx], sims[top_idx].tolist()

# ---------- LLM PROMPT & ANSWER ----------

def build_system_prompt():
    return textwrap.dedent(
        """
        You are a helpful chatbot for "Option Bistro & Billiard" called "Option Bot".

        You have three responsibilities:
        1) Answer questions about Option Bistro & Billiard (menu prices, billiard prices, opening hours,
           house rules, and billiards tips) using the provided CONTEXT.
        2) Detect when the user wants to manage billiard table reservations (reserve or cancel a table).
        3) Detect when the user is asking for general billiards knowledge or tutorials (not specific to Option),
           and delegate those questions to a separate Billiards Coach Agent.

        The knowledge base (CONTEXT) may be written partly in Indonesian.
        Your responses MUST ALWAYS be in clear English.

        You do NOT have direct access to the database, but the surrounding Python program can perform actions
        if you output structured JSON describing what should happen.

        === OUTPUT FORMAT (VERY IMPORTANT) ===

        You must ALWAYS respond with a single JSON object and nothing else.
        No extra commentary, no markdown, no code fences. Just raw JSON.

        The JSON object must be one of the following shapes:

        1) Normal chat answer (Option-related):
           {
             "action": "chat",
             "reply": "<message to show to the user>"
           }

        2) When the user clearly wants to RESERVE / BOOK a billiard table:
           {
             "action": "reserve_table",
             "table_number": <integer 1-8 or null if not clear>,
             "customer_name_from_text": "<name extracted from the user's message if present, otherwise empty string>",
             "reply": "<message to show to the user>"
           }

        3) When the user clearly wants to CANCEL a billiard table reservation:
           {
             "action": "cancel_reservation",
             "table_number": <integer 1-8 or null if not clear>,
             "customer_name_from_text": "<name extracted from the user's message if present, otherwise empty string>",
             "reply": "<message to show to the user>"
           }

        4) When the user is asking about GENERAL BILLIARDS KNOWLEDGE or TUTORIALS
           that are NOT specific to Option's prices, opening hours, promotions, or reservations
           (for example: how to play 8-ball, how to aim better, how to break, drills, strategy,
           or requests for tutorial videos), you MUST NOT answer the question yourself.
           Instead, delegate to the external Billiards Coach Agent by returning:

           {
             "action": "coach_chat",
             "reply": ""
           }

           In this case, leave "reply" as an empty string.
           The surrounding Python program will call a separate model (Cohere) to generate
           the actual explanation and tutorial. Your job is ONLY to detect and route.

        Guidelines:
        - Use the CONTEXT for any facts about Option: prices, menu items, promotions, opening hours,
          number of tables, minimum order, house rules, etc.
        - For Option-related questions (1–3), answer normally using the CONTEXT and return "action": "chat",
          "reserve_table", or "cancel_reservation" as appropriate.
        - For pure general billiards questions or tutorials, DO NOT answer directly. Instead, choose
          "action": "coach_chat" with "reply": "" exactly, so the Python code can hand off to the coach agent.
        - If a question mixes both (e.g., "How much is billiards per hour at Option, and how do I break properly?"),
          you may either:
            * answer the Option-specific part and still return "action": "chat", OR
            * if most of the question is general tutorial, you may return "action": "coach_chat" and let
              the coach agent handle all the explanation.
          Choose the behavior that seems most helpful.
        - If the question is completely unrelated to Option or billiards, return action "chat" with a short reply
          saying that you are only for Option and billiards-related questions.
        - Never invent Option prices or promotions that are not in the CONTEXT. If something is unknown, say so.

        Remember: respond in English and output pure JSON only.
        """
    )

def format_context(chunks):
    if not chunks:
        return "CONTEXT: (no additional documents found)"
    joined = "\n\n---\n\n".join(chunks)
    return f"CONTEXT:\n{joined}"

def extract_yt_links(text: str):
    return [
        line.strip()
        for line in text.splitlines()
        if ("youtu.be" in line or "youtube.com" in line)
    ]

# ---------- BILLIARDS COACH AGENT (COHERE) ----------

def call_coach_agent(question: str, chat_history=None, context_chunks=None, all_links=None) -> str:
    """
    Use Cohere as the billiards coach agent.
    Returns a plain text explanation/tutorial answer.
    """

    if cohere_client is None:
        return (
            "The billiards coach agent is not configured yet "
            "(COHERE_API_KEY is missing). Please contact the administrator."
        )

    history_lines = []
    for role, msg in (chat_history or []):
        prefix = "User: " if role == "user" else "Bot: "
        history_lines.append(prefix + msg)
    history_text = "\n".join(history_lines[-6:]) if history_lines else "(no previous messages)"

    context_text = "\n\n".join(context_chunks or [])
    links_text = "\n".join(all_links or [])

    coach_system = textwrap.dedent(
        """
        You are "Billiards Coach Agent", a friendly virtual coach for billiards / pool.

        Your job:
        - Explain how to play billiards/pool (especially 8-ball) for beginners.
        - Explain rules, fouls, and table etiquette.
        - Give tips on stance, bridge, grip, stroke, aiming, and cue ball control.
        - Suggest simple practice drills and strategy for improvement.
        - Include ONLY relevant YouTube tutorial links from the provided list; do not invent URLs.
        - Match the user's language (Indonesian or English).
        Keep the explanation clear and structured.
        """
    )

    message = (
        coach_system
        + "\n\n[Recent chat]\n" + history_text
        + "\n\n[Retrieved context]\n" + (context_text or "(none)")
        + "\n\n[Available tutorial links]\n" + (links_text or "(none)")
        + "\n\n[User question]\n" + question
    )

    response = cohere_client.chat(
        model="command-a-03-2025",
        message=message,
    )
    return response.text.strip()


def generate_answer(question, retrieved_chunks, chat_history=None, user_name=None):
    if chat_history is None:
        chat_history = []

    system_prompt = build_system_prompt()
    context = format_context(retrieved_chunks)

    history_parts = []
    for role, msg in chat_history:
        api_role = "user" if role == "user" else "model"
        history_parts.append({"role": api_role, "parts": [msg]})

    user_name_info = (
        f"The current user's display name (from the app, may be empty) is: {user_name}."
        if user_name
        else "The app did not provide a display name for the user."
    )

    final_user_message = (
        f"{context}\n\n"
        f"User question: {question}\n\n"
        f"{user_name_info}\n"
        "Remember to respond with a single JSON object as specified."
    )

    messages = (
        [{"role": "user", "parts": [system_prompt]}]
        + history_parts
        + [{"role": "user", "parts": [final_user_message]}]
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(messages)
    raw_text = resp.text.strip()

    try:
        data = json.loads(raw_text)
    except Exception:
        return raw_text

    action = data.get("action")

    if action == "coach_chat":
        full_text = load_knowledge_file("option_knowledge.txt")
        all_links = extract_yt_links(full_text)
        coach_reply = call_coach_agent(
            question,
            chat_history=chat_history,
            context_chunks=retrieved_chunks,
            all_links=all_links
        )
        final_payload = {
            "action": "chat",
            "reply": coach_reply,
        }
        return json.dumps(final_payload, ensure_ascii=False)

    return raw_text


# ---------- CONVENIENCE: build index once ----------

def build_index(path="option_knowledge.txt"):
    text = load_knowledge_file(path)
    chunks = split_into_chunks(text)
    embeddings = embed_texts(chunks)
    return chunks, embeddings
