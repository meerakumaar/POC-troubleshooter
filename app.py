import os
from pathlib import Path
import re

import streamlit as st
from pypdf import PdfReader
from google import genai

# ----------------------------
# UI (PRD: plain, prototype copy)
# ----------------------------
st.set_page_config(page_title="POC Troubleshooting Chat (Prototype)", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")  # PRD framing :contentReference[oaicite:3]{index=3}

# ----:contentReference[oaicite:4]{index=4}---
# Gemini client (new SDK)
# ----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"  # current supported line :contentReference[oaicite:5]{index=5}

# ----------------------------
# Manual loading + lightweight retrieval
# Put PDFs in a folder named `manuals/` in your repo
# ----------------------------
APP_DIR = Path(__file__).parent
MANUAL_DIR = APP_DIR / "manuals"

MANUAL_FILES = [
    "gluco.pdf",
    "hemocue correct.pdf",
    "hepatic piccolo.pdf",
    "istat.pdf",
    "piccolo op manual .pdf",
    "renal piccolo.pdf",
]

def _safe_extract(page) -> str:
    txt = page.extract_text()
    return txt if txt else ""

@st.cache_data(show_spinner=False)
def load_manual_texts() -> dict:
    manuals = {}
    for fname in MANUAL_FILES:
        fpath = MANUAL_DIR / fname
        if fpath.exists():
            reader = PdfReader(str(fpath))
            manuals[fname] = "\n".join(_safe_extract(p) for p in reader.pages)
    return manuals

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 150):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

@st.cache_data(show_spinner=False)
def build_chunks():
    manuals = load_manual_texts()
    all_chunks = []
    for fname, text in manuals.items():
        for ch in chunk_text(text):
            all_chunks.append((fname, ch))
    return all_chunks

def retrieve_relevant_chunks(query: str, k: int = 4):
    chunks = build_chunks()
    if not chunks:
        return []

    terms = [t for t in re.findall(r"[a-zA-Z0-9\-]+", query.lower()) if len(t) >= 3]
    if not terms:
        return []

    scored = []
    for fname, ch in chunks:
        low = ch.lower()
        score = sum(low.count(t) for t in terms)
        if score > 0:
            scored.append((score, fname, ch))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:k]
    return [(fname, ch) for _, fname, ch in top]

# ----------------------------
# Chat state (in-session only)
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ----------------------------
# Main interaction
# ----------------------------
user_prompt = st.chat_input("Enter issue")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    excerpts = retrieve_relevant_chunks(user_prompt, k=4)

    if not excerpts:
        assistant_text = "Step 1:\nNot in manual.\nIs there an error code displayed? (yes/no)"
        with st.chat_message("assistant"):
            st.write(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.stop()

    # PRD style: single step, single action, single yes/no question, no teaching, no extra text :contentReference[oaicite:6]{index=6}
    excerpt_block = "\n\n".join([f"[SOURCE: {fname}]\n{txt}" :contentReference[oaicite:7]{index=7}erpts])

    instruction = f"""
You are a prototype manual-based troubleshooting assistant.
You MUST follow these rules:

- Use ONLY the provided manual excerpts below. Do NOT guess. Do NOT add “typical” advice.
- Output EXACTLY ONE step in this format (and nothing else):

Step 1:
<Exact instruction or constraint from the manual excerpt>
<ONE yes/no or concrete question>

- Only one action or question total.
- No bullet lists. No explanations. No teaching.
- If the excerpt does not contain a relevant instruction, output:

Step 1:
Not in manual.
Is there an error code displayed? (yes/no)

Manual excerpts:
{excerpt_block}
""".strip()

    with st.chat_message("assistant"):
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=instruction,
        )
        assistant_text = (resp.text or "").strip() or "Step 1:\nNot in manual.\nIs there an error code displayed? (yes/no)"
        st.write(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
