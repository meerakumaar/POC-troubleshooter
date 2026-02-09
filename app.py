from pathlib import Path
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai

# ============================
# CACHE BUSTER (increment when you change chunking/indexing)
# ============================
APP_VERSION = "2026-02-09c"  # bump so Streamlit cache refreshes

# ============================
# CONFIG
# ============================
MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# Bigger chunks => fewer chunks => faster/cheaper
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 200

# Gemini embed batch hard-limit
EMBED_BATCH = 100

# Retrieval / safety knobs
TOP_K = 5
SIM_THRESHOLD = 0.20  # lower = more willing to answer; higher = asks more questions

# UI knob: show sources (for auditing). Default off.
DEFAULT_SHOW_SOURCES = False

st.set_page_config(page_title="POC Troubleshooter (RAG)", layout="centered")
st.title("POC Troubleshooter (RAG Prototype)")
st.caption("Manual-grounded troubleshooting assistant (prototype). Not a clinical tool.")
st.caption(f"Build: {APP_VERSION}")

# ============================
# AUTH
# ============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ============================
# PDF LOADING (/manuals)
# ============================
def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        if txt.strip():
            pages.append(txt)
    return "\n".join(pages).strip()

def load_repo_manuals() -> dict:
    manuals = {}
    manual_dir = Path(__file__).parent / "manuals"
    for pdf in manual_dir.glob("*.pdf"):
        try:
            manuals[pdf.name] = extract_pdf_text(pdf)
        except Exception:
            manuals[pdf.name] = ""
    return manuals

manual_texts = load_repo_manuals()

with st.expander("Loaded manuals (debug)"):
    if not manual_texts:
        st.write("No manuals found in /manuals.")
    else:
        for k, v in manual_texts.items():
            st.write(f"{k} → {len(v)} chars")

total_chars = sum(len(v or "") for v in manual_texts.values())
if total_chars < 1500:
    st.warning("Very little readable text found. Reboot the app after pushing changes.")

# Optional UI toggle for auditing sources
show_sources = st.toggle("Show sources (for auditing)", value=DEFAULT_SHOW_SOURCES)

# ============================
# RAG: chunking + embeddings index
# ============================
def chunk_text(text: str, size: int, overlap: int):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += max(1, size - overlap)
    return chunks

def vec_from_embedding(e) -> np.ndarray:
    if hasattr(e, "values"):
        return np.array(e.values, dtype=np.float32)
    return np.array(e, dtype=np.float32)

def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def detect_device(query: str) -> str | None:
    q = query.lower()
    if "piccolo" in q:
        return "piccolo"
    if "hemocue" in q or "hemo" in q:
        return "hemocue"
    if "istat" in q or "i-stat" in q:
        return "istat"
    if "gluco" in q or "glucose" in q or "glucometer" in q:
        return "gluco"
    return None

def filter_manuals(manuals: dict, device: str | None) -> dict:
    """
    Piccolo FIX:
    - If user says Piccolo, include all piccolo-related PDFs (hepatic/renal/op manual),
      even if the filename doesn't contain "piccolo" exactly the same way.
    """
    if not device:
        return manuals

    device = device.lower()

    if device == "piccolo":
        allowed = []
        for name in manuals.keys():
            n = name.lower()
            if "piccolo" in n or "hepatic" in n or "renal" in n:
                allowed.append(name)
        return {k: manuals[k] for k in allowed}

    return {k: v for k, v in manuals.items() if device in k.lower()}

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

@st.cache_data(show_spinner=True)
def build_index(manuals: dict, version: str):
    """
    Build embeddings index for a given manual subset.
    version is a cache-buster (APP_VERSION).
    """
    rows = []
    for fname, text in manuals.items():
        if not text or len(text) < 200:
            continue
        for ch in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            ch = ch.strip()
            if len(ch) >= 200:
                rows.append({"fname": fname, "chunk": ch})

    if not rows:
        return []

    contents = [r["chunk"] for r in rows]
    all_vecs = []

    # IMPORTANT: batch <= 100
    for batch in batched(contents, EMBED_BATCH):
        res = client.models.embed_content(model=EMBED_MODEL, contents=batch)
        all_vecs.extend([vec_from_embedding(e) for e in res.embeddings])

    mat = normalize(np.vstack(all_vecs))
    for i, r in enumerate(rows):
        r["vec"] = mat[i]

    return rows

def retrieve(index, query: str, k: int = TOP_K):
    if not index:
        return []

    qres = client.models.embed_content(model=EMBED_MODEL, contents=[query])
    qvec = vec_from_embedding(qres.embeddings[0])
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

    mat = np.vstack([r["vec"] for r in index])
    sims = mat @ qvec
    top = np.argsort(-sims)[:k]
    return [(index[i]["fname"], index[i]["chunk"], float(sims[i])) for i in top]

# ============================
# GROUNDED GENERATION (better language, no evidence shown by default)
# ============================
def grounded_response(query: str):
    device = detect_device(query)

    if not device:
        return (
            "Quick check—what device is this for: Piccolo, i-STAT, HemoCue, or Gluco?"
        ), None

    scoped = filter_manuals(manual_texts, device)
    if not scoped:
        scoped = manual_texts

    index = build_index(scoped, APP_VERSION)

    hits = retrieve(index, query, k=TOP_K)
    if not hits or hits[0][2] < SIM_THRESHOLD:
        return (
            "Got it. What *exactly* do you see—an on-screen error code/message, or what happens right before it fails?"
        ), None

    # Context for the model (keep sources internally)
    context = "\n\n".join([f"[SOURCE: {f}]\n{c}" for f, c, _ in hits])

    # We instruct the model to be grounded, but we DON'T require evidence in the user-facing output
    prompt = f"""
You are a troubleshooting assistant grounded ONLY in the provided manuals.

Rules:
- Use only the Manual Excerpts below as your factual source.
- Do NOT guess or add outside knowledge.
- Be concise and practical.
- Output exactly TWO lines:
  1) One next step the user can try (one action).
  2) One short question to narrow down what to do next.
- If the excerpts do not support a safe next step, output:
  1) Ask for the missing detail (one question), and
  2) Ask one follow-up question only.

Manual Excerpts:
{context}
""".strip()

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = (resp.text or "").strip()

    # Ensure clean formatting (2 lines)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        # fallback
        return (
            "What exact error code/message do you see (or what happens right before it fails)?"
        ), None

    user_facing = f"Step: {lines[0]}\nQuestion: {lines[1]}"

    # Provide sources separately for auditing if toggle is on
    sources_for_audit = hits
    return user_facing, sources_for_audit

# ============================
# CHAT UI
# ============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("Describe the issue (device + any error code helps)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            answer, sources = grounded_response(user_input)
        except Exception as e:
            st.error(f"Error while building RAG index / calling API: {type(e).__name__}")
            raise

        st.write(answer)

        if show_sources and sources:
            with st.expander("Sources used"):
                for fname, chunk, score in sources:
                    st.write(f"**{fname}** (score={score:.3f})")
                    st.write(chunk[:1200] + ("…" if len(chunk) > 1200 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
