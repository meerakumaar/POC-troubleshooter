import re
from pathlib import Path
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

st.set_page_config(page_title="POC Troubleshooter (RAG)", layout="centered")
st.title("POC Troubleshooter (RAG Prototype)")
st.caption("Prototype: manual-grounded troubleshooting assistant. Not a clinical tool.")

# ----------------------------
# AUTH
# ----------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ----------------------------
# PDF LOADING
# ----------------------------
def extract_pdf_text_from_path(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts).strip()


def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    # pypdf can read from bytes via a BytesIO-like object
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts).strip()


def load_repo_manuals() -> dict:
    """
    Loads PDFs from ./manuals if present.
    Returns {filename: text}
    """
    manuals = {}
    manual_dir = Path(__file__).parent / "manuals"
    if not manual_dir.exists():
        return manuals

    for pdf in manual_dir.glob("*.pdf"):
        try:
            text = extract_pdf_text_from_path(pdf)
            manuals[pdf.name] = text
        except Exception:
            manuals[pdf.name] = ""
    return manuals


uploaded = st.file_uploader(
    "Upload one or more manual PDFs (optional, helpful on Streamlit Cloud)",
    type=["pdf"],
    accept_multiple_files=True,
)

manual_texts = load_repo_manuals()

if uploaded:
    for f in uploaded:
        try:
            manual_texts[f.name] = extract_pdf_text_from_bytes(f.read())
        except Exception:
            manual_texts[f.name] = ""

# Basic sanity check
total_chars = sum(len(t or "") for t in manual_texts.values())
if total_chars < 1500:
    st.warning(
        "I don't have much readable manual text yet. "
        "If your PDFs are scanned images, run OCR and re-upload, or upload text-based PDFs."
    )

# Show loaded manuals
with st.expander("Loaded manuals"):
    if not manual_texts:
        st.write("None loaded yet.")
    else:
        for k, v in manual_texts.items():
            st.write(f"- {k} ({len(v)} chars extracted)")


# ----------------------------
# CHUNKING + EMBEDDINGS INDEX
# ----------------------------
def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 150):
    chunks = []
    i = 0
    text = text or ""
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks


def _vec_from_embedding(e) -> np.ndarray:
    # Handles common shapes in google-genai responses
    if hasattr(e, "values"):
        return np.array(e.values, dtype=np.float32)
    if isinstance(e, list):
        return np.array(e, dtype=np.float32)
    raise TypeError(f"Unexpected embedding type: {type(e)}")


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def detect_device_scope(query: str) -> str | None:
    q = query.lower()
    # crude but effective router
    if "piccolo" in q:
        return "piccolo"
    if "hemo" in q or "hemocue" in q:
        return "hemocue"
    if "i-stat" in q or "istat" in q:
        return "istat"
    if "gluco" in q or "glucose" in q:
        return "gluco"
    return None


def filter_manuals(manuals: dict, scope: str | None) -> dict:
    if not scope:
        return manuals
    s = scope.lower()
    return {k: v for k, v in manuals.items() if s in k.lower()}


@st.cache_data(show_spinner=True)
def build_rag_index(manual_texts_in: dict):
    """
    Build embedding index:
    returns list of rows: [{"fname":..., "chunk":..., "vec": np.array}, ...]
    Cached by Streamlit (depends on manual_texts content).
    """
    rows = []
    for fname, text in manual_texts_in.items():
        if not text or len(text.strip()) < 200:
            continue
        for ch in chunk_text(text):
            if len(ch.strip()) >= 150:
                rows.append({"fname": fname, "chunk": ch})

    if not rows:
        return []

    contents = [r["chunk"] for r in rows]

    # Try batching (fast). If SDK/server rejects large batches, fallback to per-item.
    try:
        emb_res = client.models.embed_content(model=EMBED_MODEL, contents=contents)
        embs = emb_res.embeddings
    except Exception:
        embs = []
        for c in contents:
            one = client.models.embed_content(model=EMBED_MODEL, contents=[c])
            embs.append(one.embeddings[0])

    vecs = [_vec_from_embedding(e) for e in embs]
    mat = _normalize(np.vstack(vecs))

    for i, r in enumerate(rows):
        r["vec"] = mat[i]

    return rows


def retrieve_top_k(index_rows, query: str, k: int = 5):
    if not index_rows:
        return []

    qres = client.models.embed_content(model=EMBED_MODEL, contents=[query])
    qvec = _vec_from_embedding(qres.embeddings[0])
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

    mat = np.vstack([r["vec"] for r in index_rows])
    sims = mat @ qvec
    top_idx = np.argsort(-sims)[:k]
    hits = [(index_rows[i]["fname"], index_rows[i]["chunk"], float(sims[i])) for i in top_idx]
    return hits


# ----------------------------
# CHAT STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


# ----------------------------
# GENERATION (GROUNDED)
# ----------------------------
def grounded_step(query: str) -> str:
    scope = detect_device_scope(query)
    scoped = filter_manuals(manual_texts, scope)

    # If we still have multiple manuals and no scope, ask a device question first
    if not scope and len(scoped) > 1:
        return (
            "Step 1:\n"
            "Need more info.\n"
            "Which device is this for (Piccolo, i-STAT, HemoCue, Gluco)?"
        )

    index_rows = build_rag_index(scoped)
    hits = retrieve_top_k(index_rows, query, k=5)

    # Similarity threshold: tune as you like
    top_score = hits[0][2] if hits else 0.0
    if not hits or top_score < 0.22:
        return (
            "Step 1:\n"
            "Need more info.\n"
            "What exact error message/code is shown (or what happens right before it fails)?"
        )

    context = "\n\n".join(
        [f"[SOURCE: {fname} | score={score:.3f}]\n{chunk}" for fname, chunk, score in hits]
    )

    prompt = f"""
You are a troubleshooting assistant grounded in the provided device manuals.

Rules:
- Use ONLY the Manual Excerpts below as your factual source.
- You may reason, but every instruction MUST be directly supported by a quoted excerpt.
- If excerpts do not support a safe instruction, output "Need more info" and ask ONE targeted question.

Output format (exactly):
Step 1:
<one instruction OR "Need more info">
<Question: one short question>
Evidence:
"<exact quote from the manual excerpts that supports the instruction>"

Manual Excerpts:
{context}
""".strip()

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = (resp.text or "").strip()

    # Guard: if model violates format, fall back safely
    if "Evidence:" not in text or "Step 1:" not in text:
        return (
            "Step 1:\n"
            "Need more info.\n"
            "What exact error message/code is shown (or what happens right before it fails)?"
        )

    return text


# ----------------------------
# MAIN LOOP
# ----------------------------
user_prompt = st.chat_input("Describe the issue (include device + any error code if possible)")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        answer = grounded_step(user_prompt)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
