from pathlib import Path
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai

# ============================
# CONFIG
# ============================
MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# Bigger chunks => fewer chunks => fewer embeddings => faster/cheaper
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 200

# Embed API batch limit (from your error)
EMBED_BATCH = 100

st.set_page_config(page_title="POC Troubleshooter (RAG)", layout="centered")
st.title("POC Troubleshooter (RAG Prototype)")
st.caption("Prototype: manual-grounded troubleshooting assistant. Not a clinical tool.")

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
    st.warning(
        "Very little readable manual text found. If this persists, reboot Streamlit to ensure latest commit is deployed."
    )

# ============================
# RAG: chunking + embeddings
# ============================
def chunk_text(text: str, size: int, overlap: int):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
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
    if "gluco" in q or "glucose" in q:
        return "gluco"
    return None

def filter_manuals(manuals: dict, device: str | None) -> dict:
    if not device:
        return manuals
    return {k: v for k, v in manuals.items() if device in k.lower()}

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

@st.cache_data(show_spinner=True)
def build_index(manuals: dict):
    """
    Build embeddings index for the given manuals.
    Uses batching to respect the 100-requests-per-batch limit.
    """
    rows = []
    for fname, text in manuals.items():
        if not text or len(text) < 200:
            continue
        for ch in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            if len(ch.strip()) >= 200:
                rows.append({"fname": fname, "chunk": ch})

    if not rows:
        return []

    contents = [r["chunk"] for r in rows]
    all_vecs = []

    # ---- FIX: batch embedding calls to <=100 chunks each ----
    for batch in batched(contents, EMBED_BATCH):
        res = client.models.embed_content(model=EMBED_MODEL, contents=batch)
        batch_vecs = [vec_from_embedding(e) for e in res.embeddings]
        all_vecs.extend(batch_vecs)

    mat = normalize(np.vstack(all_vecs))

    for i, r in enumerate(rows):
        r["vec"] = mat[i]

    return rows

def retrieve(index, query: str, k: int = 5):
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
# GROUNDED GENERATION
# ============================
def grounded_response(query: str) -> str:
    device = detect_device(query)
    if not device:
        return (
            "Step 1:\n"
            "Need more info.\n"
            "Which device is this for (Piccolo, i-STAT, HemoCue, Gluco)?"
        )

    scoped = filter_manuals(manual_texts, device)

    # If device filter resulted in nothing, fall back to all manuals
    if not scoped:
        scoped = manual_texts

    index = build_index(scoped)
    hits = retrieve(index, query, k=5)

    # Threshold can be tuned; higher => more conservative
    if not hits or hits[0][2] < 0.20:
        return (
            "Step 1:\n"
            "Need more info.\n"
            "What exact error message/code is displayed (or what happens right before it fails)?"
        )

    context = "\n\n".join([f"[SOURCE: {f}]\n{c}" for f, c, _ in hits])

    prompt = f"""
You are a troubleshooting assistant grounded ONLY in the provided manuals.

Rules:
- Use only the Manual Excerpts.
- Do NOT guess or add outside knowledge.
- Provide exactly ONE step and ONE question.
- Every instruction must be supported by a quoted excerpt.

Output format:
Step 1:
<one instruction>
<one question>
Evidence:
"<exact quote from the manual>"

Manual Excerpts:
{context}
""".strip()

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = (resp.text or "").strip()

    # Hard guard against format drift
    if "Evidence:" not in text or "Step 1:" not in text:
        return (
            "Step 1:\n"
            "Need more info.\n"
            "What exact error message/code is displayed (or what happens right before it fails)?"
        )

    return text

# ============================
# CHAT UI
# ============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("Describe the issue (include device + any error code if possible)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        answer = grounded_response(user_input)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
