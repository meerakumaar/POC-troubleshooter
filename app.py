import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# 10. Required Framing Text
st.set_page_config(page_title="POC Assistant", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant. Not a clinical tool.")

# Connection Setup - Forcing the STABLE API version
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Add GEMINI_API_KEY to Streamlit Secrets.")
    st.stop()

# This is the secret sauce: forcing the library to use the stable v1 path
genai.configure(api_key=st.secrets["GEMINI_API_KEY"], transport='rest')

# Using the full model path to avoid 404s
model = genai.GenerativeModel('models/gemini-1.5-flash')

# 7. Grounding - Extract Manual Text
@st.cache_data
def load_manuals():
    text = ""
    files = ["gluco.pdf", "hemocue correct.pdf", "hepatic piccolo.pdf", "istat.pdf", "piccolo op manual .pdf", "renal piccolo.pdf"]
    for f in files:
        if os.path.exists(f):
            try:
                reader = PdfReader(f)
                text += f"\n[DOC: {f}]\n" + "".join([p.extract_text() or "" for p in reader.pages])
            except: continue
    return text

manual_context = load_manuals()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f"**{m['content']}**")

if prompt := st.chat_input("Enter device issue"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**{prompt}**")

    # Instruction for the model
    instr = f"""
    Use ONLY: {manual_context[:30000]}
    Rule 1: Format as 'Step X: [Instruction] [One Question]'.
    Rule 2: One step at a time. Neutral tone.
    Rule 3: If Piccolo mentioned, ask 'Renal or Hepatic?' first.
    Rule 4: If not in manual, say 'I couldn't find a specific instruction for this in the manual.'
    """

    with st.chat_message("assistant"):
        try:
            response = model.generate_content([instr] + [m["content"] for m in st.session_state.messages])
            st.markdown(f"**{response.text}**")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Technical Error: {e}")
