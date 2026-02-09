import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# Required Prototype Framing
st.set_page_config(page_title="POC Assistant", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")

# Connection and Diagnostic Logic
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY in Secrets dashboard.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Diagnostic: Identify what models this key can actually see
@st.cache_resource
def get_model():
    try:
        # We try to initialize the model directly
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        # If it fails, list available models to the log
        models = [m.name for m in genai.list_models()]
        st.error(f"Model Load Failure. Available models for this key: {models}")
        return None

model = get_model()

# Manual Extraction
@st.cache_data
def load_manuals():
    text = ""
    files = ["gluco.pdf", "hemocue correct.pdf", "hepatic piccolo.pdf", "istat.pdf", "piccolo op manual .pdf", "renal piccolo.pdf"]
    for f in files:
        if os.path.exists(f):
            try:
                reader = PdfReader(f)
                text += f"\n[SOURCE: {f}]\n" + "".join([p.extract_text() or "" for p in reader.pages])
            except: continue
    return text

manual_data = load_manuals()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(f"**{m['content']}**")

if prompt := st.chat_input("Enter device issue"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(f"**{prompt}**")

    instr = f"""
    Use ONLY: {manual_data[:30000]}
    Format: 'Step X: [Instruction] [One Question]'.
    If Piccolo mentioned, ask 'Renal or Hepatic?' first.
    If not in manual, say 'I couldn't find a specific instruction for this in the manual.'
    """

    with st.chat_message("assistant"):
        try:
            response = model.generate_content([instr] + [m["content"] for m in st.session_state.messages])
            st.markdown(f"**{response.text}**")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Generation Error: {str(e)}")
