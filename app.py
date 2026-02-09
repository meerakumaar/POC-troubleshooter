import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

st.set_page_config(page_title="POC Assistant", layout="centered")
st.markdown("### Prototype: Manual-based POC Assistant. Not a clinical tool.")

# Connect to Gemini
genai.configure(api_key=st.secrets[" AIzaSyAP4wxWaItNDOKWbiRqK2QAy-Rr2LCUwt8 "])
model = genai.GenerativeModel('gemini-1.5-flash')

# 7. Manual Grounding (Extract Text)
@st.cache_data
def load_manuals():
    text = ""
    files = ["gluco.pdf", "hemocue correct.pdf", "hepatic piccolo.pdf", "istat.pdf", "piccolo op manual .pdf", "renal piccolo.pdf"]
    for f in files:
        if os.path.exists(f):
            reader = PdfReader(f)
            text += f"\n[SOURCE: {f}]\n" + "".join([p.extract_text() for p in reader.pages])
    return text

context = load_manuals()

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Enter issue"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    # 6. Strict Style Requirements
    instr = f"Use ONLY this text: {context[:30000]}. Rule: Format as 'Step X: [Instruction] [Yes/No Question]'. If Piccolo is mentioned, ask 'Renal or Hepatic?' first. No friendly filler. If not found, say 'Not in manual'."
    
    with st.chat_message("assistant"):
        response = model.generate_content([instr] + [m["content"] for m in st.session_state.messages])
        st.write(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
