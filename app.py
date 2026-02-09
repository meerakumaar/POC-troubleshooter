import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# 10. Required Prototype Framing
st.set_page_config(page_title="POC Assistant", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")

# Using your provided key directly to bypass Secret issues for now
genai.configure(api_key="AIzaSyAP4wxWaItNDOKWbiRqK2QAy-Rr2LCUwt8")
model = genai.GenerativeModel('gemini-1.5-flash')

# 7. Grounding - Extract text
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

manual_context = load_manuals()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(f"**{m['content']}**")

if prompt := st.chat_input("Enter issue"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(f"**{prompt}**")

    instr = f"""
    Use ONLY: {manual_context[:30000]}
    Format: 'Step X: [Instruction] [One Question]'.
    One step at a time. Neutral tone. No friendly filler.
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
