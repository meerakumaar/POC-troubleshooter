import streamlit as st
import requests
import json
from pypdf import PdfReader
import os

# 10. Required Framing Text
st.set_page_config(page_title="POC Troubleshooting Assistant", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")

# Safety check for the API Key
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

API_KEY = st.secrets["GEMINI_API_KEY"]

# 7. Grounding: Extracting text from the 6 uploaded manuals
@st.cache_data
def load_manual_context():
    manual_text = ""
    files = ["gluco.pdf", "hemocue correct.pdf", "hepatic piccolo.pdf", "istat.pdf", "piccolo op manual .pdf", "renal piccolo.pdf"]
    for f in files:
        if os.path.exists(f):
            try:
                reader = PdfReader(f)
                manual_text += f"\n[START: {f}]\n"
                for page in reader.pages:
                    text = page.extract_text()
                    if text: manual_text += text
            except: continue
    return manual_text

context_data = load_manual_context()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f"**{m['content']}**")

if prompt := st.chat_input("Enter device issue or error code..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**{prompt}**")

    # 6. Response Style & 7. Zero-Hallucination Rules
    system_instruction = f"""
    You are a professional POC troubleshooting assistant. Use ONLY this manual data: {context_data[:30000]}
    
    RULES:
    1. FORMAT: 'Step X: [Instruction] [One Yes/No or concrete question]'.
    2. Respond with ONLY one step at a time.
    3. PICCOLO RULE: If 'Piccolo' is mentioned, ask 'Is this for a Renal or Hepatic perfusion?' first.
    4. TONE: Neutral and concise. No conversational filler.
    5. GROUNDING: If info is missing, say 'I couldn't find a specific instruction for this in the manual.'
    """

    with st.chat_message("assistant"):
        # Bypassing the Google library version issues with a direct call
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"
        headers = {'Content-Type': 'application/json'}
        
        # Build the conversation history
        contents = [{"role": "user", "parts": [{"text": system_instruction}]}]
        contents.append({"role": "model", "parts": [{"text": "Understood. I will follow those rules strictly."}]})
        
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        try:
            response = requests.post(url, headers=headers, data=json.dumps({"contents": contents}))
            result = response.json()
            
            if 'candidates' in result:
                bot_text = result['candidates'][0]['content']['parts'][0]['text']
                st.markdown(f"**{bot_text}**")
                st.session_state.messages.append({"role": "assistant", "content": bot_text})
            else:
                st.error(f"API Error: {result.get('error', {}).get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Connection Error: {e}")
