import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# 10. Required Framing Text
st.set_page_config(page_title="POC Troubleshooting Assistant", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")

# Connection Setup
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Use the stable model ID
model = genai.GenerativeModel('gemini-1.5-flash')

# 7. Grounding - Extract text from the 6 manuals
@st.cache_data
def load_manuals():
    text_content = ""
    # List of files in the repository
    files = ["gluco.pdf", "hemocue correct.pdf", "hepatic piccolo.pdf", "istat.pdf", "piccolo op manual .pdf", "renal piccolo.pdf"]
    for f in files:
        if os.path.exists(f):
            try:
                reader = PdfReader(f)
                text_content += f"\n[DOC: {f}]\n"
                for page in reader.pages:
                    text_content += page.extract_text() or ""
            except:
                continue
    return text_content

manual_data = load_manuals()

if "messages" not in st.session_state:
    st.session_state.messages = []

# High-contrast display for readability
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f"**{m['content']}**")

if prompt := st.chat_input("Enter device error or issue"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**{prompt}**")

    # 6. Response Style & 7. Zero-Hallucination Prompt
    instruction = f"""
    You are a professional technical assistant. Use ONLY the manual data provided.
    RULES:
    1. Respond ONLY in format: 'Step X: [Instruction] [One Yes/No or concrete question]'.
    2. Provide only ONE step per turn.
    3. If Piccolo is mentioned, ask: 'Is this for a Renal or Hepatic perfusion?' before giving steps.
    4. Neutral tone. No friendly filler or 'lovable' language.
    5. If not in manuals, say: 'I couldn't find a specific instruction for this in the manual.'
    
    MANUAL DATA:
    {manual_data[:30000]}
    """

    with st.chat_message("assistant"):
        try:
            # Combine history for context
            chat_history = [m["content"] for m in st.session_state.messages]
            response = model.generate_content([instruction] + chat_history)
            st.markdown(f"**{response.text}**")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"System error. Verify the API key is valid and active in Google AI Studio.")
