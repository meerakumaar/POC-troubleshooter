import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# 10. Required Framing Text
st.set_page_config(page_title="POC Troubleshooting Assistant", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")

# Initialize Model with Fallback Logic to prevent 404
if "model_name" not in st.session_state:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # Try Flash first, then Pro as a fallback
        for m in ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro']:
            try:
                test_model = genai.GenerativeModel(m)
                test_model.generate_content("test")
                st.session_state.model_name = m
                break
            except:
                continue
    except Exception as e:
        st.error(f"Configuration Error: Check your API Key in Secrets.")
        st.stop()

model = genai.GenerativeModel(st.session_state.get("model_name", "gemini-1.5-flash"))

# 7. Grounding - Extract text from the manuals
@st.cache_data
def load_manuals():
    text_content = ""
    files = ["gluco.pdf", "hemocue correct.pdf", "hepatic piccolo.pdf", "istat.pdf", "piccolo op manual .pdf", "renal piccolo.pdf"]
    for f in files:
        if os.path.exists(f):
            try:
                reader = PdfReader(f)
                text_content += f"\n[SOURCE: {f}]\n"
                for page in reader.pages:
                    text_content += page.extract_text() or ""
            except:
                continue
    return text_content

manual_data = load_manuals()

if "messages" not in st.session_state:
    st.session_state.messages = []

# High-contrast display for tired users
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
    - Format: 'Step X: [Instruction] [One Yes/No or concrete question]'.
    - Provide only ONE step per turn.
    - If Piccolo or chemistry disc is mentioned, ask: 'Is this for a Renal or Hepatic perfusion?'
    - Tone: Neutral and concise. No friendly filler.
    - If not in manuals, say: 'I couldn't find a specific instruction for this in the manual.'
    
    MANUAL DATA:
    {manual_data[:30000]}
    """

    with st.chat_message("assistant"):
        try:
            response = model.generate_content([instruction] + [m["content"] for m in st.session_state.messages])
            st.markdown(f"**{response.text}**")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"System Error: Please ensure your API key is active and the PDFs are uploaded to GitHub.")
