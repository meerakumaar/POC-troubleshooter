import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# 10. Required Framing Text
st.set_page_config(page_title="POC Troubleshooting", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")

# Set up the model - Using the most stable name
try:
    # Use the key from secrets for security
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# 7. Grounding - Extract text from the 6 manuals
@st.cache_data
def load_manuals():
    text_content = ""
    # Ensure these names match your files on GitHub exactly
    files = ["gluco.pdf", "hemocue correct.pdf", "hepatic piccolo.pdf", "istat.pdf", "piccolo op manual .pdf", "renal piccolo.pdf"]
    for f in files:
        if os.path.exists(f):
            reader = PdfReader(f)
            text_content += f"\n[SOURCE: {f}]\n"
            for page in reader.pages:
                text_content += page.extract_text() or ""
    return text_content

manual_data = load_manuals()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history with high-contrast bold text
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
    - If Piccolo is mentioned, ask: 'Is this for a Renal or Hepatic perfusion?' before giving steps.
    - Tone: Neutral and concise. No conversational filler.
    - If the answer is not in the manuals, say: 'I couldn't find a specific instruction for this in the manual.'
    
    MANUAL DATA:
    {manual_data[:30000]}
    """

    with st.chat_message("assistant"):
        try:
            # Generate response
            response = model.generate_content([instruction] + [m["content"] for m in st.session_state.messages])
            st.markdown(f"**{response.text}**")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Generation Error: {e}. Please ensure your API key is active and your region is supported.")
