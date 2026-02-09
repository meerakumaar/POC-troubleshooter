import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# 10. Required Framing Text (PRD Compliance)
st.set_page_config(page_title="POC Troubleshooting Assistant", layout="centered")
st.markdown("### Prototype: Manual-based troubleshooting assistant for point-of-care measurement devices. Not a clinical tool.")

# Safety check for the API Key in Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets. Please add it to 'Advanced Settings' in your dashboard.")
    st.stop()

# Configure Gemini with the secret key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# 7. Grounding: Extracting text from the 6 uploaded manuals
@st.cache_data
def load_manual_context():
    manual_text = ""
    # These must match your GitHub filenames exactly
    files = [
        "gluco.pdf", 
        "hemocue correct.pdf", 
        "hepatic piccolo.pdf", 
        "istat.pdf", 
        "piccolo op manual .pdf", 
        "renal piccolo.pdf"
    ]
    for f in files:
        if os.path.exists(f):
            try:
                reader = PdfReader(f)
                manual_text += f"\n[START OF MANUAL: {f}]\n"
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        manual_text += text
                manual_text += f"\n[END OF MANUAL: {f}]\n"
            except Exception as e:
                manual_text += f"\n[Error reading {f}: {str(e)}]\n"
    return manual_text

# Load data once and cache it for speed
context_data = load_manual_context()

# Initialize Chat State (PRD: No data persistence across sessions)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History (Neutral, high-contrast formatting for tired users)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"**{message['content']}**")

# 5. Core Interaction Model
if prompt := st.chat_input("Enter device issue or error code..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"**{prompt}**")

    # 6. Response Style & 7. Zero-Hallucination Prompt
    system_instruction = f"""
    You are a professional POC troubleshooting assistant. Use ONLY the manual data provided below.
    
    STRICT RULES:
    1. ZERO HALLUCINATION: If the answer is not in the manuals, say: 'I couldn't find a specific instruction for this in the manual.'
    2. FORMAT: Respond ONLY as 'Step X: [Manual Instruction] [One Yes/No or concrete question]'.
    3. SINGLE STEP: Provide only one step at a time.
    4. PICCOLO BRANCHING: If the user mentions 'Piccolo', 'disc', or chemistry errors, your FIRST step must ask: 'Is this for a Renal (Kidney) or Hepatic (Liver) perfusion?'
    5. TONE: Neutral, professional, and concise. No conversational filler or 'friendly' language.
    6. SCOPE: Do not troubleshoot the perfusion machine or patient status.

    MANUAL DATA:
    {context_data[:30000]} 
    """

    with st.chat_message("assistant"):
        try:
            # Combine instructions with chat history
            full_prompt = [system_instruction] + [m["content"] for m in st.session_state.messages]
            response = model.generate_content(full_prompt)
            bot_response = response.text
            
            st.markdown(f"**{bot_response}**")
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
        except Exception as e:
            st.error("The system encountered an error. Please check your API key.")
