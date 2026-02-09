import streamlit as st
import google.generativeai as genai

st.title("Connection Diagnostic")

# Replace this with your key one more time for this test
api_key = "AIzaSyAP4wxWaItNDOKWbiRqK2QAy-Rr2LCUwt8"

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Is the connection working? Answer with 'YES' only.")
    st.success(f"Response from Gemini: {response.text}")
except Exception as e:
    st.error(f"Connection Failed: {e}")
