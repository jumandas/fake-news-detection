# app/app_streamlit.py
import streamlit as st
from predict import predict_news

st.title("📰 Fake News Detector")
st.write("Paste any news article below and check if it's FAKE or REAL.")

user_input = st.text_area("Enter news text here:")

if st.button("Check"):
    if user_input.strip():
        result = predict_news(user_input)
        if result == "REAL":
            st.success("✅ This looks like REAL News")
        else:
            st.error("❌ This looks like FAKE News")
    else:
        st.warning("⚠️ Please enter some text.")
