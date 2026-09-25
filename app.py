import streamlit as st
from complaint_classifier import classify_complaint

st.set_page_config(
    page_title="Complaint Classification System",
    page_icon="📢",
    layout="centered"
)

st.title("📢 Complaint Classification System")
st.write(
    "Enter a customer complaint and the NLP machine learning model "
    "will predict its category."
)

complaint = st.text_area(
    "Enter your complaint:",
    placeholder="Example: My internet connection is not working..."
)

if st.button("🔍 Classify Complaint"):
    if complaint.strip():
        category, confidence = classify_complaint(complaint)

        st.success(f"Predicted Category: **{category}**")
        st.metric("Model Confidence", f"{confidence:.2f}%")
    else:
        st.warning("Please enter a complaint first.")

st.divider()

st.caption(
    "Built with Python, Scikit-learn, NLP and Streamlit."
)
