import pandas as pd
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ LOAD JOB DATA ------------------
jobs = pd.read_csv("jobs.csv")

# ------------------ RESUME PARSER ------------------
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ------------------ SKILL EXTRACTION ------------------
skills_list = [
    "python","java","sql","machine learning","deep learning",
    "html","css","javascript","react","nodejs",
    "pandas","numpy","excel","powerbi","tensorflow","pytorch"
]

def extract_skills(text):
    text = text.lower()
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return found

# ------------------ JOB MATCHING ------------------
def match_jobs(user_skills):
    documents = jobs["skills"].tolist()
    documents.append(" ".join(user_skills))

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])

    scores = similarity.flatten()
    jobs["score"] = scores

    return jobs.sort_values(by="score", ascending=False)

# ------------------ LLM STYLE ADVICE ------------------
def generate_advice(skills, role):
    return f"""
    🔹 Based on your skills: {', '.join(skills)}

    🎯 Recommended Role: {role}

    📚 To improve:
    - Learn advanced concepts in this domain
    - Build 2-3 real projects
    - Practice DSA + system design

    🚀 Career Path:
    Beginner → Intermediate → Advanced → Specialized Engineer
    """

# ------------------ STREAMLIT UI ------------------
st.title("🚀 AI Career Advisor")

option = st.radio("Choose input method:", ["Upload Resume", "Enter Skills"])

user_skills = []

# ---- OPTION 1: Resume Upload ----
if option == "Upload Resume":
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if uploaded_file:
        text = extract_text(uploaded_file)
        user_skills = extract_skills(text)

# ---- OPTION 2: Manual Input ----
else:
    skills_input = st.text_input("Enter skills (comma separated)")
    if skills_input:
        user_skills = [s.strip().lower() for s in skills_input.split(",")]

# ------------------ OUTPUT ------------------
if user_skills:
    st.subheader("🧠 Detected Skills:")
    st.write(user_skills)

    matched = match_jobs(user_skills)

    st.subheader("🏆 Top Job Matches:")
    st.write(matched[["role", "score"]].head(3))

    best_role = matched.iloc[0]["role"]

    st.subheader("📈 Career Advice:")
    st.write(generate_advice(user_skills, best_role))