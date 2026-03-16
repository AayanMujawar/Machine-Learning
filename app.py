import pandas as pd
import streamlit as st
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# dataset
data = {
    "career":[
        "Data Scientist",
        "Machine Learning Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Android Developer",
        "DevOps Engineer"
    ],

    "skills":[
        "python statistics pandas numpy machine learning data visualization",
        "python tensorflow pytorch deep learning ai ml",
        "java spring database api system design",
        "html css javascript react ui ux",
        "java kotlin android mobile development",
        "docker kubernetes aws linux ci cd cloud"
    ],

    "roadmap":[
        "statistics -> machine learning -> deep learning -> data engineering",
        "linear algebra -> neural networks -> distributed training",
        "databases -> system design -> microservices",
        "javascript frameworks -> advanced css -> web performance",
        "android ui -> networking -> play store deployment",
        "cloud computing -> kubernetes -> infrastructure automation"
    ]
}

df = pd.DataFrame(data)

# clean text
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]','',text)
    return text

df["skills"] = df["skills"].apply(clean)

# vectorization
vectorizer = TfidfVectorizer()
skill_vectors = vectorizer.fit_transform(df["skills"])


# skill graph (next skill recommendation)
skill_graph = nx.DiGraph()

edges = [
("python","numpy"),
("numpy","pandas"),
("pandas","machine learning"),
("machine learning","deep learning"),
("html","css"),
("css","javascript"),
("javascript","react"),
("docker","kubernetes"),
("kubernetes","cloud")
]

skill_graph.add_edges_from(edges)


def recommend(user_skills):

    user_skills = clean(user_skills)
    user_vector = vectorizer.transform([user_skills])

    similarity = cosine_similarity(user_vector,skill_vectors)[0]
    best = similarity.argmax()

    career = df.iloc[best]["career"]
    roadmap = df.iloc[best]["roadmap"]
    score = round(similarity[best]*100,2)

    return career, roadmap, score


def next_skills(user_skills):

    skills = user_skills.lower().split()
    suggestions = []

    for skill in skills:
        if skill in skill_graph:
            suggestions.extend(list(skill_graph.successors(skill)))

    return list(set(suggestions))


# Streamlit UI
st.title("AI Career Roadmap Recommender")

skills = st.text_input("Enter your current skills")

if st.button("Recommend Career"):

    career, roadmap, score = recommend(skills)

    st.subheader("Best Career Match")
    st.write(career)
    st.write("Match Score:",score,"%")

    st.subheader("Learning Roadmap")
    st.write(roadmap)

    st.subheader("Next Skills to Learn")

    ns = next_skills(skills)

    if ns:
        for s in ns:
            st.write("•",s)
    else:
        st.write("No suggestions found")