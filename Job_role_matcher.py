import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Job dataset
jobs = {
    "role": [
        "Data Scientist",
        "Machine Learning Engineer",
        "Web Developer",
        "Android Developer",
        "Data Analyst"
    ],
    "skills": [
        "python machine learning statistics pandas numpy",
        "python deep learning tensorflow pytorch",
        "html css javascript react node",
        "java kotlin android studio",
        "sql excel data visualization python"
    ]
}

df = pd.DataFrame(jobs)

# Convert text to vectors
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(df["skills"])

def recommend_jobs(resume_skills):
    
    resume_vector = vectorizer.transform([resume_skills])
    
    similarity = cosine_similarity(resume_vector, job_vectors)
    
    scores = list(enumerate(similarity[0]))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    print("Recommended Roles:\n")
    
    for i in scores[:3]:
        print(df.iloc[i[0]]["role"])

# Example input
resume = " web developer kotlin"

recommend_jobs(resume)