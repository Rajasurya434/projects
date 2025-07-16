from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def match_resume_with_jobs(resume_path, job_descriptions):
    
    resume_text = extract_text_from_pdf(resume_path)

    resume_emb = model.encode([resume_text])[0]
    job_embs = [model.encode([desc])[0] for _, desc in job_descriptions]

    scores = cosine_similarity([resume_emb], job_embs)[0]
    results = sorted(zip(job_descriptions, scores), key=lambda x: x[1], reverse=True)

  
    print(" Top Matches:")
    for (title, _), score in results:
        print(f"**{title}** → Match Score: `{score:.2f}`")

if __name__ == "__main__":
    
    resume_path = "/content/Ai_Updated_Resume.pdf"
    
    
    job_descs = [
        ("Job 1", "Data Scientist with experience in Python and machine learning."),
        ("Job 2", "Software Engineer with expertise in Java and cloud technologies."),
        ("Job 3", "Data Analyst with strong SQL and Excel skills.")
    ]
    
    
    match_resume_with_jobs(resume_path, job_descs)
