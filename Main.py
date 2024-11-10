import streamlit as st
import pdfplumber
import spacy
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize spaCy NER and BERT model
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

st.title("Advanced Candidate Selection Tool")
st.subheader("NLP-Based Resume Screening with Bias Mitigation")
st.caption("This tool matches resumes to job descriptions using advanced NLP models (BERT), extracts structured information, and applies bias mitigation techniques to ensure fair hiring.")

# Step 1: Allow the user to upload multiple resumes (CSV)
uploaded_resumes = st.file_uploader("Upload a CSV file with multiple resumes (Resume column)", type="csv", accept_multiple_files=False)
uploadedJD = st.file_uploader("Upload Job Description", type="pdf")
uploadedResume = st.file_uploader("Upload Resume (single file)", type="pdf")

# Initialize variables
job_description = ""
resume = ""

# Step 2: Extract text from the PDFs (job description and resume)
def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            pages = pdf.pages[0]
            text = pages.extract_text()
        return text
    except Exception as e:
        st.write(f"Error extracting text: {e}")
        return ""

if uploadedJD:
    job_description = extract_text_from_pdf(uploadedJD)

if uploadedResume:
    resume = extract_text_from_pdf(uploadedResume)

# Step 3: Extract Named Entities (skills, experience, etc.)
def extract_skills_and_experience(text):
    doc = nlp(text)
    skills = []
    experience = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            skills.append(ent.text)
        elif ent.label_ == "DATE":
            experience.append(ent.text)
    return skills, experience

# Step 4: Get BERT embedding
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Mean pooling to get the sentence embedding

# Step 5: Cosine Similarity for Job Description and Resume
def get_similarity(job_description, resume):
    job_embedding = get_bert_embedding(job_description)
    resume_embedding = get_bert_embedding(resume)
    similarity = cosine_similarity(job_embedding, resume_embedding)
    return similarity[0][0] * 100  # Return similarity percentage

# Step 6: Categorize match result based on Overall Weighted Match Score
def categorize_match(weighted_score):
    if weighted_score >= 90:
        return "Highly Recommended", "green"
    elif 75 <= weighted_score < 90:
        return "Recommended", "blue"
    elif 60 <= weighted_score < 75:
        return "Moderately Recommended", "yellow"
    elif 40 <= weighted_score < 60:
        return "Not Recommended", "orange"
    elif 20 <= weighted_score < 40:
        return "Weak Fit", "red"
    else:
        return "No Match", "gray"

# Step 7: Custom scoring system based on section weights
def get_weighted_score(job_description, resume, skills_weight=0.5, experience_weight=0.3, education_weight=0.2):
    skills_job, experience_job = extract_skills_and_experience(job_description)
    skills_resume, experience_resume = extract_skills_and_experience(resume)

    skill_similarity = cosine_similarity(get_bert_embedding(' '.join(skills_job)), get_bert_embedding(' '.join(skills_resume)))
    experience_similarity = cosine_similarity(get_bert_embedding(' '.join(experience_job)), get_bert_embedding(' '.join(experience_resume)))
    
    weighted_score = (skills_weight * skill_similarity[0][0]) + (experience_weight * experience_similarity[0][0])
    return weighted_score * 100

# Step 8: Handle batch processing for multiple resumes (CSV file upload)
def batch_process_resumes(job_description, resumes_df):
    results = []
    for index, row in resumes_df.iterrows():
        resume = row['Resume']
        match_score = get_similarity(job_description, resume)
        
        # Get weighted score and decide category based on it
        weighted_score = get_weighted_score(job_description, resume)
        category, category_color = categorize_match(weighted_score)

        # Add details to results
        results.append({
            'Resume': resume,
            'Match Score': f"{match_score:.2f}%",  # Display match percentage with 2 decimals
            'Category': category,
            'Weighted Score': f"{weighted_score:.2f}%",  # Display weighted score with 2 decimals
        })
    
    # Convert results into a DataFrame for better display
    return pd.DataFrame(results)

# Step 9: Process button logic
if uploaded_resumes:
    if uploadedJD:
        resumes_df = pd.read_csv(uploaded_resumes)
        batch_results = batch_process_resumes(job_description, resumes_df)
        st.write("### Batch Results")
        st.dataframe(batch_results, width=1000, height=500)  # Increase size of the table
    else:
        st.write("Please upload a job description to process the resumes.")

elif uploadedJD and uploadedResume:
    # Single resume processing
    match = get_similarity(job_description, resume)
    match = round(match, 2)  # Round only the match percentage

    # Calculate weighted score and determine category based on it
    weighted_score = get_weighted_score(job_description, resume)
    category, category_color = categorize_match(weighted_score)

    # Prepare results to display in a table
    skill_match = weighted_score * 0.5
    experience_match = weighted_score * 0.3

    results = {
        "Field": ["Description Match Percentage", "Skills Match", "Experience Match", "Overall Weighted Match Score","Category"],
        "Value": [
            f"{match:.2f}%",  
            f"{skill_match:.2f}%",  # Display skill match with 2 decimals
            f"{experience_match:.2f}%",  # Display experience match with 2 decimals
            f"{weighted_score:.2f}%",
            category  # Display overall weighted match score with 2 decimals
        ]
    }
    
    # Display results in a table
    results_df = pd.DataFrame(results)
    st.write("### Match Result")
    st.dataframe(results_df, width=1000, height=300)  # Increase size of the table
    
else:
    st.write("Please upload a job description and resume to process.")
