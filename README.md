# Candidate-Fit-Prediction
# Advanced Candidate Selection Tool

This repository contains an NLP-based candidate selection tool built with Streamlit, BERT, and spaCy. Designed to assist HR professionals in resume screening, the tool matches resumes to job descriptions using BERT embeddings, extracts key skills and experience, and applies bias mitigation techniques. It provides detailed match scores and supports both single and batch resume processing.

## Features

- **NLP-Powered Matching**: Uses BERT embeddings and cosine similarity to match resumes against job descriptions.
- **Named Entity Recognition (NER)**: Extracts key skills and experience with spaCy NER.
- **Bias Mitigation**: Custom scoring system to ensure fair candidate evaluation.
- **Batch Processing**: Upload a CSV file with multiple resumes for bulk processing.
- **Weighted Scoring**: Apply weights to different resume sections to get an overall match score.
- **Category Recommendation**: Categorizes candidates into levels such as "Highly Recommended" or "Weak Fit."

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/advanced-candidate-selection-tool.git
    cd advanced-candidate-selection-tool
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download spaCy's English language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1. **Start the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Upload files**:
   - **Single Resume Mode**: Upload a job description (PDF) and a single resume (PDF).
   - **Batch Mode**: Upload a job description (PDF) and a CSV file with multiple resumes (each in a column named "Resume").

3. **View Results**:
   - For single resume mode, view detailed match results, including skills and experience matches.
   - In batch mode, a table displays match scores and categories for each resume.

## File Structure

- `app.py`: Main Streamlit application.
- `requirements.txt`: Required Python packages.

## Key Functions

- **`extract_text_from_pdf`**: Extracts text from uploaded PDFs.
- **`extract_skills_and_experience`**: Identifies key skills and experience using spaCy NER.
- **`get_bert_embedding`**: Generates BERT embeddings for given text.
- **`get_similarity`**: Computes cosine similarity between job description and resume.
- **`get_weighted_score`**: Calculates a custom weighted score for the resume match.
- **`batch_process_resumes`**: Processes multiple resumes in batch mode and outputs match results.

## Dependencies

- `streamlit`: Web app framework
- `pdfplumber`: PDF text extraction
- `spacy`: NLP library for Named Entity Recognition
- `transformers`: For BERT model and tokenizer
- `torch`: Backend for BERT embeddings
- `sklearn`: For cosine similarity calculation
- `pandas`: Data manipulation for batch processing
- `numpy`: Array handling


## License

This project is licensed under the MIT License.
