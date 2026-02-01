import streamlit as st
import numpy as np
import re
import os
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------
MODEL_PATH = os.path.join("pickled_data", "model.h5")
SCALER_PATH = os.path.join("pickled_data", "scaler.pkl")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

CURRENT_YEAR = datetime.now().year

# --------------------------------------------------
# Helper functions (RULE-BASED, NO NLP)
# --------------------------------------------------
def split_and_clean(text):
    if not text:
        return []
    return [t.strip().lower() for t in text.split(",") if t.strip()]

def extract_number(text):
    if not text:
        return 0
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0

def degree_to_level(degree):
    degree = degree.lower()
    if "phd" in degree:
        return 3
    if "master" in degree or "m sc" in degree or "mba" in degree:
        return 2
    if "bachelor" in degree or "b tech" in degree:
        return 1
    return 0

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="ANN-Based Resume–Job Compatibility Scoring System",
    layout="wide"
)

st.title("📄 ANN-Based Resume–Job Compatibility Scoring System")
st.write(
    "This application predicts how well a candidate matches a job role using a **pure Artificial Neural Network**. "
    "Text inputs are converted into structured numeric features using rule-based logic."
)

st.divider()

# --------------------------------------------------
# Candidate Inputs
# --------------------------------------------------
st.subheader("👤 Candidate Information")

skills_text = st.text_area("Skills (comma-separated)", placeholder="Python, Java, SQL, Machine Learning")
responsibilities_text = st.text_area("Responsibilities (comma-separated)", placeholder="Data analysis, Model development")
companies_text = st.text_input("Companies Worked (comma-separated)", placeholder="TCS, Infosys")
certifications_text = st.text_input("Certifications (comma-separated)", placeholder="AWS, Azure")
languages_text = st.text_input("Languages Known (comma-separated)", placeholder="English, Hindi")

degree = st.selectbox("Highest Degree", ["Bachelor", "Master", "PhD"])
education_years_text = st.text_input("Education Years (comma-separated)", placeholder="2017, 2019")

experience_start_year = st.number_input(
    "Experience Start Year",
    min_value=2010,
    max_value=CURRENT_YEAR,
    step=1
)

st.divider()

# --------------------------------------------------
# Job Inputs
# --------------------------------------------------
st.subheader("💼 Job Information")

job_skills_text = st.text_area("Job Required Skills (comma-separated)", placeholder="Python, SQL, Machine Learning")
job_responsibilities_text = st.text_area("Job Responsibilities (comma-separated)", placeholder="Model development, Data analysis")

experience_requirement_text = st.text_input(
    "Experience Requirement",
    placeholder="At least 3 years"
)

education_requirement_text = st.text_input(
    "Education Requirement",
    placeholder="Master in Computer Science"
)

age_requirement_flag = st.radio(
    "Is there an age requirement for the job?",
    ["No", "Yes"]
)

st.divider()

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if st.button("🔍 Predict Match Score"):
    # Candidate feature extraction
    candidate_skills = set(split_and_clean(skills_text))
    job_skills = set(split_and_clean(job_skills_text))

    num_skills = len(candidate_skills)
    required_skill_count = len(job_skills)
    skill_overlap_count = len(candidate_skills & job_skills)

    years_of_experience = CURRENT_YEAR - experience_start_year
    min_experience_required = extract_number(experience_requirement_text)
    experience_gap = years_of_experience - min_experience_required

    edu_years = [int(y) for y in split_and_clean(education_years_text) if y.isdigit()]
    education_duration_years = max(edu_years) - min(edu_years) if len(edu_years) >= 2 else 0

    degree_level = degree_to_level(degree)

    num_certifications = len(split_and_clean(certifications_text))
    num_languages = len(split_and_clean(languages_text))
    num_companies = len(split_and_clean(companies_text))

    num_responsibilities = len(split_and_clean(responsibilities_text))
    job_responsibility_count = len(split_and_clean(job_responsibilities_text))

    education_match_flag = (
        1 if degree.lower() in education_requirement_text.lower() else 0
    )
    age_req_flag = 1 if age_requirement_flag == "Yes" else 0

    # Final ANN input vector (ORDER MUST MATCH TRAINING)
    input_vector = np.array([[
        num_skills,
        required_skill_count,
        skill_overlap_count,
        years_of_experience,
        experience_gap,
        education_duration_years,
        degree_level,
        num_certifications,
        num_languages,
        num_companies,
        num_responsibilities,
        job_responsibility_count,
        education_match_flag,
        age_req_flag
    ]])

    # Scaling + prediction
    input_scaled = scaler.transform(input_vector)
    predicted_score = model.predict(input_scaled)[0][0]

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    st.success(f"✅ Predicted Match Score: **{predicted_score:.2f}**")

    if predicted_score >= 0.75:
        st.markdown("🟢 **Excellent Match**")
    elif predicted_score >= 0.50:
        st.markdown("🟡 **Moderate Match**")
    else:
        st.markdown("🔴 **Low Match**")

    st.caption(
        "Note: This prediction is based on structured numeric features derived from user input. "
        "The ANN does not directly process text."
    )
