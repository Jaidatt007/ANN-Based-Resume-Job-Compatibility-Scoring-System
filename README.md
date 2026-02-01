# ANN-Based Resume–Job Compatibility Scoring System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package_manager-purple.svg)](https://github.com/astral-sh/uv)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)

> An explainable deep learning system that predicts resume-job compatibility using structured feature engineering and Artificial Neural Networks.

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/Jaidatt007/ANN-Based-Resume-Job-Compatibility-Scoring-System.git
cd ANN

# Install dependencies with uv
uv sync

# Run the application
uv run streamlit run main.py
```

Access the app at `http://localhost:8501`

---

## 🚀 Project Overview

An **Artificial Neural Network (ANN)** system that predicts resume-job compatibility by converting text inputs into structured numerical features and processing them through a deep learning model.

**Key Features:**
- ✨ Rule-based feature extraction for transparency
- 🧠 Pure ANN architecture on numerical features
- 📊 Explainable compatibility scores (0.0 - 1.0)
- 🎨 Interactive Streamlit web interface

> This project demonstrates end-to-end ML pipeline development, from data preprocessing to deployment.

---

## 🧠 Core Concept

**Text Inputs → Numerical Features → ANN Model → Compatibility Score**

This project uses structured feature engineering instead of black-box NLP:

- Resume and job descriptions converted to numerical features
- Pure ANN processes only structured data
- Output: Explainable match score (0.0 - 1.0)

**Benefits:**
- Full transparency in predictions
- Feature-level interpretability
- Easy to extend and modify
- No dependency on pre-trained language models

---

## 🏗️ System Architecture

```
Text Inputs (Resume + Job Description)
           ↓
Rule-Based Feature Extraction
           ↓
Numerical Feature Vector (14+ features)
           ↓
StandardScaler (Normalization)
           ↓
ANN Model (Deep Learning)
           ↓
Match Score (0.0 – 1.0)
           ↓
Category: 🟢 Excellent | 🟡 Moderate | 🔴 Low
```

---

## 📂 Project Structure

```
ANN/
│
├── dataset/
│   ├── resume_data.csv              # Raw dataset from Kaggle
│   ├── cleaned_resume_data.csv      # After preprocessing
│   └── numerified_resume_data.csv   # Final feature-engineered data
│
├── feature_engineering/
│   ├── data_preprocessing_nb.ipynb  # Data cleaning notebook
│   └── feature_aggregation_nb.ipynb # Feature engineering notebook
│
├── neural_network/
│   ├── ann.ipynb                    # Model training notebook
│   └── logs/                        # Training logs and metrics
│
├── pickled_data/
│   ├── model.h5                     # Trained ANN model
│   └── scaler.pkl                   # Fitted StandardScaler
│
├── main.py                          # Streamlit application
├── requirements.txt                 # Project dependencies
├── pyproject.toml                   # uv project configuration (optional)
├── uv.lock                          # uv lock file (optional)
└── README.md                        # Project documentation
```

---

## 🧪 Feature Engineering

**14+ structured numerical features extracted using rule-based logic:**

**Skills & Experience:**
- Candidate skills count
- Job required skills count
- Skill overlap count
- Total years of experience
- Experience gap (candidate vs requirement)

**Education:**
- Education duration (years)
- Degree level encoding
- Education match flag

**Additional:**
- Number of certifications
- Number of known languages
- Number of companies worked at
- Number of responsibilities
- Age requirement flag

---

## 🤖 Model Details

- **Architecture**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow / Keras
- **Input**: Numerical feature vector
- **Hidden Layers**: Multiple dense layers with ReLU activation
- **Regularization**: Dropout to prevent overfitting
- **Output**: Sigmoid activation → Score (0.0 - 1.0)
- **Scaling**: StandardScaler for feature normalization

---

## 🖥️ Application

**Streamlit-based web interface** with:

- Manual input forms for candidate and job details
- Real-time compatibility prediction
- Match categorization:
  - 🟢 **Excellent Match** (≥0.7)
  - 🟡 **Moderate Match** (0.4-0.69)
  - 🔴 **Low Match** (<0.4)

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- [uv package manager](https://github.com/astral-sh/uv)

### Setup

```bash
# Clone repository
git clone https://github.com/Jaidatt007/ANN-Based-Resume-Job-Compatibility-Scoring-System.git
cd ANN

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/MacOS
# or
pip install uv

# Install dependencies
uv sync

# Run application
uv run streamlit run main.py
```

### Troubleshooting

```bash
# Clear cache and reinstall
uv cache clean
uv sync --reinstall

# Different port
uv run streamlit run main.py --server.port 8502
```

---

## 🛠️ Tech Stack

- **Python** 3.8+
- **uv** - Fast package manager
- **TensorFlow/Keras** - Deep learning
- **Scikit-learn** - Feature scaling
- **Streamlit** - Web interface
- **NumPy & Pandas** - Data processing

---

## 📈 What I Learned

- End-to-end ML pipeline design
- Feature engineering without NLP
- ANN training and evaluation
- Model deployment with Streamlit
- Working with structured data in deep learning

---

## 🔮 Future Improvements

- Add NLP-based semantic features
- Incorporate transformer embeddings
- Expand dataset size and diversity
- Deploy as REST API
- Automated resume parsing from PDF/DOCX
- Feature importance visualization

---

## 📌 Note

This project demonstrates end-to-end ML development skills including data preprocessing, feature engineering, model training, and deployment. Built for educational purposes to showcase practical AI/ML expertise.

**Dataset**: Kaggle

---

⭐ **If you found this project helpful, please consider giving it a star!**