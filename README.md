# 🤖 AI Fake Job Detection System

A Machine Learning-based web application that analyzes job posting information and predicts whether a job posting is **FAKE** or **REAL**.

## 🎯 Project Objective

The main objective of this project is to help job seekers identify potentially fraudulent job postings and reduce the risk of falling victim to fake recruitment scams.

The system uses Natural Language Processing (NLP) and Machine Learning techniques to analyze job posting content.

## 🧠 How It Works
AI-Fake-Job-Detection.png
1. The user enters job posting details.
2. The system combines the relevant text fields.
3. TF-IDF Vectorization converts the text into numerical features.
4. A Machine Learning classification model analyzes the features.
5. The system predicts whether the job is:
   - 🚨 FAKE JOB
   - ✅ REAL JOB
6. The application displays the prediction and confidence score.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- HTML
- CSS
- Git & GitHub

## 📊 Dataset

The project uses a job-posting dataset containing information such as:

- Job title
- Company profile
- Job description
- Requirements
- Benefits
- Employment type
- Required experience
- Required education
- Industry
- Function
- Fraudulent label

The `fraudulent` field is used as the target variable.

## 🤖 Machine Learning

The text information from the job posting is converted into numerical features using:

**TF-IDF (Term Frequency-Inverse Document Frequency)**

The trained classification model then predicts the class of an unseen job posting.

## 📈 Model Performance

The model achieved approximately:

**Accuracy: 97.29%**

For the fake-job class:

- Precision: 0.71
- Recall: 0.90
- F1 Score: 0.80
- Support: 173

The high recall for fake jobs is particularly useful because the system is designed to identify potentially fraudulent job postings.

## 🖥️ Web Application

The project includes a Flask-based web interface where users can enter job information and receive a prediction.

Example output:

**🚨 FAKE JOB DETECTED**

or

**✅ REAL JOB**

The application also provides a confidence score.

## 📂 Project Structure

```text
AI-Fake-Job-Detection/
│
├── Dataset/
│   └── fake_job_postings.csv
│
├── Model/
│
├── static/
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── predict.py
├── test_model.py
├── train_model.py
├── dataset.csv
└── requirements.txt
