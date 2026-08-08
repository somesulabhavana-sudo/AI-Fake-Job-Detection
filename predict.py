import joblib

# Load the trained model
model = joblib.load("Model/fake_job_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")


def predict_job(job_text):

    # Convert job text into TF-IDF features
    job_vector = vectorizer.transform([job_text])

    # Make prediction
    prediction = model.predict(job_vector)[0]

    # Convert prediction into readable result
    if prediction == 1:
        return "FAKE JOB"
    else:
        return "REAL JOB"


# Test the system
job = input("\nEnter a job description: ")

result = predict_job(job)

print("\nPrediction:", result)