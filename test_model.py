import joblib

# Load the trained model
model = joblib.load("Model/fake_job_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")


# Test job postings
test_jobs = [

    # FAKE JOB 1
    """
    Work from home and earn 50000 per week.
    No experience required.
    Pay a registration fee of 2000 to get started.
    Send your bank details and OTP immediately.
    Limited vacancies. Apply now.
    """,

    # REAL JOB 1
    """
    Data Analyst required at a technology company.
    Responsibilities include analyzing datasets, creating reports,
    preparing dashboards and working with business teams.
    Candidates should have knowledge of Python, SQL, Excel and statistics.
    Bachelor's degree in Data Science or a related field preferred.
    Full-time position with a formal interview process.
    """,

    # FAKE JOB 2
    """
    Congratulations! You have been selected for a high-paying online job.
    Earn 100000 per month without any interview or experience.
    Pay a small security deposit to receive your employee ID.
    Send your bank account information to complete registration.
    """,

    # REAL JOB 2
    """
    Software Developer Intern.
    The intern will assist the development team with software testing,
    debugging and documentation.
    Knowledge of Python or Java is preferred.
    Candidates will participate in a technical interview.
    This is a three-month paid internship.
    """,

    # FAKE JOB 3
    """
    Make money instantly from home.
    We guarantee huge weekly income with zero skills required.
    No interview is necessary.
    Pay a joining fee today and start earning immediately.
    Contact us through the provided messaging number.
    """
]


print("\n====================================")
print("AI FAKE JOB DETECTION - TESTING")
print("====================================\n")


for number, job in enumerate(test_jobs, start=1):

    # Convert text into TF-IDF features
    job_vector = vectorizer.transform([job])

    # Predict
    prediction = model.predict(job_vector)[0]

    # Probability
    probabilities = model.predict_proba(job_vector)[0]
    confidence = max(probabilities) * 100

    if prediction == 1:
        result = "FAKE JOB"
    else:
        result = "REAL JOB"

    print(f"Test {number}: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print("------------------------------------")


print("\nTesting completed successfully!")