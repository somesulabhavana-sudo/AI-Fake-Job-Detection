from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("Model/fake_job_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        job_text = request.form["job_description"]

        # Convert job description into TF-IDF features
        job_vector = vectorizer.transform([job_text])

        # Get prediction
        result = model.predict(job_vector)[0]

        # Get probability
        probabilities = model.predict_proba(job_vector)[0]

        confidence = round(max(probabilities) * 100, 2)

        if result == 1:
            prediction = "FAKE JOB"
        else:
            prediction = "REAL JOB"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)