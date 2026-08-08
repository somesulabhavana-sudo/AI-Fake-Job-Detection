import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data = pd.read_csv("Dataset/fake_job_postings.csv")

# 2. Clean column names
data.columns = data.columns.str.strip().str.lower()

# 3. Fill missing values
data = data.fillna("")

# 4. Combine important text fields
data["text"] = (
    data["title"].astype(str) + " " +
    data["company_profile"].astype(str) + " " +
    data["description"].astype(str) + " " +
    data["requirements"].astype(str) + " " +
    data["benefits"].astype(str)
)

# 5. Features and target
X_text = data["text"]
y = data["fraudulent"].astype(int)

# 6. Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(X_text)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print("TF-IDF shape:", X.shape)

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Train improved model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 9. Predictions
y_pred = model.predict(X_test)

# 10. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Real Job", "Fake Job"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Save model
joblib.dump(model, "Model/fake_job_model.pkl")
joblib.dump(vectorizer, "Model/tfidf_vectorizer.pkl")

print("\n==============================")
print("MODEL SAVED SUCCESSFULLY!")
print("==============================")