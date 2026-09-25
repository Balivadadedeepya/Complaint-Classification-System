# Complaint Classification System
# NLP + Machine Learning

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Sample complaint dataset
complaints = [
    "My internet connection is not working",
    "The WiFi keeps disconnecting",
    "My broadband speed is very slow",
    "I cannot connect to the internet",

    "I was charged extra on my bill",
    "My monthly bill is incorrect",
    "There is an unexpected charge on my account",
    "I received a wrong billing amount",

    "I cannot login to my account",
    "My password is not working",
    "I forgot my account password",
    "I am unable to access my account",

    "My payment failed",
    "The transaction was unsuccessful",
    "My card payment did not go through",
    "I cannot complete the payment",
]

categories = [
    "Internet Issue",
    "Internet Issue",
    "Internet Issue",
    "Internet Issue",

    "Billing Issue",
    "Billing Issue",
    "Billing Issue",
    "Billing Issue",

    "Account Issue",
    "Account Issue",
    "Account Issue",
    "Account Issue",

    "Payment Issue",
    "Payment Issue",
    "Payment Issue",
    "Payment Issue",
]


# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train the model
model.fit(complaints, categories)


def classify_complaint(text):
    """Classify a complaint into a category."""
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# Interactive prediction
if __name__ == "__main__":
    print("=" * 50)
    print("      COMPLAINT CLASSIFICATION SYSTEM")
    print("=" * 50)

    print("\nAvailable categories:")
    print("• Internet Issue")
    print("• Billing Issue")
    print("• Account Issue")
    print("• Payment Issue")

    complaint = input("\nEnter your complaint: ")

    category, confidence = classify_complaint(complaint)

    print("\nPredicted Category:", category)
    print(f"Confidence: {confidence:.2f}%")
