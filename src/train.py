import string
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Download stopwords if they are not already available
nltk.download("stopwords")


# ---------------------------------------------------------
# Text preprocessing setup
# ---------------------------------------------------------
# stop_words contains common words like:
# "the", "is", "are", "and", "to", etc.
#
# stemmer helps convert words to root form:
# "playing", "played", "plays" -> "play"

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    """
    Clean one text message before giving it to the ML model.

    Steps:
    1. Convert text to lowercase
    2. Remove punctuation
    3. Remove stopwords
    4. Apply stemming
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    clean_text = ""

    for char in text:
        if char not in string.punctuation:
            clean_text += char

    # Split sentence into words
    words = clean_text.split()

    # Remove stopwords and apply stemming
    final_words = []

    for word in words:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            final_words.append(stemmed_word)

    # Join words back into a single sentence
    return " ".join(final_words)


# ---------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------
# encoding="latin-1" is used because this dataset has some special
# characters. Without it, Python may give UnicodeDecodeError.

df = pd.read_csv("data/spam.csv", encoding="latin-1")


# ---------------------------------------------------------
# Step 2: Keep only useful columns
# ---------------------------------------------------------
# The original dataset has extra columns, but we only need:
# v1 -> label
# v2 -> message

df = df.iloc[:, 0:2]


# ---------------------------------------------------------
# Step 3: Rename columns
# ---------------------------------------------------------

df.columns = ["label", "message"]


# ---------------------------------------------------------
# Step 4: Remove wrong header row if present
# ---------------------------------------------------------

df = df[df["label"] != "v1"].copy()


# ---------------------------------------------------------
# Step 5: Convert labels into numbers
# ---------------------------------------------------------
# ham  -> 0
# spam -> 1

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ---------------------------------------------------------
# Step 6: Clean all messages
# ---------------------------------------------------------

df["message"] = df["message"].apply(preprocess_text)


# ---------------------------------------------------------
# Step 7: Prepare input and output
# ---------------------------------------------------------
# X_text = cleaned messages
# y      = labels

X_text = df["message"]
y = df["label"]


# ---------------------------------------------------------
# Step 8: Convert text into numbers using TF-IDF
# ---------------------------------------------------------
# max_features=3000 means we keep the top 3000 useful words.

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(X_text).toarray()


# ---------------------------------------------------------
# Step 9: Split data into training and testing
# ---------------------------------------------------------
# 80% data -> training
# 20% data -> testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------------
# Step 10: Train the model
# ---------------------------------------------------------

model = MultinomialNB()
model.fit(X_train, y_train)


# ---------------------------------------------------------
# Step 11: Test the model
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Step 12: Evaluate the model
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------------
# Step 13: Save model and vectorizer
# ---------------------------------------------------------
# We save both:
# 1. model  -> for prediction
# 2. tfidf  -> to convert future messages into numbers

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")