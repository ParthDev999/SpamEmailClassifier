import string
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Download stopwords if not already available
nltk.download("stopwords", quiet=True)


# ---------------------------------------------------------
# Text preprocessing setup
# ---------------------------------------------------------
# These steps must be the same as train.py and predict.py.
# The model was trained on cleaned text, so new input must also
# be cleaned in the same way.

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    """
    Clean one input message before prediction.

    Steps:
    1. Convert text to lowercase
    2. Remove punctuation
    3. Remove stopwords
    4. Apply stemming
    """

    text = text.lower()

    clean_text = ""

    for char in text:
        if char not in string.punctuation:
            clean_text += char

    words = clean_text.split()

    final_words = []

    for word in words:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            final_words.append(stemmed_word)

    return " ".join(final_words)


# ---------------------------------------------------------
# Load saved model and vectorizer
# ---------------------------------------------------------
# model -> predicts spam or not spam
# tfidf -> converts cleaned text into numbers

model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("Spam Email Classifier")

st.write("Enter a message below and the model will predict whether it is spam or not.")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned_message = preprocess_text(message)

        message_vector = tfidf.transform([cleaned_message]).toarray()

        prediction = model.predict(message_vector)

        if prediction[0] == 1:
            st.error("This message is Spam")
        else:
            st.success("This message is Not Spam")