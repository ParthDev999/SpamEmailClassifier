import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already downloaded
nltk.download("stopwords")

# ---------------------------------------------------------
# Step 1: Load saved model and vectorizer
# ---------------------------------------------------------
# spam_model.pkl contains the trained ML model.
# tfidf_vectorizer.pkl contains the TF-IDF converter.
#
# We need both:
# 1. vectorizer -> converts new text into numbers
# 2. model      -> predicts spam or ham

model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# ---------------------------------------------------------
# Step 2: Prepare text cleaning tools
# ---------------------------------------------------------
# These must be the same cleaning steps used in train.py.
# If training text was cleaned one way and prediction text is cleaned
# another way, model performance can become poor.

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ---------------------------------------------------------
# Step 3: Create preprocessing function
# ---------------------------------------------------------
# This function cleans a new message using the same steps:
# 1. lowercase
# 2. remove punctuation
# 3. remove stopwords
# 4. apply stemming

def preprocess_message(message):
    # Convert message to lowercase
    message = message.lower()

    # Remove punctuation
    clean_text = ""

    for char in message:
        if char not in string.punctuation:
            clean_text += char

    # Split message into words
    words = clean_text.split()

    # Remove stopwords and apply stemming
    final_words = []

    for word in words:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            final_words.append(stemmed_word)

    # Join words back into one sentence
    return " ".join(final_words)


# ---------------------------------------------------------
# Step 4: Take input from user
# ---------------------------------------------------------
# input() allows us to type a message in the terminal.

message = input("Enter a message: ")


# ---------------------------------------------------------
# Step 5: Clean the input message
# ---------------------------------------------------------

cleaned_message = preprocess_message(message)


# ---------------------------------------------------------
# Step 6: Convert cleaned message into numbers
# ---------------------------------------------------------
# IMPORTANT:
# Here we use transform(), not fit_transform().
#
# fit_transform() was used during training because TF-IDF had to learn
# the vocabulary from the training data.
#
# During prediction, vocabulary is already learned.
# So we only transform the new message using the saved vectorizer.

message_vector = tfidf.transform([cleaned_message]).toarray()


# ---------------------------------------------------------
# Step 7: Make prediction
# ---------------------------------------------------------
# prediction will be:
# 0 -> ham
# 1 -> spam

prediction = model.predict(message_vector)


# ---------------------------------------------------------
# Step 8: Show result
# ---------------------------------------------------------

if prediction[0] == 1:
    print("Result: Spam message")
else:
    print("Result: Not spam message")