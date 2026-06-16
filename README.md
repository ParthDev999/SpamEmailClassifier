# Spam Email Classifier

A machine learning project that classifies text messages as **Spam** or **Not Spam** using NLP techniques. The project includes text preprocessing, TF-IDF vectorization, model training, evaluation, prediction, and a Streamlit web app for easy usage.

## Project Overview

This project is built to detect spam messages from text data. Since machine learning models cannot directly understand raw text, the messages are first cleaned and converted into numerical features using **TF-IDF Vectorization**. A **Multinomial Naive Bayes** classifier is then trained to classify messages as spam or ham.

## Features

- Loads and cleans spam message dataset
- Converts labels into numerical format
- Applies NLP preprocessing:
  - Lowercasing
  - Punctuation removal
  - Stopword removal
  - Stemming
- Converts text into numerical features using TF-IDF
- Trains a Multinomial Naive Bayes model
- Evaluates model using:
  - Accuracy
  - Confusion Matrix
  - Precision
  - Recall
  - F1-score
- Saves trained model and vectorizer
- Provides prediction through terminal
- Includes a Streamlit web app for user-friendly prediction

## Tech Stack

- Python
- Pandas
- Scikit-learn
- NLTK
- Joblib
- Streamlit

## Project Structure

```text
SpamEmailClassifier/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
├── requirements.txt
└── README.md
