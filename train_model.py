"""
train_model.py
Trains a TF-IDF + Logistic Regression Fake News Classification Pipeline
and saves the trained model artifact to model/trained_model.pkl.
"""

import os
import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "news_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")


def clean_text(text: str) -> str:
    """Clean and normalize raw news text prior to TF-IDF vectorization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("=" * 60)
    print("AI-Based Fake News Detection System - Model Training")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # 1. Load dataset
    df = pd.read_csv(DATASET_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset CSV must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(["REAL", "FAKE"])]
    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 0]

    print(f"Loaded {len(df)} valid records from {DATASET_PATH}")
    print("Class Distribution:")
    print(df["label"].value_counts().to_string())
    print("-" * 60)

    # 2. Train-test split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 3. Build TF-IDF + Logistic Regression Pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_df=0.95,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 4. Fit model
    pipeline.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="FAKE", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="FAKE", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="FAKE", zero_division=0)

    print("\nModel Evaluation Metrics (Test Set):")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}% (FAKE class)")
    print(f"  Recall   : {rec * 100:.2f}% (FAKE class)")
    print(f"  F1-Score : {f1 * 100:.2f}% (FAKE class)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"]))

    # 6. Save model + metadata using Joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "metrics": {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "total_samples": int(len(df)),
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved trained model successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
