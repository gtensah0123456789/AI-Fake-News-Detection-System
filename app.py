"""
app.py
Flask Backend for the AI-Based Fake News Detection System.
Provides web pages, ML inference API, SQLite persistence, and dashboard analytics.
"""

import os
import re
import string
import sqlite3
from datetime import datetime
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.pkl")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "predictions.db")

app = Flask(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize input text identically to train_model.py."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_db_connection():
    """Create and return a SQLite connection with Row factory enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite predictions table if it does not already exist."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_ml_artifact():
    """Load the trained Joblib pipeline and evaluation metrics."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Please run 'python train_model.py' first."
        )
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact.get("metrics", {})


# Initialize database and load ML model on startup
init_db()
try:
    ml_pipeline, model_metrics = load_ml_artifact()
except Exception as exc:
    ml_pipeline = None
    model_metrics = {}
    print(f"[WARNING] Model load failed at startup: {exc}")


def extract_top_features(cleaned_text: str, predicted_label: str, top_n: int = 4):
    """
    Extract influential TF-IDF n-grams from the user's text based on
    Logistic Regression coefficients to explain the ML prediction transparently.
    """
    if ml_pipeline is None:
        return []
    try:
        vectorizer = ml_pipeline.named_steps["tfidf"]
        classifier = ml_pipeline.named_steps["clf"]
        feature_names = np.array(vectorizer.get_feature_names_out())
        tfidf_vector = vectorizer.transform([cleaned_text])
        nonzero_indices = tfidf_vector.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []

        # classifier.classes_ is typically ['FAKE', 'REAL']
        classes = list(classifier.classes_)
        coefs = classifier.coef_[0]  # positive favors classes[1], negative favors classes[0]
        target_sign = 1 if predicted_label == classes[1] else -1

        scored_terms = []
        for idx in nonzero_indices:
            weight = tfidf_vector[0, idx] * coefs[idx] * target_sign
            if weight > 0:
                scored_terms.append((feature_names[idx], float(weight)))

        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in scored_terms[:top_n]]
    except Exception:
        return []


# -------------------- Web Page Routes --------------------


@app.route("/")
def index():
    """Render Home & News Detection page."""
    return render_template("index.html", active_page="home", metrics=model_metrics)


@app.route("/dashboard")
def dashboard():
    """Render Prediction History & Analytics Dashboard."""
    return render_template("dashboard.html", active_page="dashboard", metrics=model_metrics)


@app.route("/about")
def about():
    """Render About Project & ML Workflow page."""
    return render_template("about.html", active_page="about", metrics=model_metrics)


# -------------------- REST API Endpoints --------------------


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Classify submitted news text using the trained TF-IDF + Logistic Regression model."""
    global ml_pipeline, model_metrics

    if ml_pipeline is None:
        try:
            ml_pipeline, model_metrics = load_ml_artifact()
        except Exception as exc:
            return jsonify({"error": f"Model not loaded: {str(exc)}"}), 500

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Invalid request payload. Please provide news text."}), 400

    raw_text = str(data.get("text", "")).strip()
    if len(raw_text) < 15:
        return jsonify(
            {"error": "Please enter a complete news headline or article (at least 15 characters)."}
        ), 400

    if len(raw_text) > 15000:
        return jsonify(
            {"error": "Input text exceeds the 15,000-character limit. Please shorten the article."}
        ), 400

    cleaned = clean_text(raw_text)
    if len(cleaned.split()) < 3:
        return jsonify(
            {"error": "Input must contain at least 3 meaningful words after text cleaning."}
        ), 400

    # Perform ML prediction & probability estimation
    raw_pred = ml_pipeline.predict([cleaned])[0]
    probabilities = ml_pipeline.predict_proba([cleaned])[0]
    classes = list(ml_pipeline.classes_)
    pred_index = classes.index(raw_pred)
    confidence = round(float(probabilities[pred_index]) * 100, 2)

    display_label = "Likely Real" if raw_pred == "REAL" else "Likely Fake"
    top_terms = extract_top_features(cleaned, raw_pred, top_n=4)

    if raw_pred == "FAKE":
        explanation = (
            "The model detected linguistic and statistical patterns frequently associated with "
            "sensationalist, unverified, or misleading articles in the training corpus."
        )
    else:
        explanation = (
            "The model identified structured reporting patterns, neutral phrasing, and vocabulary "
            "consistent with credible news articles in the training corpus."
        )

    if top_terms:
        explanation += f" Key influential terms detected: {', '.join(top_terms)}."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save result to SQLite database
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO predictions (article_text, prediction, confidence, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (raw_text, display_label, confidence, timestamp),
        )
        conn.commit()
        record_id = cursor.lastrowid
    finally:
        conn.close()

    return jsonify(
        {
            "id": record_id,
            "prediction": display_label,
            "raw_label": raw_pred,
            "confidence": confidence,
            "explanation": explanation,
            "top_terms": top_terms,
            "created_at": timestamp,
            "disclaimer": (
                "Important: This output is a Machine Learning probability estimate based on "
                "text patterns, not a guaranteed human fact-check."
            ),
        }
    )


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return aggregated prediction counts and confidence averages for the dashboard."""
    conn = get_db_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        real_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 'Likely Real'"
        ).fetchone()[0]
        fake_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 'Likely Fake'"
        ).fetchone()[0]
        avg_conf_row = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
        avg_confidence = round(float(avg_conf_row), 2) if avg_conf_row is not None else 0.0

        # Recent 7 trend points for confidence line chart
        trend_rows = conn.execute(
            """
            SELECT id, prediction, confidence, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 10
            """
        ).fetchall()
        trend_data = [
            {
                "id": row["id"],
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
            }
            for row in reversed(trend_rows)
        ]
    finally:
        conn.close()

    return jsonify(
        {
            "total": total,
            "likely_real": real_count,
            "likely_fake": fake_count,
            "avg_confidence": avg_confidence,
            "trend": trend_data,
            "model_metrics": model_metrics,
        }
    )


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return recent prediction history ordered by latest first."""
    limit = request.args.get("limit", default=25, type=int)
    limit = max(1, min(limit, 100))

    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, article_text, prediction, confidence, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        history = [
            {
                "id": row["id"],
                "article_text": row["article_text"],
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()

    return jsonify({"history": history})


@app.route("/api/history/clear", methods=["DELETE", "POST"])
def api_clear_history():
    """Delete all stored records from the predictions table."""
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        conn.commit()
    finally:
        conn.close()

    return jsonify({"message": "Prediction history cleared successfully."})


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Requested resource was not found."}), 404


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "An unexpected internal server error occurred."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
