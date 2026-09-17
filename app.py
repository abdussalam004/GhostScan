from flask import Flask, render_template, request, jsonify
import pickle
import re
import traceback

from werkzeug.exceptions import HTTPException

import pandas as pd
from scipy.sparse import hstack, csr_matrix

from helper import get_qa_chain

app = Flask(__name__)

# Flip to False once GhostBot is confirmed working - True sends the real
# exception text into the chat bubble (via /chat) which is handy for
# debugging but shouldn't stay on in production.
DEBUG_CHAT_ERRORS = True

# ---------------------------------------------------------------------------
# Phishing scanner setup - loads the shared TF-IDF vectorizer and every
# trained model from Phishing_Detection_System.ipynb. Keys here are what the
# <select> in index.html submits as `model`.
# ---------------------------------------------------------------------------
VECTORIZER_PATH = "phishing_tfidf.pkl"  # notebook saves the fitted TfidfVectorizer as `tfidf` -> phishing_tfidf.pkl

MODEL_FILES = {
    "svm": "phishing_svm.pkl",
    "lr": "phishing_lr.pkl",
    "mnb": "phishing_mnb.pkl",
    "dt": "phishing_dt.pkl",
    "rf": "phishing_rf.pkl",
    "xgb": "phishing_xgb.pkl",
}

# Display name + test-set accuracy, pulled straight from the notebook's
# "Model Comparison" cell (section 13) for these exact .pkl files. An earlier
# version of that cell paired the Model-name list and the score list in
# different orders, silently swapping Random Forest's and XGBoost's numbers -
# fixed in the notebook now, so XGBoost is actually the top performer.
MODEL_META = {
    "xgb": {"name": "XGBoost - HIGH", "accuracy": 0.977647},
    "rf": {"name": "Random Forest - HIGH", "accuracy": 0.967884},
    "dt": {"name": "Decision Tree - MODERATE", "accuracy": 0.962871},
    "svm": {"name": "SVM (Linear) - MODERATE", "accuracy": 0.956992},
    "lr": {"name": "Logistic Regression - MODERATE", "accuracy": 0.954469},
    "mnb": {"name": "Naive Bayes - NOT RECOMMENDED", "accuracy": 0.894385},
}

DEFAULT_MODEL = "xgb"  # highest test accuracy among the benchmarked models

# XGBoost was fit on integer-encoded labels (bad -> 0, good -> 1), unlike the
# other models which were fit directly on the 'good'/'bad' strings. Its
# predict() therefore returns 0/1, not a string, and needs this explicit map.
XGB_LABEL_MAP = {0: "bad", 1: "good"}

tfidf = pickle.load(open(VECTORIZER_PATH, "rb"))
models = {key: pickle.load(open(path, "rb")) for key, path in MODEL_FILES.items()}

# ---------------------------------------------------------------------------
# Feature building must match the notebook's build_features() *exactly*
# (section 4, "Feature Extraction"), or the input shape/scale won't line up
# with what each model was fit on:
#
#   features = hstack([ tfidf.transform(urls), <10 handcrafted columns> ])
#
# The models were trained on the RAW url string run through tfidf directly -
# no scheme/www stripping, no tokenizing, no stemming. A previous version of
# this file preprocessed the url (strip scheme -> tokenize -> stem -> join)
# before vectorizing, which was for an older/different notebook pipeline;
# that produces a 3000-column vector when these models expect 3010 columns
# (3000 tfidf + 10 handcrafted), and predictions would be unreliable or
# error outright. This must stay byte-for-byte identical to the notebook.
# ---------------------------------------------------------------------------
SUSPICIOUS_WORDS = ["login", "verify", "account", "secure", "update", "banking", "confirm"]


def url_features(url):
    url_l = url.lower()
    return {
        "length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_at": url.count("@"),
        "num_slash": url.count("/"),
        "has_https": int("https" in url_l),
        "has_ip": int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url))),
        "num_subdomains": url.split("/")[0].count("."),
        "suspicious_words": sum(w in url_l for w in SUSPICIOUS_WORDS),
    }


def build_features(urls):
    """Same combined TF-IDF (char n-grams) + handcrafted feature matrix the
    notebook builds, using the already-fitted `tfidf` (fit=False equivalent)."""
    x_text = tfidf.transform(urls)
    feat_df = pd.DataFrame([url_features(u) for u in urls])
    return hstack([x_text, csr_matrix(feat_df.values)]).tocsr()


# ---------------------------------------------------------------------------
# GhostBot setup - the RAG chain is built lazily on first chat request
# (not at import time), so the scanner still starts up fast even before
# anyone opens the chat widget. helper.py handles building/loading the
# FAISS index itself.
# ---------------------------------------------------------------------------
_ghostbot_chain = None


def get_ghostbot_chain():
    global _ghostbot_chain
    if _ghostbot_chain is None:
        _ghostbot_chain = get_qa_chain()
    return _ghostbot_chain


@app.route("/", methods=["GET", "POST"])
def index():
    predict = None
    selected_model = DEFAULT_MODEL
    confidence = None

    if request.method == "POST":
        url = request.form.get("url", "")
        selected_model = request.form.get("model", DEFAULT_MODEL)
        if selected_model not in models:
            selected_model = DEFAULT_MODEL

        model = models[selected_model]
        vec = build_features([url])
        raw_result = model.predict(vec)[0]

        if selected_model == "xgb":
            # raw_result is 0 or 1 here, not a string - map it explicitly.
            result = XGB_LABEL_MAP.get(int(raw_result))
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = round(float(proba[int(raw_result)]) * 100, 1)
        else:
            result = raw_result
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                classes = list(model.classes_)
                if result in classes:
                    confidence = round(float(proba[classes.index(result)]) * 100, 1)

        if result == "bad":
            predict = "This is a phishing website !!"
        elif result == "good":
            predict = "This is legit website !!"
        else:
            predict = "Something went wrong !!"

    return render_template(
        "index.html",
        predict=predict,
        models=MODEL_META,
        selected_model=selected_model,
        confidence=confidence,
    )


@app.route("/chat", methods=["POST"])
def chat():
    """JSON endpoint for the GhostBot popup widget."""
    data = request.get_json(silent=True) or {}
    # The widget's JS sends {"question": ...}, not {"message": ...}.
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Ask me something about GhostScan!"}), 400

    try:
        chain = get_ghostbot_chain()
        answer = chain.invoke(question)
        if not isinstance(answer, str):
            answer = str(answer)
    except Exception as exc:
        traceback.print_exc()
        if DEBUG_CHAT_ERRORS:
            # Shows the real error in the chat bubble so it's easy to see
            # without digging through the terminal. Set DEBUG_CHAT_ERRORS =
            # False above once GhostBot is confirmed working.
            answer = f"[DEBUG ERROR] {type(exc).__name__}: {exc}"
        else:
            answer = ("Sorry, I'm having trouble answering right now. "
                       "Please try again in a moment.")

    return jsonify({"answer": answer})


# Belt-and-braces: if something throws outside the /chat route's own
# try/except (e.g. inside Flask/Werkzeug itself), still return JSON for
# /chat calls instead of an HTML error page, so the widget's fetch() never
# gets a response it can't parse.
#
# IMPORTANT: this must let ordinary HTTPExceptions (404 for a missing route
# like the browser's automatic /favicon.ico request, 405, etc.) through
# untouched. `raise exc` from inside an errorhandler does NOT hand it back
# to Flask's normal error page the way you'd expect - it turns routine 404s
# into broken 500s instead. Returning the exception itself is the correct
# way to let Flask render its normal response for it.
@app.errorhandler(Exception)
def handle_uncaught(exc):
    if isinstance(exc, HTTPException):
        return exc
    traceback.print_exc()
    if request.path.startswith("/chat"):
        return jsonify({"answer": "Sorry, something went wrong on the server."}), 500
    raise exc


if __name__ == "__main__":
    app.run(debug=True)
