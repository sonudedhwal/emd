"""All model and helper functions for the Spam Email Detector.

Sections
  1. Imports and constants
  2. Training data and text cleaning
  3. Model: build_pipeline, train, predict
  4. Message checks: red_flags, apply_rules
  5. Link scanner: extract_urls, analyze_url, url_level
  6. Header analyzer: analyze_headers
  7. Model comparison and top words
"""
import re
from email import message_from_string
from email.utils import parseaddr
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sample_data import HAM, SPAM

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "free", "win", "winner", "prize", "urgent", "click", "verify", "password",
    "account", "bank", "lottery", "cash", "offer", "limited", "claim", "credit",
    "congratulations", "guaranteed", "bitcoin", "loan", "suspended", "act now",
]
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.IGNORECASE)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "cutt.ly", "rb.gy",
}
RISKY_TLDS = {
    "xyz", "top", "click", "loan", "work", "gq", "tk",
    "ml", "cf", "ga", "zip", "country", "win",
}


# ---------------------------------------------------------------------------
# 2. Training data and text cleaning
# ---------------------------------------------------------------------------
def builtin_dataframe():
    """Small built-in dataset (text, label) so the app works without files."""
    return pd.DataFrame({
        "text": SPAM + HAM,
        "label": ["spam"] * len(SPAM) + ["ham"] * len(HAM),
    })


def clean(text):
    """Lowercase, replace links with 'url', strip odd characters."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^a-z0-9$! ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _prepare(df):
    """Validate a (text, label) dataframe and return it with clean labels."""
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    if df["label"].nunique() < 2 or len(df) < 10:
        raise ValueError("Need at least 10 rows containing both 'spam' and 'ham' labels.")
    return df


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train(df, test_size=0.25):
    """Train on df, return (final_pipeline, metrics_dict)."""
    df = _prepare(df)
    y = (df["label"] == "spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=test_size, random_state=42, stratify=y)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "confusion": pd.DataFrame(
            confusion_matrix(y_test, pred, labels=[0, 1]),
            index=["true ham", "true spam"], columns=["pred ham", "pred spam"]),
        "rows": len(df),
    }
    pipe.fit(df["text"], y)  # final model uses all the data
    return pipe, metrics


def predict(pipe, text):
    """Return (label, spam_probability) for one message."""
    prob = float(pipe.predict_proba([text])[0][1])
    return ("spam" if prob >= 0.5 else "ham"), prob


# ---------------------------------------------------------------------------
# 4. Message checks
# ---------------------------------------------------------------------------
def red_flags(text):
    """Human-readable warning signs found in a message."""
    low = text.lower()
    found = [w for w in SUSPICIOUS_WORDS if w in low]
    if re.search(r"http\S+|www\.", low):
        found.append("contains link")
    if text.count("!") >= 2:
        found.append("many exclamation marks")
    if re.search(r"\$\s?\d|\d+\s?(dollars|usd)", low):
        found.append("money amount")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.4:
        found.append("mostly CAPITAL letters")
    return found


def apply_rules(prob, text, blocklist, allowlist):
    """Adjust the model probability using the user's keyword rules."""
    low = text.lower()
    hit_block = [w for w in blocklist if w and w.lower() in low]
    hit_allow = [w for w in allowlist if w and w.lower() in low]
    notes = []
    if hit_block:
        prob = max(prob, 0.9)
        notes.append("Blocklist match: " + ", ".join(hit_block))
    if hit_allow and not hit_block:
        prob = min(prob, 0.2)
        notes.append("Allowlist match: " + ", ".join(hit_allow))
    return prob, notes


# ---------------------------------------------------------------------------
# 5. Link scanner
# ---------------------------------------------------------------------------
def extract_urls(text):
    return [u.rstrip(".,;!?") for u in URL_PATTERN.findall(text)]


def analyze_url(url):
    """Return (risk_score, [reasons]) for one URL using simple heuristics."""
    full = url if url.lower().startswith("http") else "http://" + url
    parsed = urlparse(full)
    host = (parsed.hostname or "").lower()
    reasons = []
    score = 0

    is_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
    if is_ip:
        reasons.append("uses an IP address instead of a domain"); score += 3
    if host in SHORTENERS:
        reasons.append("link shortener hides the real destination"); score += 2
    if parsed.scheme == "http":
        reasons.append("not encrypted (http)"); score += 1
    if "@" in parsed.netloc:
        reasons.append("contains '@' in the address"); score += 3
    if not is_ip and host.count(".") >= 3:
        reasons.append("many subdomains"); score += 2
    if host.startswith("xn--") or ".xn--" in host:
        reasons.append("punycode (look-alike characters)"); score += 3
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in RISKY_TLDS:
        reasons.append(f"risky domain ending .{tld}"); score += 2
    if len(url) > 75:
        reasons.append("very long address"); score += 1
    if any(w in host for w in ["login", "verify", "secure", "account", "update", "bank", "paypal"]):
        reasons.append("sensitive keyword in domain name"); score += 2
    return score, reasons


def url_level(score):
    if score >= 4:
        return "High risk"
    if score >= 2:
        return "Medium risk"
    return "Low risk"


# ---------------------------------------------------------------------------
# 6. Header analyzer
# ---------------------------------------------------------------------------
def analyze_headers(raw):
    """Parse raw email headers and return (summary dict, [warnings])."""
    msg = message_from_string(raw)
    from_name, from_addr = parseaddr(msg.get("From", ""))
    _, reply_addr = parseaddr(msg.get("Reply-To", ""))
    _, return_addr = parseaddr(msg.get("Return-Path", ""))
    auth = (msg.get("Authentication-Results", "") + " " + msg.get("Received-SPF", "")).lower()

    def result(name):
        m = re.search(name + r"=(\w+)", auth)
        if m:
            return m.group(1)
        if name == "spf":
            m = re.match(r"\s*(\w+)", msg.get("Received-SPF", "").lower())
            return m.group(1) if m else "not found"
        return "not found"

    summary = {
        "From": msg.get("From", "(missing)"),
        "Reply-To": msg.get("Reply-To", "(none)"),
        "Return-Path": msg.get("Return-Path", "(none)"),
        "Subject": msg.get("Subject", "(none)"),
        "Date": msg.get("Date", "(missing)"),
        "Hops (Received lines)": len(msg.get_all("Received", [])),
        "SPF": result("spf"),
        "DKIM": result("dkim"),
        "DMARC": result("dmarc"),
    }

    def domain(addr):
        return addr.split("@")[-1].lower() if "@" in addr else ""

    warnings = []
    if not from_addr:
        warnings.append("No valid From address found.")
    if reply_addr and domain(reply_addr) != domain(from_addr):
        warnings.append(f"Reply-To domain ({domain(reply_addr)}) differs from From domain ({domain(from_addr)}).")
    if return_addr and domain(return_addr) != domain(from_addr):
        warnings.append(f"Return-Path domain ({domain(return_addr)}) differs from From domain ({domain(from_addr)}).")
    for check in ("SPF", "DKIM", "DMARC"):
        if summary[check] in ("fail", "softfail", "none", "permerror", "temperror"):
            warnings.append(f"{check} check result: {summary[check]}.")
        elif summary[check] == "not found":
            warnings.append(f"{check} result not present in the headers.")
    if from_name and "@" in from_name and domain(from_name) != domain(from_addr):
        warnings.append("Display name contains a different email address (spoofing trick).")
    if not msg.get("Date"):
        warnings.append("Date header is missing.")
    return summary, warnings


# ---------------------------------------------------------------------------
# 7. Model comparison and top words
# ---------------------------------------------------------------------------
def compare_models(df, test_size=0.25):
    """Train four algorithms and return a table sorted by F1 score."""
    df = _prepare(df)
    y = (df["label"] == "spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=test_size, random_state=42, stratify=y)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }
    rows = []
    for name, clf in classifiers.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2))),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 3),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 3),
            "F1": round(f1_score(y_test, pred, zero_division=0), 3),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def top_words(pipe, n=15):
    """Words that push a message toward spam / ham (needs a linear model)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None, None
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = clf.coef_[0]
    order = coef.argsort()
    spam = pd.Series(coef[order[-n:]][::-1], index=names[order[-n:]][::-1])
    ham = pd.Series(-coef[order[:n]], index=names[order[:n]])
    return spam, ham
