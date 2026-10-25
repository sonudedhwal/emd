import pandas as pd
import streamlit as st

import spam_model as sm

# Page configuration (must be the first Streamlit call)
st.set_page_config(page_title="Spam Email Detector", layout="wide")


@st.cache_resource(show_spinner="Training model...")
def load_default_model():
    return sm.train(sm.builtin_dataframe())


def set_model(pipe, metrics, source):
    st.session_state.pipe = pipe
    st.session_state.metrics = metrics
    st.session_state.model_source = source


def score_text(text):
    """Model probability + keyword rules + settings threshold -> (label, prob, notes)."""
    _, prob = sm.predict(st.session_state.pipe, text)
    prob, notes = sm.apply_rules(
        prob, text, st.session_state.blocklist, st.session_state.allowlist
    )
    label = "spam" if prob >= st.session_state.threshold else "ham"
    return label, prob, notes


def read_csv(uploaded):
    try:
        return pd.read_csv(uploaded, encoding="latin-1")
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None


# --- SESSION STATE ---
st.session_state.setdefault("history", [])
st.session_state.setdefault("blocklist", [])
st.session_state.setdefault("allowlist", [])
st.session_state.setdefault("threshold", 0.5)
st.session_state.setdefault("training_df", sm.builtin_dataframe())
if "pipe" not in st.session_state:
    pipe, metrics = load_default_model()
    set_model(pipe, metrics, "Built-in sample data")

# --- SIDEBAR OPTIONS ---
st.sidebar.title("📧 Spam Detector")
st.sidebar.markdown("Check emails and manage your spam tools")

sidebar_option = st.sidebar.radio(
    "Navigation Options",
    [
        "Dashboard Overview",
        "Check Email",
        "Link Scanner",
        "Header Analyzer",
        "Bulk CSV Check",
        "Keyword Rules",
        "Train Model",
        "Model Comparison",
        "Analytics",
        "History",
        "Settings",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Model Status:** Ready 🟢\n\n"
    f"**Trained on:** {st.session_state.model_source}\n\n"
    f"**Threshold:** {st.session_state.threshold:.0%}\n\n"
    f"**Emails checked:** {len(st.session_state.history)}"
)

# --- MAIN CONTENT AREA ---

if sidebar_option == "Dashboard Overview":
    st.title("📧 Spam Email Detector Dashboard")
    st.write("Welcome back! Here is how your model is performing.")

    m = st.session_state.metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{m['accuracy']:.0%}")
    col2.metric("Precision", f"{m['precision']:.0%}")
    col3.metric("Recall", f"{m['recall']:.0%}")
    col4.metric("Training Rows", m["rows"])

    hist = st.session_state.history
    spam_count = sum(1 for h in hist if h["Result"] == "spam")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spam found this session", spam_count)
    c2.metric("Safe emails this session", len(hist) - spam_count)
    c3.metric("Custom rules", len(st.session_state.blocklist) + len(st.session_state.allowlist))

    st.subheader("Confusion Matrix (test split)")
    st.dataframe(m["confusion"])
    st.markdown(
        """
        * **Check Email:** classify one message with red flags
        * **Link Scanner:** inspect every link in a message
        * **Header Analyzer:** spot spoofed senders (SPF / DKIM / DMARC)
        * **Bulk CSV Check:** classify many emails at once
        * **Keyword Rules:** your own block and allow words
        * **Train Model / Model Comparison:** retrain and compare algorithms
        * **Analytics, History, Settings:** charts, past checks and threshold
        """
    )

elif sidebar_option == "Check Email":
    st.title("🔍 Check Email")
    st.write("Paste an email or SMS message to see if it is spam.")

    subject = st.text_input("Subject (optional)")
    body = st.text_area("Email body", height=200)

    if st.button("Check for Spam", type="primary"):
        text = f"{subject} {body}".strip()
        if not text:
            st.warning("Please enter some text first.")
        else:
            label, prob, notes = score_text(text)
            flags = sm.red_flags(text)
            if label == "spam":
                st.error(f"🚨 SPAM detected ({prob:.0%} spam probability)")
            else:
                st.success(f"✅ Looks safe ({1 - prob:.0%} confidence)")
            st.progress(min(max(prob, 0.0), 1.0), text=f"Spam probability: {prob:.0%}")
            for n in notes:
                st.info(n)
            if flags:
                st.write("**Red flags found:** " + ", ".join(flags))
            urls = sm.extract_urls(text)
            if urls:
                st.write(f"**Links found:** {len(urls)} (see the Link Scanner page)")
            st.session_state.history.append(
                {"Message": text[:120], "Result": label, "Spam probability": round(prob, 3)}
            )

elif sidebar_option == "Link Scanner":
    st.title("🔗 Link Scanner")
    st.write("Paste a message or a list of links. Every link is checked for phishing signs.")

    text = st.text_area("Message or links", height=180)
    if st.button("Scan Links", type="primary"):
        urls = sm.extract_urls(text)
        if not urls:
            st.warning("No links found. Links must start with http://, https:// or www.")
        else:
            rows = []
            for u in urls:
                score, reasons = sm.analyze_url(u)
                rows.append({
                    "Link": u,
                    "Risk": sm.url_level(score),
                    "Score": score,
                    "Reasons": "; ".join(reasons) if reasons else "no warning signs",
                })
            result = pd.DataFrame(rows)
            high = int((result["Risk"] == "High risk").sum())
            if high:
                st.error(f"{high} high-risk link(s) found. Do not click them.")
            else:
                st.success("No high-risk links found.")
            st.dataframe(result)
            st.caption("This is a simple rule-based check, not a full safe-browsing database.")

elif sidebar_option == "Header Analyzer":
    st.title("🧾 Header Analyzer")
    st.write(
        "Paste the raw email headers (in Gmail: open the email, then "
        "⋮ → Show original) to check the sender."
    )

    raw = st.text_area("Raw headers", height=250)
    if st.button("Analyze Headers", type="primary"):
        if not raw.strip():
            st.warning("Please paste the headers first.")
        else:
            summary, warnings = sm.analyze_headers(raw)
            st.dataframe(pd.DataFrame({"Field": list(summary), "Value": [str(v) for v in summary.values()]}))
            if warnings:
                st.error(f"{len(warnings)} warning(s):")
                for w in warnings:
                    st.write("• " + w)
            else:
                st.success("No sender warnings found.")

elif sidebar_option == "Bulk CSV Check":
    st.title("📂 Bulk CSV Check")
    st.write("Upload a CSV with a text column to classify many emails at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="bulk")
    if uploaded is not None:
        df = read_csv(uploaded)
        if df is not None:
            column = st.selectbox("Column containing the email text", df.columns)
            if st.button("Classify All", type="primary"):
                texts = df[column].fillna("").astype(str)
                results = [score_text(t) for t in texts]
                out = df.copy()
                out["spam_probability"] = [round(r[1], 3) for r in results]
                out["prediction"] = [r[0] for r in results]
                spam_n = int((out["prediction"] == "spam").sum())
                st.success(f"Done. {spam_n} of {len(out)} messages flagged as spam.")
                st.dataframe(out.head(200))
                st.download_button(
                    "Download results",
                    out.to_csv(index=False).encode("utf-8"),
                    "spam_results.csv",
                    "text/csv",
                )

elif sidebar_option == "Keyword Rules":
    st.title("🛡️ Keyword Rules")
    st.write(
        "Blocklist words push a message toward spam. Allowlist words push it toward safe "
        "(for example your company name). Blocklist wins if both match."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Blocklist")
        new_block = st.text_input("Add blocked word or phrase", key="nb")
        if st.button("Add to blocklist") and new_block.strip():
            if new_block.strip().lower() not in st.session_state.blocklist:
                st.session_state.blocklist.append(new_block.strip().lower())
            st.rerun()
        for w in st.session_state.blocklist:
            st.write(f"🚫 {w}")
        if st.session_state.blocklist and st.button("Clear blocklist"):
            st.session_state.blocklist = []
            st.rerun()
    with col2:
        st.subheader("Allowlist")
        new_allow = st.text_input("Add trusted word or phrase", key="na")
        if st.button("Add to allowlist") and new_allow.strip():
            if new_allow.strip().lower() not in st.session_state.allowlist:
                st.session_state.allowlist.append(new_allow.strip().lower())
            st.rerun()
        for w in st.session_state.allowlist:
            st.write(f"✅ {w}")
        if st.session_state.allowlist and st.button("Clear allowlist"):
            st.session_state.allowlist = []
            st.rerun()

    st.caption("Rules last for the current session. Use the allowlist carefully: scammers can copy trusted words.")

elif sidebar_option == "Train Model":
    st.title("🧠 Train Model")
    st.write(
        "Upload a labeled CSV to train on your own data. "
        "Needs a text column and a label column containing `spam` / `ham`."
    )

    uploaded = st.file_uploader("Upload labeled CSV", type=["csv"], key="train")
    if uploaded is not None:
        df = read_csv(uploaded)
        if df is not None:
            st.dataframe(df.head(10))
            text_col = st.selectbox("Text column", df.columns, index=0)
            label_col = st.selectbox("Label column", df.columns, index=min(1, len(df.columns) - 1))
            test_size = st.slider("Test size (%)", 10, 40, 25) / 100

            if st.button("Train", type="primary"):
                data = pd.DataFrame({"text": df[text_col], "label": df[label_col]})
                try:
                    pipe, metrics = sm.train(data, test_size)
                    set_model(pipe, metrics, uploaded.name)
                    st.session_state.training_df = data
                    st.success(f"Model trained. Accuracy: **{metrics['accuracy']:.2%}**")
                    st.dataframe(metrics["confusion"])
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if st.button("Reset to built-in model"):
        pipe, metrics = load_default_model()
        set_model(pipe, metrics, "Built-in sample data")
        st.session_state.training_df = sm.builtin_dataframe()
        st.rerun()

elif sidebar_option == "Model Comparison":
    st.title("⚖️ Model Comparison")
    st.write(
        f"Compare four algorithms on the current training data "
        f"({len(st.session_state.training_df)} rows, from {st.session_state.model_source})."
    )

    test_size = st.slider("Test size (%)", 10, 40, 25, key="cmp") / 100
    if st.button("Compare Models", type="primary"):
        try:
            table = sm.compare_models(st.session_state.training_df, test_size)
            st.dataframe(table)
            st.bar_chart(table.set_index("Model")[["Accuracy", "F1"]])
            best = table.iloc[0]["Model"]
            st.success(f"Best F1 score: **{best}**")
        except Exception as e:
            st.error(f"Comparison failed: {e}")

elif sidebar_option == "Analytics":
    st.title("📈 Analytics")
    st.write("Charts from your checks this session and what the model learned.")

    hist = st.session_state.history
    if hist:
        hist_df = pd.DataFrame(hist)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Spam vs Safe")
            st.bar_chart(hist_df["Result"].value_counts())
        with c2:
            st.subheader("Spam probability per check")
            st.line_chart(hist_df["Spam probability"])
    else:
        st.info("Check some emails first to see session charts.")

    st.subheader("Words the model links to spam / safe")
    spam_w, ham_w = sm.top_words(st.session_state.pipe)
    if spam_w is None:
        st.info("Top words are only available for the linear model.")
    else:
        c1, c2 = st.columns(2)
        c1.write("Top spam words")
        c1.bar_chart(spam_w)
        c2.write("Top safe words")
        c2.bar_chart(ham_w)

elif sidebar_option == "History":
    st.title("🗂️ History")
    st.write("Emails you checked in this session.")

    if not st.session_state.history:
        st.info("Nothing checked yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        filt = st.selectbox("Show", ["All", "spam", "ham"])
        shown = hist_df if filt == "All" else hist_df[hist_df["Result"] == filt]
        st.dataframe(shown)
        st.download_button(
            "Download history",
            hist_df.to_csv(index=False).encode("utf-8"),
            "spam_history.csv",
            "text/csv",
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

elif sidebar_option == "Settings":
    st.title("⚙️ Settings")
    st.write("Tune how strict the detector is.")

    st.session_state.threshold = st.slider(
        "Spam threshold (messages at or above this probability are marked spam)",
        0.10, 0.90, float(st.session_state.threshold), 0.05,
    )
    st.caption("Lower = catches more spam but may flag safe mail. Higher = fewer false alarms.")

    if st.button("Reset all session data"):
        for key in ["history", "blocklist", "allowlist"]:
            st.session_state[key] = []
        st.session_state.threshold = 0.5
        st.rerun()
