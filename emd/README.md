# Spam Email Detector (Streamlit)

Sidebar-step Streamlit app. No database and no external files, so it deploys cleanly.

## Sidebar pages
Dashboard Overview, Check Email, Link Scanner, Header Analyzer, Bulk CSV Check,
Keyword Rules, Train Model, Model Comparison, Analytics, History, Settings

## Files (keep all in the repo root)
app.py, spam_model.py, sample_data.py, requirements.txt

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
Push the files to GitHub, set Main file path to app.py. If it fails, open
"Manage app" and read the logs.
