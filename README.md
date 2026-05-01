# AI SIEM Triage System

## Overview

This project is a simple SIEM-style alert triage system built using machine learning.
It takes network log data, classifies whether an alert is suspicious or not, and provides basic reasoning for the decision.

The goal of this project was to understand how SIEM systems work and how ML can be used to reduce manual analysis.

---

## Features

* Classifies alerts as high-risk or normal using a trained model
* Basic explanation of why an alert is flagged
* Groups similar alerts to identify patterns
* Simple interface to view logs and results
* Optional LLM-based analysis for generating reports

---

## Tech Stack

* Python
* scikit-learn (Random Forest)
* FastAPI (for backend APIs)
* Streamlit (for dashboard)
* SQLite (for storing logs)

---

## Project Structure

* `app.py` – main dashboard
* `api_server.py` – backend API for log ingestion
* `triage.py` – model training and prediction
* `log_generator.py` – generates sample logs
* `llm_helper.py` – handles LLM-based analysis
* `expected_features.json` – feature schema
* `live_logs.json` – sample logs

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/FAYZAL-69/AI-SIEM-TRIAGE.git
   cd AI-SIEM-TRIAGE
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

4. (Optional) Run backend API:

   ```bash
   uvicorn api_server:app --reload
   ```

---

## Notes

* The trained model file is not included due to size.
* You can retrain the model using `triage.py`.
* This project is for learning purposes and not production-ready.

---

## What I Learned

* Basics of SIEM and alert triage
* Using machine learning for classification
* Building simple APIs with FastAPI
* Working with real-time style data
* Integrating ML with a dashboard

---

## Future Improvements

* Improve model accuracy
* Add better visualization
* Handle larger datasets
* Integrate with real log sources

---

## Author

FAISAL
