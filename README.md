# Yavuzhan Canli - 1905bash


# AI Casino Optimization System

This project is a modular AI-based backend designed to optimize customer engagement, promotion decisions, and slot machine management in physical casino environments.

---

## Features

- K-Means based customer segmentation (A–E)
- Random Forest model for promotion response prediction
- Rule Engine for RTP adjustment and machine relocation
- Full AI analysis output in CSV
- REST API built with FastAPI
- Streamlit dashboard for customer interaction demo

---

### Project Structure

```
casino-ai-system/
├── app/
│   ├── main.py              # FastAPI app
│   ├── preprocessing.py     # Feature extraction logic
│   ├── rule_engine.py       # RTP & zone adjustment logic
│   ├── schemas.py           # API input/output models
│   └── utils.py             # Helper functions
├── data/
│   └── final_ai_output_dataset.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

#### Installation

```bash
git clone https://github.com/YOUR_USERNAME/casino-ai-system.git
cd casino-ai-system
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

##### Run the API

```bash
uvicorn app.main:app --reload
```

Then open your browser to: [http://localhost:8000/docs](http://localhost:8000/docs)

---

####### License

MIT License - see LICENSE file for details.

###### 21 may 2025
## Version History and Progress
## Current Version: `v0.3.0`

---

## v0.3.1 – Feature Engineering & Data Merging

This version includes:
- `build_customer_features()` created in `app/preprocessing.py`
- Reads slot, tito, CRM, and heatmap data
- Outputs a customer-level dataset for training AI models
- `/features` endpoint generates the dataset live
- `/debug` endpoint confirms session-level data integrity

Next: `train_kmeans_model()` → customer segmentation
