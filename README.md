
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
