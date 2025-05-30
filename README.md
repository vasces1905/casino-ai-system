# Yavuzhan Canli


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

###### 28 may 2025
# Casino AI Optimization System

This project builds a full pipeline for casino player segmentation and promotion targeting using AI algorithms.

## Version: v1.0 - KMeans Segmentation Completed

## Project Structure

- **Data Sources:** Slot logs (XML), TITO logs (CSV), CRM profiles (JSON), heatmap logs (CSV)
- **Preprocessing Module (`preprocessing.py`)**: Combines and processes all sources into a single feature dataset per customer
- **Model Training (`train_model.py`)**: Trains a KMeans clustering model to segment customers into 3 behavioral segments
- **Segment Labels:** Casual Gambler, Regular Player, High Roller
- **Output:** A labeled dataset is saved to `data/labeled_customer_dataset.csv`
- **Visualization:** Segment distribution and cluster results can be visualized using bar/pie plots before proceeding to supervised modeling

## Workflow Overview

| Step                | Completed? | Description                                                                 |
|---------------------|------------|-----------------------------------------------------------------------------|
| Data Ingestion      | ✅          | XML (Slot), CSV (TITO), JSON (CRM), CSV (Heatmap) files were successfully read |
| Data Merging        | ✅          | Merged into one row per customer using `build_customer_features()`         |
| Feature Engineering | ✅          | Behavioral features like `avg_loss`, `zone_diversity`, `jackpot_total` were generated |
| Scaling             | ✅          | All numeric features were normalized using `StandardScaler`                |
| K-Means Training    | ✅          | Model trained with `n_clusters=3`                                          |
| Segment Assignment  | ✅          | Model outputs (0 / 1 / 2) were mapped to Casual / Regular / High Roller     |
| CSV Export          | ✅          | Output saved as `data/labeled_customer_dataset.csv`                         |
| Visualization       | ✅          | Cluster membership can be visualized to validate segmentation distribution |
| Random Forest Prep  | ⏳          | Synthetic target to be generated and RF model training to begin             |

---
#### 29/05/2025
## Version: v1.1 - Modular KMeans Training Function 
### Update Summary:
- `train_kmeans.py` refactored into a modular, callable structure
- Now compatible with FastAPI endpoints, automation scripts, and Jupyter pipelines
- External modules (like Streamlit, API, or cron jobs) can call:
  ```python
  from train_kmeans import train_kmeans_model, run_training_pipeline
  run_training_pipeline()

## Upcoming: Version v2.0 - Random Forest Module

### Objective:
Predict customer responsiveness to promotions using supervised learning.

### Key Features:

1. **Synthetic Target Generation**
   - Business logic-based: customer value, diversity, usage patterns
   - Segment-aware probabilities
   - Classes: Low, Medium, High response

2. **Advanced Feature Engineering**
   - Engagement: engagement_intensity, loyalty_score
   - Financial: loss_per_session, win_loss_ratio
   - Risk: risk_appetite derived from RTP
   - Demographics: age buckets, categorical encodings

3. **Model Training and Optimization**
   - Classifier: `RandomForestClassifier`
   - Hyperparameter tuning via `GridSearchCV`
   - Stratified 5-fold cross-validation
   - Optional scaling depending on model behavior

4. **Comprehensive Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Confusion matrix and cross-validation statistics
   - Feature importance interpretation

5. **Business Intelligence Output**
   - Segment-based recommendation guidance
   - Promotion strategy alignment
   - Insights for ROI estimation

---

Run `python -m app.models.train_model` to reproduce the clustering results.
Random Forest development will continue in `models/train_random_forest.py` as of version v2.0.

<<<<<<< Updated upstream
<<<<<<< HEAD
<<<<<<< HEAD
=======

