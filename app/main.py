from fastapi import FastAPI
from app.preprocessing import build_customer_features
import pandas as pd

app = FastAPI()

# Home check
@app.get("/")
def read_root():
    return {"message": "Casino AI Optimization API is running."}

# Test customer endpoint
@app.get("/customers")
def read_customers():
    return {"message": "Customer endpoint is active!"}

# Feature engineering endpoint
@app.get("/features")
def generate_features():
    try:
        # Define data paths (relative to your /data folder)
        slot_path = "data/final_slot_log_100.xml"
        tito_path = "data/final_tito_log_matched.csv"
        crm_path = "data/crm_profiles_100.json"
        heatmap_path = "data/heatmap_100_customers.csv"

        # Build features
        features_df = build_customer_features(
            slot_path=slot_path,
            tito_path=tito_path,
            crm_path=crm_path,
            heatmap_path=heatmap_path
        )

        # Return sample of output
        return {
            "total_customers": len(features_df),
            "columns": features_df.columns.tolist(),
            "sample": features_df.head(5).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}

    
    
    

