from fastapi import FastAPI
from app.preprocessing import build_customer_features
import pandas as pd
import xml.etree.ElementTree as ET

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Casino AI Optimization API is running."}

@app.get("/customers")
def read_customers():
    return {"message": "Customer endpoint is active!"}

@app.get("/features")
def generate_features():
    try:
        slot_path = "data/final_slot_log_100.xml"
        tito_path = "data/final_tito_log_matched.csv"
        crm_path = "data/crm_profiles_100.json"
        heatmap_path = "data/heatmap_100_customers.csv"

        features_df = build_customer_features(
            slot_path=slot_path,
            tito_path=tito_path,
            crm_path=crm_path,
            heatmap_path=heatmap_path
        )

        if features_df is None or len(features_df) == 0:
            return {"error": "No features generated. Check if data files are valid and match."}

        return {
            "total_customers": len(features_df),
            "columns": features_df.columns.tolist(),
            "sample": features_df.head(5).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/debug")
def debug_merge():
    try:
        slot_tree = ET.parse("data/final_slot_log_100.xml")
        slot_root = slot_tree.getroot()
        slot_sessions = []
        for session in slot_root.findall("SlotSession"):
            slot_sessions.append({
                "session_id": session.findtext("SessionID"),
                "customer_id": session.findtext("CustomerID"),
            })
        slot_df = pd.DataFrame(slot_sessions)
        tito_df = pd.read_csv("data/final_tito_log_matched.csv")

        return {
            "slot_columns": slot_df.columns.tolist(),
            "tito_columns": tito_df.columns.tolist(),
            "slot_sample": slot_df.head(1).to_dict(orient="records"),
            "tito_sample": tito_df.head(1).to_dict(orient="records"),
            "matching_session_ids": len(pd.merge(slot_df, tito_df, on=["session_id", "customer_id"]))
        }

    except Exception as e:
        return {"error": str(e)}
