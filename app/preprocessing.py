import pandas as pd
import xml.etree.ElementTree as ET
import json
from typing import Literal

def build_customer_features(
    slot_path: str,
    tito_path: str,
    crm_path: str,
    heatmap_path: str,
    merge_method: Literal["strict", "loose"] = "loose"
) -> pd.DataFrame:
    """
    Generates a customer-level feature dataset from raw casino data sources.

    Parameters:
    - slot_path (str): Path to XML file with slot sessions.
    - tito_path (str): Path to CSV file with TITO logs.
    - crm_path (str): Path to JSON file with CRM customer profiles.
    - heatmap_path (str): Path to CSV file with machine usage heatmap.
    - merge_method (Literal): 'strict' matches exact timestamp, 'loose' uses ±30 min window.

    Returns:
    - features_df (pd.DataFrame): One row per customer with all engineered features.
    """

    # 1. Load Slot Log (XML)
    slot_tree = ET.parse(slot_path)
    slot_root = slot_tree.getroot()
    slot_sessions = []
    for session in slot_root.findall("SlotSession"):
        slot_sessions.append({
            "customer_id": session.findtext("CustomerID"),
            "session_id": session.findtext("SessionID"),
            "machine_id": session.findtext("MachineID"),
            "rtp": float(session.findtext("RTP")),
            "game_type": session.findtext("GameType"),
            "symbols": session.findtext("Symbols"),
            "bet_amount": float(session.findtext("BetAmount")),
            "win_amount": float(session.findtext("WinAmount")),
            "session_duration": int(session.findtext("SessionDuration")),
            "timestamp": pd.to_datetime(session.findtext("Timestamp"))
        })

    slot_df = pd.DataFrame(slot_sessions)

    # 2. Load TITO Log (CSV)
    tito_df = pd.read_csv(tito_path, parse_dates=["timestamp"])

    # 3. Load CRM Profiles (JSON)
    with open(crm_path, "r", encoding="utf-8") as f:
        crm_profiles = json.load(f)
    crm_df = pd.DataFrame(crm_profiles)

    # 4. Load Heatmap (CSV)
    heatmap_df = pd.read_csv(heatmap_path, parse_dates=["timestamp"])

    # 5. Merge Slot + TITO
    slot_tito_df = pd.merge(slot_df, tito_df, on=["session_id", "customer_id"], suffixes=("", "_tito"))

    # 6. Merge CRM
    full_df = pd.merge(slot_tito_df, crm_df, on="customer_id")

    # 7. Merge Heatmap by timestamp window
    heatmap_df['merge_key'] = heatmap_df['machine_id'] + heatmap_df['timestamp'].dt.floor('30min').astype(str)
    full_df['merge_key'] = full_df['machine_id'] + full_df['timestamp'].dt.floor('30min').astype(str)
    full_df = pd.merge(full_df, heatmap_df[['merge_key', 'zone', 'usage_level']], on='merge_key', how='left')

    # 8. Feature Engineering
    features_df = full_df.groupby('customer_id').agg({
        'age': 'first',
        'gender': 'first',
        'nationality': 'first',
        'game_type': pd.Series.nunique,
        'rtp': 'mean',
        'bet_amount': 'mean',
        'win_amount': 'mean',
        'session_duration': 'mean',
        'ticket_in': 'sum',
        'ticket_out': 'sum',
        'session_loss': 'mean',
        'jackpot_contribution': 'sum',
        'usage_level': 'mean',
        'zone': pd.Series.nunique
    }).rename(columns={
        'game_type': 'game_type_diversity',
        'rtp': 'avg_rtp',
        'bet_amount': 'avg_bet',
        'win_amount': 'avg_win',
        'session_duration': 'avg_session_duration',
        'session_loss': 'avg_loss',
        'jackpot_contribution': 'jackpot_total',
        'usage_level': 'avg_usage_level',
        'zone': 'zone_diversity'
    }).reset_index()
    
    # Prevents errors during the .fit() operation of the K-Means model.
    features_df = features_df.dropna()

    return features_df
