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

