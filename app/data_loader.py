"""
Data Loader Module - Supports both file-based and database loading
"""

import pandas as pd
import xml.etree.ElementTree as ET
import json
from typing import Dict, Optional
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

# Import pyodbc only when needed
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def load_slot_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def load_tito_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def load_crm_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def load_heatmap_data(self) -> pd.DataFrame:
        pass

class FileDataSource(DataSource):
    """Loads data from local files"""
    
    def __init__(self, customer_count: int = 1500):
        self.customer_count = customer_count
        self.base_path = "../data"
    
    def load_slot_data(self) -> pd.DataFrame:
        """Load slot sessions from XML"""
        slot_path = f"{self.base_path}/final_slot_log_{self.customer_count}.xml"
        slot_tree = ET.parse(slot_path)
        slot_root = slot_tree.getroot()
        
        slot_sessions = []
        for session in slot_root.findall("SlotSession"):
            slot_sessions.append({
                "customer_id": session.findtext("Customerid"),
                "session_id": session.findtext("Sessionid"),
                "machine_id": session.findtext("Machineid"),
                "rtp": float(session.findtext("Rtp") or 90.0),
                "game_type": session.findtext("Gametype"),
                "symbols": session.findtext("Symbols"),
                "bet_amount": float(session.findtext("Betamount") or 50.0),
                "win_amount": float(session.findtext("Winamount") or 45.0),
                "session_duration": int(session.findtext("Sessionduration") or 15),
                "timestamp": pd.to_datetime(session.findtext("Timestamp"))
            })
        
        return pd.DataFrame(slot_sessions)
    
    def load_tito_data(self) -> pd.DataFrame:
        """Load TITO data from CSV"""
        tito_path = f"{self.base_path}/final_tito_log_matched_{self.customer_count}.csv"
        return pd.read_csv(tito_path, parse_dates=["timestamp"])
    
    def load_crm_data(self) -> pd.DataFrame:
        """Load CRM profiles from JSON"""
        crm_path = f"{self.base_path}/crm_profiles_{self.customer_count}.json"
        with open(crm_path, "r", encoding="utf-8") as f:
            crm_profiles = json.load(f)
        return pd.DataFrame(crm_profiles)
    
    def load_heatmap_data(self) -> pd.DataFrame:
        """Load heatmap data from CSV"""
        heatmap_path = f"{self.base_path}/heatmap_{self.customer_count}_customers.csv"
        return pd.read_csv(heatmap_path, parse_dates=["timestamp"])

class DatabaseDataSource(DataSource):
    """Loads data from MSSQL database"""
    
    def __init__(self, connection_string: str, months_back: int = 6):
        if not PYODBC_AVAILABLE:
            raise ImportError("pyodbc is required for database connections. Install with: pip install pyodbc")
        
        self.connection_string = connection_string
        self.months_back = months_back
        self.start_date = datetime.now() - timedelta(days=months_back*30)
    
    def _get_connection(self):
        """Create database connection"""
        return pyodbc.connect(self.connection_string)
    
    def load_slot_data(self) -> pd.DataFrame:
        """Load slot sessions from database"""
        query = f"""
        SELECT 
            CustomerID as customer_id,
            SessionID as session_id,
            MachineID as machine_id,
            RTP as rtp,
            GameType as game_type,
            Symbols as symbols,
            BetAmount as bet_amount,
            WinAmount as win_amount,
            SessionDuration as session_duration,
            Timestamp as timestamp
        FROM SlotSessions
        WHERE Timestamp >= ?
        """
        
        with self._get_connection() as conn:
            return pd.read_sql(query, conn, params=[self.start_date])
    
    def load_tito_data(self) -> pd.DataFrame:
        """Load TITO data from database"""
        query = f"""
        SELECT 
            customer_id,
            session_id,
            machine_id,
            ticket_in,
            ticket_out,
            session_loss,
            jackpot_contribution,
            timestamp
        FROM TitoLogs
        WHERE timestamp >= ?
        """
        
        with self._get_connection() as conn:
            return pd.read_sql(query, conn, params=[self.start_date])
    
    def load_crm_data(self) -> pd.DataFrame:
        """Load CRM profiles from database"""
        query = """
        SELECT 
            customer_id,
            age,
            gender,
            nationality
        FROM CustomerProfiles
        WHERE is_active = 1
        """
        
        with self._get_connection() as conn:
            return pd.read_sql(query, conn)
    
    def load_heatmap_data(self) -> pd.DataFrame:
        """Load heatmap data from database"""
        query = f"""
        SELECT 
            machine_id,
            zone,
            usage_level,
            timestamp
        FROM MachineUsageHeatmap
        WHERE timestamp >= ?
        """
        
        with self._get_connection() as conn:
            return pd.read_sql(query, conn, params=[self.start_date])

class DataLoader:
    """Main data loader that can switch between sources"""
    
    def __init__(self, source: str = "file", **kwargs):
        """
        Initialize data loader
        
        Parameters:
        - source: "file" or "database"
        - For file source: customer_count (default 1500)
        - For database source: connection_string, months_back (default 6)
        """
        if source == "file":
            self.data_source = FileDataSource(kwargs.get('customer_count', 1500))
        elif source == "database":
            self.data_source = DatabaseDataSource(
                kwargs.get('connection_string'),
                kwargs.get('months_back', 6)
            )
        else:
            raise ValueError("Source must be 'file' or 'database'")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data sources"""
        return {
            'slot': self.data_source.load_slot_data(),
            'tito': self.data_source.load_tito_data(),
            'crm': self.data_source.load_crm_data(),
            'heatmap': self.data_source.load_heatmap_data()
        }
    
    def build_features(self) -> pd.DataFrame:
        """Build customer features from all data sources"""
        # Load all data
        data = self.load_all_data()
        
        # Merge slot + TITO
        slot_tito_df = pd.merge(
            data['slot'], 
            data['tito'], 
            on=["session_id", "customer_id"], 
            suffixes=("", "_tito")
        )
        
        # Merge CRM
        full_df = pd.merge(slot_tito_df, data['crm'], on="customer_id")
        
        # Merge Heatmap by timestamp window
        data['heatmap']['merge_key'] = (
            data['heatmap']['machine_id'] + 
            data['heatmap']['timestamp'].dt.floor('30min').astype(str)
        )
        full_df['merge_key'] = (
            full_df['machine_id'] + 
            full_df['timestamp'].dt.floor('30min').astype(str)
        )
        full_df = pd.merge(
            full_df, 
            data['heatmap'][['merge_key', 'zone', 'usage_level']], 
            on='merge_key', 
            how='left'
        )
        
        # Feature engineering
        full_df['usage_level'] = full_df['usage_level'].fillna(0.5)
        
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
            'zone': lambda x: x.nunique() if len(x) > 0 else 1
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
        
        return features_df.dropna()

# Example usage
if __name__ == "__main__":
    # File-based loading (current)
    file_loader = DataLoader(source="file", customer_count=1500)
    features_df = file_loader.build_features()
    print(f"Loaded {len(features_df)} customers from files")
    
    # Database loading (future) - example
    # db_connection_string = "DRIVER={SQL Server};SERVER=casino-db;DATABASE=CasinoData;UID=user;PWD=pass"
    # db_loader = DataLoader(source="database", connection_string=db_connection_string, months_back=6)
    # features_df = db_loader.build_features()
    # print(f"Loaded {len(features_df)} customers from database")