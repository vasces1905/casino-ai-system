import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import random
import os

class RawDataExpander:
    """Expands raw casino data files maintaining original formats"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def load_existing_data(self):
        """Load existing data to understand patterns"""
        print("Loading existing data for pattern analysis...")
        
        # Load slot data from XML
        slot_tree = ET.parse("../../data/final_slot_log_100.xml")
        slot_root = slot_tree.getroot()
        self.slot_sessions = []
        for session in slot_root.findall("SlotSession"):
            self.slot_sessions.append({
                "customer_id": session.findtext("CustomerID"),
                "session_id": session.findtext("SessionID"),
                "machine_id": session.findtext("MachineID"),
                "rtp": float(session.findtext("RTP")),
                "game_type": session.findtext("GameType"),
                "symbols": session.findtext("Symbols"),
                "bet_amount": float(session.findtext("BetAmount")),
                "win_amount": float(session.findtext("WinAmount")),
                "session_duration": int(session.findtext("SessionDuration")),
                "timestamp": session.findtext("Timestamp")
            })
        
        # Load TITO data
        self.tito_df = pd.read_csv("../../data/final_tito_log_matched.csv")
        
        # Load CRM data
        with open("../../data/crm_profiles_100.json", "r", encoding="utf-8") as f:
            self.crm_profiles = json.load(f)
            
        # Load heatmap data
        self.heatmap_df = pd.read_csv("../../data/heatmap_100_customers.csv")
        
        print(f"Loaded {len(self.slot_sessions)} slot sessions")
        print(f"Loaded {len(self.tito_df)} TITO records")
        print(f"Loaded {len(self.crm_profiles)} CRM profiles")
        print(f"Loaded {len(self.heatmap_df)} heatmap records")
        
    def generate_slot_sessions(self, num_customers=1500):
        """Generate expanded slot session data"""
        print(f"\nGenerating slot sessions for {num_customers} customers...")
        
        # Get patterns from existing data
        game_types = list(set([s['game_type'] for s in self.slot_sessions]))
        symbols_patterns = list(set([s['symbols'] for s in self.slot_sessions]))
        
        # Statistics from existing data
        rtp_stats = {'mean': 91.5, 'std': 3.0, 'min': 85, 'max': 96}
        bet_stats = {'mean': 50, 'std': 20, 'min': 10, 'max': 100}
        duration_stats = {'mean': 15, 'std': 5, 'min': 5, 'max': 30}
        
        new_sessions = []
        session_counter = 10000  # Start from a high number to avoid conflicts
        
        # Base timestamp
        base_date = datetime(2024, 1, 1)
        
        for customer_num in range(100, num_customers):
            customer_id = f"CUST_{customer_num:04d}"
            
            # Each customer has 3-15 sessions
            num_sessions = np.random.randint(3, 16)
            
            for session_num in range(num_sessions):
                session_counter += 1
                
                # Generate session timestamp
                days_offset = np.random.randint(0, 90)
                hours_offset = np.random.randint(0, 24)
                minutes_offset = np.random.randint(0, 60)
                timestamp = base_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
                
                # Generate session data
                rtp = np.clip(np.random.normal(rtp_stats['mean'], rtp_stats['std']), 
                             rtp_stats['min'], rtp_stats['max'])
                bet_amount = np.clip(np.random.normal(bet_stats['mean'], bet_stats['std']), 
                                   bet_stats['min'], bet_stats['max'])
                
                # Win amount based on RTP
                win_ratio = (rtp / 100) * np.random.uniform(0.5, 1.5)
                win_amount = bet_amount * win_ratio
                
                session = {
                    "customer_id": customer_id,
                    "session_id": f"SES_{session_counter:08d}",
                    "machine_id": f"SLOT_{np.random.randint(1, 51):03d}",
                    "rtp": round(rtp, 2),
                    "game_type": np.random.choice(game_types),
                    "symbols": np.random.choice(symbols_patterns),
                    "bet_amount": round(bet_amount, 2),
                    "win_amount": round(win_amount, 2),
                    "session_duration": np.random.randint(duration_stats['min'], duration_stats['max']),
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                new_sessions.append(session)
        
        # Combine with existing sessions
        all_sessions = self.slot_sessions + new_sessions
        
        # Create XML
        root = ET.Element("SlotLog")
        for session in all_sessions:
            session_elem = ET.SubElement(root, "SlotSession")
            for key, value in session.items():
                elem = ET.SubElement(session_elem, key.replace("_", "").title().replace("Id", "ID"))
                elem.text = str(value)
        
        # Save XML
        tree = ET.ElementTree(root)
        output_file = f"../../data/final_slot_log_{num_customers}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"Saved: {output_file}")
        
        return new_sessions
    
    def generate_tito_data(self, slot_sessions, num_customers=1500):
        """Generate TITO data matching slot sessions"""
        print(f"\nGenerating TITO data for {num_customers} customers...")
        
        # Group sessions by customer and session_id
        session_map = {}
        for session in slot_sessions:
            key = (session['customer_id'], session['session_id'])
            session_map[key] = session
        
        tito_records = []
        
        # Keep existing TITO records
        for _, row in self.tito_df.iterrows():
            tito_records.append(row.to_dict())
        
        # Generate new TITO records
        for (customer_id, session_id), session in session_map.items():
            if customer_id.startswith("CUST_") and int(customer_id.split("_")[1]) >= 100:
                ticket_in = session['bet_amount'] * np.random.uniform(10, 20)
                win_loss_ratio = session['win_amount'] / session['bet_amount']
                ticket_out = ticket_in * win_loss_ratio * np.random.uniform(0.8, 1.2)
                session_loss = max(0, ticket_in - ticket_out)
                jackpot_contribution = ticket_in * np.random.uniform(0.01, 0.05)
                
                tito_record = {
                    'customer_id': customer_id,
                    'session_id': session_id,
                    'ticket_in': round(ticket_in, 2),
                    'ticket_out': round(ticket_out, 2),
                    'session_loss': round(session_loss, 2),
                    'jackpot_contribution': round(jackpot_contribution, 2),
                    'timestamp': session['timestamp']
                }
                
                tito_records.append(tito_record)
        
        # Create DataFrame and save
        tito_df = pd.DataFrame(tito_records)
        output_file = f"../../data/final_tito_log_matched_{num_customers}.csv"
        tito_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        
        return tito_df
    
    def generate_crm_profiles(self, num_customers=1500):
        """Generate CRM profiles for customers"""
        print(f"\nGenerating CRM profiles for {num_customers} customers...")
        
        # Get patterns from existing data
        nationalities = [p['nationality'] for p in self.crm_profiles]
        nationality_dist = pd.Series(nationalities).value_counts(normalize=True)
        
        all_profiles = []
        
        # Keep existing profiles
        all_profiles.extend(self.crm_profiles)
        
        # Generate new profiles
        for customer_num in range(100, num_customers):
            customer_id = f"CUST_{customer_num:04d}"
            
            profile = {
                "customer_id": customer_id,
                "age": int(np.random.normal(35, 10)),
                "gender": np.random.choice(["Male", "Female"], p=[0.6, 0.4]),
                "nationality": np.random.choice(nationality_dist.index, p=nationality_dist.values),
                "vip_level": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], 
                                            p=[0.4, 0.3, 0.2, 0.1]),
                "registration_date": (datetime.now() - timedelta(days=np.random.randint(30, 730))).strftime("%Y-%m-%d"),
                "communication_preference": np.random.choice(["email", "sms", "app", "none"], 
                                                           p=[0.4, 0.2, 0.3, 0.1])
            }
            
            # Age boundaries
            profile['age'] = max(21, min(75, profile['age']))
            
            all_profiles.append(profile)
        
        # Save JSON
        output_file = f"../../data/crm_profiles_{num_customers}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_profiles, f, indent=4, ensure_ascii=False)
        print(f"Saved: {output_file}")
        
        return all_profiles
    
    def generate_heatmap_data(self, slot_sessions, num_customers=1500):
        """Generate heatmap data based on slot sessions"""
        print(f"\nGenerating heatmap data for {num_customers} customers...")
        
        # Extract unique machine_ids and timestamps
        heatmap_records = []
        
        # Keep existing heatmap records
        for _, row in self.heatmap_df.iterrows():
            heatmap_records.append(row.to_dict())
        
        # Generate new heatmap records
        zones = ['A', 'B', 'C', 'D', 'E']
        
        for session in slot_sessions:
            if session['customer_id'].startswith("CUST_") and int(session['customer_id'].split("_")[1]) >= 100:
                # Round timestamp to nearest 30 minutes
                timestamp = pd.to_datetime(session['timestamp'])
                rounded_timestamp = timestamp.floor('30min')
                
                record = {
                    'machine_id': session['machine_id'],
                    'timestamp': rounded_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'zone': np.random.choice(zones),
                    'usage_level': round(np.random.beta(2, 5), 3)  # Skewed towards lower values
                }
                
                heatmap_records.append(record)
        
        # Remove duplicates and save
        heatmap_df = pd.DataFrame(heatmap_records)
        heatmap_df = heatmap_df.drop_duplicates(subset=['machine_id', 'timestamp'])
        
        output_file = f"../../data/heatmap_{num_customers}_customers.csv"
        heatmap_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        
        return heatmap_df

def main():
    """Main function to expand all raw data files"""
    print("="*60)
    print("RAW DATA EXPANSION PROCESS")
    print("="*60)
    
    expander = RawDataExpander()
    
    # Load existing data
    expander.load_existing_data()
    
    # Generate data for different sizes
    for num_customers in [1500, 2000]:
        print(f"\n{'='*60}")
        print(f"Generating data for {num_customers} customers...")
        print("="*60)
        
        # Generate all data types
        slot_sessions = expander.generate_slot_sessions(num_customers)
        tito_data = expander.generate_tito_data(slot_sessions, num_customers)
        crm_profiles = expander.generate_crm_profiles(num_customers)
        heatmap_data = expander.generate_heatmap_data(slot_sessions, num_customers)
        
        print(f"\nSummary for {num_customers} customers:")
        print(f"- Slot sessions: {len(slot_sessions)} records")
        print(f"- TITO records: {len(tito_data)} records")
        print(f"- CRM profiles: {len(crm_profiles)} profiles")
        print(f"- Heatmap records: {len(heatmap_data)} records")
    
    print("\n" + "="*60)
    print("RAW DATA EXPANSION COMPLETE!")
    print("Files created in data/ folder:")
    print("- final_slot_log_1500.xml & final_slot_log_2000.xml")
    print("- final_tito_log_matched_1500.csv & final_tito_log_matched_2000.csv")
    print("- crm_profiles_1500.json & crm_profiles_2000.json")
    print("- heatmap_1500_customers.csv & heatmap_2000_customers.csv")
    print("="*60)

if __name__ == "__main__":
    main()