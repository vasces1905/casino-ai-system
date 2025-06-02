import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_promo_response_data(segmented_data_path: str, output_path: str):
    """
    Generates realistic promotional response data for each customer segment
    """
    
    print("=== PROMO RESPONSE DATA GENERATOR ===")
    
    # Load segmented customer data
    df = pd.read_csv(segmented_data_path)
    print(f"Loaded {len(df)} customers from {segmented_data_path}")
    
    # Segment-specific response rates (realistic casino industry rates)
    segment_response_rates = {
        "High Roller": {
            "base_rate": 0.30,  # 30% base response rate
            "variance": 0.10,   # ±10% variation
            "factors": {
                "avg_loss": 0.0002,     # Higher loss = slightly higher response
                "zone_diversity": 0.02,  # More diverse = higher response
                "avg_session_duration": 0.001
            }
        },
        "Regular Player": {
            "base_rate": 0.20,  # 20% base response rate
            "variance": 0.08,
            "factors": {
                "avg_loss": 0.0003,
                "zone_diversity": 0.025,
                "avg_session_duration": 0.0015
            }
        },
        "Casual Gambler": {
            "base_rate": 0.40,  # 40% base response rate (price sensitive!)
            "variance": 0.12,
            "factors": {
                "avg_loss": 0.0001,     # Less sensitive to loss amount
                "zone_diversity": 0.015,
                "avg_session_duration": 0.002
            }
        }
    }
    
    # Promo types with different characteristics
    promo_types = {
        "Free_Play": {"multiplier": 1.0, "description": "Free slot credits"},
        "Cashback": {"multiplier": 0.8, "description": "Percentage of losses returned"},
        "Bonus_Match": {"multiplier": 1.2, "description": "Match deposit bonus"},
        "VIP_Event": {"multiplier": 1.5, "description": "Exclusive event invitation"},
        "Dining_Comp": {"multiplier": 0.9, "description": "Free meal vouchers"}
    }
    
    # Generate promotional campaigns for each customer
    promo_data = []
    
    print("\nGenerating promotional response data...")
    
    for idx, customer in df.iterrows():
        segment = customer['segment_name']
        customer_id = customer['customer_id']
        
        # Generate 3-5 promotional campaigns per customer over last 6 months
        num_promos = np.random.randint(3, 6)
        
        for promo_idx in range(num_promos):
            # Random promo type
            promo_type = np.random.choice(list(promo_types.keys()))
            promo_info = promo_types[promo_type]
            
            # Calculate response probability based on segment and customer characteristics
            base_rate = segment_response_rates[segment]["base_rate"]
            variance = segment_response_rates[segment]["variance"]
            factors = segment_response_rates[segment]["factors"]
            
            # Apply customer-specific factors
            response_prob = base_rate
            response_prob += factors["avg_loss"] * customer["avg_loss"]
            response_prob += factors["zone_diversity"] * customer["zone_diversity"]
            response_prob += factors["avg_session_duration"] * customer["avg_session_duration"]
            
            # Apply promo type multiplier
            response_prob *= promo_info["multiplier"]
            
            # Add random variance
            response_prob += np.random.normal(0, variance)
            
            # Ensure probability is between 0 and 1
            response_prob = max(0, min(1, response_prob))
            
            # Determine actual response (1 or 0)
            actual_response = 1 if np.random.random() < response_prob else 0
            
            # Generate campaign date (last 6 months)
            days_ago = np.random.randint(1, 180)
            campaign_date = datetime.now() - timedelta(days=days_ago)
            
            # Additional features for modeling
            promo_data.append({
                "customer_id": customer_id,
                "segment_name": segment,
                "segment_id": customer["segment"],
                "promo_type": promo_type,
                "promo_description": promo_info["description"],
                "campaign_date": campaign_date.strftime("%Y-%m-%d"),
                "days_since_campaign": days_ago,
                
                # Customer features (for modeling)
                "avg_bet": customer["avg_bet"],
                "avg_loss": customer["avg_loss"],
                "avg_session_duration": customer["avg_session_duration"],
                "zone_diversity": customer["zone_diversity"],
                "ticket_in": customer["ticket_in"],
                "ticket_out": customer["ticket_out"],
                "jackpot_total": customer["jackpot_total"],
                "game_type_diversity": customer["game_type_diversity"],
                "avg_usage_level": customer["avg_usage_level"],
                
                # Target variable
                "response_probability": response_prob,
                "actual_response": actual_response
            })
    
    # Convert to DataFrame
    promo_df = pd.DataFrame(promo_data)
    
    # Add some additional features for modeling
    promo_df["is_weekend_campaign"] = pd.to_datetime(promo_df["campaign_date"]).dt.dayofweek >= 5
    promo_df["month"] = pd.to_datetime(promo_df["campaign_date"]).dt.month
    promo_df["is_holiday_season"] = promo_df["month"].isin([11, 12, 1])  # Nov, Dec, Jan
    
    # Save the promotional response dataset
    promo_df.to_csv(output_path, index=False)
    print(f"\n✅ Promotional response data saved to: {output_path}")
    
    # Generate summary statistics
    print("\n=== PROMO RESPONSE SUMMARY ===")
    print(f"Total promotional campaigns: {len(promo_df)}")
    print(f"Unique customers: {promo_df['customer_id'].nunique()}")
    print(f"Overall response rate: {promo_df['actual_response'].mean():.2%}")
    
    print("\n📊 Response rates by segment:")
    segment_summary = promo_df.groupby('segment_name').agg({
        'actual_response': ['count', 'sum', 'mean'],
        'response_probability': 'mean'
    }).round(3)
    
    segment_summary.columns = ['Total_Campaigns', 'Responses', 'Response_Rate', 'Avg_Probability']
    print(segment_summary)
    
    print("\n📧 Response rates by promo type:")
    promo_summary = promo_df.groupby('promo_type').agg({
        'actual_response': ['count', 'mean']
    }).round(3)
    promo_summary.columns = ['Total_Campaigns', 'Response_Rate']
    print(promo_summary)
    
    # Save summary as JSON
    summary = {
        "total_campaigns": len(promo_df),
        "unique_customers": int(promo_df['customer_id'].nunique()),
        "overall_response_rate": float(promo_df['actual_response'].mean()),
        "segment_response_rates": promo_df.groupby('segment_name')['actual_response'].mean().to_dict(),
        "promo_type_response_rates": promo_df.groupby('promo_type')['actual_response'].mean().to_dict(),
        "features_for_modeling": [
            "avg_bet", "avg_loss", "avg_session_duration", "zone_diversity",
            "ticket_in", "ticket_out", "jackpot_total", "game_type_diversity",
            "avg_usage_level", "is_weekend_campaign", "month", "is_holiday_season"
        ]
    }
    
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {summary_path}")
    
    return promo_df

if __name__ == "__main__":
    # Generate promo response data
    segmented_data_path = "../data/segmented_customers_1500.csv"
    output_path = "../data/promo_response_data.csv"
    
    promo_df = generate_promo_response_data(segmented_data_path, output_path)
    
    print("\n🎯 Ready for Random Forest modeling!")
    print("Next step: Train prediction model using this promotional response data")