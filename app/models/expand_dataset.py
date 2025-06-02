import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def expand_customer_dataset(original_df, target_size=500, seed=42):
    """
    Expands the customer dataset by generating synthetic customers
    based on the statistical properties of the original data.
    
    Parameters:
    - original_df: Original customer dataset
    - target_size: Target number of customers
    - seed: Random seed for reproducibility
    
    Returns:
    - expanded_df: Expanded dataset with synthetic customers
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Get original statistics for each column
    stats = {}
    for col in original_df.columns:
        if original_df[col].dtype in ['int64', 'float64']:
            stats[col] = {
                'mean': original_df[col].mean(),
                'std': original_df[col].std(),
                'min': original_df[col].min(),
                'max': original_df[col].max()
            }
    
    # Get categorical distributions
    gender_dist = original_df['gender'].value_counts(normalize=True)
    nationality_dist = original_df['nationality'].value_counts(normalize=True)
    
    # Create new customers
    new_customers = []
    existing_count = len(original_df)
    
    for i in range(existing_count, target_size):
        customer = {
            'customer_id': f'CUST_{1000 + i}'
        }
        
        # Age (normal distribution, bounded)
        age = int(np.random.normal(stats['age']['mean'], stats['age']['std']))
        customer['age'] = np.clip(age, 21, 75)
        
        # Gender (based on original distribution)
        customer['gender'] = np.random.choice(gender_dist.index, p=gender_dist.values)
        
        # Nationality (based on original distribution)
        customer['nationality'] = np.random.choice(nationality_dist.index, p=nationality_dist.values)
        
        # Game type diversity (1-5 range typically)
        customer['game_type_diversity'] = np.random.randint(1, 6)
        
        # Average RTP (typically 80-96%)
        customer['avg_rtp'] = np.random.normal(stats['avg_rtp']['mean'], 3)
        customer['avg_rtp'] = np.clip(customer['avg_rtp'], 80, 96)
        
        # Create correlated betting behavior
        # High rollers have higher bets
        bet_multiplier = np.random.lognormal(0, 0.5)
        customer['avg_bet'] = np.random.normal(stats['avg_bet']['mean'] * bet_multiplier, 15)
        customer['avg_bet'] = np.clip(customer['avg_bet'], 10, 100)
        
        # Win amount correlates with bet amount and RTP
        win_ratio = customer['avg_rtp'] / 100 * np.random.normal(1, 0.2)
        customer['avg_win'] = customer['avg_bet'] * win_ratio
        customer['avg_win'] = np.clip(customer['avg_win'], 0, customer['avg_bet'] * 3)
        
        # Session duration (5-30 minutes typically)
        customer['avg_session_duration'] = np.random.normal(
            stats['avg_session_duration']['mean'], 
            stats['avg_session_duration']['std']
        )
        customer['avg_session_duration'] = np.clip(customer['avg_session_duration'], 5, 30)
        
        # Financial metrics
        # Ticket in/out based on betting patterns
        sessions_estimate = np.random.randint(3, 20)
        customer['ticket_in'] = customer['avg_bet'] * sessions_estimate * np.random.uniform(0.8, 1.2)
        customer['ticket_out'] = customer['ticket_in'] * (customer['avg_rtp'] / 100) * np.random.uniform(0.7, 1.1)
        
        # Average loss
        customer['avg_loss'] = (customer['ticket_in'] - customer['ticket_out']) / sessions_estimate
        customer['avg_loss'] = np.clip(customer['avg_loss'], 0, customer['avg_bet'] * 2)
        
        # Jackpot contribution (typically 1-5% of total wagered)
        customer['jackpot_total'] = customer['ticket_in'] * np.random.uniform(0.01, 0.05)
        
        # Usage level (0-1)
        customer['avg_usage_level'] = np.random.beta(2, 5)  # Skewed towards lower values
        
        # Zone diversity (1-5 typically)
        customer['zone_diversity'] = np.random.randint(1, 6)
        
        # Add segment label based on behavioral patterns
        if customer['avg_bet'] > 70 and customer['avg_loss'] > 150:
            customer['segment_label'] = 'High Roller'
        elif customer['avg_bet'] < 30 and customer['avg_loss'] < 50:
            customer['segment_label'] = 'Casual Gambler'
        else:
            customer['segment_label'] = 'Regular Player'
        
        new_customers.append(customer)
    
    # Create DataFrame for new customers
    new_df = pd.DataFrame(new_customers)
    
    # Combine with original data
    expanded_df = pd.concat([original_df, new_df], ignore_index=True)
    
    # Shuffle the dataset
    expanded_df = expanded_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return expanded_df


def add_promo_response_to_expanded(df):
    """
    Adds promo_response to the expanded dataset using the same rules
    as the original generate_target.py
    """
    def assign_promo_response(row):
        # Original rules
        if row["avg_loss"] > 250 and row["zone_diversity"] > 4:
            return "High"
        elif row["avg_loss"] > 100 or row["zone_diversity"] > 2:
            return "Medium"
        else:
            return "Low"
    
    df["promo_response"] = df.apply(assign_promo_response, axis=1)
    return df


if __name__ == "__main__":
    # Load original dataset
    print("Loading original dataset...")
    original_df = pd.read_csv("../data/labeled_customer_dataset.csv")
    print(f"Original dataset size: {len(original_df)} customers")
    
    # Remove promo_response if it exists (we'll regenerate it)
    if 'promo_response' in original_df.columns:
        original_df = original_df.drop('promo_response', axis=1)
    
    # Let's create multiple dataset sizes
    target_sizes = [1000, 1500, 2000]
    
    for target_size in target_sizes:
        print(f"\n{'='*50}")
        print(f"Expanding dataset to {target_size} customers...")
        expanded_df = expand_customer_dataset(original_df, target_size=target_size)
        
        # Add promo response
        print("Adding promo_response...")
        expanded_df = add_promo_response_to_expanded(expanded_df)
        
        # Save expanded dataset
        output_path = f"../data/labeled_customer_dataset_{target_size}.csv"
        expanded_df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        
        # Show statistics
        print(f"\n=== Dataset Statistics ({target_size} customers) ===")
        print(f"\nSegment distribution:")
        segment_dist = expanded_df['segment_label'].value_counts()
        print(segment_dist)
        print("\nSegment percentages:")
        print((segment_dist / len(expanded_df) * 100).round(2))
        
        print(f"\nPromo response distribution:")
        promo_dist = expanded_df['promo_response'].value_counts()
        print(promo_dist)
        print("\nPromo response percentages:")
        print((promo_dist / len(expanded_df) * 100).round(2))
    
    # Show comparison
    print(f"\n{'='*50}")
    print("DATASET SIZE COMPARISON")
    print(f"Original: {len(original_df)} customers")
    print(f"Option 1: 1000 customers")
    print(f"Option 2: 1500 customers")
    print(f"Option 3: 2000 customers")
    print("\nFiles created:")
    print("- labeled_customer_dataset_1000.csv")
    print("- labeled_customer_dataset_1500.csv")
    print("- labeled_customer_dataset_2000.csv")