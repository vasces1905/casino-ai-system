import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import json
from preprocessing import build_customer_features
import warnings
warnings.filterwarnings('ignore')

def perform_segmentation(features_df: pd.DataFrame, n_clusters: int = 3) -> tuple:
    """
    Performs K-Means clustering on customer features with smart naming
    """
    
    # Key features for clustering (simplified for better interpretation)
    # We tell K-means which features to use for grouping.
    # Why these features: The 9 features that best reflect customer behavior have been selected.
    
    clustering_features = [
        'avg_bet', 'avg_loss', 'avg_session_duration',
        'ticket_in', 'ticket_out', 'jackpot_total',
        'game_type_diversity', 'zone_diversity', 'avg_usage_level'
    ]
    
    # Prepare data
    X = features_df[clustering_features].copy()
    
    # Standardize features / It brings all features to the same scale.
    # Why it is necessary:
    # avg_loss = $500 (large number)
    #zone_diversity = 2.3 (small number)
    # Since K-means uses distance, big numbers crush small numbers! Each feature has "mean=0, standard deviation=1".
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means 
    # Finds 3 center points - Assigns every customer to the nearest center 
    # Conclusion: Each customer receives 0, 1, or 2 segment numbers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['segment'] = kmeans.fit_predict(X_scaled)
    
    # SMART NAMING: Analyze cluster centers to assign correct names
    # It turns the 3 center points found by K-means into a DataFrame
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=clustering_features)
    
    # Calculate key metrics for each center. It saves key metrics for each center in the dictionary.
    center_profiles = {}
    for i in range(n_clusters):
        center_profiles[i] = {
            'avg_loss': centers_df.loc[i, 'avg_loss'],
            'avg_bet': centers_df.loc[i, 'avg_bet'], 
            'diversity': centers_df.loc[i, 'zone_diversity'],
            'ticket_in': centers_df.loc[i, 'ticket_in']
        }
    
    # Sort centers by avg_loss (spending level)
    sorted_centers = sorted(center_profiles.items(), 
                          key=lambda x: x[1]['avg_loss'])
    
    # Assign names based on spending level
    segment_names = {}
    for idx, (center_id, profile) in enumerate(sorted_centers):
        if idx == 0:  # Lowest spending
            segment_names[center_id] = "Casual Gambler"
        elif idx == 1:  # Medium spending  
            segment_names[center_id] = "Regular Player"
        else:  # Highest spending
            segment_names[center_id] = "High Roller"
    
    features_df['segment_name'] = features_df['segment'].map(segment_names)
    
    # Print cluster analysis
    print("\n=== CLUSTER CENTER ANALYSIS ===")
    for i in range(n_clusters):
        profile = center_profiles[i]
        name = segment_names[i]
        print(f"\nSegment {i} -> {name}:")
        print(f"  Avg Loss: {profile['avg_loss']:.3f} (standardized)")
        print(f"  Avg Bet: {profile['avg_bet']:.3f} (standardized)")
        print(f"  Zone Diversity: {profile['diversity']:.3f} (standardized)")
        print(f"  Ticket In: {profile['ticket_in']:.3f} (standardized)")
    
    return features_df, kmeans, scaler, segment_names

def analyze_segments(features_df: pd.DataFrame):
    """
    Analyzes and visualizes customer segments with actual values
    """
    
    print("\n=== SEGMENT ANALYSIS (ACTUAL VALUES) ===")
    print(f"Total customers: {len(features_df)}")
    print(f"\nSegment distribution:")
    print(features_df['segment_name'].value_counts())
    
    # Key metrics by segment (ACTUAL VALUES)
    print("\n=== KEY METRICS BY SEGMENT ===")
    for segment_name in features_df['segment_name'].unique():
        segment_data = features_df[features_df['segment_name'] == segment_name]
        print(f"\n{segment_name} ({len(segment_data)} customers):")
        print(f"  Avg Loss: ${segment_data['avg_loss'].mean():.0f}")
        print(f"  Avg Bet: ${segment_data['avg_bet'].mean():.0f}")
        print(f"  Avg Session: {segment_data['avg_session_duration'].mean():.1f} min")
        print(f"  Zone Diversity: {segment_data['zone_diversity'].mean():.2f}")
        print(f"  Ticket In: ${segment_data['ticket_in'].mean():.0f}")
        
        # Add promo strategy recommendation
        if segment_name == "Casual Gambler":
            print("  → Promo Strategy: Price Oriented, Simple Promotions ** ")
        elif segment_name == "Regular Player":
            print("  → Promo Strategy: Loyality Programs, bonus offers *** ")  
        elif segment_name == "High Roller":
            print("  → Promo Strategy: Premium, VIP oriented campaigns **** ")
    
    # Create scatter plot with actual values
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green']
    
    for i, segment_name in enumerate(features_df['segment_name'].unique()):
        segment_data = features_df[features_df['segment_name'] == segment_name]
        plt.scatter(segment_data['avg_loss'], segment_data['zone_diversity'], 
                   c=colors[i], label=segment_name, alpha=0.6, s=60)
        
        # Add cluster center
        center_x = segment_data['avg_loss'].mean()
        center_y = segment_data['zone_diversity'].mean()
        plt.scatter(center_x, center_y, c='black', marker='X', s=200, 
                   edgecolors='white', linewidth=2)
    
    plt.xlabel('Average loss (avg_loss)')
    plt.ylabel('Zone Divesity (zone_diversity)')
    plt.title('Casino Customer Segmentation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations for cluster centers
    for segment_name in features_df['segment_name'].unique():
        segment_data = features_df[features_df['segment_name'] == segment_name]
        center_x = segment_data['avg_loss'].mean()
        center_y = segment_data['zone_diversity'].mean()
        plt.annotate('Group Center', (center_x, center_y), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('segmentation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_segmented_data(features_df: pd.DataFrame, output_path: str, segment_names: dict):
    """
    Saves segmented customer data with segment mapping
    """
    features_df.to_csv(output_path, index=False)
    print(f"\nSegmented data saved to: {output_path}")
    
    # Save detailed segment summary
    summary = {
        "total_customers": len(features_df),
        "n_segments": 3,
        "segment_mapping": segment_names,
        "segment_distribution": features_df['segment_name'].value_counts().to_dict(),
        "segment_profiles": {}
    }
    
    for segment_name in features_df['segment_name'].unique():
        segment_data = features_df[features_df['segment_name'] == segment_name]
        summary["segment_profiles"][segment_name] = {
            "count": len(segment_data),
            "avg_bet": float(segment_data['avg_bet'].mean()),
            "avg_loss": float(segment_data['avg_loss'].mean()),
            "avg_session_duration": float(segment_data['avg_session_duration'].mean()),
            "zone_diversity": float(segment_data['zone_diversity'].mean()),
            "ticket_in": float(segment_data['ticket_in'].mean()),
            "promo_strategy": get_promo_strategy(segment_name)
        }
    
    with open(output_path.replace('.csv', '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Segment summary saved to: {output_path.replace('.csv', '_summary.json')}")

def get_promo_strategy(segment_name: str) -> str:
    """Returns promo strategy for each segment"""
    strategies = {
        "Casual Gambler": "Price oriented, simple promotions ** ",
        "Regular Player": "Loyalty programs, bonus offers *** ",
        "High Roller": "Premium, VIP Oriented Campaigns **** "
    }
    return strategies.get(segment_name, "Özel strateji gerekli")

if __name__ == "__main__":
    print("=== CUSTOMER SEGMENTATION MODULE (3 SEGMENTS) ===")
    
    # Import the new data loader
    from data_loader import DataLoader
    
    # Choose data source and parameters
    USE_DATABASE = False
    customer_count = 1500
    
    if USE_DATABASE:
        print("\nConnecting to database...")
        connection_string = "DRIVER={SQL Server};SERVER=casino-db;DATABASE=CasinoData;UID=user;PWD=pass"
        loader = DataLoader(source="database", connection_string=connection_string, months_back=6)
    else:
        print(f"\nLoading {customer_count} customer dataset from files...")
        loader = DataLoader(source="file", customer_count=customer_count)
    
    # Load features
    features_df = loader.build_features()
    print(f"Loaded {len(features_df)} customers with {len(features_df.columns)} features")
    
    # Perform segmentation with smart naming
    print("\nPerforming K-Means clustering with smart segment naming...")
    features_df, kmeans_model, scaler, segment_names = perform_segmentation(features_df, n_clusters=3)
    
    # Analyze segments
    analyze_segments(features_df)
    
    # Save results  
    save_segmented_data(features_df, f"../data/segmented_customers_{customer_count}.csv", segment_names)
    
    print("\n=== SEGMENTATION COMPLETE ===")
    print("\nSegment Mapping:")
    for seg_id, seg_name in segment_names.items():
        print(f"  Segment {seg_id} = {seg_name}")
    
    print("\nNext steps:")
    print("1. Review segmentation_analysis.png for visual insights")
    print("2. Check segmented_customers_1500.csv for full data") 
    print("3. Ready for Random Forest promo response modeling!")