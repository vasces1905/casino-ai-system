import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # To normalize the values (if we don't scale, some features will dominate)

def train_kmeans_model(features_df: pd.DataFrame, n_clusters: int = 5):
    """
    Trains a KMeans model on selected features and returns the model and labeled dataframe.

    Parameters:
    - features_df: DataFrame with customer features
    - n_clusters: number of customer segments (default=5)

    Returns:
    - kmeans: trained KMeans model
    - features_df: dataframe with added 'segment_label' column
    """
    # Select numerical features only
    X = features_df[[
        "avg_loss", "avg_bet", "avg_rtp", "jackpot_total", 
        "avg_session_duration", "zone_diversity", "avg_usage_level"
    ]].copy()

    # Normalize values for clustering
    # When AVG_loss is around 300, zone_diversity can only be between 0–5
    # This difference causes the model to attach much importance to some features.
    # Standardscaler: converts each column to mean = 0, std = 1
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    # n_clusters=5: 5 customer groups will be created
    # random_state=42: Fixed randomness to repeat the same results
    # n_init=10: Try 10 different starting points and get the best result (recommended for stability)
    
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) old
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    kmeans.fit(X_scaled)

    # Add cluster labels to DataFrame
    # Calculates which cluster each customer belongs to
    # Adds the results to the data frame as the segment_label column (e.g.: 0, 1, 2, 3, 4)
    
    features_df["segment_label"] = kmeans.predict(X_scaled)

   # Trained model (kmeans)
   # Labeled customer data (features_df + segment_label)
   

    return kmeans, features_df

   

