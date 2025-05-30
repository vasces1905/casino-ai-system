import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_kmeans_model(features_df: pd.DataFrame, n_clusters: int = 3):
    """
    Trains a KMeans model on customer behavioral features.

    Parameters:
    - features_df: DataFrame containing customer metrics
    - n_clusters: number of desired customer segments (default = 3)

    Returns:
    - kmeans: trained KMeans model
    - features_df: DataFrame with 'segment_label' column added
    """
    # Select numerical behavioral features
    X = features_df[[
        "avg_loss", "avg_bet", "avg_rtp", "jackpot_total", 
        "avg_session_duration", "zone_diversity", "avg_usage_level"
    ]].copy()

    # Normalize values (important for fair clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Assign segment labels to the original DataFrame
    features_df["segment_label"] = kmeans.predict(X_scaled)

    return kmeans, features_df

def run_training_pipeline(
    input_csv_path="../data/labeled_customer_dataset.csv",
    output_model_path="models/kmeans_model.pkl"
):
    """
    Loads data, trains KMeans, saves results and model to disk.

    Parameters:
    - input_csv_path: path to feature CSV
    - output_model_path: path to save trained model
    """
    print("Loading data for KMeans training...")
    df = pd.read_csv(input_csv_path)

    print("Training KMeans model...")
    model, labeled_df = train_kmeans_model(df)

    print(f"Saving model to: {output_model_path}")
    joblib.dump(model, output_model_path)

    print(f"Saving labeled dataset to: {input_csv_path}")
    labeled_df.to_csv(input_csv_path, index=False)

    print("KMeans training complete.")

# Optional CLI entrypoint
if __name__ == "__main__":
    run_training_pipeline()
