import pandas as pd
from app.preprocessing import build_customer_features
from app.models.train_kmeans import train_kmeans_model

# Step 1 – Build Features set
features_df = build_customer_features(
    slot_path="data/final_slot_log_100.xml",
    tito_path="data/final_tito_log_matched.csv",
    crm_path="data/crm_profiles_100.json",
    heatmap_path="data/heatmap_100_customers.csv"
)

# Step 2 – Train KMeans Model
kmeans_model, labeled_df = train_kmeans_model(features_df, n_clusters=3)

# Step 3 – Map Segment Labels to Business requirement meaning
segment_map = {
    0: "Casual Gambler",
    1: "Regular Player",
    2: "High Roller"
}
labeled_df["segment_label"] = labeled_df["segment_label"].map(segment_map)

# Step 4 – Export to CSV
labeled_df.to_csv("data/labeled_customer_dataset.csv", index=False)
print(" OK - Labeled customer dataset saved to: data/labeled_customers.csv")
