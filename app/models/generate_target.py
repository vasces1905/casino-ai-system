# generate_target.py

"""
Add a synthetic 'promo_response' column to labeled_customer_dataset.csv
based on business rules for simulated engagement levels.
"""

import pandas as pd

# Load final labeled dataset
df = pd.read_csv("labeled_customer_dataset.csv")

# Rule-based impact level simulation
def assign_promo_response(row):
    if row["avg_loss"] > 250 and row["zone_diversity"] > 4:
        return "High"
    elif row["avg_loss"] > 100 or row["zone_diversity"] > 2:
        return "Medium"
    else:
        return "Low"

# Apply target function
df["promo_response"] = df.apply(assign_promo_response, axis=1)

# Save updated dataset
df.to_csv("labeled_customer_dataset.csv", index=False)

print("promo_response' column added to labeled_customer_dataset.csv")
