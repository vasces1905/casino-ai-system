#import pandas as pd
#df = pd.read_csv("../data/labeled_customer_dataset.csv")
#print(df["segment_label"].value_counts())
#print(df[["customer_id", "segment_label"]].head(10))


import pandas as pd
df = pd.read_csv("../data/labeled_customer_dataset.csv")
print(df["segment_label"].value_counts())

print(df[["customer_id", "avg_loss", "avg_bet", "zone_diversity", "segment_label"]].head(10))
