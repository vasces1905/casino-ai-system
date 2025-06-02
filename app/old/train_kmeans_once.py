# train_kmeans_once.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# 1. Veri yükle
df = pd.read_csv('../data/labeled_customer_dataset.csv')

# 2. Feature kolonları
features = [
    'avg_loss', 'avg_rtp', 'avg_bet', 'avg_win',
    'avg_session_duration', 'avg_usage_level',
    'zone_diversity', 'game_type_diversity'
]

# 3. Ölçeklendir
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. KMeans eğit
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 5. Klasör oluştur (varsa geç)
os.makedirs('models', exist_ok=True)

# 6. Modeli kaydet
joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("✅ Model kaydedildi: models/kmeans_model.pkl")
