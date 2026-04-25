import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

os.makedirs(os.path.join('API_analyzer','Models'), exist_ok=True)

# Synthetic training data
X = np.random.rand(200, 6)
# Labels: normal, attack, bot, outlier
y = np.random.choice(['normal','attack','bot','outlier'], size=200, p=[0.7,0.1,0.15,0.05])

# Scaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Label encoder
le = LabelEncoder().fit(y)
y_enc = le.transform(y)

# Simple classifier
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_scaled, y_enc)

# Save artifacts
joblib.dump(clf, os.path.join('API_analyzer','Models','api_model.pkl'))
joblib.dump(scaler, os.path.join('API_analyzer','Models','api_scaler.pkl'))
joblib.dump(le, os.path.join('API_analyzer','Models','api_label_encoder.pkl'))

print('API model artifacts created at API_analyzer/Models')
