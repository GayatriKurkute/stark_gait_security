# src/train_model.py
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = os.path.join("data", "uci_har", "UCI HAR Dataset")

def load_uci_har_data():
    """Loads feature matrices, activity labels, and subject mappings."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "train", "X_train.txt"), sep=r'\s+', header=None, engine='python').values
    y_activity_train = pd.read_csv(os.path.join(DATA_DIR, "train", "y_train.txt"), sep=r'\s+', header=None, engine='python').values.ravel()
    subject_train = pd.read_csv(os.path.join(DATA_DIR, "train", "subject_train.txt"), sep=r'\s+', header=None, engine='python').values.ravel()

    X_test = pd.read_csv(os.path.join(DATA_DIR, "test", "X_test.txt"), sep=r'\s+', header=None, engine='python').values
    y_activity_test = pd.read_csv(os.path.join(DATA_DIR, "test", "y_test.txt"), sep=r'\s+', header=None, engine='python').values.ravel()
    subject_test = pd.read_csv(os.path.join(DATA_DIR, "test", "subject_test.txt"), sep=r'\s+', header=None, engine='python').values.ravel()

    return X_train, y_activity_train, subject_train, X_test, y_activity_test, subject_test

print("Loading dataset...")
X_tr_raw, y_act_tr, sub_tr_raw, X_te_raw, y_act_te, sub_te_raw = load_uci_har_data()

# Combine all data
X_all = np.vstack((X_tr_raw, X_te_raw))
y_act_all = np.concatenate((y_act_tr, y_act_te))
sub_all = np.concatenate((sub_tr_raw, sub_te_raw))

# Filter strictly for Dynamic Gait/Walking activities (1: Walking, 2: Upstairs, 3: Downstairs)
walking_mask = np.isin(y_act_all, [1, 2, 3])
X_gait = X_all[walking_mask]
y_gait = sub_all[walking_mask]

# Stratified Train/Val/Test Split to ensure uniform subject coverage across window frames
X_train, X_temp, y_train, y_temp = train_test_split(
    X_gait, y_gait, test_size=0.30, random_state=42, stratify=y_gait
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

# Feature Scaling & Dimensionality Reduction (PCA to eliminate noise)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=120, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nTraining Stark Security Identification Model (Random Forest Ensemble)...")

# Optimized Random Forest Classifier
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train_pca, y_train)

val_preds = clf.predict(X_val_pca)
test_preds = clf.predict(X_test_pca)

val_acc = accuracy_score(y_val, val_preds) * 100
test_acc = accuracy_score(y_test, test_preds) * 100

print("\n================ EVALUATION METRICS ================")
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy:       {test_acc:.2f}%")
print("====================================================\n")

if val_acc >= 80.0:
    print("SUCCESS: Target accuracy threshold (>80%) achieved!")
else:
    print("Accuracy below 80%, consider adjusting hyperparameters.")


# Create models directory if missing
os.makedirs("models", exist_ok=True)

# Save pipeline artifacts
joblib.dump(clf, "models/gait_rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
print("Model artifacts successfully saved to models/")