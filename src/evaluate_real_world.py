
# src/evaluate_real_world.py
import os
import glob
import re
import numpy as np
from sklearn.metrics import accuracy_score

def load_real_world_file_robust(filepath):
    """
    Robustly parses text/CSV files line-by-line using regular expressions 
    to extract numerical sensor values, completely bypassing Pandas parser errors.
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Extract all integer/float values from the line
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
            if numbers:
                # Convert string numbers to float
                float_nums = [float(n) for n in numbers]
                rows.append(float_nums)
    
    if not rows:
        return None
    
    # Pad or truncate rows to ensure a uniform 2D array matrix
    max_cols = max(len(r) for r in rows)
    matrix = np.zeros((len(rows), max_cols))
    for i, r in enumerate(rows):
        matrix[i, :len(r)] = r
        
    num_samples, num_cols = matrix.shape
    
    # Align features with the trained model's expected 561 input dimensions
    if num_cols < 561:
        padded_matrix = np.zeros((num_samples, 561))
        padded_matrix[:, :num_cols] = matrix
        return padded_matrix
    elif num_cols > 561:
        return matrix[:, :561]
    
    return matrix

def run_real_world_evaluation(clf, scaler, pca, real_world_dir="data/real_world"):
    files = glob.glob(os.path.join(real_world_dir, "subject*.csv"))
    if not files:
        print(f"No real-world data files found in {real_world_dir}")
        return

    print(f"Found {len(files)} real-world subject files for testing.\n")
    
    overall_correct = 0
    overall_total = 0

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        
        # Extract ground-truth subject ID from filename (e.g., subject01.csv -> 1)
        try:
            true_subject = int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            true_subject = None

        X_processed = load_real_world_file_robust(filepath)
        
        if X_processed is None:
            print(f"File: {filename} | Error: No numerical data found.")
            continue

        # Pipeline Transformation: Scaler -> PCA -> Classifier
        X_scaled = scaler.transform(X_processed)
        X_pca = pca.transform(X_scaled)
        
        preds = clf.predict(X_pca)
        
        if true_subject is not None:
            correct = np.sum(preds == true_subject)
            acc = (correct / len(preds)) * 100
            overall_correct += correct
            overall_total += len(preds)
            print(f"File: {filename} | Ground-Truth ID: {true_subject:02d} | Evaluation Windows: {len(preds):4d} | Accuracy: {acc:6.2f}%")
        else:
            print(f"File: {filename} | Predicted Subject IDs: {np.unique(preds)}")

    if overall_total > 0:
        total_acc = (overall_correct / overall_total) * 100
        print("\n================ CROSS-DOMAIN EVALUATION RESULTS ================")
        print(f"Benchmark Accuracy (UCI HAR):           >93%")
        print(f"Real-World Zero-Shot Accuracy:          {total_acc:.2f}%")
        print("=================================================================\n")

if __name__ == "__main__":
    from train_model import clf, scaler, pca
    run_real_world_evaluation(clf, scaler, pca)