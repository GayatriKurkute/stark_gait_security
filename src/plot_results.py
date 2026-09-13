# src/plot_results.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from train_model import clf, scaler, pca, X_test_pca, y_test

# Create output folder for figures
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

def plot_confusion_matrix():
    """Generates and saves a confusion matrix for benchmark test dataset."""
    print("Generating Benchmark Confusion Matrix...")
    y_pred = clf.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, square=True)
    plt.title("UCI HAR Benchmark Dynamic Gait Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Subject ID", fontsize=12, labelpad=10)
    plt.ylabel("True Subject ID", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "benchmark_confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_domain_gap():
    """Generates and saves the Cross-Domain Performance Gap comparison chart."""
    print("Generating Cross-Domain Performance Gap Chart...")
    domains = ['UCI HAR Benchmark\n(Controlled)', 'Real-World Mobile\n(Zero-Shot Uncalibrated)']
    accuracies = [93.72, 0.00]
    colors = ['#2b5c8f', '#d9534f']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(domains, accuracies, color=colors, width=0.45, edgecolor='black', linewidth=1.2)

    # Annotate exact percentage values over bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            yval + 2, 
            f"{yval:.2f}%", 
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )

    plt.ylim(0, 110)
    plt.ylabel("Classification Accuracy (%)", fontsize=12, labelpad=10)
    plt.title("Gait Authentication Performance: Controlled vs. Real-World", fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "domain_gap_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_confusion_matrix()
    plot_domain_gap()
    print("\nVisualizations successfully created in the 'reports/' directory!")