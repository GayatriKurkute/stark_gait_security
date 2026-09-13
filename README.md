\# Continuous Biometric Gait Authentication \& Domain Gap Analysis



An end-to-end machine learning pipeline for continuous biometric user authentication using smartphone gait data. This repository features a pre-processed benchmark training pipeline achieving \*\*>93% accuracy\*\* on controlled signals, along with an evaluation framework detailing zero-shot cross-domain performance degradation on raw mobile sensor streams.



\## 📌 Project Overview

\- \*\*Benchmark Model Accuracy:\*\* 94.01% (Validation) | 93.72% (Test)

\- \*\*Real-World Zero-Shot Accuracy:\*\* 0.00% (Quantified Domain Gap)

\- \*\*Pipeline Architecture:\*\* Dynamic activity filtering, Z-score normalization, 120-component PCA, and `RandomForestClassifier` ensembles.



\## 📁 Repository Architecture

```text

stark\_gait\_security/

├── data/

│   ├── uci\_har/           # Controlled benchmark dataset

│   └── real\_world/        # 5 raw smartphone sensor CSV logs

├── models/                # Pipeline artifacts (Scaler, PCA, Classifier)

├── reports/               # Evaluation plots and visual artifacts

├── src/

│   ├── train\_model.py     # Benchmark model training pipeline

│   ├── evaluate\_real\_world.py # Robust cross-domain testing script

│   └── plot\_results.py    # Metric visualizer \& chart generator

├── requirements.txt       # Project dependencies

└── README.md              # Documentation

