# Project Documentation

## Objective
Build an educational machine-learning prototype that estimates whether an industrial machine is likely to report a failure from operating-condition measurements.

## Dataset
UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset (10,000 records). The project downloads it through `ucimlrepo` on the first run and stores it at `data/raw/ai4i2020.csv`.

## Workflow
1. Download/load public dataset.
2. Create temperature-difference and mechanical-power features.
3. Exclude failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to avoid target leakage.
4. Split data using stratification.
5. Train a class-weighted Random Forest classifier.
6. Evaluate predictions and save plots.
7. Use Streamlit for interactive estimates and Flask for a JSON API.

## Limitations
This prototype classifies failure labels in the AI4I dataset. It does not estimate remaining useful life, prove real-world future failure, connect to physical IoT sensors, or replace engineering inspection. Model results depend on the dataset and train/test split.
