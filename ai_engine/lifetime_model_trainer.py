#!/usr/bin/env python3
"""
Lifetime model trainer using XGBoost matching LAVA paper performance.
Supports data augmentation and exports model for Go inference.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def load_and_augment_data(data_path):
    """Load JSON data and apply augmentation per LAVA paper."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Features from LAVA paper
    categorical_features = ['zone', 'vm_shape', 'vm_category', 'metadata_id', 'provisioning_model', 'priority', 'admission_policy']
    numeric_features = ['uptime_hours']

    # Data augmentation: create examples at 0%, 12.5%, 25%, 50%, 75% of actual lifetime
    augmented_rows = []
    for _, row in df.iterrows():
        T_hours = row['actual_lifetime'].total_seconds() / 3600.0
        for frac in [0.0, 0.125, 0.25, 0.5, 0.75]:
            uptime = frac * T_hours
            if uptime >= T_hours:
                continue
            label = math.log10(T_hours - uptime)  # Log10(remaining)

            aug_row = row.copy()
            aug_row['uptime_hours'] = uptime
            aug_row['label'] = label
            augmented_rows.append(aug_row)

    df_aug = pd.DataFrame(augmented_rows)
    return df_aug, categorical_features


def preprocess_features(df, categorical_features):
    """Collapse rare categories and one-hot encode."""
    for cat in categorical_features:
        counts = df[cat].value_counts()
        rare = counts[counts < 10].index
        df[cat] = df[cat].replace(rare, 'Other')

    # Prepare X, y
    feature_cols = categorical_features + ['uptime_hours']
    X = pd.get_dummies(df[feature_cols], columns=categorical_features)
    y = df['label']

    return X, y


def train_model(X_train, y_train, X_test, y_test, output_path):
    """Train XGBoost model with LAVA hyperparameters."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'max_bins': 32,
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
    }

    evals = [(dtrain, 'train'), (dtest, 'val')]
    model = xgb.train(params, dtrain, num_boost_round=2000, evals=evals, early_stopping_rounds=50, verbose_eval=100)

    # Evaluate 7-day threshold classification (log10(168h) ~ 2.225)
    pred = model.predict(dtest)
    threshold = math.log10(168)
    pred_class = (pred > threshold).astype(int)
    y_class = (y_test > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_class, pred_class, average='binary', zero_division=0)
    print(f"7-day threshold - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # Save model
    model.save_model(output_path)
    print(f"Model saved to {output_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train lifetime prediction model")
    parser.add_argument('--data', required=True, help="Input JSON training data path")
    parser.add_argument('--output', required=True, help="Output model path")
    parser.add_argument('--test-size', type=float, default=0.15, help="Test split size")
    args = parser.parse_args()

    # Load and augment
    df, cat_features = load_and_augment_data(args.data)

    # Preprocess
    X, y = preprocess_features(df, cat_features)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # Train
    train_model(X_train, y_train, X_test, y_test, args.output)


if __name__ == '__main__':
    main()