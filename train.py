#!/usr/bin/env python
"""
Concrete Compressive Strength Prediction - Training Script
This script trains multiple models and saves the best performing one.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='Concrete_Data.xls'):
    """Load the concrete dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    print(f"Data loaded successfully! Shape: {df.shape}")
    return df


def preprocess_data(df):
    """Preprocess the dataset."""
    print("\nPreprocessing data...")

    # Define features and target
    target_col = 'Concrete compressive strength(MPa, megapascals) '
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(df)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}] MPa")

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Split data and apply scaling."""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train baseline models without tuning."""
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVM': SVR(kernel='rbf')
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results[name] = {
            'model': model,
            'r2': r2,
            'mae': mae,
            'mse': mse
        }

        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")

    return results


def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters."""
    print("\n" + "="*60)
    print("TUNING RANDOM FOREST")
    print("="*60)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )

    print("Running randomized search (20 iterations, 5-fold CV)...")
    rf_search.fit(X_train, y_train)

    print(f"\nBest parameters: {rf_search.best_params_}")
    print(f"Best CV R² score: {rf_search.best_score_:.4f}")

    return rf_search.best_estimator_


def tune_xgboost(X_train, y_train):
    """Tune XGBoost hyperparameters."""
    print("\n" + "="*60)
    print("TUNING XGBOOST")
    print("="*60)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    xgb_search = RandomizedSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0),
        param_distributions=param_grid,
        n_iter=30,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )

    print("Running randomized search (30 iterations, 5-fold CV)...")
    xgb_search.fit(X_train, y_train)

    print(f"\nBest parameters: {xgb_search.best_params_}")
    print(f"Best CV R² score: {xgb_search.best_score_:.4f}")

    return xgb_search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model on test set."""
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n{model_name} - Test Set Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f} MPa")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f} MPa")

    return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}


def save_model_artifacts(model, scaler, feature_names, output_dir='models'):
    """Save model, scaler, and feature names."""
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")

    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved to: {features_path}")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("CONCRETE COMPRESSIVE STRENGTH PREDICTION")
    print("Model Training Pipeline")
    print("="*60)

    # Load and preprocess data
    df = load_data('Concrete_Data.xls')
    X, y = preprocess_data(df)

    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Train baseline models
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test)

    # Tune best performing models
    rf_tuned = tune_random_forest(X_train, y_train)
    xgb_tuned = tune_xgboost(X_train, y_train)

    # Evaluate tuned models
    rf_metrics = evaluate_model(rf_tuned, X_test, y_test, "Random Forest (Tuned)")
    xgb_metrics = evaluate_model(xgb_tuned, X_test, y_test, "XGBoost (Tuned)")

    # Select best model
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)

    if xgb_metrics['r2'] > rf_metrics['r2']:
        best_model = xgb_tuned
        best_name = "XGBoost (Tuned)"
        best_metrics = xgb_metrics
    else:
        best_model = rf_tuned
        best_name = "Random Forest (Tuned)"
        best_metrics = rf_metrics

    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   R² Score: {best_metrics['r2']:.4f}")
    print(f"   MAE: {best_metrics['mae']:.4f} MPa")
    print(f"   RMSE: {best_metrics['rmse']:.4f} MPa")

    # Save artifacts
    save_model_artifacts(best_model, scaler, list(X.columns))

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now use the trained model for predictions.")
    print("Run: python predict.py")


if __name__ == "__main__":
    main()
