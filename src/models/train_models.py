from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.models.knn_model import train_knn_model
from src.models.evaluate_models import get_probability

def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float,
    random_state: int
) -> Tuple[Pipeline, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train the k-NN model and produce validation predictions and probabilities.

    Returns
    -------
    knn_model : Pipeline
        Trained k-NN model.
    y_val : pd.Series
        True labels for the validation split.
    y_pred_baseline : np.ndarray
        Baseline predictions (majority class).
    y_pred_knn : np.ndarray
        k-NN hard predictions on the validation split.
    val_prob : np.ndarray
        k-NN predicted probabilities (for the positive class) on the validation split.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y
    )

    # Train k-NN
    knn_model = train_knn_model(X_train, y_train, random_state)

    # Baseline: always predict the majority class from the training labels
    majority_class = y_train.value_counts().idxmax()
    y_pred_baseline = np.full(shape=len(y_val), fill_value=majority_class)

    # k-NN hard predictions on validation set
    y_pred_knn = knn_model.predict(X_val)

    # k-NN probabilities on validation set (for ROC)
    val_prob = get_probability(knn_model, X_val)

    return knn_model, y_val, y_pred_baseline, y_pred_knn, val_prob