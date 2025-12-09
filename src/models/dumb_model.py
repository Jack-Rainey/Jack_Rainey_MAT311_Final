import pandas as pd
from sklearn.dummy import DummyClassifier


def train_dumb_model(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """Train a model that always predicts the majority class (never fraud)."""
    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(X_train, y_train)
    return model
