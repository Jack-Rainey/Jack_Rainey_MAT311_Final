from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def get_probability(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]