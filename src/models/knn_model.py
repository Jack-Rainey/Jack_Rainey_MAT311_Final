from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Pipeline:
    # Pipeline: scaler -> KNN
    base_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    # Note the "knn__" prefix: we are tuning params of the "knn" step in the pipeline.
    param_dist = {
        "knn__n_neighbors": np.arange(1, 101),
        "knn__weights": ["uniform", "distance"],
    }

    rand = RandomizedSearchCV(
        estimator=base_knn,
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    rand.fit(X_train, y_train)

    print("Best KNN params:", rand.best_params_)
    print("Best KNN CV ROC-AUC:", rand.best_score_)

    # This is a Pipeline(scaler -> knn) with the best hyperparameters baked in.
    return rand.best_estimator_