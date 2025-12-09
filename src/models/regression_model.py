import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
def train_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    logit = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )

    param_grid = {
        "C": [0.01, 0.1, 0.3, 1, 3, 10],
        "class_weight": [None, "balanced"],
    }

    logit_cv = GridSearchCV(
        estimator=logit,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )

    logit_cv.fit(X_train, y_train)
    print("Best logistic params:", logit_cv.best_params_)
    print("Best logistic CV AUC:", logit_cv.best_score_)
    
    return logit_cv.best_estimator_
"""


def train_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    base_logit = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        "logit__C": [0.01, 0.1, 0.3, 1, 3, 10],
        "logit__class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=base_logit,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    print("Best logistic params:", grid.best_params_)
    print("Best logistic CV ROC-AUC:", grid.best_score_)

    return grid.best_estimator_