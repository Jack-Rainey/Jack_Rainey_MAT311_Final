import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

def train_neural_net_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train and return a neural-net-based classifier with hyperparameter tuning.

    Uses RandomizedSearchCV over an MLPClassifier wrapped in a Pipeline
    with StandardScaler.
    """

    base_pipeline = Pipeline([
        # ("scaler", StandardScaler()),  # <-- REMOVE this
        ("mlp", MLPClassifier(
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        )),
    ])

    # NOTE: use double underscores to refer to step params
    param_distributions = {
        "mlp__hidden_layer_sizes": [
            (2, 2),
            (32,32),
            (64, 32),
            (128, 64),
        ],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [5e-4, 1e-3, 5e-3],
        "mlp__batch_size": [32, 64, 128, 256],
    }

    tuner = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=20,          # main runtime knob
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=4,
        refit=True,
    )

    tuner.fit(X_train, y_train)

    print("\nBest neural net params:", tuner.best_params_)
    print("Best neural net CV ROC-AUC:", tuner.best_score_)

    neural_net_model = tuner.best_estimator_
    return neural_net_model