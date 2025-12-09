import pandas as pd

from src.data.clean_dataset import clean_dataset

def clean_datasets(
        raw_train_df: pd.DataFrame,
        raw_test_df: pd.DataFrame,
        categorical_columns: list[str],
        fill_na_method,
        one_hot_encode
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    
    y = raw_train_df["Churn"].copy()
    train_features = raw_train_df.drop(columns=["Churn"])
    test_features  = raw_test_df.copy()  # no Churn col here
    clean_train_df = clean_dataset(
        train_features,
        categorical_columns,
        fill_na_method,
        one_hot_encode
    )
    clean_test_df  = clean_dataset(
        test_features,
        categorical_columns,
        fill_na_method,
        one_hot_encode
    )
    X = clean_train_df.drop(columns=["Customer Status"], errors="ignore")
    feature_cols = X.columns
    X_test = clean_test_df.reindex(columns=feature_cols, fill_value=0)

    return clean_train_df, clean_test_df, X, X_test, y