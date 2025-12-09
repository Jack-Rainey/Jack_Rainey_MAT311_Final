# Personal imports
from src.data.load_data import load_dataset
from src.data.clean_datasets import clean_datasets
from src.data.generate_files import save_to_csv, generate_submission_file
from src.models.evaluate_models import get_probability
from src.models.train_models import train_models
from src.visualization.performance import plot_confusion_matrices, plot_roc_curve


def run_core_functionality(
        train_df_path: str,
        test_df_path: str,
        train_cleaned_df_path: str,
        test_cleaned_df_path: str,
        validation_size: float,
        random_state: int,
        categorical_columns: list[str],
        fill_na_method: str,
        one_hot_encode: bool
    ) -> None:

    print("---Loading Datasets...")
    raw_train_df = load_dataset(train_df_path)
    raw_test_df  = load_dataset(test_df_path)

    print("\n---Cleaning datasets...")
    clean_train_df, clean_test_df, X, X_test, y = clean_datasets(
        raw_train_df,
        raw_test_df,
        categorical_columns,
        fill_na_method,
        one_hot_encode
    )

    print("\n\n---Saving cleaned datasets...")
    save_to_csv(clean_train_df, train_cleaned_df_path)
    save_to_csv(clean_test_df, test_cleaned_df_path)

    print("\n\n---Training Models (tuning on training set)...")
    knn_model, y_val, y_pred_baseline, y_pred_knn, val_prob = train_models(
        X, y, validation_size, random_state
    )

    print("\n\n---Confusion matrices on validation set...")
    plot_confusion_matrices(y_val, y_pred_baseline, y_pred_knn)

    print("\n\n---ROC curve on validation set...")
    plot_roc_curve(y_val, val_prob, "KNN - Validation ROC")

    print("\n\n---Getting model probabilities for submission...")
    test_prob = get_probability(knn_model, X_test)

    print("\n\n---Generating submission file...")
    generate_submission_file(
        df=clean_test_df,
        columns_to_save=["CustomerID", "Churn"],
        file_path="data/processed/submission.csv",
        prob=test_prob
    )


def main() -> None:
    run_core_functionality(
        train_df_path="data/raw/train.csv",
        test_df_path="data/raw/test.csv",
        train_cleaned_df_path="data/processed/train_cleaned.csv",
        test_cleaned_df_path="data/processed/test_cleaned.csv",
        validation_size=0.3,
        random_state=123,
        categorical_columns=["Gender", "Subscription Type", "Contract Length"],
        fill_na_method="smart",
        one_hot_encode=True
    )

if __name__ == '__main__':
    main()