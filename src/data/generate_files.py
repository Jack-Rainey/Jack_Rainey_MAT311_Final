import pandas as pd


def save_to_csv(df: pd.DataFrame, filePath: str) -> None:
    df.to_csv(filePath, index=False)


def generate_submission_file(df: pd.DataFrame, columns_to_save: list[str], file_path: str, prob) -> None:
    prediciton_df = df.assign(Churn = prob)
    prediciton_df[columns_to_save].to_csv(file_path, index=False)