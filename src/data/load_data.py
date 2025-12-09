import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the customer churn dataset."""
    return pd.read_csv(file_path)


if __name__ == "__main__":
    df = load_dataset("data/raw/train.csv")
    print(df.head())
