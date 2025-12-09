import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    """Create exploratory plots including bivariate views colored by fraud."""
    sns.countplot(x='fraud', data=df)
    plt.title('Fraud Class Distribution')
    plt.show()

    sns.countplot(x='used_pin_number', data=df)
    plt.title('Used PIN Distribution')
    plt.show()

    sns.countplot(x='repeat_retailer', data=df)
    plt.title('Repeat Retailer Distribution')
    plt.show()

    sns.histplot(df['ratio_to_median_purchase_price'], bins=30)
    plt.title('Ratio to Median Purchase Price Distribution')
    plt.show()

    # Bivariate visualizations colored by fraud
    sns.countplot(data=df, x='used_pin_number', hue='fraud', palette=['green', 'red'])
    plt.title('Transactions with PIN vs Fraudulent Transactions')
    plt.xlabel('Used PIN')
    plt.ylabel('Count')
    plt.legend(title='Fraud', labels=['Non-Fraudulent', 'Fraudulent'])
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='distance_from_home',
        y='ratio_to_median_purchase_price',
        hue='fraud',
        palette={0: 'green', 1: 'red'},
    )
    plt.title(
        'Fraudulent vs Non-Fraudulent: Distance from Home vs Ratio to Median Purchase Price'
    )
    plt.xlabel('Distance from Home')
    plt.ylabel('Ratio to Median Purchase Price')
    plt.show()


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/card_transdata.csv")
    clean = clean_dataset(raw)
    plot_eda(clean)
