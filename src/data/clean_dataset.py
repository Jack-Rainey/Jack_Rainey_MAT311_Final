import pandas as pd

"""
Credit to ChatGPT 5.1 Thinking for helping with implementation of this file
"""


def fill_na_df(df: pd.DataFrame, fill_na_method: str) -> pd.DataFrame:
    
    # This mask is necessary to avoid accidental identification as a categorical column that should be a numeric column
    missing_tokens = ["none", "None", "NONE", "Missing"]
    df = df.replace(missing_tokens, "")
    df = df.replace("", pd.NA)
    
    # Fill NAs using specified method
    match fill_na_method:
        case "mode":
            modes = df.mode().iloc[0]
            df = df.fillna(modes)
        case "smart": # Median for numeric and "Missing" for categorical
            df = df.copy()

            # Let pandas tell us what is numeric
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = [c for c in df.columns if c not in num_cols]

            for col in num_cols:
                df[f"{col}_was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].astype(float).fillna(df[col].median())

            for col in cat_cols:
                df[f"{col}_was_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna("Missing")

            return df
        case _:
            raise NotImplementedError
    
    return df


def process_date_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    # Columns with MM-DD format
    date_cols = ["Last Payment Date", "Last Due Date"]

    for col in date_cols:
        # Work with clean strings; pd.NA stays as <NA>
        s = cleaned[col].astype("string").str.strip()

        # Prepend a dummy year so '07-08' -> '2000-07-08'
        cleaned[col] = pd.to_datetime(
            "2000-" + s,
            format="%Y-%m-%d",
            errors="coerce"     # anything weird -> NaT, no crash
        )

    # Now compute difference in days
    cleaned["Days Since Last Payment"] = (
        cleaned["Last Payment Date"] - cleaned["Last Due Date"]
    ).dt.days
    cleaned["Payment Days Late"] = cleaned["Days Since Last Payment"].clip(lower=0)


    # Drop raw datetime columns so they never hit sklearn
    cleaned = cleaned.drop(columns=[ "Last Payment Date", "Last Due Date"])

    return cleaned


"""
TODO:
    - Implement function
    - Add more versions of datetime processing
"""
def one_hot_encode_df(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    encoded = df.copy()

    cat_cols_present = [c for c in categorical_columns if c in encoded.columns]

    if cat_cols_present:
        encoded = pd.get_dummies(
            encoded,
            columns=cat_cols_present,
            drop_first=False,  # keep all dummies; fine for most models
            dtype=float,       # ensure dummy columns are numeric
        )

    return encoded


"""
TODO:
    - Finish the rest of the implementation of this function
    - Fix type annotations
    - Add other information
"""
def clean_dataset(
        df: pd.DataFrame,
        categorical_columns: list[str],
        fill_na_method,
        one_hot_encode
    ) -> pd.DataFrame:
    cleaned = df.copy()

    # Drop columns that are resulting in insane ROC-AUC correlations
    cleaned = cleaned.drop(columns=["Customer Status", "Payment Delay"])

    # Deal with annoying problem of a numeric column having the word "Missing" in it
    if "Support Calls" in cleaned.columns:
        cleaned["Support Calls"] = pd.to_numeric(cleaned["Support Calls"], errors="coerce")


    cleaned = fill_na_df(cleaned, fill_na_method)

    cleaned = process_date_features(cleaned)

    if one_hot_encode:
        cleaned = one_hot_encode_df(cleaned, categorical_columns)

    return cleaned