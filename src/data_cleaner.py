def clean_data(df):
    # Remove unnamed index columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Convert object columns safely to string
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df

