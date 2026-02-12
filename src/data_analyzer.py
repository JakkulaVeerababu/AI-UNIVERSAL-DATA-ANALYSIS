import pandas as pd

def analyze_data(df: pd.DataFrame):
    """
    Performs automatic data analysis:
    - Summary statistics
    - Correlation matrix
    """

    summary = df.describe(include="all")
    correlation = df.corr(numeric_only=True)

    return summary, correlation
