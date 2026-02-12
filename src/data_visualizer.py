import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numeric_distributions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    figures = []

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        figures.append(fig)

    return figures


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)

    if corr.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")

    return fig
