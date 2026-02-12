def generate_insights(df, summary, correlation):
    insights = []

    # Dataset size
    insights.append(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Numeric insights
    if "CGPA" in df.columns:
        avg_cgpa = df["CGPA"].mean()
        insights.append(f"The average CGPA of students is {avg_cgpa:.2f}.")

    if "Attendance" in df.columns:
        avg_attendance = df["Attendance"].mean()
        insights.append(f"The average attendance is {avg_attendance:.2f}%.")

    # Correlation insight
    if not correlation.empty:
        insights.append("Some numeric features show correlations, which may impact performance.")

    return insights
