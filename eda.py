import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

DATA_PATH = os.path.join("data", "processed", "cleaned_air_quality.csv")
OUT_DIR = os.path.join("outputs", "eda")

NUM_COLS = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temperature", "Humidity", "Wind Speed", "AQI"
]


def save_fig(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=200)
    plt.close()


def univariate_analysis(df):
    print("[EDA] Univariate analysis")
    for col in NUM_COLS:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col], bins=40)
            plt.title(f"Univariate Distribution: {col}")
            save_fig(f"univariate_{col}.png")


def bivariate_analysis(df):
    print("[EDA] Bivariate analysis")
    pairs = ["PM2.5", "PM10", "NO2", "Wind Speed"]
    for col in pairs:
        if col in df.columns:
            plt.figure()
            sns.scatterplot(x=df[col], y=df["AQI"], alpha=0.5)
            plt.xlabel(col)
            plt.ylabel("AQI")
            plt.title(f"AQI vs {col}")
            save_fig(f"bivariate_AQI_vs_{col}.png")


def correlation_analysis(df):
    print("[EDA] Correlation analysis")
    corr = df[NUM_COLS].corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    save_fig("correlation_heatmap.png")

    corr.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))


def comparative_analysis(df):
    print("[EDA] Comparative analysis")

    # Top cities by AQI
    city_avg = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=city_avg.values, y=city_avg.index)
    plt.title("Top 10 Cities by Average AQI")
    save_fig("top_cities_aqi.png")

    # Top countries by AQI
    country_avg = df.groupby("Country")["AQI"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=country_avg.values, y=country_avg.index)
    plt.title("Top 10 Countries by Average AQI")
    save_fig("top_countries_aqi.png")


def cycle_detection(df):
    print("[EDA] Cycle detection")

    if "Month" not in df.columns:
        df["Month"] = pd.to_datetime(df["Date"]).dt.month
        df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek

    monthly = df.groupby("Month")["AQI"].mean()
    plt.figure()
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o")
    plt.title("Monthly AQI Cycle")
    save_fig("cycle_monthly_aqi.png")

    dow = df.groupby("DayOfWeek")["AQI"].mean()
    plt.figure()
    sns.barplot(x=dow.index, y=dow.values)
    plt.title("Day-of-Week AQI Cycle")
    save_fig("cycle_dayofweek_aqi.png")


def main():
    print("=== PART 3: Exploratory Data Analysis ===")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    univariate_analysis(df)
    bivariate_analysis(df)
    correlation_analysis(df)
    comparative_analysis(df)
    cycle_detection(df)

    print(f"EDA completed. Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()