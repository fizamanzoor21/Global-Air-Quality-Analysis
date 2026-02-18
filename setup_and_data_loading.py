import os
import pandas as pd

def main():
    csv_path = os.path.join("data", "raw", "global_air_quality_data_10000.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at: {csv_path}\n"
            "Fix: Put your dataset in data/raw/ and name it exactly: global_air_quality_data_10000.csv"
        )

    df = pd.read_csv(csv_path)

    print(" Dataset loaded successfully")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()
