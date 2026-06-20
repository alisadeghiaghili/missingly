import os

import numpy as np
import pandas as pd

from missingly.stats import (
    mcar_test,
    diagnose_missing,
    mar_mnar_test,
)

OUTPUT_DIR = "manual_tests/output_stats"


def load_data():
    df = pd.read_csv(
        "./manual_tests/petrochemical_tank_sensors.csv",
        parse_dates=["زمان ثبت"],
    )

    df = df.sample(
        n=1000,
        random_state=42,
    )

    return df


def test_mcar(df):
    print("\n" + "=" * 80)
    print("MCAR TEST")
    print("=" * 80)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    result = mcar_test(numeric_df)

    print("Chi Square:", result["chi_square"])
    print("DF:", result["df"])
    print("P Value:", result["p_value"])
    print("Missing Patterns:", result["missing_patterns"])

    result["amount_missing"].to_csv(
        f"{OUTPUT_DIR}/amount_missing.csv",
        encoding="utf-8-sig",
    )


def test_diagnose(df):
    print("\n" + "=" * 80)
    print("DIAGNOSE MISSING")
    print("=" * 80)

    result = diagnose_missing(df)

    print("Mechanism:", result["mechanism"])
    print()
    print(result["recommendation"])
    print()
    print("Hint:", result["strategy_hint"])
    print()
    print("High Missing:", result["high_missingness_cols"])


def test_high_missing(df):
    print("\n" + "=" * 80)
    print("HIGH MISSING TEST")
    print("=" * 80)

    work = df.copy()

    idx = work.sample(
        frac=0.60,
        random_state=42,
    ).index

    work.loc[idx, "PH"] = np.nan

    result = diagnose_missing(work)

    print(
        result["high_missingness_cols"]
    )


def test_mar_mnar(df):
    print("\n" + "=" * 80)
    print("MAR / MNAR TEST")
    print("=" * 80)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    y = (
        numeric_df["دما (سانتی‌گراد)"]
        .fillna(
            numeric_df["دما (سانتی‌گراد)"]
            .median()
        )
        .values
    )

    result = mar_mnar_test(
        numeric_df,
        y,
    )

    for row in result:
        print(row)


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    df = load_data()

    test_mcar(df)

    test_diagnose(df)

    test_high_missing(df)

    test_mar_mnar(df)

    print("\nStats tests completed.")


if __name__ == "__main__":
    main()