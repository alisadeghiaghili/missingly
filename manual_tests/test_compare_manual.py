import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from missingly.compare import (
    compare_imputations,
    cv_compare_imputations,
)

OUTPUT_DIR = "manual_tests/output_compare"


def load_data():
    df = pd.read_csv(
        "./manual_tests/timeseries_duplicate_index.csv",
        parse_dates=["زمان ثبت"],
def create_complete_dataset(df):
    """
    compare_imputations نیاز به دیتاست کامل دارد.
    """

    complete_df = df.dropna().copy()

    # زمان برای مقایسه بازسازی مهم نیست
    complete_df = complete_df.drop(
        columns=["زمان ثبت"],
        errors="ignore",
    )

    return complete_df


def test_compare_reconstruction(df):
    print("\n" + "=" * 80)
    print("RECONSTRUCTION BENCHMARK")
    print("=" * 80)

    result = compare_imputations(
        df,
        mask_frac=0.20,
        random_state=42,
    )

    print(result)

    result.to_csv(
        f"{OUTPUT_DIR}/compare_reconstruction.csv",
        encoding="utf-8-sig",
    )


def test_compare_mask_levels(df):
    print("\n" + "=" * 80)
    print("MASK FRACTION SENSITIVITY")
    print("=" * 80)

    for frac in [0.05, 0.10, 0.20, 0.30]:
        print(f"\nMask Fraction = {frac}")

        result = compare_imputations(
            df,
            mask_frac=frac,
            random_state=42,
        )

        print(result)

        result.to_csv(
            f"{OUTPUT_DIR}/compare_mask_{int(frac * 100)}.csv",
            encoding="utf-8-sig",
        )


def create_cv_dataset(df):
    """
    هدف:
    پیش‌بینی دما از روی سایر سنسورها
    """

    y = df["دما (سانتی‌گراد)"].copy()

    X = df.drop(
        columns=[
            "دما (سانتی‌گراد)",
            "زمان ثبت",
        ],
        errors="ignore",
    )

    # فقط ستون‌های عددی
    X = X.select_dtypes(
        include=[np.number]
    )

    mask = y.notna()

    X = X.loc[mask]
    y = y.loc[mask]

    return X, y


def test_cv_compare(df):
    print("\n" + "=" * 80)
    print("CROSS VALIDATION BENCHMARK")
    print("=" * 80)

    X, y = create_cv_dataset(df)

    results = cv_compare_imputations(
        X=X,
        y=y,
        estimator=RandomForestRegressor(
            n_estimators=20,
            random_state=42,
            n_jobs=-1,
        ),
        strategies=[
            "mean",
            "median",
            "mode",
            "knn",
        ],
        n_splits=2,
        random_state=42,
    )

    print(results)

    results.to_csv(
        f"{OUTPUT_DIR}/cv_compare.csv",
        encoding="utf-8-sig",
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    df = load_data()

    print("=" * 80)
    print("DATASET SHAPE")
    print(df.shape)
    print("=" * 80)

    complete_df = create_complete_dataset(df)

    print("\nComplete Dataset Shape")
    print(complete_df.shape)

    test_compare_reconstruction(
        complete_df
    )

    test_compare_mask_levels(
        complete_df
    )

    test_cv_compare(
        df
    )

    print("\nAll compare manual tests completed.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()