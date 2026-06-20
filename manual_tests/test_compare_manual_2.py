import os

import pandas as pd

from missingly.compare import (
    compare_imputations,
)

OUTPUT_DIR = "manual_tests/output_compare"


def load_data():
    df = pd.read_csv(
        "./manual_tests/timeseries_duplicate_index.csv",
        parse_dates=["timestamp"],
    )
    return df


def prepare_complete_dataset(df):
    """
    compare_imputations نیاز به دیتاست کامل دارد.
    """
    complete_df = df.dropna().copy()
    print("Rows after dropna:", len(complete_df))
    return complete_df


def test_basic_compare(df):
    print("\n" + "=" * 80)
    print("BASIC COMPARE")
    print("=" * 80)

    result = compare_imputations(
        df,
        mask_frac=0.20,
        random_state=42,
    )

    print(result)

    result.to_csv(
        f"{OUTPUT_DIR}/compare_basic.csv",
        encoding="utf-8-sig",
    )


def test_multiple_mask_levels(df):
    print("\n" + "=" * 80)
    print("MASK LEVELS")
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
            f"{OUTPUT_DIR}/compare_mask_{int(frac*100)}.csv",
            encoding="utf-8-sig",
        )


def test_numeric_only(df):
    print("\n" + "=" * 80)
    print("NUMERIC ONLY")
    print("=" * 80)

    numeric_df = df.select_dtypes(include="number")

    result = compare_imputations(
        numeric_df,
        mask_frac=0.20,
        random_state=42,
    )

    print(result)

    result.to_csv(
        f"{OUTPUT_DIR}/compare_numeric_only.csv",
        encoding="utf-8-sig",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    print("=" * 80)
    print("DATASET SHAPE")
    print(df.shape)
    print("=" * 80)

    complete_df = prepare_complete_dataset(df)

    test_basic_compare(complete_df)
    test_multiple_mask_levels(complete_df)
    test_numeric_only(complete_df)

    print("\nCompare manual tests completed.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
