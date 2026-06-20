import os

import pandas as pd

from missingly.transformer import MissinglyImputer

OUTPUT_DIR = "manual_tests/output_transformer_tree"


def load_data():
    df = pd.read_csv(
        "./manual_tests/petrochemical_tank_sensors.csv",
        parse_dates=["زمان ثبت"],
    )

    # برای RF و GB حجم کم نگه می‌داریم
    df = df.sample(
        n=500,
        random_state=42,
    )

    return df


def run_strategy(df, strategy):
    print("\n" + "=" * 80)
    print(f"TESTING {strategy.upper()}")
    print("=" * 80)

    before = int(df.isna().sum().sum())

    imputer = MissinglyImputer(
        strategy=strategy,
        random_state=42,
    )

    result = imputer.fit_transform(df)

    after = int(result.isna().sum().sum())

    print("Missing Before:", before)
    print("Missing After :", after)

    print("\nColumn Missing Counts:")
    print(result.isna().sum())

    result.to_csv(
        f"{OUTPUT_DIR}/{strategy}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result


def test_fit_transform_equivalence(df):
    print("\n" + "=" * 80)
    print("FIT VS FIT_TRANSFORM")
    print("=" * 80)

    imputer = MissinglyImputer(
        strategy="rf",
        random_state=42,
    )

    result1 = imputer.fit_transform(df)

    imputer = MissinglyImputer(
        strategy="rf",
        random_state=42,
    )

    imputer.fit(df)
    result2 = imputer.transform(df)

    same_shape = result1.shape == result2.shape

    print("Same Shape:", same_shape)
    print("Rows:", result1.shape[0])
    print("Cols:", result1.shape[1])


def test_column_order(df):
    print("\n" + "=" * 80)
    print("COLUMN ORDER")
    print("=" * 80)

    original_cols = list(df.columns)

    imputer = MissinglyImputer(
        strategy="gb",
        random_state=42,
    )

    result = imputer.fit_transform(df)

    transformed_cols = list(result.columns)

    print(
        "Column Order Preserved:",
        original_cols == transformed_cols
    )


def test_no_missing_remaining(df):
    print("\n" + "=" * 80)
    print("FINAL MISSING CHECK")
    print("=" * 80)

    for strategy in ["rf", "gb"]:

        imputer = MissinglyImputer(
            strategy=strategy,
            random_state=42,
        )

        result = imputer.fit_transform(df)

        remaining = int(
            result.isna().sum().sum()
        )

        print(
            f"{strategy}: remaining missing = {remaining}"
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

    print("\nOriginal Missing Values:")
    print(df.isna().sum())

    run_strategy(
        df,
        "rf",
    )

    run_strategy(
        df,
        "gb",
    )

    test_fit_transform_equivalence(df)

    test_column_order(df)

    test_no_missing_remaining(df)

    print("\nTree transformer tests completed.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()