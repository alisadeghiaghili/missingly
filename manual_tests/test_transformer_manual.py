
import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from missingly.transformer import MissinglyImputer

OUTPUT_DIR = "manual_tests/output_transformer"


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


def print_missing(df, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.isna().sum())
    print("TOTAL:", int(df.isna().sum().sum()))


def test_strategy(df, strategy):
    print("\n" + "=" * 80)
    print(f"STRATEGY: {strategy}")
    print("=" * 80)

    imp = MissinglyImputer(
        strategy=strategy,
        random_state=42,
    )

    result = imp.fit_transform(df)

    before = int(df.isna().sum().sum())
    after = int(result.isna().sum().sum())

    print("Missing Before:", before)
    print("Missing After :", after)

    result.to_csv(
        f"{OUTPUT_DIR}/{strategy}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def test_fit_then_transform(df):
    print("\n" + "=" * 80)
    print("FIT / TRANSFORM")
    print("=" * 80)

    train = df.iloc[:700].copy()
    test = df.iloc[700:].copy()

    imp = MissinglyImputer(
        strategy="mean"
    )

    imp.fit(train)

    transformed = imp.transform(test)

    print("Train Shape:", train.shape)
    print("Test Shape :", test.shape)

    print(
        "Remaining Missing:",
        int(transformed.isna().sum().sum())
    )


def test_feature_names(df):
    print("\n" + "=" * 80)
    print("FEATURE NAMES")
    print("=" * 80)

    imp = MissinglyImputer(
        strategy="mean"
    )

    imp.fit(df)

    print(
        imp.get_feature_names_out()
    )


def test_pipeline(df):
    print("\n" + "=" * 80)
    print("SKLEARN PIPELINE")
    print("=" * 80)

    y = df["دما (سانتی‌گراد)"]

    X = df.drop(
        columns=[
            "دما (سانتی‌گراد)",
            "زمان ثبت",
        ],
        errors="ignore",
    )

    # فقط ستون‌های عددی
    X = X.select_dtypes(
        include="number"
    )

    mask = y.notna()

    X = X.loc[mask]
    y = y.loc[mask]

    pipe = Pipeline(
        [
            (
                "imputer",
                MissinglyImputer(
                    strategy="mean"
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=10,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe.fit(X, y)

    preds = pipe.predict(
        X.head(20)
    )

    print("Predictions:")
    print(preds[:10])


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    df = load_data()

    print_missing(
        df,
        "ORIGINAL DATA"
    )

    for strategy in [
        "mean",
        "median",
        "mode",
        "knn",
    ]:
        test_strategy(
            df,
            strategy,
        )

    test_fit_then_transform(df)

    test_feature_names(df)

    test_pipeline(df)

    print("\nTransformer manual tests completed.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()