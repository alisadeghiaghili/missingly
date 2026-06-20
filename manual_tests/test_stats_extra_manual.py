import pandas as pd

from missingly.stats_extra import (
    pattern_monotone_test,
    missing_correlation_matrix,
)


def load_data():
    return pd.read_csv(
        "./manual_tests/petrochemical_tank_sensors.csv",
        parse_dates=["زمان ثبت"],
    ).sample(
        n=1000,
        random_state=42,
    )


def test_monotone():
    print("\n" + "=" * 80)
    print("MONOTONE PATTERN")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "A": [1, 2, 3, None, None],
            "B": [5, 6, None, None, None],
            "C": [8, None, None, None, None],
        }
    )

    result = pattern_monotone_test(df)

    print(result)


def test_non_monotone():
    print("\n" + "=" * 80)
    print("NON MONOTONE PATTERN")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "A": [1, None, 3],
            "B": [None, 2, None],
            "C": [3, None, 5],
        }
    )

    result = pattern_monotone_test(df)

    print(result)


def test_missing_corr(df):
    print("\n" + "=" * 80)
    print("MISSING CORRELATION")
    print("=" * 80)

    corr = missing_correlation_matrix(df)

    print(corr)

    corr.to_csv(
        "manual_tests/output_stats_extra/correlation.csv",
        encoding="utf-8-sig",
    )


def test_no_missing():
    print("\n" + "=" * 80)
    print("NO MISSING")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )

    corr = missing_correlation_matrix(df)

    print(corr)


def main():
    import os

    os.makedirs(
        "manual_tests/output_stats_extra",
        exist_ok=True,
    )

    df = load_data()

    test_monotone()

    test_non_monotone()

    test_missing_corr(df)

    test_no_missing()

    print("\nStats Extra tests completed.")


if __name__ == "__main__":
    main()