import pandas as pd

from missingly.simulate import (
    simulate_mcar,
    simulate_mar,
    simulate_mnar,
    simulate_mixed,
)


def load_data():

    df = pd.read_csv(
        "./manual_tests/petrochemical_tank_sensors.csv",
        parse_dates=["زمان ثبت"],
    )

    # simulate نیاز به دیتای کامل دارد
    df = df.dropna().copy()

    if len(df) > 1000:
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

    total = int(df.isna().sum().sum())

    print()
    print("Total Missing:", total)


def test_mcar(df):

    result = simulate_mcar(
        df,
        frac=0.10,
        random_state=42,
    )

    print_missing(
        result,
        "MCAR",
    )


def test_mcar_single_column(df):

    result = simulate_mcar(
        df,
        frac=0.20,
        columns=["PH"],
        random_state=42,
    )

    print_missing(
        result,
        "MCAR SINGLE COLUMN",
    )


def test_mar(df):

    result = simulate_mar(
        df,
        frac=0.15,
        target_col="PH",
        predictor_col="دما (سانتی‌گراد)",
        random_state=42,
    )

    print_missing(
        result,
        "MAR",
    )


def test_mnar(df):

    result = simulate_mnar(
        df,
        frac=0.15,
        target_col="PH",
        random_state=42,
    )

    print_missing(
        result,
        "MNAR",
    )


def test_mixed(df):

    result = simulate_mixed(
        df,
        mechanisms=[
            {
                "mechanism": "mcar",
                "frac": 0.10,
                "columns": ["PH"],
            },
            {
                "mechanism": "mar",
                "frac": 0.10,
                "target_col": "فشار (bar)",
                "predictor_col": "دما (سانتی‌گراد)",
            },
            {
                "mechanism": "mnar",
                "frac": 0.10,
                "target_col": "سطح مایع (%)",
            },
        ],
        random_state=42,
    )

    print_missing(
        result,
        "MIXED",
    )


def test_reproducibility(df):

    print("\n" + "=" * 80)
    print("REPRODUCIBILITY")
    print("=" * 80)

    a = simulate_mcar(
        df,
        frac=0.10,
        random_state=42,
    )

    b = simulate_mcar(
        df,
        frac=0.10,
        random_state=42,
    )

    print(
        "Equal:",
        a.equals(b)
    )


def test_invalid_frac(df):

    print("\n" + "=" * 80)
    print("INVALID FRAC")
    print("=" * 80)

    try:

        simulate_mcar(
            df,
            frac=1.5,
        )

    except Exception as e:

        print(type(e).__name__)
        print(e)


def main():

    df = load_data()

    print("=" * 80)
    print("DATASET SHAPE")
    print(df.shape)
    print("=" * 80)

    test_mcar(df)

    test_mcar_single_column(df)

    test_mar(df)

    test_mnar(df)

    test_mixed(df)

    test_reproducibility(df)

    test_invalid_frac(df)

    print("\nSimulate tests completed.")


if __name__ == "__main__":
    main()
