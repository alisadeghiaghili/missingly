import os

import matplotlib.pyplot as plt
import pandas as pd

import missingly


OUTPUT_DIR = "manual_tests/output_accessor"


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


def save_plot(name):
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{name}.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def test_summary_api(df):
    print("\n" + "=" * 80)
    print("SUMMARY API")
    print("=" * 80)

    print("n_miss:", df.miss.n_miss())
    print("n_complete:", df.miss.n_complete())
    print("pct_miss:", df.miss.pct_miss())
    print("pct_complete:", df.miss.pct_complete())

    print("\nVariable Summary")
    print(df.miss.miss_var_summary().head())

    print("\nCase Summary")
    print(df.miss.miss_case_summary().head())


def test_bind_shadow(df):
    print("\n" + "=" * 80)
    print("BIND SHADOW")
    print("=" * 80)

    shadow = df.miss.bind_shadow()

    print("Original Shape:", df.shape)
    print("Shadow Shape:", shadow.shape)

    shadow.to_csv(
        f"{OUTPUT_DIR}/bind_shadow.csv",
        index=False,
        encoding="utf-8-sig",
    )


def test_manipulation(df):
    print("\n" + "=" * 80)
    print("MANIPULATION")
    print("=" * 80)

    result = (
        df
        .miss
        .miss_as_feature()
    )

    print(
        "After miss_as_feature:",
        result.shape,
    )

    result.to_csv(
        f"{OUTPUT_DIR}/miss_as_feature.csv",
        index=False,
        encoding="utf-8-sig",
    )


def test_clean_names():
    print("\n" + "=" * 80)
    print("CLEAN NAMES")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "First Name": [1, 2],
            "Last Name": [3, 4],
            "Phone Number": [5, 6],
        }
    )

    result = df.miss.clean_names()

    print(result.columns.tolist())


def test_imputation(df):
    print("\n" + "=" * 80)
    print("IMPUTATION")
    print("=" * 80)

    mean_df = df.miss.impute_mean()

    print(
        "Remaining Missing:",
        int(mean_df.isna().sum().sum())
    )

    median_df = df.miss.impute_median()

    print(
        "Median Remaining:",
        int(median_df.isna().sum().sum())
    )


def test_stats(df):
    print("\n" + "=" * 80)
    print("STATS")
    print("=" * 80)

    numeric_df = df.select_dtypes(
        include="number"
    )

    result = numeric_df.miss.mcar_test()

    print(result)

    try:
        mar_result = (
            numeric_df
            .miss
            .mar_mnar_test()
        )

        print("\nMAR/MNAR")
        print(mar_result)

    except Exception as exc:
        print(
            "mar_mnar_test FAILED:"
        )
        print(exc)


def test_visualisation(df):
    print("\n" + "=" * 80)
    print("VISUALISATION")
    print("=" * 80)

    df.miss.matrix()
    save_plot("matrix")

    df.miss.bar()
    save_plot("bar")

    df.miss.vis_miss()
    save_plot("vis_miss")

    df.miss.heatmap()
    save_plot("heatmap")


def test_chain(df):
    print("\n" + "=" * 80)
    print("CHAINING")
    print("=" * 80)

    result = (
        df
        .miss
        .impute_mean()
        .miss
        .miss_as_feature()
    )

    print(
        "Final Shape:",
        result.shape
    )

    result.to_csv(
        f"{OUTPUT_DIR}/chain_result.csv",
        index=False,
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

    test_summary_api(df)

    test_bind_shadow(df)

    test_manipulation(df)

    test_clean_names()

    test_imputation(df)

    test_stats(df)

    test_visualisation(df)

    test_chain(df)

    print("\nAccessor tests completed.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()