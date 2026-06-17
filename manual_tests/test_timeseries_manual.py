import os

import matplotlib.pyplot as plt
import pandas as pd

from missingly.timeseries import (
    gap_table,
    impute_ts,
    miss_ts_summary,
    vis_gap_lengths,
    vis_miss_over_time,
    vis_ts_miss,
)


OUTPUT_DIR = "manual_tests/output_timeseries"


def save_plot(name):
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/{name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def load_data():
    df = pd.read_csv(
        "./manual_tests/petrochemical_tank_sensors.csv",
        parse_dates=["زمان ثبت"],
    )

    df = (
        df
        .sort_values("زمان ثبت")
        .drop_duplicates(subset=["زمان ثبت"])
        .set_index("زمان ثبت")
    )
    # df = df.sort_values("زمان ثبت")
    # df = df.set_index("زمان ثبت")

    return df


def test_miss_ts_summary(df):
    print("\n" + "=" * 80)
    print("TIME SERIES SUMMARY")
    print("=" * 80)

    result = miss_ts_summary(df)

    print(result)

    result.to_csv(
        f"{OUTPUT_DIR}/miss_ts_summary.csv",
        encoding="utf-8-sig",
    )


def test_gap_table(df):
    print("\n" + "=" * 80)
    print("GAP TABLE")
    print("=" * 80)

    gaps = gap_table(df)

    print(gaps.head(20))

    gaps.to_csv(
        f"{OUTPUT_DIR}/gap_table.csv",
        index=False,
        encoding="utf-8-sig",
    )


def test_vis_ts_miss(df):
    vis_ts_miss(df)

    save_plot("vis_ts_miss")


def test_vis_gap_lengths_hist(df):
    vis_gap_lengths(
        df,
        kind="hist",
    )

    save_plot("gap_lengths_hist")


def test_vis_gap_lengths_box(df):
    vis_gap_lengths(
        df,
        kind="box",
    )

    save_plot("gap_lengths_box")


def test_vis_miss_over_time(df):
    vis_miss_over_time(
        df,
        window=100,
    )

    save_plot("miss_over_time")


def test_impute_linear(df):
    result = impute_ts(
        df,
        method="linear",
    )

    print("\nLinear imputation")

    before = int(df.isna().sum().sum())
    after = int(result.isna().sum().sum())

    print("Missing before:", before)
    print("Missing after :", after)

    result.to_csv(
        f"{OUTPUT_DIR}/imputed_linear.csv",
        encoding="utf-8-sig",
    )


def test_impute_ffill(df):
    result = impute_ts(
        df,
        method="ffill",
    )

    print("\nForward Fill")

    before = int(df.isna().sum().sum())
    after = int(result.isna().sum().sum())

    print("Missing before:", before)
    print("Missing after :", after)

    result.to_csv(
        f"{OUTPUT_DIR}/imputed_ffill.csv",
        encoding="utf-8-sig",
    )


def test_impute_bfill(df):
    result = impute_ts(
        df,
        method="bfill",
    )

    print("\nBackward Fill")

    before = int(df.isna().sum().sum())
    after = int(result.isna().sum().sum())

    print("Missing before:", before)
    print("Missing after :", after)

    result.to_csv(
        f"{OUTPUT_DIR}/imputed_bfill.csv",
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

    test_miss_ts_summary(df)

    test_gap_table(df)

    test_vis_ts_miss(df)

    test_vis_gap_lengths_hist(df)
    test_vis_gap_lengths_box(df)

    test_vis_miss_over_time(df)

    test_impute_linear(df)
    test_impute_ffill(df)
    test_impute_bfill(df)

    print("\nAll time-series manual tests completed.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()