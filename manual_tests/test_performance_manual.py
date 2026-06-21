import pandas as pd

from missingly.performance import (
    memory_usage_mb,
    optimize_dtypes,
    chunk_apply,
)


def load_data():

    df = pd.read_csv(
        "./manual_tests/petrochemical_tank_sensors.csv",
        parse_dates=["زمان ثبت"],
    )

    return df


def test_memory_usage(df):

    print("\n" + "=" * 80)
    print("MEMORY USAGE")
    print("=" * 80)

    try:

        result = memory_usage_mb(df)

        print(result)

    except Exception as e:

        print(type(e).__name__)
        print(e)


def test_optimize_dtypes(df):

    print("\n" + "=" * 80)
    print("OPTIMIZE DTYPES")
    print("=" * 80)

    try:

        before = df.memory_usage(
            deep=True
        ).sum()

        optimized = optimize_dtypes(df)

        after = optimized.memory_usage(
            deep=True
        ).sum()

        print(
            "Before:",
            before,
        )

        print(
            "After:",
            after,
        )

        print(
            "Reduction:",
            round(
                100 * (before - after) / before,
                2,
            ),
            "%",
        )

        print()
        print(optimized.dtypes)

    except Exception as e:

        print(type(e).__name__)
        print(e)


def test_chunk_apply(df):

    print("\n" + "=" * 80)
    print("CHUNK APPLY")
    print("=" * 80)

    try:

        def process(chunk):

            chunk = chunk.copy()

            chunk["processed"] = 1

            return chunk

        result = chunk_apply(
            df=df,
            func=process,
            chunk_size=500,
        )

        print(result.shape)

        print(
            "processed" in result.columns
        )

    except Exception as e:

        print(type(e).__name__)
        print(e)


def test_small_chunk(df):

    print("\n" + "=" * 80)
    print("SMALL CHUNK")
    print("=" * 80)

    try:

        result = chunk_apply(
            df=df.head(100),
            func=lambda x: x,
            chunk_size=10,
        )

        print(result.shape)

    except Exception as e:

        print(type(e).__name__)
        print(e)


def main():

    df = load_data()

    print("=" * 80)
    print("DATASET SHAPE")
    print(df.shape)
    print("=" * 80)

    test_memory_usage(df)

    test_optimize_dtypes(df)

    test_chunk_apply(df)

    test_small_chunk(df)

    print("\nPerformance tests completed.")


if __name__ == "__main__":
    main()