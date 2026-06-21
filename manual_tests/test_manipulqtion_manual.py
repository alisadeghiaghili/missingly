import pandas as pd
import numpy as np

from missingly.manipulation import (
    replace_with_na,
    replace_with_na_all,
    add_any_miss_var,
    bind_shadow_matrix,
    miss_as_feature,
)


def test_replace_with_na():
    print("\n" + "=" * 80)
    print("REPLACE WITH NA")
    print("=" * 80)

    df = pd.DataFrame({
        "A": [1, -99, 3],
        "B": ["ok", "N/A", "ok"]
    })

    result = replace_with_na(
        df,
        {
            "A": -99,
            "B": "N/A",
        }
    )

    print(result)


def test_replace_with_na_all():
    print("\n" + "=" * 80)
    print("REPLACE WITH NA ALL")
    print("=" * 80)

    df = pd.DataFrame({
        "A": [1, -99, 3],
        "B": [-99, 2, -99]
    })

    result = replace_with_na_all(
        df,
        lambda x: x == -99
    )

    print(result)


def test_add_any_miss_var():
    print("\n" + "=" * 80)
    print("ADD ANY MISS VAR")
    print("=" * 80)

    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [4, 5, np.nan]
    })

    result = add_any_miss_var(df)

    print(result)


def test_bind_shadow_matrix():
    print("\n" + "=" * 80)
    print("BIND SHADOW MATRIX")
    print("=" * 80)

    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [np.nan, 2, 3]
    })

    result = bind_shadow_matrix(df)

    print(result)


def test_miss_as_feature():
    print("\n" + "=" * 80)
    print("MISS AS FEATURE")
    print("=" * 80)

    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [np.nan, 2, 3],
        "C": [10, 11, 12]
    })

    result = miss_as_feature(df)

    print(result)

    print("Columns:")
    print(result.columns.tolist())


def test_miss_as_feature_drop_original():
    print("\n" + "=" * 80)
    print("MISS AS FEATURE DROP ORIGINAL")
    print("=" * 80)

    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [np.nan, 2, 3]
    })

    result = miss_as_feature(
        df,
        keep_original=False
    )

    print(result)


def main():

    test_replace_with_na()

    test_replace_with_na_all()

    test_add_any_miss_var()

    test_bind_shadow_matrix()

    test_miss_as_feature()

    test_miss_as_feature_drop_original()

    print("\nManipulation tests completed.")


if __name__ == "__main__":
    main()