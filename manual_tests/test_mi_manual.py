
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from missingly.impute import impute_mice
from missingly.mi import (
    pool_scalar_estimates,
    pool_linear_regression_results,
)

OUTPUT_DIR = "manual_tests/output_mi"


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


def test_multiple_imputations(df):
    print("\n" + "=" * 80)
    print("MULTIPLE IMPUTATIONS")
    print("=" * 80)

    imputations = impute_mice(
        df,
        n_imputations=5,
        random_state=42,
        max_iter=5,
    )

    print("Number of imputations:", len(imputations))

    for i, imp_df in enumerate(imputations):
        missing = int(imp_df.isna().sum().sum())

        print(
            f"Imputation {i + 1}: missing={missing}"
        )

        imp_df.to_csv(
            f"{OUTPUT_DIR}/imputation_{i+1}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return imputations


def test_scalar_pooling():
    print("\n" + "=" * 80)
    print("SCALAR POOLING")
    print("=" * 80)

    estimates = [
        10.1,
        10.4,
        9.9,
        10.3,
        10.0,
    ]

    variances = [
        0.25,
        0.22,
        0.27,
        0.21,
        0.24,
    ]

    result = pool_scalar_estimates(
        estimates,
        variances,
    )

    for k, v in result.items():
        print(f"{k}: {v}")


def test_regression_pooling(imputations):
    print("\n" + "=" * 80)
    print("REGRESSION POOLING")
    print("=" * 80)

    coefs = []
    covs = []

    for df in imputations:

        work = df.copy()

        work = work.select_dtypes(
            include="number"
        )

        if len(work.columns) < 3:
            continue

        y = work["دما (سانتی‌گراد)"]

        X = work.drop(
            columns=["دما (سانتی‌گراد)"]
        )

        model = LinearRegression()
        model.fit(X, y)

        coef = model.coef_

        residuals = y - model.predict(X)

        sigma2 = np.var(
            residuals,
            ddof=len(coef),
        )

        xtx_inv = np.linalg.pinv(
            X.T @ X
        )

        cov = sigma2 * xtx_inv

        coefs.append(coef)
        covs.append(cov)

    coefs = np.asarray(coefs)
    covs = np.asarray(covs)

    result = pool_linear_regression_results(
        coefs,
        covs,
    )

    print("\nPooled Coefficients:")
    print(result["coef_"])

    print("\nStandard Errors:")
    print(result["se_"])

    print("\nDegrees of Freedom:")
    print(result["df_"])

    np.save(
        f"{OUTPUT_DIR}/pooled_coef.npy",
        result["coef_"],
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

    imputations = test_multiple_imputations(df)

    test_scalar_pooling()

    test_regression_pooling(
        imputations
    )

    print("\nMI manual tests completed.")
    print(f"Check: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()