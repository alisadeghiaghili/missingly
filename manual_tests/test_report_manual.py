import os
import webbrowser

import pandas as pd

from missingly.report import create_report

OUTPUT_DIR = "manual_tests/output_report"


def load_data():
    df = pd.read_csv(
        "./manual_tests/petrochemical_tank_sensors.csv",
        parse_dates=["\u0632\u0645\u0627\u0646 \u062b\u0628\u062a"],
    )
    df = df.sample(n=2000, random_state=42)
    return df


def test_basic_report(df):
    print("\n" + "=" * 80)
    print("BASIC REPORT")
    print("=" * 80)

    output = create_report(
        df,
        output_path=f"{OUTPUT_DIR}/basic_report.html",
        title="Petrochemical Missing Data Report",
    )

    print("Report Created:")
    print(output)
    return output


def test_custom_missing_values(df):
    print("\n" + "=" * 80)
    print("CUSTOM MISSING VALUES")
    print("=" * 80)

    work = df.copy()
    work.loc[work.sample(20, random_state=42).index, "PH"] = -999

    output = create_report(
        work,
        output_path=f"{OUTPUT_DIR}/custom_missing_report.html",
        title="Custom Missing Values Report",
        missing_values=[-999],
    )

    print("Report Created:")
    print(output)


def test_small_dataset():
    print("\n" + "=" * 80)
    print("SMALL DATASET")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "A": [1, None, 3, None],
            "B": [10, 20, None, 40],
            "C": ["X", None, "Y", "Z"],
        }
    )

    output = create_report(
        df,
        output_path=f"{OUTPUT_DIR}/small_dataset_report.html",
        title="Small Dataset Report",
    )
    print(output)


def test_mcar_heavy_dataset(df):
    print("\n" + "=" * 80)
    print("HEAVY MISSING DATASET")
    print("=" * 80)

    work = df.copy()

    for col in [
        "\u062f\u0645\u0627 (\u0633\u0627\u0646\u062a\u06cc\u200c\u06af\u0631\u0627\u062f)",
        "\u0641\u0634\u0627\u0631 (bar)",
        "PH",
    ]:
        idx = work.sample(frac=0.40, random_state=42).index
        work.loc[idx, col] = None

    output = create_report(
        work,
        output_path=f"{OUTPUT_DIR}/heavy_missing_report.html",
        title="Heavy Missing Dataset",
    )
    print(output)


def test_report_file_exists():
    print("\n" + "=" * 80)
    print("FILE CHECK")
    print("=" * 80)

    files = os.listdir(OUTPUT_DIR)
    for f in files:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"{f:<35} {size/1024:.1f} KB")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    print("=" * 80)
    print("DATASET SHAPE")
    print(df.shape)
    print("=" * 80)

    report_path = test_basic_report(df)
    test_custom_missing_values(df)
    test_small_dataset()
    test_mcar_heavy_dataset(df)
    test_report_file_exists()

    print("\nReport manual tests completed.")

    try:
        webbrowser.open(report_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
