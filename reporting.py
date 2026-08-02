"""Self-contained, escaped HTML reports for reproducible descriptive analysis."""

from __future__ import annotations

import html
from typing import Any

from analytics import missingness_report, profile_dataset


__all__ = ["build_descriptive_report"]


def _cell(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_cell(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_descriptive_report(records: list[dict[str, Any]], title: str) -> str:
    """Build an escaped, self-contained HTML descriptive-analysis report in memory.

    Args:
        records: JSON-like row objects accepted by the profiling and missingness APIs.
        title: Reader-facing report title; HTML-sensitive characters are escaped.

    Returns:
        A complete UTF-8-compatible HTML document with dataset, variable, missingness,
        correlation, and interpretation-boundary sections.

    Raises:
        AnalyticsError: If records cannot be profiled as a valid rectangular dataset.

    Examples:
        >>> "Dataset overview" in build_descriptive_report([{"score": 10}], "Study")
        True
    """
    profile = profile_dataset(records)
    missingness = missingness_report(records)
    dataset = profile["dataset"]
    profile_rows = [
        [
            column["name"],
            column["kind"],
            column["non_missing"],
            column["missing"],
            column["missing_rate"],
            column.get("mean"),
            column.get("median"),
            column.get("std_dev"),
            column["distinct"],
        ]
        for column in profile["columns"]
    ]
    correlation_rows = [[item["left"], item["right"], item["n"], item["pearson_r"], item["p_value"]] for item in profile["correlations"]]
    missingness_rows = [
        [", ".join(item["missing_columns"]) if item["missing_columns"] else "Complete row", item["rows"], item["row_rate"]]
        for item in missingness["patterns"]
    ]
    warnings = "".join(f"<li>{_cell(warning)}</li>" for warning in missingness["warnings"])
    fingerprint = profile["reproducibility"]["input_sha256"]
    safe_title = html.escape(title.strip() or "Missingly statistical report")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title><style>
body{{font-family:Arial,sans-serif;color:#14213d;max-width:1100px;margin:36px auto;padding:0 24px;line-height:1.5}}
h1,h2{{color:#0b3d5c}} .meta{{color:#4b6175}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px;font-size:13px}}
th,td{{border:1px solid #c8d4df;padding:8px;text-align:left}}th{{background:#e9f2f8}}code{{word-break:break-all}} .note{{background:#fff8df;padding:12px;border-left:4px solid #d99f00}}
</style></head><body>
<h1>{safe_title}</h1><p class="meta">Engine: {html.escape(profile['reproducibility']['engine'])} · Input SHA-256: <code>{html.escape(fingerprint)}</code></p>
<h2>Dataset overview</h2>{_table(["Rows", "Columns", "Complete rows", "Duplicate rows", "Missing cells", "Missing-cell rate"], [[dataset['rows'], dataset['columns'], dataset['complete_rows'], dataset['duplicate_rows'], dataset['missing_cells'], dataset['missing_cell_rate']]])}
<h2>Variable profile</h2>{_table(["Column", "Kind", "Observed", "Missing", "Missing rate", "Mean", "Median", "SD", "Distinct"], profile_rows)}
<h2>Missing-data notes</h2><div class="note"><ul>{warnings}</ul></div>
<h2>Missing-data patterns</h2>{_table(["Missing columns", "Rows", "Row rate"], missingness_rows) if missingness_rows else '<p>No missing-data patterns were observed.</p>'}
<h2>Pearson correlations</h2>{_table(["Left", "Right", "n", "r", "p-value"], correlation_rows) if correlation_rows else '<p>No eligible numeric correlation pairs.</p>'}
<h2>Interpretation boundary</h2><div class="note">This report is descriptive. It does not establish causality, the missing-data mechanism, model fit, proportional hazards, or a complex-survey design. Review assumptions, measurement quality, and study design before inference.</div>
</body></html>"""
