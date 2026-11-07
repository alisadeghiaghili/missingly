"""Security lock for the HTML report (SQ-13).

These tests pin two properties of :func:`missingly.report.create_report`:

1. The Jinja2 environment that renders the report is created with
   ``autoescape=True``.
2. Untrusted strings (column names, row index labels, categorical values and
   the report title) reach the output only in HTML-escaped form.
"""

from __future__ import annotations

from unittest import mock

import jinja2
import pandas as pd
import pytest

from missingly.report import create_report

SCRIPT_NAME = "<script>alert(1)</script>"
SCRIPT_ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"
IMG_NAME = '"><img src=x onerror=alert(1)>'
IMG_ESCAPED = "&#34;&gt;&lt;img src=x onerror=alert(1)&gt;"


def _render(df: pd.DataFrame, output_path: str, **kwargs) -> str:
    """Generate a report with ``create_report`` and return its HTML text."""
    create_report(df, output_path=output_path, **kwargs)
    with open(output_path, encoding="utf-8") as fh:
        return fh.read()


def test_jinja_environment_has_autoescape_enabled(tmp_path):
    """The real Environment object used by create_report autoescapes output."""
    created: list[jinja2.Environment] = []
    real_environment = jinja2.Environment

    def _spy(*args, **kwargs):
        env = real_environment(*args, **kwargs)
        created.append(env)
        return env

    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0]})
    with mock.patch("missingly.report.Environment", side_effect=_spy):
        create_report(df, output_path=str(tmp_path / "report.html"))

    assert len(created) == 1
    assert created[0].autoescape is True


@pytest.mark.parametrize(
    ("raw_name", "escaped_name"),
    [
        (SCRIPT_NAME, SCRIPT_ESCAPED),
        (IMG_NAME, IMG_ESCAPED),
    ],
)
def test_column_name_appears_only_escaped(tmp_path, raw_name, escaped_name):
    """A column name containing HTML is rendered escaped and never raw."""
    df = pd.DataFrame({raw_name: [1.0, None, 3.0], "plain": [None, 2.0, 3.0]})

    html = _render(df, str(tmp_path / "report.html"))

    assert raw_name not in html
    assert escaped_name in html


def test_index_label_title_and_categorical_value_are_escaped(tmp_path):
    """HTML in the title, row index labels and categorical values is escaped.

    The row index labels are rendered in the "Top 10 Rows" table. Categorical
    cell values are not rendered by the template; the assertion still checks
    that their raw markup never reaches the output.
    """
    index = ["<img src=x onerror=alert(1)>", "row-2", "row-3"]
    df = pd.DataFrame(
        {
            "cat": pd.Categorical([None, "<b>bold</b>", "plain"]),
            "x": [None, 2.0, 3.0],
            "y": [1.0, None, 3.0],
        },
        index=index,
    )

    html = _render(
        df,
        str(tmp_path / "report.html"),
        title=SCRIPT_NAME,
    )

    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<b>bold</b>" not in html
    assert SCRIPT_NAME not in html
    assert SCRIPT_ESCAPED in html
