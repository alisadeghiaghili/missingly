import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO
import os

from .summary import miss_var_summary
from .visualise import matrix, bar, upset, dendrogram

def create_report(df: pd.DataFrame, output_path: str = "missing_data_report.html"):
    """
    Generates an HTML report summarizing the missing data analysis.

    Args:
        df: The dataframe to analyze.
        output_path: The path to save the HTML report to.
    """
    # 1. Generate summary table
    summary_df = miss_var_summary(df)
    summary_table_html = summary_df.to_html(index=False)

    # 2. Generate plots and encode them
    plots = {}
    plot_functions = {
        "matrix_plot": matrix,
        "bar_plot": bar,
        "upset_plot": upset,
        "dendrogram_plot": dendrogram,
    }

    for name, plot_func in plot_functions.items():
        plt.figure() # Create a new figure for each plot
        if name == "upset_plot":
            plot_func(df)
        else:
            ax = plt.gca()
            plot_func(df, ax=ax)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()

        plots[name] = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 3. Render the HTML template
    # Set up Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html")

    # Render the template with the data
    html_content = template.render(
        summary_table=summary_table_html,
        **plots
    )

    # 4. Save the rendered HTML to a file
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Report saved to {output_path}")
