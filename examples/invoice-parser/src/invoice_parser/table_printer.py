"""Pretty-print invoice results as a table with box-drawing characters."""

from pathlib import Path

import click


def print_results_table(results: list[dict]) -> None:
    """Print invoice results as a formatted table."""
    headers = ["File", "Utility", "Amount", "Currency"]
    rows = [
        [
            Path(r["file_path"]).name,
            r["utility"],
            str(r["amount"]),
            r["currency"],
        ]
        for r in results
    ]

    col_widths = [
        max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)
    ]

    def fmt_row(row: list[str]) -> str:
        cells = " │ ".join(v.ljust(w) for v, w in zip(row, col_widths))
        return f"│ {cells} │"

    top = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    sep = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    bot = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    click.echo(top)
    click.echo(fmt_row(headers))
    for row in rows:
        click.echo(sep)
        click.echo(fmt_row(row))
    click.echo(bot)
