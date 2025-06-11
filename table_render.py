"""Utility script for debugging table rendering."""

from __future__ import annotations

import pandas as pd

try:
    from tabulate import tabulate
    from colorama import Fore, Style, init as colorama_init
except ImportError:  # pragma: no cover - dev dependency
    tabulate = None
    Fore = Style = None

from main import should_highlight_row


def render_table(rows: list[dict]) -> None:
    """Render ``rows`` as a table highlighting recommended bets."""

    if not rows:
        print("No rows to display")
        return

    if tabulate is not None:
        colorama_init(autoreset=True)
        df = pd.DataFrame(rows)
        df["highlight"] = df["edge"].apply(should_highlight_row)

        table_data = []
        use_color = Fore is not None

        for row in df.to_dict("records"):
            rec = "★" if row.get("highlight") else " "
            if use_color and row.get("highlight"):
                rec = f"{Fore.GREEN}{rec}{Style.RESET_ALL}"

            table_data.append([
                rec,
                row.get("team"),
                row.get("edge"),
            ])

        print(tabulate(table_data, headers=["Rec", "Team", "Edge"], tablefmt="pretty"))
    else:  # pragma: no cover - fallback
        for row in rows:
            rec = "★" if should_highlight_row(row.get("edge")) else " "
            print(f"{rec} {row.get('team')}: edge={row.get('edge')}")


if __name__ == "__main__":  # pragma: no cover - manual debug
    SAMPLE = [
        {"team": "Team A", "edge": 0.05},
        {"team": "Team B", "edge": 0.08},
    ]

    render_table(SAMPLE)

