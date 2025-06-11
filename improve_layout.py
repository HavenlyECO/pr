#!/usr/bin/env python3
import os


def improve_table_layout():
    """Update main.py with improved table layout"""

    with open('main.py', 'r') as f:
        content = f.read()

    # Find the print_h2h_projections_table function
    start_marker = "def print_h2h_projections_table(projections: list) -> None:"
    end_marker = "def log_bet_recommendations("

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)

    if start_idx == -1 or end_idx == -1:
        print("Could not find the table function in main.py")
        return False

    # Create improved function
    improved_function = '''def print_h2h_projections_table(projections: list) -> None:
    """Display a visually appealing table for h2h projections."""

    if not projections:
        print("No projection data available.")
        return

    if tabulate is not None:
        # Define columns for a more compact and readable table
        table_data = []

        # Set up colorization if available
        use_color = Fore is not None

        for row in projections:
            prob = row.get(K_PROJECTED_WIN)
            prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"

            edge = row.get(K_EDGE)
            edge_color = ""
            edge_reset = ""

            if use_color and edge is not None:
                if edge > EDGE_THRESHOLD:
                    edge_color = Fore.GREEN
                    edge_reset = Style.RESET_ALL
                elif edge < 0:
                    edge_color = Fore.RED
                    edge_reset = Style.RESET_ALL

            edge_str = f"{edge_color}{edge*100:+.1f}%{edge_reset}" if edge is not None else "N/A"

            # Format the team names with home team highlighted
            team1 = row.get(K_TEAM1, "")
            team2 = row.get(K_TEAM2, "")

            if use_color:
                team1 = f"{Fore.CYAN}{team1}{Style.RESET_ALL}"

            # Format odds with colors based on favorites/underdogs
            price1 = row.get(K_PRICE1, 0)
            price2 = row.get(K_PRICE2, 0)

            price1_color = Fore.GREEN if use_color and price1 > 0 else ""
            price2_color = Fore.GREEN if use_color and price2 > 0 else ""
            price1_str = f"{price1_color}{'+'+ str(price1) if price1 > 0 else price1}{Style.RESET_ALL if use_color else ''}"
            price2_str = f"{price2_color}{'+'+ str(price2) if price2 > 0 else price2}{Style.RESET_ALL if use_color else ''}"

            # Recommendation indicator
            rec = ""
            if edge is not None and edge > EDGE_THRESHOLD and not row.get(K_RISK_BLOCK_FLAG):
                rec = f"{Fore.GREEN}★{Style.RESET_ALL}" if use_color else "★"

            # Add the row to table data
            table_data.append([
                rec,
                team1,
                price1_str,
                team2,
                price2_str,
                prob_str,
                edge_str,
                row.get(K_BOOKMAKER, "")
            ])

        # Print the table with nice formatting
        print(tabulate(
            table_data,
            headers=["Rec", "Team", "Odds", "Opponent", "Odds", "Win%", "Edge", "Book"],
            tablefmt="pretty",
            stralign="left",
        ))

        # Print legend at the bottom
        if use_color:
            print(f"\n{Fore.GREEN}★{Style.RESET_ALL} = Recommended bet (Edge > {EDGE_THRESHOLD*100:.1f}%)")

    else:
        # Fallback for environments without tabulate
        headers = ["REC", "TEAM1", "ODDS", "TEAM2", "ODDS", "WIN%", "EDGE", "BOOK"]

        def col_width(key: str, minimum: int) -> int:
            return max(minimum, max(len(str(row.get(key, ""))) for row in projections))

        # Define minimum widths for each column
        widths = {
            "REC": 3,
            "TEAM1": col_width(K_TEAM1, 15),
            "ODDS": 6,
            "TEAM2": col_width(K_TEAM2, 15),
            "ODDS2": 6,
            "WIN%": 6,
            "EDGE": 7,
            "BOOK": col_width(K_BOOKMAKER, 10),
        }

        # Print header
        header_line = " ".join(h.ljust(widths[h if h != "ODDS2" else "ODDS"]) for h in headers)
        print(header_line)
        print("-" * len(header_line))

        # Print data rows
        for row in projections:
            prob = row.get(K_PROJECTED_WIN)
            prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"

            edge = row.get(K_EDGE)
            edge_str = f"{edge*100:+.1f}%" if edge is not None else "N/A"

            rec = "★" if edge is not None and edge > EDGE_THRESHOLD and not row.get(K_RISK_BLOCK_FLAG) else " "

            price1 = row.get(K_PRICE1, 0)
            price2 = row.get(K_PRICE2, 0)
            price1_str = f"+{price1}" if price1 > 0 else f"{price1}"
            price2_str = f"+{price2}" if price2 > 0 else f"{price2}"

            values = [
                rec,
                row.get(K_TEAM1, ""),
                price1_str,
                row.get(K_TEAM2, ""),
                price2_str,
                prob_str,
                edge_str,
                row.get(K_BOOKMAKER, ""),
            ]

            print(" ".join(str(v).ljust(widths[h if i != 4 else "ODDS2"]) for i, (v, h) in enumerate(zip(values, headers))))

        print(f"\n★ = Recommended bet (Edge > {EDGE_THRESHOLD*100:.1f}%)")
'''

    # Replace the function in the content
    new_content = content[:start_idx] + improved_function + content[end_idx:]

    # Create backup
    os.rename('main.py', 'main.py.bak')

    # Write updated content
    with open('main.py', 'w') as f:
        f.write(new_content)

    print("Updated main.py with improved table layout")
    return True


if __name__ == "__main__":
    improve_table_layout()
