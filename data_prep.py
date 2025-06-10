import pickle
import json
import pprint
from pathlib import Path
import pandas as pd


def american_odds_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


DEFAULT_CACHE_DIR = Path("h2h_data") / "api_cache"


def examine_cache_structure(cache_dir: str | Path = DEFAULT_CACHE_DIR, max_files: int = 5) -> None:
    """Print a detailed look at cached API responses for debugging."""
    cache_path = Path(cache_dir)
    files = list(cache_path.glob("*.pkl"))
    if not files:
        print(f"No cache files found in {cache_path}")
        return

    print(
        f"Found {len(files)} cache files. Examining {min(max_files, len(files))} files in detail."
    )

    for i, file_path in enumerate(files[:max_files]):
        print(f"\n{'='*80}\nFile {i+1}: {file_path.name}")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "data" in data:
                print("Structure: Dictionary with 'data' key")
                other_keys = [k for k in data.keys() if k != "data"]
                if other_keys:
                    print(f"Other keys: {other_keys}")
                inner_data = data["data"]
                if isinstance(inner_data, list):
                    print(f"Data: List of {len(inner_data)} events")
                    events = inner_data
                elif isinstance(inner_data, dict):
                    print("Data: Single event dictionary")
                    events = [inner_data]
                else:
                    print(f"Data is of unexpected type: {type(inner_data)}")
                    continue
            elif isinstance(data, list):
                print(f"Structure: Direct list of {len(data)} events")
                events = data
            elif isinstance(data, dict):
                print(f"Structure: Dictionary without 'data' key. Keys: {list(data.keys())}")
                if "id" in data and "bookmakers" in data:
                    print("Appears to be a single event")
                    events = [data]
                else:
                    print("Not recognized as event data")
                    continue
            else:
                print(f"Unrecognized data structure: {type(data)}")
                continue

            if not events:
                print("No events found in the data")
                continue

            for event_idx, event in enumerate(events[:2]):
                print(f"\nExamining Event {event_idx + 1}:")
                if not isinstance(event, dict):
                    print(f"Event is not a dictionary: {type(event)}")
                    continue
                print(f"Event ID: {event.get('id', 'N/A')}")
                print(f"Sport: {event.get('sport_key', 'N/A')}")
                print(f"Teams: {event.get('home_team', 'N/A')} vs {event.get('away_team', 'N/A')}")

                bookmakers = event.get("bookmakers", [])
                if not bookmakers:
                    print("No bookmakers found!")
                    continue
                print(f"Found {len(bookmakers)} bookmakers")
                book = bookmakers[0]
                print(f"First bookmaker: {book.get('key', 'unknown')}")
                markets = book.get("markets", [])
                if not markets:
                    print("No markets found!")
                    continue
                print(f"Found {len(markets)} markets")
                h2h_markets = [m for m in markets if m.get("key") == "h2h"]
                if not h2h_markets:
                    print("No h2h markets found!")
                    continue
                print(f"Found {len(h2h_markets)} h2h markets")
                market = h2h_markets[0]
                outcomes = market.get("outcomes", [])
                if not outcomes:
                    print("No outcomes found!")
                    continue
                print(f"H2h market has {len(outcomes)} outcomes:")
                for outcome_idx, outcome in enumerate(outcomes):
                    print(f"\n  Outcome {outcome_idx + 1}:")
                    print(f"    Name: {outcome.get('name', 'N/A')}")
                    print(f"    Price: {outcome.get('price', 'N/A')}")
                    print(f"    Result: {outcome.get('result', 'N/A')}")
                    print(f"    All fields: {list(outcome.keys())}")

                print("\nFull event structure:")
                try:
                    print(json.dumps(event, indent=2, default=str)[:500] + "...")
                except Exception:
                    pprint.pprint(event)
        except Exception as e:
            print(f"Error examining file {file_path}: {e}")


def build_moneyline_dataset_from_cache(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    verbose: bool = False,
    require_results: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame built from cached API responses."""
    cache_path = Path(cache_dir)
    files = list(cache_path.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No cache files found in {cache_path}")

    rows: list[dict] = []
    for fp in files:
        try:
            with open(fp, "rb") as f:
                cached = pickle.load(f)
        except Exception as e:  # pragma: no cover - passthrough unexpected errors
            if verbose:
                print(f"Error reading {fp}: {e}")
            continue

        data = cached.get("data") if isinstance(cached, dict) and "data" in cached else cached

        # Some API responses store a single event as a dictionary rather than a
        # list. Wrap it so the existing iteration logic still works.
        if isinstance(data, dict):
            if verbose:
                print(f"Wrapping single event from {fp.name} into list")
            data = [data]

        if not isinstance(data, list):
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            for book in item.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) != 2:
                        continue
                    team1 = outcomes[0].get("name")
                    team2 = outcomes[1].get("name")
                    price1 = outcomes[0].get("price")
                    price2 = outcomes[1].get("price")
                    result1 = outcomes[0].get("result")
                    result2 = outcomes[1].get("result")

                    # Skip entries with missing basic data
                    if None in (team1, team2, price1, price2):
                        continue

                    # If requiring results, skip entries with missing results
                    if require_results and None in (result1, result2):
                        continue

                    # Set team1_win to None if results are missing
                    team1_win = None
                    if result1 is not None:
                        team1_win = 1 if result1 == "win" else 0
                    rows.append(
                        {
                            "team1": team1,
                            "team2": team2,
                            "price1": price1,
                            "price2": price2,
                            "implied_prob": american_odds_to_prob(price1),
                            "team1_win": team1_win,
                        }
                    )
                    if verbose:
                        result_info = f"result={result1}" if result1 else "no result yet"
                        print(f"Added {team1} vs {team2} from {fp.name} ({result_info})")
                    break
    if not rows:
        raise RuntimeError("No valid data found in cache")
    return pd.DataFrame(rows)


def save_dataset_from_cache(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    csv_out: str | Path = "training_data.csv",
    *,
    verbose: bool = False,
    require_results: bool = False,
) -> Path:
    """Build dataset from cache and save as CSV."""
    df = build_moneyline_dataset_from_cache(
        cache_dir, verbose=verbose, require_results=require_results
    )
    out = Path(csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if verbose:
        total_rows = len(df)
        missing_results = df["team1_win"].isna().sum()
        print(f"Saved {total_rows} rows to {out}")
        print(
            f"Note: {missing_results} rows ({missing_results/total_rows:.1%}) have missing results"
        )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert cached API pickles to a training CSV")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory containing .pkl cache files")
    parser.add_argument("--output", help="Path to write the CSV dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Only include entries with result fields",
    )
    parser.add_argument(
        "--examine-cache",
        action="store_true",
        help="Print information about cached files and exit",
    )
    args = parser.parse_args()

    if args.examine_cache:
        examine_cache_structure(args.cache_dir)
        raise SystemExit(0)

    if not args.output:
        parser.error("--output is required unless --examine-cache is used")

    try:
        save_dataset_from_cache(
            args.cache_dir,
            args.output,
            verbose=args.verbose,
            require_results=args.require_results,
        )
    except Exception as exc:
        print(f"Failed to build dataset: {exc}")
        raise SystemExit(1)
