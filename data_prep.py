import pickle
from pathlib import Path
import pandas as pd


def american_odds_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


DEFAULT_CACHE_DIR = Path("h2h_data") / "api_cache"


def build_moneyline_dataset_from_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR, *, verbose: bool = False) -> pd.DataFrame:
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
                    if None in (team1, team2, price1, price2, result1, result2):
                        continue
                    rows.append(
                        {
                            "team1": team1,
                            "team2": team2,
                            "price1": price1,
                            "price2": price2,
                            "implied_prob": american_odds_to_prob(price1),
                            "team1_win": 1 if result1 == "win" else 0,
                        }
                    )
                    if verbose:
                        print(f"Added {team1} vs {team2} from {fp.name}")
                    break
    if not rows:
        raise RuntimeError("No valid data found in cache")
    return pd.DataFrame(rows)


def save_dataset_from_cache(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    csv_out: str | Path = "training_data.csv",
    *,
    verbose: bool = False,
) -> Path:
    """Build dataset from cache and save as CSV."""
    df = build_moneyline_dataset_from_cache(cache_dir, verbose=verbose)
    out = Path(csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if verbose:
        print(f"Saved {len(df)} rows to {out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert cached API pickles to a training CSV")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory containing .pkl cache files")
    parser.add_argument("--output", required=True, help="Path to write the CSV dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    try:
        save_dataset_from_cache(args.cache_dir, args.output, verbose=args.verbose)
    except Exception as exc:
        print(f"Failed to build dataset: {exc}")
        raise SystemExit(1)
