#!/usr/bin/env python3
import os
import pandas as pd
import zipfile

# Retrosheet provides game logs as fixed-width text files inside ZIP archives.
import requests
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import json
from difflib import SequenceMatcher

# Configuration
RETROSHEET_DIR = Path("retrosheet_data")
PROCESSED_DIR = RETROSHEET_DIR / "processed"
CACHE_DIR = Path("h2h_data") / "api_cache"  # Your existing odds API cache
OUTPUT_FILE = "integrated_training_data.csv"
# Process one year at a time for now.  Start with just 2023 while validating the
# integration logic.
YEARS_TO_PROCESS = [2023]
RETROSHEET_BASE_URL = "https://www.retrosheet.org/gamelogs"


def setup_directories():
    """Create necessary directories"""
    RETROSHEET_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)


def load_odds_from_cache():
    """Load historical odds data from The Odds API cache files"""
    print("Loading odds data from API cache...")

    # Find all cache files
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    if not cache_files:
        print(f"No cache files found in {CACHE_DIR}")
        return pd.DataFrame()

    print(f"Found {len(cache_files)} cache files")

    odds_data = []
    for cache_path in cache_files:
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            data = (
                cached.get("data")
                if isinstance(cached, dict) and "data" in cached
                else cached
            )

            # Handle different data structures
            if isinstance(data, dict):
                events = [data]
            elif isinstance(data, list):
                events = data
            else:
                continue

            # Process each event
            for event in events:
                if not isinstance(event, dict):
                    continue

                # Get event details
                event_id = event.get("id")
                commence_time = event.get("commence_time")
                home_team = event.get("home_team")
                away_team = event.get("away_team")

                if not all([event_id, commence_time, home_team, away_team]):
                    continue

                # Process bookmakers for h2h markets
                for book in event.get("bookmakers", []):
                    bookmaker = book.get("key")

                    for market in book.get("markets", []):
                        if market.get("key") != "h2h":
                            continue

                        outcomes = market.get("outcomes", [])
                        if len(outcomes) != 2:
                            continue

                        # Find home and away teams in outcomes
                        home_outcome = next(
                            (o for o in outcomes if o.get("name") == home_team), None
                        )
                        away_outcome = next(
                            (o for o in outcomes if o.get("name") == away_team), None
                        )

                        if not home_outcome or not away_outcome:
                            continue

                        # Extract prices and results
                        home_price = home_outcome.get("price")
                        away_price = away_outcome.get("price")
                        home_result = home_outcome.get("result")
                        away_result = away_outcome.get("result")

                        if None in (home_price, away_price):
                            continue

                        # Create a record
                        odds_record = {
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "date": datetime.fromisoformat(
                                commence_time.replace("Z", "+00:00")
                            ).strftime("%Y-%m-%d"),
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker,
                            "price1": home_price,  # Home team price
                            "price2": away_price,  # Away team price
                            "home_result": home_result,
                            "away_result": away_result,
                        }

                        odds_data.append(odds_record)
                        # Take the first market found for each event-bookmaker
                        break

        except Exception as e:
            print(f"Error processing {cache_path}: {e}")

    if not odds_data:
        print("No valid odds data found in cache")
        return pd.DataFrame()

    odds_df = pd.DataFrame(odds_data)

    if "home_result" in odds_df.columns and not odds_df["home_result"].isna().all():
        odds_df["home_team_win"] = (odds_df["home_result"] == "win").astype(int)

    odds_df = odds_df.sort_values("date")

    print(f"Loaded {len(odds_df)} odds records from cache")

    return odds_df


def download_retrosheet_data(year):
    """Download Retrosheet game logs for a specific year"""
    url = f"{RETROSHEET_BASE_URL}/gl{year}.zip"
    local_file = RETROSHEET_DIR / f"gl{year}.zip"

    if local_file.exists():
        print(f"File for {year} already exists, skipping download")
        return local_file

    print(f"Downloading {year} data...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(local_file, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {local_file}")
            return local_file
        else:
            print(f"Failed to download {year} data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading {year} data: {e}")
        return None


def extract_zip_file(zip_path):
    """Extract the GL####.TXT log from a Retrosheet ZIP archive"""
    if not zip_path or not zip_path.exists():
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            game_log_files = [
                f
                for f in zip_ref.namelist()
                if Path(f).name.upper().startswith("GL")
                and Path(f).name.upper().endswith(".TXT")
            ]
            if not game_log_files:
                print(f"No game log files found in {zip_path}")
                return None
            game_log_file = game_log_files[0]
            extract_name = Path(game_log_file).name
            extract_path = PROCESSED_DIR / extract_name
            with open(extract_path, "wb") as f:
                f.write(zip_ref.read(game_log_file))
            print(f"Extracted {game_log_file} to {extract_path}")
            return extract_path
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return None


def parse_retrosheet_file(file_path, year):
    """Parse a GL####.TXT game log and return a dataframe"""
    if not file_path or not file_path.exists():
        return None

    try:
        # Retrosheet gamelog files are comma separated, not fixed width.  The
        # previous implementation attempted to parse them with ``read_fwf`` which
        # resulted in truncated team codes like ``hu`` and missing dates.  Read
        # the file as CSV instead and then rename the handful of columns we need.

        df = pd.read_csv(file_path, header=None, encoding="latin1")
        df = df.rename(
            columns={
                0: "date",
                1: "game_number",
                2: "day_of_week",
                3: "visiting_team",
                4: "visiting_league",
                5: "visiting_game_num",
                6: "home_team",
                7: "home_league",
                8: "home_game_num",
                9: "visiting_score",
                10: "home_score",
                11: "outs",
                12: "day_night",
                16: "park_id",
                17: "attendance",
            }
        )

        df["year"] = year
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        df["visiting_score"] = pd.to_numeric(df["visiting_score"], errors="coerce")
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["home_team_win"] = (df["home_score"] > df["visiting_score"]).astype(int)
        df["is_night_game"] = (df["day_night"] == "N").astype(int)
        df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
        team_code_map = {
            "ANA": "Los Angeles Angels",
            "LAA": "Los Angeles Angels",
            "ARI": "Arizona Diamondbacks",
            "ATL": "Atlanta Braves",
            "BAL": "Baltimore Orioles",
            "BOS": "Boston Red Sox",
            "CHA": "Chicago White Sox",
            "CHW": "Chicago White Sox",
            "CHN": "Chicago Cubs",
            "CIN": "Cincinnati Reds",
            "CLE": "Cleveland Guardians",
            "COL": "Colorado Rockies",
            "DET": "Detroit Tigers",
            "FLO": "Miami Marlins",
            "MIA": "Miami Marlins",
            "HOU": "Houston Astros",
            "KCA": "Kansas City Royals",
            "KCR": "Kansas City Royals",
            "LAN": "Los Angeles Dodgers",
            "MIL": "Milwaukee Brewers",
            "MIN": "Minnesota Twins",
            "NYA": "New York Yankees",
            "NYN": "New York Mets",
            "OAK": "Oakland Athletics",
            "PHI": "Philadelphia Phillies",
            "PIT": "Pittsburgh Pirates",
            "SDN": "San Diego Padres",
            "SDP": "San Diego Padres",
            "SEA": "Seattle Mariners",
            "SFN": "San Francisco Giants",
            "SFG": "San Francisco Giants",
            "SLN": "St. Louis Cardinals",
            "TBA": "Tampa Bay Rays",
            "TBR": "Tampa Bay Rays",
            "TEX": "Texas Rangers",
            "TOR": "Toronto Blue Jays",
            "WAS": "Washington Nationals",
            "WSH": "Washington Nationals",
        }
        # Preserve the original Retrosheet team codes
        df["home_team_code"] = df["home_team"]
        df["away_team_code"] = df["visiting_team"]

        # Map the Retrosheet codes to full team names for easier matching with
        # the odds data.  Perform the mapping in place so we don't end up with
        # duplicate column names which can cause pandas to return a DataFrame
        # instead of a Series when selecting by column name.
        df["home_team"] = df["home_team"].map(team_code_map).fillna(df["home_team"])
        df["away_team"] = df["visiting_team"].map(team_code_map).fillna(df["visiting_team"])

        # The original visiting_team column is no longer needed after we create
        # ``away_team``.  Dropping it prevents accidental duplication of column
        # names which previously caused issues when using the ``.str`` accessor.
        df = df.drop(columns=["visiting_team"])
        return df
    except Exception as e:
        print(f"Error parsing Retrosheet file {file_path}: {e}")
        return None


def _normalize_team_name(name: str) -> str:
    """Return a lower-case team name without surrounding whitespace"""
    return str(name).strip().lower()


def _ensure_normalized_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has normalized team name columns."""
    if "home_team_norm" not in df.columns and "home_team" in df.columns:
        df["home_team_norm"] = df["home_team"].apply(_normalize_team_name)
    if "away_team_norm" not in df.columns:
        if "away_team" not in df.columns and "visiting_team" in df.columns:
            df["away_team"] = df["visiting_team"]
        if "away_team" in df.columns:
            df["away_team_norm"] = df["away_team"].apply(_normalize_team_name)
    return df


def _is_close_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """Return True when strings are an approximate match."""
    if not a or not b:
        return False
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def match_odds_with_results(odds_df, results_df):
    """Match odds data with game results by team and date"""
    print("\nMatching odds data with game results...")
    if odds_df.empty or results_df.empty:
        print("No data to match")
        return pd.DataFrame()

    # Normalize team names once to avoid repeated string operations
    odds_df = odds_df.copy()
    results_df = results_df.copy()
    odds_df = _ensure_normalized_team_columns(odds_df)
    results_df = _ensure_normalized_team_columns(results_df)

    print(f"Odds date range: {odds_df['date'].min()} to {odds_df['date'].max()}")
    print(f"Results date range: {results_df['date'].min()} to {results_df['date'].max()}")
    print(
        f"Sample odds teams: {odds_df[['home_team_norm', 'away_team_norm']].head(3).values}"
    )
    print(
        f"Sample results teams: {results_df[['home_team_norm', 'away_team_norm']].head(3).values}"
    )

    odds_df['date_dt'] = pd.to_datetime(odds_df['date'])
    results_df['date_dt'] = pd.to_datetime(results_df['date'])

    matched_records = []
    matched_count = 0
    total_odds = len(odds_df)
    for _, odds_row in odds_df.iterrows():
        odds_date = odds_row["date_dt"]
        odds_home = odds_row["home_team_norm"]
        odds_away = odds_row["away_team_norm"]

        matches = results_df[
            (results_df["date_dt"] >= odds_date - pd.Timedelta(days=1))
            & (results_df["date_dt"] <= odds_date + pd.Timedelta(days=1))
            & (
                results_df["home_team_norm"].str.contains(odds_home, case=False, na=False)
                | results_df["home_team_norm"].str.contains(odds_home[:5], case=False, na=False)
            )
            & (
                results_df["away_team_norm"].str.contains(odds_away, case=False, na=False)
                | results_df["away_team_norm"].str.contains(odds_away[:5], case=False, na=False)
            )
        ]
        if len(matches) > 0:
            result_row = matches.iloc[0]
            record = {
                "event_id": odds_row.get("event_id"),
                "price1": odds_row["price1"],
                "price2": odds_row["price2"],
                "commence_time": odds_row.get("commence_time"),
                "bookmaker": odds_row.get("bookmaker"),
                "date": result_row["date"],
                "home_team": result_row["home_team"],
                "away_team": result_row["away_team"],
                "home_score": result_row["home_score"],
                "visiting_score": result_row["visiting_score"],
                "home_team_win": result_row["home_team_win"],
                "day_night": result_row.get("day_night"),
                "attendance": result_row.get("attendance"),
                "team1": result_row["home_team"],
                "team2": result_row["away_team"],
                "pregame_price": odds_row["price1"],
                "pregame_line": odds_row["price1"],
                "implied_prob": american_odds_to_prob(odds_row["price1"]),
            }
            matched_records.append(record)
            matched_count += 1
    print(
        f"Matched {matched_count} out of {total_odds} odds records with game results."
    )
    if matched_records:
        return pd.DataFrame(matched_records)
    else:
        print("No matches found between odds and results")
        return pd.DataFrame()


def american_odds_to_prob(odds):
    """Convert American odds to implied probability"""
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)
    except:
        return 0.5


def add_ml_features(df):
    """Add additional features needed for the ML model"""
    df["live_inning_5_diff"] = None
    df["live_inning_7_diff"] = None
    df["game_day"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["is_weekend"] = df["game_day"].isin([5, 6]).astype(int)
    required_columns = [
        "team1",
        "team2",
        "price1",
        "price2",
        "pregame_price",
        "pregame_line",
        "home_team_win",
        "implied_prob",
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Required column '{col}' not found. Adding default values.")
            if "team" in col:
                df[col] = df["home_team"] if "team1" in col else df["away_team"]
            elif "price" in col or "line" in col:
                df[col] = df["price1"] if df["price1"].notna().any() else -110
            elif "prob" in col:
                df[col] = 0.5
            elif "win" in col:
                df[col] = (df["home_score"] > df["visiting_score"]).astype(int)
    return df


def main():
    """Main processing function"""
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate odds data with Retrosheet game results"
    )
    parser.add_argument(
        "--year", type=int, help="Process only a specific target year (e.g., 2023)"
    )
    parser.add_argument(
        "--years", help="Year range to process (e.g., '2018-2023')"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output CSV file path (default: {OUTPUT_FILE})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    print("Starting data integration process...")
    setup_directories()
    odds_df = load_odds_from_cache()
    if odds_df.empty:
        print("No odds data found. Please check your cache files.")
        return

    # Filter odds data to specific year if requested
    if args.year:
        year_str = str(args.year)
        odds_df = odds_df[odds_df["date"].str.startswith(year_str)]
        print(f"Filtered to {len(odds_df)} odds records from {args.year}")

    # Determine years to process
    years_to_process = YEARS_TO_PROCESS
    if args.year:
        years_to_process = [args.year]
    elif args.years:
        if "-" in args.years:
            start_year, end_year = map(int, args.years.split("-"))
            years_to_process = range(start_year, end_year + 1)
        else:
            years_to_process = [int(year.strip()) for year in args.years.split(",")]

    all_results = []
    for year in years_to_process:
        print(f"\nProcessing Retrosheet data for {year}:")
        zip_file = download_retrosheet_data(year)
        if zip_file:
            extracted_file = extract_zip_file(zip_file)
            if extracted_file:
                year_df = parse_retrosheet_file(extracted_file, year)
                if year_df is not None and not year_df.empty:
                    print(f"Processed {len(year_df)} games from {year}")
                    all_results.append(year_df)
                else:
                    print(f"No valid data extracted for {year}")

    if not all_results:
        print("No Retrosheet results were processed.")
        return

    results_df = pd.concat(all_results)
    print(f"Combined results dataset has {len(results_df)} games")
    matched_df = match_odds_with_results(odds_df, results_df)

    if matched_df.empty:
        print("No matches found between odds data and results.")
        return

    final_df = add_ml_features(matched_df)
    print(f"Saving integrated training dataset to {args.output}...")
    final_df.to_csv(args.output, index=False)

    print("\nDataset Summary:")
    print(f"Total games with odds and results: {len(final_df)}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Home team win rate: {final_df['home_team_win'].mean():.3f}")
    print(
        "\nProcess complete! You can now use this dataset for training your ML model."
    )
    print(
        f"Run: python main.py train_classifier --dataset={args.output} --features-type=dual --verbose"
    )


if __name__ == "__main__":
    main()
