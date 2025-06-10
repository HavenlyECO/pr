#!/usr/bin/env python3
import os
import pandas as pd
import zipfile
import requests
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import json

# Configuration
RETROSHEET_DIR = Path("retrosheet_data")
PROCESSED_DIR = RETROSHEET_DIR / "processed"
CACHE_DIR = Path("h2h_data") / "api_cache"  # Your existing odds API cache
OUTPUT_FILE = "integrated_training_data.csv"
YEARS_TO_PROCESS = range(2018, 2024)  # Adjust years to match your odds data
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
            
            data = cached.get("data") if isinstance(cached, dict) and "data" in cached else cached
            
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
                        home_outcome = next((o for o in outcomes if o.get("name") == home_team), None)
                        away_outcome = next((o for o in outcomes if o.get("name") == away_team), None)
                        
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
                            "date": datetime.fromisoformat(commence_time.replace('Z', '+00:00')).strftime('%Y-%m-%d'),
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker,
                            "price1": home_price,  # Home team price
                            "price2": away_price,  # Away team price
                            "home_result": home_result,
                            "away_result": away_result
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
    
    if 'home_result' in odds_df.columns and not odds_df['home_result'].isna().all():
        odds_df['home_team_win'] = (odds_df['home_result'] == 'win').astype(int)
    
    odds_df = odds_df.sort_values('date')
    
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
            with open(local_file, 'wb') as f:
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
    """Extract files from retrosheet ZIP file"""
    if not zip_path or not zip_path.exists():
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            game_log_files = [f for f in zip_ref.namelist() if f.startswith('GL') and f.endswith('.TXT')]
            if not game_log_files:
                print(f"No game log files found in {zip_path}")
                return None
            game_log_file = game_log_files[0]
            extract_path = PROCESSED_DIR / game_log_file
            with open(extract_path, 'wb') as f:
                f.write(zip_ref.read(game_log_file))
            print(f"Extracted {game_log_file} to {extract_path}")
            return extract_path
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return None

def parse_retrosheet_file(file_path, year):
    """Parse Retrosheet game log file"""
    if not file_path or not file_path.exists():
        return None
    
    try:
        columns = [
            (0, 8, 'date'),
            (8, 9, 'game_number'),
            (9, 12, 'day_of_week'),
            (12, 14, 'visiting_team'),
            (14, 15, 'visiting_league'),
            (17, 19, 'home_team'),
            (19, 20, 'home_league'),
            (22, 24, 'visiting_score'),
            (24, 26, 'home_score'),
            (28, 29, 'day_night'),
            (59, 62, 'park_id'),
            (62, 68, 'attendance'),
        ]
        colspecs = [(start, end) for start, end, _ in columns]
        names = [name for _, _, name in columns]
        df = pd.read_fwf(file_path, colspecs=colspecs, names=names, encoding='latin1')
        df['year'] = year
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
        df['visiting_score'] = pd.to_numeric(df['visiting_score'], errors='coerce')
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['home_team_win'] = (df['home_score'] > df['visiting_score']).astype(int)
        df['is_night_game'] = (df['day_night'] == 'N').astype(int)
        df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')
        team_code_map = {
            'ANA': 'Los Angeles Angels', 'LAA': 'Los Angeles Angels',
            'ARI': 'Arizona Diamondbacks',
            'ATL': 'Atlanta Braves',
            'BAL': 'Baltimore Orioles',
            'BOS': 'Boston Red Sox',
            'CHA': 'Chicago White Sox', 'CHW': 'Chicago White Sox',
            'CHN': 'Chicago Cubs',
            'CIN': 'Cincinnati Reds',
            'CLE': 'Cleveland Guardians',
            'COL': 'Colorado Rockies',
            'DET': 'Detroit Tigers',
            'FLO': 'Miami Marlins', 'MIA': 'Miami Marlins',
            'HOU': 'Houston Astros',
            'KCA': 'Kansas City Royals', 'KCR': 'Kansas City Royals',
            'LAN': 'Los Angeles Dodgers',
            'MIL': 'Milwaukee Brewers',
            'MIN': 'Minnesota Twins',
            'NYA': 'New York Yankees',
            'NYN': 'New York Mets',
            'OAK': 'Oakland Athletics',
            'PHI': 'Philadelphia Phillies',
            'PIT': 'Pittsburgh Pirates',
            'SDN': 'San Diego Padres', 'SDP': 'San Diego Padres',
            'SEA': 'Seattle Mariners',
            'SFN': 'San Francisco Giants', 'SFG': 'San Francisco Giants',
            'SLN': 'St. Louis Cardinals',
            'TBA': 'Tampa Bay Rays', 'TBR': 'Tampa Bay Rays',
            'TEX': 'Texas Rangers',
            'TOR': 'Toronto Blue Jays',
            'WAS': 'Washington Nationals', 'WSH': 'Washington Nationals'
        }
        df['home_team_std'] = df['home_team'].map(team_code_map).fillna(df['home_team'])
        df['visiting_team_std'] = df['visiting_team'].map(team_code_map).fillna(df['visiting_team'])
        df['home_team_code'] = df['home_team']
        df['visiting_team_code'] = df['visiting_team']
        df = df.rename(columns={'visiting_team_std': 'away_team', 'home_team_std': 'home_team'})
        return df
    except Exception as e:
        print(f"Error parsing Retrosheet file {file_path}: {e}")
        return None

def match_odds_with_results(odds_df, results_df):
    """Match odds data with game results by team and date"""
    print("\nMatching odds data with game results...")
    if odds_df.empty or results_df.empty:
        print("No data to match")
        return pd.DataFrame()
    matched_records = []
    matched_count = 0
    total_odds = len(odds_df)
    for _, odds_row in odds_df.iterrows():
        odds_date = odds_row['date']
        odds_home = odds_row['home_team']
        odds_away = odds_row['away_team']
        matches = results_df[
            (results_df['date'] == odds_date) &
            ((results_df['home_team'] == odds_home) | (results_df['home_team'].str.contains(odds_home, na=False))) &
            ((results_df['away_team'] == odds_away) | (results_df['away_team'].str.contains(odds_away, na=False)))
        ]
        if len(matches) > 0:
            result_row = matches.iloc[0]
            record = {
                'event_id': odds_row.get('event_id'),
                'price1': odds_row['price1'],
                'price2': odds_row['price2'],
                'commence_time': odds_row.get('commence_time'),
                'bookmaker': odds_row.get('bookmaker'),
                'date': result_row['date'],
                'home_team': result_row['home_team'],
                'away_team': result_row['away_team'],
                'home_score': result_row['home_score'],
                'visiting_score': result_row['visiting_score'],
                'home_team_win': result_row['home_team_win'],
                'day_night': result_row.get('day_night'),
                'attendance': result_row.get('attendance'),
                'team1': result_row['home_team'],
                'team2': result_row['away_team'],
                'pregame_price': odds_row['price1'],
                'pregame_line': odds_row['price1'],
                'implied_prob': american_odds_to_prob(odds_row['price1'])
            }
            matched_records.append(record)
            matched_count += 1
    print(f"Matched {matched_count} out of {total_odds} odds records with game results.")
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
    df['live_inning_5_diff'] = None
    df['live_inning_7_diff'] = None
    df['game_day'] = pd.to_datetime(df['date']).dt.dayofweek
    df['is_weekend'] = df['game_day'].isin([5, 6]).astype(int)
    required_columns = [
        'team1', 'team2', 'price1', 'price2', 'pregame_price',
        'pregame_line', 'home_team_win', 'implied_prob'
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Required column '{col}' not found. Adding default values.")
            if 'team' in col:
                df[col] = df['home_team'] if 'team1' in col else df['away_team']
            elif 'price' in col or 'line' in col:
                df[col] = df['price1'] if df['price1'].notna().any() else -110
            elif 'prob' in col:
                df[col] = 0.5
            elif 'win' in col:
                df[col] = (df['home_score'] > df['visiting_score']).astype(int)
    return df

def main():
    """Main processing function"""
    print("Starting data integration process...")
    setup_directories()
    odds_df = load_odds_from_cache()
    if odds_df.empty:
        print("No odds data found. Please check your cache files.")
        return
    all_results = []
    for year in YEARS_TO_PROCESS:
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
    print(f"Saving integrated training dataset to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("\nDataset Summary:")
    print(f"Total games with odds and results: {len(final_df)}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Home team win rate: {final_df['home_team_win'].mean():.3f}")
    print("\nProcess complete! You can now use this dataset for training your ML model.")
    print(f"Run: python main.py train_classifier --dataset={OUTPUT_FILE} --features-type=dual --verbose")

if __name__ == "__main__":
    main()
