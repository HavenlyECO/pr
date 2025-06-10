#!/usr/bin/env python3
import os
import pandas as pd
import zipfile
# Retrosheet archives contain fixed-width text files (GL####.TXT).
# This script extracts those TXT files and converts them into a CSV suitable
# for model training.
import requests
from pathlib import Path
from datetime import datetime
import numpy as np

RETROSHEET_DIR = Path("retrosheet_data")
PROCESSED_DIR = RETROSHEET_DIR / "processed"
OUTPUT_FILE = "retrosheet_training_data.csv"
YEARS_TO_PROCESS = range(2018, 2023)
RETROSHEET_BASE_URL = "https://www.retrosheet.org/gamelogs"


def setup_directories():
    """Create necessary directories"""
    RETROSHEET_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)


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
        print(f"Failed to download {year} data: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {year} data: {e}")
    return None


def extract_zip_file(zip_path):
    """Extract the GL####.TXT log from a Retrosheet ZIP archive"""
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


# Define field specifications for fixed-width format
FIELD_SPECS = [
    (0, 8, 'date'),
    (8, 9, 'game_number'),
    (9, 12, 'day_of_week'),
    (12, 14, 'visiting_team'),
    (14, 15, 'visiting_league'),
    (15, 17, 'visiting_game_num'),
    (17, 19, 'home_team'),
    (19, 20, 'home_league'),
    (20, 22, 'home_game_num'),
    (22, 24, 'visiting_score'),
    (24, 26, 'home_score'),
    (26, 28, 'game_outs'),
    (28, 29, 'day_night'),
    (29, 57, 'completion_info'),
    (57, 58, 'forfeit'),
    (58, 59, 'protest'),
    (59, 62, 'park_id'),
    (62, 68, 'attendance'),
    (68, 71, 'time_of_game'),
    (169, 173, 'home_team_errors'),
    (265, 300, 'winning_pitcher'),
    (300, 335, 'losing_pitcher'),
    (335, 370, 'save_pitcher'),
    (440, 475, 'vis_starting_pitcher'),
    (475, 510, 'home_starting_pitcher'),
]


def parse_retrosheet_file(file_path, year):
    """Parse a GL####.TXT game log and return a dataframe"""
    if not file_path or not file_path.exists():
        return None

    colspecs = [(s, e) for s, e, _ in FIELD_SPECS]
    names = [n for _, _, n in FIELD_SPECS]

    try:
        df = pd.read_fwf(
            file_path,
            colspecs=colspecs,
            names=names,
            encoding='latin1',
            error_bad_lines=False
        )
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        try:
            print("Attempting fallback parsing...")
            df = pd.read_csv(file_path, header=None, encoding='latin1', error_bad_lines=False)
            if df.shape[1] >= 97:
                df.columns = [f'field_{i+1}' for i in range(df.shape[1])]
                df = df.rename(columns={
                    'field_1': 'date',
                    'field_3': 'day_of_week',
                    'field_4': 'visiting_team',
                    'field_7': 'home_team',
                    'field_10': 'visiting_score',
                    'field_11': 'home_score',
                    'field_13': 'day_night',
                })
            else:
                return None
        except Exception as e2:
            print(f"Fallback parsing failed: {e2}")
            return None

    df['year'] = year
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
    df['visiting_score'] = pd.to_numeric(df['visiting_score'], errors='coerce')
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['home_team_win'] = (df['home_score'] > df['visiting_score']).astype(int)
    df['is_night_game'] = (df['day_night'] == 'N').astype(int)
    df['time_of_game'] = pd.to_numeric(df['time_of_game'], errors='coerce')
    df = df.dropna(subset=['visiting_score', 'home_score'])
    return df


TEAM_CODE_MAP = {
    'ANA': 'LAA',
    'LAA': 'LAA',
    'ARI': 'ARI',
    'ATL': 'ATL',
    'BAL': 'BAL',
    'BOS': 'BOS',
    'CHA': 'CWS',
    'CHW': 'CWS',
    'CHN': 'CHC',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'COL': 'COL',
    'DET': 'DET',
    'FLO': 'MIA',
    'MIA': 'MIA',
    'HOU': 'HOU',
    'KCA': 'KC',
    'KCR': 'KC',
    'LAA': 'LAA',
    'LAN': 'LAD',
    'MIL': 'MIL',
    'MIN': 'MIN',
    'NYA': 'NYY',
    'NYN': 'NYM',
    'OAK': 'OAK',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SDN': 'SD',
    'SDP': 'SD',
    'SEA': 'SEA',
    'SFN': 'SF',
    'SFG': 'SF',
    'SLN': 'STL',
    'TBA': 'TB',
    'TBR': 'TB',
    'TEX': 'TEX',
    'TOR': 'TOR',
    'WAS': 'WSH',
    'WSH': 'WSH',
}


def clean_team_codes(df):
    if 'visiting_team' in df.columns:
        df['visiting_team'] = df['visiting_team'].map(TEAM_CODE_MAP).fillna(df['visiting_team'])
    if 'home_team' in df.columns:
        df['home_team'] = df['home_team'].map(TEAM_CODE_MAP).fillna(df['home_team'])
    return df


def create_team_stats(df):
    print("Creating team performance statistics...")
    df = df.sort_values('date')
    all_teams = pd.concat([
        df[['visiting_team']].rename(columns={'visiting_team': 'team'}),
        df[['home_team']].rename(columns={'home_team': 'team'})
    ])
    unique_teams = all_teams['team'].unique()
    team_stats = {}
    for team in unique_teams:
        home_games = df[df['home_team'] == team].copy()
        away_games = df[df['visiting_team'] == team].copy()
        home_games['team_win'] = home_games['home_team_win']
        away_games['team_win'] = 1 - away_games['home_team_win']
        home_games['team_score'] = home_games['home_score']
        home_games['opponent_score'] = home_games['visiting_score']
        away_games['team_score'] = away_games['visiting_score']
        away_games['opponent_score'] = away_games['home_score']
        home_games['is_home'] = 1
        away_games['is_home'] = 0
        team_games = pd.concat([home_games, away_games])
        team_games = team_games.sort_values('date')
        team_games['rolling_win_pct_5'] = team_games['team_win'].rolling(5, min_periods=1).mean()
        team_games['rolling_win_pct_10'] = team_games['team_win'].rolling(10, min_periods=1).mean()
        team_games['rolling_win_pct_20'] = team_games['team_win'].rolling(20, min_periods=1).mean()
        team_games['rolling_runs_scored_5'] = team_games['team_score'].rolling(5, min_periods=1).mean()
        team_games['rolling_runs_allowed_5'] = team_games['opponent_score'].rolling(5, min_periods=1).mean()
        team_games['rolling_runs_scored_10'] = team_games['team_score'].rolling(10, min_periods=1).mean()
        team_games['rolling_runs_allowed_10'] = team_games['opponent_score'].rolling(10, min_periods=1).mean()
        team_games['rolling_run_diff_5'] = team_games['rolling_runs_scored_5'] - team_games['rolling_runs_allowed_5']
        team_games['rolling_run_diff_10'] = team_games['rolling_runs_scored_10'] - team_games['rolling_runs_allowed_10']
        team_stats[team] = team_games

    enhanced_df = df.copy()
    for feature in ['rolling_win_pct_5', 'rolling_win_pct_10', 'rolling_run_diff_5', 'rolling_run_diff_10',
                    'rolling_runs_scored_5', 'rolling_runs_allowed_5']:
        enhanced_df[f'home_{feature}'] = np.nan
        enhanced_df[f'visiting_{feature}'] = np.nan

    print("Adding team statistics to each game...")
    for idx, row in enhanced_df.iterrows():
        game_date = row['date']
        home_team = row['home_team']
        visiting_team = row['visiting_team']
        if home_team in team_stats:
            home_prev = team_stats[home_team][team_stats[home_team]['date'] < game_date]
            if not home_prev.empty:
                latest_home = home_prev.iloc[-1]
                for feature in ['rolling_win_pct_5', 'rolling_win_pct_10', 'rolling_run_diff_5', 'rolling_run_diff_10',
                                'rolling_runs_scored_5', 'rolling_runs_allowed_5']:
                    enhanced_df.at[idx, f'home_{feature}'] = latest_home[feature]
        if visiting_team in team_stats:
            vis_prev = team_stats[visiting_team][team_stats[visiting_team]['date'] < game_date]
            if not vis_prev.empty:
                latest_vis = vis_prev.iloc[-1]
                for feature in ['rolling_win_pct_5', 'rolling_win_pct_10', 'rolling_run_diff_5', 'rolling_run_diff_10',
                                'rolling_runs_scored_5', 'rolling_runs_allowed_5']:
                    enhanced_df.at[idx, f'visiting_{feature}'] = latest_vis[feature]

    for feature in ['rolling_win_pct_5', 'rolling_win_pct_10']:
        enhanced_df[f'home_{feature}'] = enhanced_df[f'home_{feature}'].fillna(0.5)
        enhanced_df[f'visiting_{feature}'] = enhanced_df[f'visiting_{feature}'].fillna(0.5)
    for feature in ['rolling_run_diff_5', 'rolling_run_diff_10']:
        enhanced_df[f'home_{feature}'] = enhanced_df[f'home_{feature}'].fillna(0)
        enhanced_df[f'visiting_{feature}'] = enhanced_df[f'visiting_{feature}'].fillna(0)
    for feature in ['rolling_runs_scored_5', 'rolling_runs_allowed_5']:
        enhanced_df[f'home_{feature}'] = enhanced_df[f'home_{feature}'].fillna(4.5)
        enhanced_df[f'visiting_{feature}'] = enhanced_df[f'visiting_{feature}'].fillna(4.5)
    return enhanced_df


def create_ml_features(df):
    print("Creating ML-ready features...")
    ml_df = df.copy()
    ml_df['away_team'] = ml_df['visiting_team']
    ml_df['team1'] = ml_df['home_team']
    ml_df['team2'] = ml_df['away_team']
    ml_df['pregame_win_pct_diff'] = ml_df['home_rolling_win_pct_10'] - ml_df['visiting_rolling_win_pct_10']
    ml_df['pregame_run_diff'] = ml_df['home_rolling_run_diff_10'] - ml_df['visiting_rolling_run_diff_10']
    ml_df['pregame_home_adv'] = 1.0
    ml_df['pregame_day_night'] = (ml_df['day_night'] == 'N').astype(float)

    ml_df['synthetic_home_prob'] = (
        ml_df['home_rolling_win_pct_10'] * 0.6 +
        (ml_df['home_rolling_run_diff_10'] / 10 + 0.5) * 0.4
    )
    ml_df['synthetic_home_prob'] *= 1.08
    ml_df['synthetic_home_prob'] = ml_df['synthetic_home_prob'].clip(0.05, 0.95)
    ml_df['price1'] = ml_df['synthetic_home_prob'].apply(
        lambda p: -100 * p / (1 - p) if p > 0.5 else 100 * (1 - p) / p
    )
    ml_df['price2'] = ml_df['price1'].apply(
        lambda x: 100 * (100 / abs(x)) if x < 0 else -100 * (abs(x) / 100)
    )
    ml_df['price1'] = (ml_df['price1'] / 5).round() * 5
    ml_df['price2'] = (ml_df['price2'] / 5).round() * 5
    ml_df['pregame_price'] = ml_df['price1']
    ml_df['pregame_line'] = ml_df['price1']
    ml_df['implied_prob'] = ml_df['synthetic_home_prob']
    ml_df['live_inning_5_diff'] = None
    ml_df['live_inning_7_diff'] = None
    ml_df['home_team_win'] = (ml_df['home_score'] > ml_df['visiting_score']).astype(int)
    return ml_df


def main():
    print("Starting Retrosheet data processing...")
    setup_directories()
    all_data = []
    for year in YEARS_TO_PROCESS:
        print(f"\nProcessing year {year}:")
        zip_file = download_retrosheet_data(year)
        if zip_file:
            extracted_file = extract_zip_file(zip_file)
            if extracted_file:
                year_df = parse_retrosheet_file(extracted_file, year)
                if year_df is not None and not year_df.empty:
                    year_df = clean_team_codes(year_df)
                    print(f"Processed {len(year_df)} games from {year}")
                    all_data.append(year_df)
                else:
                    print(f"No valid data extracted for {year}")
    if not all_data:
        print("No data was processed. Exiting.")
        return
    print("\nCombining data from all years...")
    combined_df = pd.concat(all_data)
    print(f"Combined dataset has {len(combined_df)} games")
    print("\nCalculating team performance metrics...")
    enhanced_df = create_team_stats(combined_df)
    ml_df = create_ml_features(enhanced_df)
    print(f"\nSaving training dataset to {OUTPUT_FILE}...")
    ml_df.to_csv(OUTPUT_FILE, index=False)
    print("\nDataset Summary:")
    print(f"Total games: {len(ml_df)}")
    print(f"Date range: {ml_df['date'].min()} to {ml_df['date'].max()}")
    print(f"Home team win rate: {ml_df['home_team_win'].mean():.3f}")
    print(
        "Key columns for ML: " + ", ".join(['team1', 'team2', 'price1', 'price2', 'pregame_price', 'home_team_win'])
    )
    print("\nProcess complete! You can now use this dataset for training your ML model.")
    print(f"Run: python main.py train_classifier --dataset={OUTPUT_FILE} --features-type=dual --verbose")


if __name__ == "__main__":
    main()

