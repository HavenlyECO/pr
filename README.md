# Odds Fetcher

This script fetches odds from [The Odds API](https://the-odds-api.com/).

## Usage

1. Set an environment variable with your API key:

```bash
export THE_ODDS_API_KEY=<your api key>
```

2. Run the script:

```bash
python main.py
```

The script prints the API endpoint containing the `h2h` market and displays
head-to-head (moneyline) lines for each game in a clean layout.
