# Odds Fetcher

This script fetches odds from [The Odds API](https://the-odds-api.com/).

## Usage

1. Set an environment variable with your API key:

```bash
export THE_ODDS_API_KEY=<your api key>
```

2. Run the script (moneyline odds shown by default):

```bash
python main.py
```

To display point spread (handicap) odds, run:

```bash
python main.py spreads
```

To display totals (over/under) odds, run:

```bash
python main.py totals
```

To display outrights (futures) odds, run:

```bash
python main.py outrights
```

The script requests head-to-head, point spread, totals, and outright markets as
needed. It prints
the API endpoint used and displays odds for the selected market in a clean
layout.
