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

To display alternate spreads odds, run:

```bash
python main.py alternate_spreads
```

To display alternate totals odds, run:

```bash
python main.py alternate_totals
```

To display team totals odds, run:

```bash
python main.py team_totals
```

To display alternate team totals odds, run:

```bash
python main.py alternate_team_totals
```

Player prop markets (hits, home runs, strikeouts and batter strikeouts)
are included in the odds request by default. Pass the ``--no-player-props``
flag to exclude them:

```bash
python main.py --no-player-props
```

To display projected pitcher strikeout props using the machine learning model,
run:

```bash
python main.py --model=pitcher_ks_classifier.pkl
```

By default the script requests only the ``batter_strikeouts`` market and
evaluates those props. Pass the ``--markets`` option to request different or
additional markets.

To display outrights (futures) odds, run:

```bash
python main.py outrights
```

To display historical odds for a specific date, run:

```bash
python main.py historical --date=2024-01-01
```

Historical data is only available for approximately the last year.

The script requests head-to-head, point spread, totals, and outright markets as
needed. It prints

the API endpoint used and displays odds for the selected market in a clean
layout.

To list upcoming events and their IDs for a sport, run:

```bash
python main.py --list-events
```
The region defaults to ``us``. Pass ``--regions`` with a comma-separated list to
see events available in other regions.

To fetch all odds for a single event and print the raw JSON response, supply the
event ID and ``--event-odds``:

```bash
python main.py --event-id=<event id> --event-odds
```
You can also customize the ``--markets``, ``--odds-format`` and ``--date-format``
options when using this endpoint.

To include game period markets (e.g. quarters or innings) in the API request,
pass them via the ``--game-period-markets`` option:

```bash
python main.py --game-period-markets=first_half_totals
```

To list all market keys and descriptions available for upcoming games, run:

```bash
python main.py --list-market-keys
```

## Moneyline Classifier


The project includes a simple logistic regression model that predicts the
probability of the home team winning a matchup. Training data can be supplied
via a CSV file **or** gathered automatically using the Odds API historical odds
endpoint. The dataset should contain a `home_team_win` column as the target
along with feature columns such as team statistics, starting pitcher ratings,
bullpen strength, park factor and injury indicators.

To train the classifier and save it to ``moneyline_classifier.pkl`` run:

```bash
python main.py train_classifier --dataset=training_data.csv
```

Or fetch historical data for a date range and train directly from it:

```bash
python main.py train_classifier --sport=baseball_mlb \
    --start-date=2024-04-01 --end-date=2024-04-07
```

To predict with a trained model supply feature values as a JSON string:

```bash
python main.py predict_classifier --features='{"home_team_stat":1.2,"away_team_stat":0.8}'
```

The command prints the home team win probability.

To keep the classifier up to date without manually running a command each time,
use the continuous training mode. This repeatedly fetches historical data from a
start date up to the current day and retrains the model on a fixed interval
(24&nbsp;hours by default):

```bash
python main.py continuous_train_classifier --sport=baseball_mlb \
    --start-date=2024-04-01 --interval-hours=24
```

The process runs indefinitely until interrupted and writes the model to the path
given by ``--model-out`` after each training cycle.
Note that The Odds API only provides historical results for roughly the last
year. If the supplied ``--start-date`` is older than that window, the
continuous training command automatically clamps it to the most recent date
allowed by the API.
