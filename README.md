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

The script now focuses solely on head-to-head matchups. Running it without
extra options will print projected win probabilities using the trained
``h2h_data/h2h_classifier.pkl`` model. The ``h2h_data`` directory is created
automatically when training the model:

```bash
python main.py
```

Use ``--model`` to specify a different classifier file if needed.

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

While the utility is called a classifier, the model is trained in regression mode to predict a continuous win probability. You can choose any threshold later to convert that score into a binary label.

If your CSV includes both ``opening_odds`` and ``closing_odds`` columns,
``train_classifier`` will automatically create a ``line_delta`` feature
representing ``opening_odds - closing_odds``. When the absolute change
exceeds 15 points an additional ``line_movement_delta`` flag is set to ``1``
or ``-1`` depending on the direction of the move.

When ticket sentiment is available an additional ``reverse_line_move`` flag is
derived. This compares ``line_delta`` to the public ticket percentage and marks
cases where the odds move away from the popular side (for example when the line
shortens on a team receiving fewer bets). Such moves can indicate sharp action
or internal risk adjustments that the public market has yet to fully price in.

When Reddit, Twitter or Telegram chatter is accessible the :func:`attach_social_scores`
helper can derive ``sharp_money_score_social``. It queries recent posts for each
team and uses OpenAI to rate how closely the language matches historical sharp
betting patterns (urgency, insider tone, etc.). This numeric feature can be
added alongside your regular statistics to capture non-price indicators of
informed action.

Set ``REDDIT_CLIENT_ID``/``REDDIT_CLIENT_SECRET`` for Reddit, ``TWITTER_BEARER_TOKEN``
for Twitter and ``TG_API_ID``/``TG_API_HASH`` for Telegram if you wish to
enable this feature. ``OPENAI_API_KEY`` must also be configured.

Columns prefixed with ``pregame_`` are treated as pregame features while those
starting with ``live_`` are considered live-game inputs. Use the
``--features-type`` option of ``train_classifier`` to train on one set or the
other and avoid mixing the two, which can lead to data leakage.
Any columns containing terms such as ``result`` or ``final`` are discarded
automatically to prevent leaking post-game information into the model.

Continuous numeric features like ERA, SLG and line delta are automatically
standardized during training to improve optimizer convergence.
During training, probability quality is reported using AUC and Brier score
rather than simple accuracy.
When live-inning columns are available the validation set is also segmented into
``pregame``, ``5th inning`` and ``7th inning`` windows to gauge how accuracy
improves as more game context becomes available.

The ``live_features.py`` module introduces an ``InningDifferentialTracker``
utility for live play. It records the run differential at the end of each
inning and exposes them as ``live_inning_X_diff`` fields. By updating this
tracker with live scoring data you can feed time-aware inning-by-inning
differentials directly into the classifier.

The same module provides ``build_win_probability_curve`` to evaluate how the
model's predicted win probability evolves throughout a game. Supply a list of
inning scores and the path to your model and it returns timestamped
probabilities suitable for plotting.

``WinProbabilitySwingTracker`` builds on this by recording the change in win
probability after each update. It exposes features like
``win_prob_delta_inning_X`` that capture sharp momentum swings from events such
as a big inning or pitching change.

To train the classifier and save it to ``moneyline_classifier.pkl`` run:

```bash
python main.py train_classifier --dataset=training_data.csv --features-type=pregame
```

Pass ``--recent-half-life`` to weight newer rows more heavily based on a date column
(defaults to the first column containing ``date`` in its name). Use ``--date-column``
to specify the exact column if needed.

Or fetch historical data for a date range and train directly from it:

```bash
python main.py train_classifier --sport=baseball_mlb \
    --start-date=2024-04-01 --end-date=2024-04-07 \
    --features-type=pregame
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
    --start-date=2024-04-01 --interval-hours=24 \
    --features-type=pregame
```

The process runs indefinitely until interrupted and writes the model to the path
given by ``--model-out`` after each training cycle.
Note that The Odds API only provides historical results for roughly the last
year. If the supplied ``--start-date`` is older than that window, the
continuous training command automatically clamps it to the most recent date
allowed by the API.

## Bet Logging

The project can log recommended bets to ``bet_log.jsonl``. Each entry records
the team, odds, predicted win probability, the implied probability from the
odds, the calculated edge and the timestamp when the bet was suggested. When a
bankroll amount is provided the recommended stake is stored as well.

After the game finishes call ``update_bet_result`` from the ``bet_logger``
module to mark the bet as a win or loss. The function records the resulting
payout and the ROI compared to the stake so you can track how profitable the
model's edge is over time.
