# Odds Fetcher

This script fetches odds from [The Odds API](https://the-odds-api.com/).

## Usage

1. Set an environment variable with your API key:

```bash
export THE_ODDS_API_KEY=<your api key>
```

If the variable is omitted the script runs in a limited *test mode* without
making any API requests.

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

Another derived field ``bet_freeze_flag`` triggers when heavy ticket
sentiment coincides with a significant line shortening on the same team. This
safeguard spots potential trap scenarios where books may be encouraging action
on an overpriced favorite, signaling you to hold off placing the bet.

``anti_correlation_flag`` extends this idea by firing whenever a team draws
over 70% of tickets yet the line moves against them. Such reverse line moves
suggest sportsbooks are welcoming public bets on a side they expect to lose,
providing a data-driven cue to fade the favorite.

When Reddit, Twitter or Telegram chatter is accessible the :func:`attach_social_scores`
helper can derive ``sharp_money_score_social``. It queries recent posts for each
team and uses OpenAI to rate how closely the language matches historical sharp
betting patterns (urgency, insider tone, etc.). This numeric feature can be
added alongside your regular statistics to capture non-price indicators of
informed action.

When beat-writer notes or insider chatter are available the
:func:`attach_managerial_signals` helper can extract tactical cues from the
text. It uses OpenAI to flag phrases such as "yanked the starter early" or
"pinch hitter coming for the lefty" and outputs boolean features like
``early_pull_flag`` and ``pinch_hit_flag``. These signals highlight managerial
patterns that may influence late-game outcomes beyond the box score.

Similarly, :func:`attach_lineup_risk_scores` looks for injury language such as
"late scratch", "day-to-day" or "questionable start" in pregame chatter pulled
from social media and news feeds. It assigns a ``lineup_risk_score`` between
0 and 1 for each team so you can capture last-minute uncertainty before it is
fully reflected in the betting lines.

To spot markets whipped into a frenzy, the :func:`attach_hype_trend_scores`
helper compiles recent social posts for each team and has OpenAI judge how much
hype or "can't lose" sentiment is present. This numeric ``hype_trend_score``
helps identify overreactions driven more by buzz than by fundamentals.

The :func:`attach_sentiment_fakeout_scores` helper goes a step further by
scanning chat messages, Twitter replies and Telegram threads for sarcasm or
exaggerated hype. OpenAI assigns a ``sentiment_fakeout_score`` from 0 to 1 where
higher values indicate that crowd sentiment is likely performative or ironic
rather than genuinely bullish. Including this feature helps avoid chasing lines
moved by viral trash talk rather than real conviction.

The ``multi_book_edge_score`` metric compares moneyline prices from Bovada,
MyBookie and BetUS. The odds from these softer books are converted to implied
probabilities. The difference between the highest and lowest value for the same
team is reported as ``soft_book_spread`` while their midpoint forms the
``multi_book_edge_score``. Large spreads highlight pricing disagreements that
can be exploited before the books adjust.

``market_maker_mirror_score`` goes a step further by modeling how sharp
bookmakers move their lines when money flow, volatility and public sentiment
shift. Train the model with ``train_market_maker_mirror_model`` on a dataset
containing opening/closing odds plus handle and ticket percentages. The model
predicts a synthetic closing price representing what a highly efficient book
would offer. Comparing that price to the current line yields the
``market_maker_mirror_score``—positive values suggest the market line is longer
than the mirrored price while negative values indicate shading.

When monitoring betting exchanges directly, the ``volume_surge.py`` module
offers a ``VolumeSurgeDetector`` utility. Provide it with a callback that
returns the latest matched volume from Betfair or Matchbook and it maintains a
short rolling history (default 10&nbsp;minutes). If the most recent reading
exceeds the window average by several standard deviations, the detector outputs
a ``volume_surge_score`` between 0 and 1. Sudden liquidity spikes often hint at
syndicates entering the market before prices move.

Set ``REDDIT_CLIENT_ID``/``REDDIT_CLIENT_SECRET`` for Reddit, ``TWITTER_BEARER_TOKEN``
for Twitter and ``TG_API_ID``/``TG_API_HASH`` for Telegram if you wish to
enable this feature. ``OPENAI_API_KEY`` must also be configured.

Columns prefixed with ``pregame_`` are treated as pregame features while those
starting with ``live_`` are considered live-game inputs. Use the
``--features-type`` option of ``train_classifier`` to train on one set or the
other and avoid mixing the two, which can lead to data leakage. Passing
``dual`` for this option builds a model with separate heads trained on each
feature group.
Any columns containing terms such as ``result`` or ``final`` are discarded
automatically to prevent leaking post-game information into the model.

Continuous numeric features like ERA, SLG and line delta are automatically
standardized during training to improve optimizer convergence.
Another useful metric is ``bullpenERA_vs_opponentSLG`` which subtracts an
opponent's slugging percentage from a team's bullpen ERA. Include
``bullpen_ERA`` and ``opponent_SLG_adjusted`` columns in your dataset and the
trainer will derive this value automatically to highlight volatile late-game
matchups.
If your dataset provides fields like ``stat_last_10`` alongside ``stat``,
``train_classifier`` will generate ``stat_weighted_recent`` using the
``--recency-multiplier`` value so recent performance carries more influence.
During training, probability quality is reported using AUC and Brier score
rather than simple accuracy.
When live-inning columns are available the validation set is also segmented into
``pregame``, ``5th inning`` and ``7th inning`` windows to gauge how accuracy
improves as more game context becomes available.

Each segment is calibrated separately so a predicted 65% chance in the
``7th inning`` truly reflects a historical 65% win rate. This segmentation
uses isotonic regression on the validation split to align probabilities within
each inning window, sharpening expected value calculations.

The ``live_features.py`` module introduces an ``InningDifferentialTracker``
utility for live play. It records the run differential at the end of each
inning and exposes them as ``live_inning_X_diff`` fields. By updating this
tracker with live scoring data you can feed time-aware inning-by-inning
differentials directly into the classifier.

``OffensivePressureTracker`` complements this by keeping rolling counts of
errors, runners left on base and RISP% over the last two innings. It exposes
features like ``errors_last_2`` and ``RISP_last_2`` that quantify recent
pressure or missed opportunities.

The same module provides ``build_win_probability_curve`` to evaluate how the
model's predicted win probability evolves throughout a game. Supply a list of
inning scores and the path to your model and it returns timestamped
probabilities suitable for plotting.

``WinProbabilitySwingTracker`` builds on this by recording the change in win
probability after each update. It exposes features like
``win_prob_delta_inning_X`` that capture sharp momentum swings from events such
as a big inning or pitching change.

``llm_inning_trend_summaries`` complements these utilities by turning raw
play-by-play logs into a concise ``trend_summary`` for each inning. OpenAI
distills the pitch sequence and scoring events into one short sentence per
inning so you can quickly gauge emotional and tactical momentum.

To train the classifier and save it to ``moneyline_classifier.pkl`` run:

```bash
python main.py train_classifier --dataset=training_data.csv --features-type=pregame
```

To train both heads at once:

```bash
python main.py train_classifier --dataset=training_data.csv --features-type=dual
```

Pass ``--recent-half-life`` to weight newer rows more heavily based on a date column
(defaults to the first column containing ``date`` in its name). Use ``--date-column``
to specify the exact column if needed.
``--recency-multiplier`` controls how much extra emphasis recent form columns
receive when paired with season-long averages. A value of ``0.7`` gives 70% weight
to stats like ``*_last_10`` when creating ``*_weighted_recent`` features.

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
odds, the calculated edge and the timestamp when the bet was suggested. The
stake is stored as a number—``0.0`` when no bankroll is supplied—so later
results can be evaluated consistently.

After the game finishes call ``update_bet_result`` from the ``bet_logger``
module to mark the bet as a win or loss. ``update_bet_result`` searches for the
first log entry where ``event_id`` and ``team`` match and ``outcome`` is still
``null`` and updates that record. If you logged multiple bets for the same
event/team pair, call the function once for each open position. The function
records the resulting payout and the ROI compared to the stake so you can track
how profitable the model's edge is over time. When ``closing_odds`` (or
``closing_implied_prob``) is supplied, ``update_bet_result`` also stores the
closing line's implied probability and logs ``deviation_score = predicted_prob -
closing_implied_prob`` to measure how far the model disagreed with the market at
close.

## Kelly Bet Sizing

When a bankroll value is supplied, stakes are calculated using the Kelly
criterion. For a predicted win probability ``p`` and the offered odds expressed
in decimal form, the fraction of bankroll to wager is::

    kelly_fraction = (b * p - (1 - p)) / b

where ``b`` is ``odds - 1``. This approach allocates more to high-edge plays and
reduces exposure when the edge is slim. Use the ``kelly_fraction`` argument to
scale down from full Kelly if desired.

## Risk Filter

To avoid overexposing the bankroll to marginal situations the ``evaluate_h2h``
logic applies a simple risk filter. Bets with a modeled edge below 5% or
moneyline odds worse than ``-170`` are assigned a zero weight. They will not be
logged or factored into weighted edge calculations, helping protect ROI by
discarding low-value opportunities.

## Memory Profiling

Training functions in ``ml.py`` accept a ``profile_memory`` flag. When set to
``True`` and the optional ``psutil`` package is installed, memory usage for
heavy pandas operations and model fitting is printed to the console. This can
help identify bottlenecks when working with large datasets.
