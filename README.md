# Odds Fetcher

This script fetches odds from [The Odds API](https://the-odds-api.com/).
It requires **Python 3**.

`cache_historical_odds.py` saves daily snapshot odds but does **not** include
timeline updates. Run `fetch_odds_timelines.py` first if you need the complete
sequence of line movements.

## Installation

Install the core dependencies with:

```bash
pip3 install -r requirements.txt
```

For optional features such as memory profiling and colored table output, also install the development extras:

```bash
pip3 install -r requirements-dev.txt
```

### Testing

Install both requirement files then run the test suite:

```bash
pip3 install -r requirements.txt -r requirements-dev.txt
pytest -q
```

You can alternatively run `make setup` to install the dependencies with a
single command.

## Usage

1. Set an environment variable with your API key:

```bash
export THE_ODDS_API_KEY=<your api key>
```

You can alternatively place the key in a `.env` file at the repository root.
Scripts such as `main.py` and `fetch_odds_cache.py` load this file
automatically so you don't have to export the variable each time.

If the variable is omitted the script runs in a limited *test mode* without
making any API requests.

### Quick start

To fetch live odds, compute all features and output today's bet recommendations:

```bash
python3 main.py --run
```

This command pulls the latest odds, computes every feature and prints a
dashboard with the top edges. Results are also saved to ``bet_log.jsonl`` and
``bet_recommendations.log``.

To build the training dataset and retrain the models in one step (example years shown):

```bash
python3 main.py --train --years 2018-2024
```

The old subcommands remain available for advanced workflows but ``--run`` and
``--train`` are the recommended one-click options.
These flags are mutually exclusive; the CLI will exit with an error if you
attempt to supply both at once. Any unknown command line options are also
reported so you can correct typos quickly. Options such as
``--game-period-markets`` only take effect with ``--event-odds`` or
``--list-market-keys`` and will trigger a warning otherwise.

2. Run the script (moneyline odds shown by default):

```bash
python3 main.py
```

To display point spread (handicap) odds, run:

```bash
python3 main.py spreads
```

To display totals (over/under) odds, run:

```bash
python3 main.py totals
```

To display alternate spreads odds, run:

```bash
python3 main.py alternate_spreads
```

To display alternate totals odds, run:

```bash
python3 main.py alternate_totals
```

To display team totals odds, run:

```bash
python3 main.py team_totals
```

To display alternate team totals odds, run:

```bash
python3 main.py alternate_team_totals
```

The script now focuses solely on head-to-head matchups. Running it without
extra options prints projected win probabilities using a trained
``h2h_data/h2h_classifier.pkl`` model. Be sure to train the classifier first —
otherwise the script will exit with an error. The ``h2h_data`` directory is
created automatically when training:

```bash
python3 main.py
```

Use ``--model`` to specify a different classifier file if needed.

If you previously generated a simple fallback model, delete that file and train
a classifier instead:

```bash
python3 main.py train_classifier --dataset=retrosheet_training_data.csv
```

To display outrights (futures) odds, run:

```bash
python3 main.py outrights
```

To display historical odds for a specific date, run:

```bash
python3 main.py historical --date=2024-01-01
```

Historical data is only available for approximately the last year.
When training with ``ml.py`` any start date older than this window is
automatically adjusted to the earliest supported day to avoid empty results.

The script requests head-to-head, point spread, totals, and outright markets as
needed. It prints

the API endpoint used and displays odds for the selected market in a clean
layout.

To list upcoming events and their IDs for a sport, run:

```bash
python3 main.py --list-events
```
The region defaults to ``us``. Pass ``--regions`` with a comma-separated list to
see events available in other regions.

To fetch all odds for a single event and print the raw JSON response, supply the
event ID and ``--event-odds``:

```bash
python3 main.py --event-id=<event id> --event-odds
```
You can also customize the ``--markets``, ``--odds-format`` and ``--date-format``
options when using this endpoint.

To include game period markets (e.g. quarters or innings) in the API request,
pass them via the ``--game-period-markets`` option:

```bash
python3 main.py --game-period-markets=first_half_totals
```

To list all market keys and descriptions available for upcoming games, run:

```bash
python3 main.py --list-market-keys
```

## Moneyline Classifier


The project includes a simple logistic regression model that predicts the
probability of the home team winning a matchup. Training data can be supplied
via a CSV file **or** gathered automatically using the Odds API historical odds
endpoint. The dataset should contain a `home_team_win` column as the target
(datasets generated with `data_prep.py` use `team1_win`, which is also accepted)
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
containing opening/closing odds and basic line movement details. The trainer
uses **linear regression** to predict a synthetic closing price representing
what a highly efficient book would offer. Comparing that price to the current
line yields the

``market_maker_mirror_score``—positive values suggest the market line is longer
than the mirrored price while negative values indicate shading.

To train the mirror model provide a CSV with these columns:

| Column Name    | Description                                                | Example |
| -------------- | ---------------------------------------------------------- | ------- |
| opening_odds   | The opening moneyline odds for the team                   | -110    |
| closing_odds   | The closing moneyline odds for the team                   | -130    |
| line_move      | opening_odds - closing_odds                              | 20      |
| volatility     | Line movement/volatility metric                          | 15.0    |
| momentum_price | Recent price change indicating market pressure          | -2      |
| acceleration_price | Change in momentum, highlighting shifts              | 1       |
| sharp_disparity | Difference between sharp book and others               | 5       |
| line_adjustment_rate | Rate of line changes per hour                       | 3       |
| oscillation_frequency | How often the price direction flips                | 0.5     |
| order_book_imbalance | Relative depth on the back vs. lay side            | 0.1     |
| mirror_target  | Target value: closing_odds or line_move (see code)       | -130    |

Example:

```csv
opening_odds,closing_odds,line_move,volatility,momentum_price,acceleration_price,sharp_disparity,line_adjustment_rate,oscillation_frequency,order_book_imbalance,mirror_target
-110,-130,20,15.0,-2,1,5,3,0.5,0.1,-130
+120,+105,15,8.0,0.5,-0.5,2,1,0.25,-0.05,105
```

All columns except ``volatility`` and the advanced metrics
(``momentum_price`` through ``order_book_imbalance``) are required.
``line_move`` represents ``opening_odds - closing_odds`` and ``mirror_target``
should match the value your training routine predicts—either ``closing_odds``
or ``line_move``. When required columns are missing the trainer raises
``ValueError: Missing required columns...``.

To generate mirror model training data run:

```bash
python3 generate_mirror_training_data.py
```

Train the model or keep it updated with:

```bash
python3 main.py continuous_train_mirror --dataset=mirror_training_data.csv --verbose
```

## Line Movement Modeling

The ``line_movement_model`` module provides utilities to predict how a moneyline
will shift from open to close. Call
``load_and_engineer_features`` to create implied probability and timing
features from a CSV. ``train_regression_model`` fits a random forest to predict
the exact shift while ``train_classification_model`` categorizes the movement
into direction bins. Running ``line_movement_model.py`` directly trains both
models on ``line_movement_data.csv`` and prints validation metrics.

When monitoring betting exchanges directly, the ``volume_surge.py`` module
offers a ``VolumeSurgeDetector`` utility. Provide it with a callback that
returns the latest matched volume from Betfair or Matchbook and it maintains a
short rolling history (default 10&nbsp;minutes). If the most recent reading
exceeds the window average by several standard deviations, the detector outputs
a ``volume_surge_score`` between 0 and 1. Sudden liquidity spikes often hint at
syndicates entering the market before prices move.

### Pricing Pressure Indices

In addition to volatility, the toolkit computes "pricing pressure" features to mimic insights typically derived from handle data:

- **Momentum:** Measures how much the price (odds) has changed over a recent window, capturing the direction and speed of market moves.
- **Acceleration:** Captures the change in momentum, highlighting periods when line movement is intensifying or slowing.
- **Cross-book Disparity:** Quantifies how much a sharp book's price diverges from the average of other books, surfacing potential leading signals.

These indices help models infer where and how quickly betting pressure is building, providing a synthetic view of market sentiment without direct access to ticket/handle data.

Each metric is computed from historical price timelines and automatically included in all model training and inference workflows.

### Liquidity & Market Depth Proxies

The toolkit estimates market liquidity and depth using price-based metrics:

- **Line Adjustment Rate:** Counts how many times the price changes per hour. High rates imply thin liquidity (lines move easily).
- **Oscillation Frequency:** Measures how often prices alternate direction (e.g., up-down-up) within a short window. Flickery lines often signal shallow markets or algorithmic adjustment.
- **Order Book Imbalance:** (For exchanges with depth data) Compares back and lay sizes to show which side dominates at a given moment.

These features help the model infer how much “resistance” a line faces, complementing volatility and pricing pressure. They serve as proxies for handle-driven movement—allowing sharp action, thin books, or liquidity vacuums to be recognized directly from public data.

### Synthetic Sentiment & Handle Features

To replace missing handle data, the toolkit generates synthetic sentiment and price-impact features:

- **Public Team Bias:** A static index for each MLB team reflecting "public side" popularity.
- **Reddit Sentiment:** Automated scores from /r/baseball and team subreddits, including:
    - *Sharp Social Score* — language resembling sharp bettors.
    - *Hype Trend Score* — public "can't lose" hype or overconfidence.
    - *Lineup Risk Score* — discussion of injuries or lineup uncertainty.
- **Implied Handle:** A numeric proxy for bet volume, derived from the price movement required to shift the line.
- **Data Augmentation:** Historical samples are synthetically perturbed to simulate extreme betting flows, improving model generalization.
- **Multi-Scale Features:** Momentum and volatility are computed over multiple time horizons (e.g., 10m, 2h, since open) for richer market context.

No fallback logic or bandages are used; features are computed directly from data and public signals.

Set ``REDDIT_CLIENT_ID``/``REDDIT_CLIENT_SECRET`` (and optionally ``REDDIT_USER_AGENT``)
for Reddit, ``TWITTER_BEARER_TOKEN`` for Twitter and
``TG_API_ID``/``TG_API_HASH`` along with ``TG_CHANNEL`` for Telegram if you wish to
enable these integrations. All of these variables are optional—the script will
fall back to the available services when any token is missing. ``OPENAI_API_KEY``
must still be configured for language model features.

#### Market Reaction Regimes

The toolkit clusters historical odds timelines into **market reaction regimes**
using unsupervised learning. Each event is assigned a regime cluster ID, which
reflects its pattern of line movement (e.g., flat, early sharp move, or late
steam). This regime feature is included in all model training and inference.

- **Feature Extraction:** For each event, indicators such as total line change,
  timing of the largest move and volatility are calculated.
- **Clustering:** These features feed a KMeans model to identify common market
  behaviors. The model trains unsupervised on historical data.
- **Usage:** The resulting ``market_regime`` cluster is appended to every event
  row so models can leverage it directly or train specialized sub-models for
  each regime.
- **Anomaly Detection:** When a new event does not fit any cluster well, it may
  signal unusual market activity.

This regime analysis lets the system recognize and adapt to diverse market
behaviors, augmenting raw volatility and pricing-pressure features for richer
modeling of line movement dynamics.

#### Fetching Historical Odds Cache

Historical API responses can be cached with ``fetch_odds_cache.py``:

```bash
python3 fetch_odds_cache.py --start-date=2024-01-01 --end-date=2024-01-31 --sport=baseball_mlb
```

Each day's JSON is saved to ``h2h_data/api_cache/YYYY-MM-DD.pkl``. Existing files
are skipped so the command can be run incrementally.

To capture multiple snapshots throughout the day use
``collect_snapshot_intervals.py``. It repeatedly queries the historical odds API
on a fixed interval and appends each response to ``h2h_data/api_cache/snapshot_data.pkl``:

```bash
python3 collect_snapshot_intervals.py --interval 5 --duration 60
```

After collecting several daily snapshots you can convert them into per-event
timelines:

```bash
python3 snapshot_to_timeline.py
```

This aggregates the prices for each ``event_id`` across the snapshots stored in
``snapshot_data.pkl`` and saves ``h2h_data/api_cache/<event_id>.pkl`` with an ``odds_timeline`` DataFrame
ready for ``prepare_autoencoder_dataset.py``.

To gather a long-range set of odds timelines in one step run:

```bash
python3 fetch_historical_timelines.py --sport=baseball_mlb \
    --start-date=2024-01-01 --end-date=2024-12-31 --interval=60
```
By default every timeline is aggregated into
``h2h_data/api_cache/snapshot_data.pkl`` which can be fed directly to the
autoencoder. Use ``--out-file`` to override the location.


#### Unsupervised Representation Learning

##### Why Odds Timeline Data Is Needed

Most advanced features depend on the full sequence of moneyline updates, not
just the final price. The autoencoder and RL modules analyze how each game's
line evolves over time. Collect the timeline data **first** and then generate
the dataset:

```bash
python3 fetch_odds_timelines.py --sport=baseball_mlb \
    --start-date=2024-04-01 --end-date=2024-04-30
python3 prepare_autoencoder_dataset.py
```

Run ``fetch_odds_timelines.py`` **before** ``prepare_autoencoder_dataset.py``.
The dataset builder relies on the ``odds_timeline`` files saved by the first
command and will be empty if they have not been fetched. The timelines script is
distinct from ``cache_historical_odds.py``—that helper stores only daily odds
snapshots and does not provide the ``odds_timeline`` data required for the
autoencoder.

``prepare_autoencoder_dataset.py`` now searches every ``.pkl`` file under
``h2h_data`` by default. Specify ``--cache-dir`` if your files live elsewhere,
and ``--out-file`` to customize where the aggregated timelines are written.

The toolkit uses a sequence autoencoder to learn latent embeddings of moneyline movement.
An LSTM-based autoencoder is trained to reconstruct normalized odds timelines; the hidden state ("latent vector") summarizes the dynamic pattern of each event’s line moves.
These embeddings (autoencoder_feature_1, autoencoder_feature_2, ...) are added to every example for model training and live inference.

**How it works:**
- Each odds timeline is normalized and encoded to a fixed-length vector.
- The autoencoder is trained unsupervised, capturing subtle behaviors such as steady drift, sharp jumps, or oscillatory moves.
- Downstream models use the latent vector as input, improving their ability to recognize complex and novel market patterns without relying solely on hand-engineered features.

This approach provides a deeper, data-driven summary of market dynamics for each game.

_No fallback or bandage models are included; the autoencoder is trained directly from market data._

```bash
python3 prepare_autoencoder_dataset.py
```

This command collects all ``odds_timeline`` entries under the specified cache
directory (``h2h_data`` by default) and writes ``odds_timelines.pkl``. Supply
this file to ``train_sequence_autoencoder``.

To prepare the dataset and train the model in one step run:

```bash
python3 train_autoencoder.py
```

### Reinforcement Learning Market Maker

The toolkit includes a reinforcement-learning agent that mimics bookmaker line
movement. Given historical odds timelines, the agent learns to move the
moneyline in response to volatility, pricing pressure and time-to-start. The
agent operates within an RL environment where actions shift the line and rewards
depend on how closely its final price matches the actual closing line.

**Key Components:**
- **MarketMakerEnv** – Simulates bookmaker decision-making with features like
  time-to-game, price, volatility and momentum.
- **DQN Agent** – Trained to move the line using discrete actions such as
  ``+5``, ``-5`` or ``0`` price points, optimizing for minimal closing-line
  error.
- **CLI Integration** – Train the RL agent with
  ``--train_rl_market_maker`` and supply a dataset of odds timelines.
- **Inference** – The agent's recommended price adjustment is appended to each
  event's feature dictionary for downstream models.

Train the policy from cached odds timelines:

```bash
python3 main.py --train_rl_market_maker --rl_dataset_path=/path/to/cache \
    --rl_model_out=market_maker_rl.pt
```

During evaluation ``main.py`` loads ``market_maker_rl.pt`` and appends
``rl_line_adjustment`` to each event snapshot so other models can anticipate
sharp moves.

**Usage:**
- The RL-adjusted line acts as a synthetic signal of bookmaker behavior, aiding
  model training or live forecasting.
- No explicit handle or ticket data is required; the agent reacts purely to
  observed market features.

**Reward Structure:**
The agent receives higher rewards when its closing price closely matches the
true market close, with optional extensions for simulated profit and loss.


No fallback logic or bandage models are used; the agent operates directly from
the environment and training data.

### Hybrid Neural Network

The hybrid neural network fuses fundamental team stats and market-reaction data in a single architecture.  
- **Fundamental branch:** Processes team statistics and contextual features via a feed-forward network.
- **Market branch:** Encodes the odds timeline (and/or multi-book price grid) using an LSTM sequence encoder.
- The outputs are concatenated and passed to a final layer to predict win probability.

**How it works:**
- Each event’s dataset row includes both fundamental features and a normalized odds sequence.
- The model is trained end-to-end, learning to extract complementary signals from team stats and live market movement.
- Cross-validation and holdout sets are recommended for reliable validation.

**Integration:**
- Inference calls `predict_hybrid_probability` with the team stats and odds sequence for each event.
- The output (`hybrid_prob`) is included in all event records and can be used as a base model in the ensemble.

This permanently integrated architecture replaces temporary or ad hoc feature logic, ensuring models always benefit from both fundamental and market perspectives.

_No fallback logic or bandages are used; both branches are always required for training and inference._

### Ensemble of Specialized Models

The system combines multiple predictors—fundamental models, market-dynamics
models, recent-form and CLV nets—into a single, robust probability using a
meta-model ensemble.

- **Base models:**
  - *Fundamental classifier* (e.g. `h2h_classifier.pkl` or
    `moneyline_classifier.pkl`)
  - *Market maker mirror* (volatility-driven score)
  - *RL market maker* (line adjustment policy)
  - *Recent-form/CLV* (optional: dual-head, CLV nets)

- **Training:**
  Use `generate_mirror_training_data.py` to generate a CSV of all base model
  predictions and true outcomes.
  Then run `train_ensemble_model` from `ensemble_models.py` to fit a meta-model
  (e.g. logistic regression) on these features.

- **Inference:**
  In `main.py`, ensemble probabilities are computed for each event by passing
  the latest base model outputs to `predict_ensemble_probability`. Use
  `build_feature_dict` from `ensemble_models.py` to assemble these values
  consistently.

This approach leverages the unique strengths of each specialized model,
yielding a win probability that is more robust than any single model alone.
No fallback or bandage logic is included; the ensemble is trained and applied
directly on the underlying predictions.

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
python3 main.py train_classifier --dataset=training_data.csv --features-type=pregame
```

To train both heads at once:

```bash
python3 main.py train_classifier --dataset=training_data.csv --features-type=dual
```

An experimental PyTorch implementation of this dual-head setup is available in
``dual_head_nn.py``. It trains a small neural network to mirror closing lines
while also predicting game outcomes.

Pass ``--recent-half-life`` to weight newer rows more heavily based on a date column
(defaults to the first column containing ``date`` in its name). Use ``--date-column``
to specify the exact column if needed.
``--recency-multiplier`` controls how much extra emphasis recent form columns
receive when paired with season-long averages. A value of ``0.7`` gives 70% weight
to stats like ``*_last_10`` when creating ``*_weighted_recent`` features.

Or fetch historical data for a date range and train directly from it:

```bash
python3 main.py train_classifier --sport=baseball_mlb \
    --start-date=2024-04-01 --end-date=2024-04-07 \
    --features-type=pregame
```

If you've already cached historical API responses under ``h2h_data/api_cache``
you can turn those ``.pkl`` files into a training CSV with:

```bash
python3 data_prep.py --output=training_data.csv
```

By default, rows without a recorded result are kept with ``team1_win`` set to
``NaN``. This column is treated as ``home_team_win`` when training.
Add the ``--require-results`` flag if you want to filter those out.

For historical MLB games ``integrate_data.py`` combines your cached odds with
official Retrosheet logs. Running it creates ``integrated_training_data.csv``
ready for model training:

```bash
python3 integrate_data.py
```

If you only need the Retrosheet results without any odds data, run
``process_retrosheet.py``. Place any ``GLYYYY.TXT`` gamelog files in the
``retrosheet_data`` directory first to skip those downloads. The script will
grab any seasons not already present (2018–2025 by default) and combine the
logs into ``retrosheet_training_data.csv`` with rolling team stats. When
``THE_ODDS_API_KEY`` is set, historical moneyline prices from The Odds API are
merged automatically so no synthetic odds are created. Manager and home plate
umpire columns are also included. Your ``GL`` files stay in that folder so they
can be fed to the Retrosheet trainer or other tools:

```bash
python3 process_retrosheet.py
```

The resulting CSV can then be supplied to ``train_classifier`` as shown above.

To predict with a trained model supply feature values as a JSON string:

```bash
python3 main.py predict_classifier --features='{"home_team_stat":1.2,"away_team_stat":0.8}'
```

The command prints the home team win probability.

Training column names are persisted in the ``.pkl`` file. Prediction helpers
reindex the provided feature mapping to this saved order, filling any missing
columns with ``0``. A warning is emitted when expected columns are absent so
mismatches can be caught early.

To keep the classifier up to date without manually running a command each time,
use the continuous training mode. This repeatedly fetches historical data from a
start date up to the current day and retrains the model on a fixed interval
(24&nbsp;hours by default):

```bash
python3 main.py continuous_train_classifier --sport=baseball_mlb \
    --start-date=2024-04-01 --interval-hours=24 \
    --features-type=pregame
```

The process runs indefinitely until interrupted and writes the model to the path
given by ``--model-out`` after each training cycle.
Note that The Odds API only provides historical results for roughly the last
year. If the supplied ``--start-date`` is older than that window, the
continuous training command automatically clamps it to the most recent date
allowed by the API.

To record live and recent scores for training data, fetch them using the ``scores`` command:

```bash
python3 main.py scores --sport=baseball_mlb --days-from=1 --save-history
```

This saves the latest results to ``h2h_data/scores_history.jsonl`` which can later
be converted into a training dataset.

To keep a dataset-driven model up to date, run the moneyline continuous training
command. It retrains from the specified CSV on a fixed interval and also records
recent scores:

```bash
python3 main.py continuous_train_moneyline --dataset=training_data.csv \
    --interval-hours=24 --sport=baseball_mlb
```

Likewise the market maker mirror model can be refreshed automatically:

```bash
python3 main.py continuous_train_mirror --dataset=mirror_training_data.csv --verbose
```

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

## Verbose Output and Debugging

Pass the ``--verbose`` flag to ``main.py`` to see progress messages while data
is fetched and evaluated. This prints a short notice for each game as odds are
retrieved. For interactive troubleshooting launch the script with Python's
built-in debugger:

```bash
python3 -m pdb main.py --verbose
```

The debugger lets you inspect variables, step through code and continue
execution at your own pace.

