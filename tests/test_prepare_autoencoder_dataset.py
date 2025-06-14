import pickle
import pandas as pd
from prepare_autoencoder_dataset import extract_odds_timelines


def test_extract_odds_timelines_top_level(tmp_path):
    df = pd.DataFrame({"timestamp": [1, 2], "price": [100, 101]})
    with open(tmp_path / "a.pkl", "wb") as f:
        pickle.dump({"odds_timeline": df}, f)

    timelines = extract_odds_timelines(tmp_path)
    assert len(timelines) == 1
    pd.testing.assert_frame_equal(timelines[0], df[["timestamp", "price"]])
