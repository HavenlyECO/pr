import pickle
import pandas as pd
from prepare_autoencoder_dataset import extract_odds_timelines, main


def test_extract_odds_timelines_top_level(tmp_path):
    df = pd.DataFrame({"timestamp": [1, 2], "price": [100, 101]})
    with open(tmp_path / "a.pkl", "wb") as f:
        pickle.dump({"odds_timeline": df}, f)

    timelines, files = extract_odds_timelines(tmp_path)
    assert len(timelines) == 1
    assert files == ["a.pkl"]
    pd.testing.assert_frame_equal(timelines[0], df[["timestamp", "price"]])


def test_extract_odds_timelines_missing(tmp_path, capsys):
    with open(tmp_path / "missing.pkl", "wb") as f:
        pickle.dump({}, f)

    timelines, files = extract_odds_timelines(tmp_path)
    out = capsys.readouterr().out
    assert "No odds_timeline in missing.pkl" in out
    assert timelines == []
    assert files == ["missing.pkl"]


def test_main_no_timelines(monkeypatch, tmp_path, capsys):
    with open(tmp_path / "x.pkl", "wb") as f:
        pickle.dump({}, f)

    monkeypatch.setattr("prepare_autoencoder_dataset.CACHE_DIR", tmp_path)
    monkeypatch.setattr("prepare_autoencoder_dataset.OUT_FILE", tmp_path / "out.pkl")
    main()
    out = capsys.readouterr().out
    assert "No odds timelines found in cache" in out
    assert "x.pkl" in out
    assert "Inspected files:" in out
    assert not (tmp_path / "out.pkl").exists()
