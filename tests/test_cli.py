import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import builtins
import main
import pytest


def test_unknown_argument(capsys):
    with pytest.raises(SystemExit) as exc:
        main.main(['--bogus'])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert 'unrecognized arguments: --bogus' in captured.err.lower()


def test_mutually_exclusive_run_train():
    with pytest.raises(SystemExit) as exc:
        main.main(['--run', '--train'])
    assert exc.value.code == 2


def test_game_period_markets_warning(monkeypatch, capsys):
    monkeypatch.setattr(main, 'evaluate_h2h_all_tomorrow', lambda *a, **k: [])
    # stub out functions that would do I/O
    monkeypatch.setattr(main, 'print_h2h_projections_table', lambda *a, **k: None)
    monkeypatch.setattr(main, 'log_bet_recommendations', lambda *a, **k: None)
    monkeypatch.setattr(main, 'log_bets', lambda *a, **k: None)

    main.main(['--game-period-markets', '1st_half'])
    out = capsys.readouterr().out
    assert '--game-period-markets has no effect' in out


def test_missing_model_file(monkeypatch, capsys):
    def fake_run_pipeline(*args, **kwargs):
        raise FileNotFoundError('missing model')
    monkeypatch.setattr(main, 'run_pipeline', fake_run_pipeline)
    main.main(['--run'])
    captured = capsys.readouterr().out
    assert 'missing model' in captured
