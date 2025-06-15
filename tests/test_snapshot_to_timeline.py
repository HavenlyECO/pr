import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import snapshot_to_timeline as st


def test_parse_timestamp_hyphenated():
    fp = Path("2024-06-08T12-30-45Z.pkl")
    ts = st._parse_timestamp(fp)
    assert ts == pd.Timestamp(datetime(2024, 6, 8, 12, 30, 45))

