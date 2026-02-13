from datetime import datetime, timedelta

import numpy as np

from src.feature_utils import count_prev_within, time_since_signup_hours


def test_count_prev_within_basic_windows():
    base = datetime(2024, 1, 1, 12, 0, 0)
    timestamps = [
        base,
        base + timedelta(minutes=30),
        base + timedelta(hours=2),
        base + timedelta(hours=2, minutes=30),
    ]

    counts_1h = count_prev_within(timestamps, window_hours=1)
    assert counts_1h == [0, 1, 0, 1]

    counts_3h = count_prev_within(timestamps, window_hours=3)
    assert counts_3h == [0, 1, 2, 3]


def test_time_since_signup_hours():
    signup = datetime(2024, 1, 1, 10, 0, 0)
    purchase = signup + timedelta(hours=5, minutes=30)

    hours = time_since_signup_hours(signup, purchase)
    assert np.isclose(hours, 5.5)


def test_time_since_signup_hours_large_gap():
    signup = datetime(2024, 1, 1, 0, 0, 0)
    purchase = signup + timedelta(days=10, hours=3)

    hours = time_since_signup_hours(signup, purchase)
    assert np.isclose(hours, 243.0)


def test_count_prev_within_single_timestamp():
    base = datetime(2024, 1, 1, 12, 0, 0)
    counts = count_prev_within([base], window_hours=1)
    assert counts == [0]
