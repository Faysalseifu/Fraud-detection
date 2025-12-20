"""Lightweight feature utilities for time-based fraud features.

Functions avoid pandas dependencies so unit tests stay fast and self-contained.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List

import numpy as np


def count_prev_within(timestamps: Iterable[datetime], window_hours: float) -> List[int]:
    """Count prior events within a rolling window for each timestamp.

    Args:
        timestamps: Iterable of datetime objects, assumed in ascending order.
        window_hours: Lookback window in hours.

    Returns:
        List of counts where each element i is the number of events strictly before
        timestamps[i] and within the preceding window.
    """

    ts_list = list(timestamps)
    if not ts_list:
        return []

    arr = np.array([ts.timestamp() for ts in ts_list], dtype=np.float64)
    window_secs = window_hours * 3600.0
    # For each time, find the leftmost index within the window using searchsorted.
    left_idx = np.searchsorted(arr, arr - window_secs, side="left")
    counts = np.arange(len(arr)) - left_idx
    return counts.tolist()


def time_since_signup_hours(signup: datetime, purchase: datetime) -> float:
    """Compute hours between signup and purchase."""

    delta: timedelta = purchase - signup
    return delta.total_seconds() / 3600.0
