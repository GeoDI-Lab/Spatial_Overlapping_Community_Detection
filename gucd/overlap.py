"""Overlap count and overlap zone (Figs 4 and 5).

A block group's overlap count is the number of communities in which its membership exceeds 0.5. The overlap zone of a
city comprises the block groups with an overlap count of at least 2; the rest are the other block groups with
membership (overlap count 1, or 0 with nonzero memberships).
"""
import numpy as np


def overlap_count(F, threshold=0.5):
    F = np.asarray(F, float)
    return np.where(np.isnan(F).any(1), -1, (np.nan_to_num(F) > threshold).sum(1))


def overlap_zone(F, threshold=0.5):
    """Boolean masks (zone, rest) over block groups; block groups without memberships are in neither."""
    F = np.asarray(F, float)
    cnt = overlap_count(F, threshold)
    member = (cnt >= 0) & (np.nan_to_num(F).sum(1) > 0)
    return member & (cnt >= 2), member & (cnt < 2)
