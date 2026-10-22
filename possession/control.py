"""Refine existing ownership changes with observed physical contact onsets."""

import numpy as np


def control_onsets(nearest, stable, observed):
    nearest, stable = np.asarray(nearest), np.asarray(stable)
    observed = np.asarray(observed, dtype=bool)
    if nearest.shape != (len(observed), 2) or stable.shape != nearest.shape:
        raise ValueError('Expected per-team distances/stability and one visibility flag per frame')
    onsets = []
    for team in (0, 1):
        contact = observed & (nearest[:, team] < 1.5) & (nearest[:, 1-team] > 3) & (stable[:, team] > .2)
        edges = np.diff(np.r_[False, contact, False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        onsets.append(starts[stops - starts >= 8])
    return onsets


def snap_control_boundaries(prediction, onsets, global_start):
    """Move only existing H/A edges to the closest new-team onset within 2 s.

    Contact is a physical proxy, so it only refines a model-supported change.
    It cannot insert a spell or move a live/dead boundary.
    """
    prediction = np.asarray(prediction)
    if prediction.ndim != 1 or not np.isin(prediction, (0, 1, 2, 3)).all() or len(onsets) != 2:
        raise ValueError('Expected H/A/D/X codes and two global contact-onset arrays')
    out = prediction.copy()
    if not len(out):
        return out, []
    edges = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1, len(out)]
    labels = out[edges[:-1]].copy()
    changes = []
    for i in range(1, len(edges) - 1):
        old = int(edges[i])
        before, after = int(labels[i-1]), int(labels[i])
        if before >= 2 or after >= 2:
            continue
        candidates = np.asarray(onsets[after]) - global_start
        candidates = candidates[(np.abs(candidates - old) <= 50)
            & (candidates > edges[i-1]) & (candidates < edges[i+1])]
        if len(candidates):
            new = int(candidates[np.argmin(np.abs(candidates - old))])
            edges[i] = new
            if new != old:
                changes.append((old, new))
    for i, label in enumerate(labels):
        out[edges[i]:edges[i+1]] = label
    original_edges = np.r_[0, np.flatnonzero(prediction[1:] != prediction[:-1]) + 1, len(out)]
    new_edges = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1, len(out)]
    assert len(original_edges) == len(new_edges)
    assert np.array_equal(prediction[original_edges[:-1]], out[new_edges[:-1]])
    assert np.all(np.diff(new_edges) > 0)
    assert np.array_equal(prediction == 2, out == 2)
    return out, changes
