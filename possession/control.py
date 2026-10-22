"""Refine existing ownership changes with sustained contact or observed kick onsets."""

import numpy as np

CAMERA_MAX_DISTANCE=2.5
CAMERA_OTHER_DISTANCE=3.
CAMERA_MIN_FRAMES=12
CAMERA_MIN_PROBABILITY=.4


def control_onsets(nearest, stable, observed, acceleration=None):
    """Optional supported acceleration confirms one-touch proximity intervals too."""
    nearest, stable = np.asarray(nearest), np.asarray(stable)
    observed = np.asarray(observed, dtype=bool)
    if nearest.shape != (len(observed), 2) or stable.shape != nearest.shape:
        raise ValueError('Expected per-team distances/stability and one visibility flag per frame')
    if acceleration is not None:
        acceleration = np.asarray(acceleration)
        if acceleration.shape != observed.shape or not np.isfinite(acceleration).all():
            raise ValueError('Expected finite supported ball acceleration per frame')
    onsets = []
    for team in (0, 1):
        proximity = observed & (nearest[:, team] < 1.5) & (nearest[:, 1-team] > 3)
        contact = proximity & (stable[:, team] > .2)
        edges = np.diff(np.r_[False, contact, False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        onsets.append(starts[stops - starts >= 8])
        if acceleration is not None:
            edges = np.flatnonzero(np.diff(np.r_[False, proximity, False].astype(np.int8)))
            kicks = [a for a, b in zip(edges[::2], edges[1::2]) if np.any(acceleration[a:b] >= 4)]
            onsets[-1] = np.union1d(onsets[-1], np.asarray(kicks, dtype=int))
    return onsets


def recover_camera_spells(prediction, probability, corrected_nearest, visible, corrected):
    """Restore short spells supported by cross-camera corrected proximity.

    Only frames where the broadcast-to-tactical team correction was active are
    eligible. Within those spans a short exclusive contact run is accepted when
    the calibrated classifier also assigns the alternative team at least 0.4
    mean probability. Dead-ball labels and existing H/A edges are preserved.
    """
    prediction=np.asarray(prediction)
    probability=np.asarray(probability,dtype=float)
    corrected_nearest=np.asarray(corrected_nearest,dtype=float)
    visible=np.asarray(visible,dtype=bool)
    corrected=np.asarray(corrected,dtype=bool)
    n=len(prediction)
    if prediction.ndim!=1 or not np.isin(prediction,(0,1,2,3)).all():
        raise ValueError('Expected H/A/D/X codes')
    if probability.shape!=(n,3) or not np.isfinite(probability).all():
        raise ValueError('Expected finite H/A/D probabilities')
    if corrected_nearest.shape!=(n,2) or not np.isfinite(corrected_nearest).all():
        raise ValueError('Expected two finite corrected distances per frame')
    if visible.shape!=corrected.shape or corrected.shape!=(n,):
        raise ValueError('Expected one visibility and correction flag per frame')
    out=prediction.copy()
    changes=[]
    eligible=corrected & visible
    for team in (0,1):
        proximity=eligible & (corrected_nearest[:,team]<CAMERA_MAX_DISTANCE) \
            & (corrected_nearest[:,1-team]>CAMERA_OTHER_DISTANCE)
        edges=np.flatnonzero(np.diff(np.r_[False,proximity,False].astype(np.int8)))
        for start,stop in zip(edges[::2],edges[1::2]):
            if stop-start<CAMERA_MIN_FRAMES:
                continue
            current=prediction[start]
            if current not in (0,1) or current==team:
                continue
            if not np.all(prediction[start:stop]==current):
                continue
            if probability[start:stop,team].mean()<CAMERA_MIN_PROBABILITY:
                continue
            out[start:stop]=team
            changes.append((int(start),int(stop),int(current),int(team)))
    return out,changes


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
