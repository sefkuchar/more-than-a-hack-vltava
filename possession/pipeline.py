"""Shared decode step: calibrated H/A/D evidence -> whole possession sequence."""

import numpy as np

from possession.control import control_onsets, snap_control_boundaries
from possession.duration import decode


def onsets_from(features, names, raw):
    stable = features[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    return control_onsets(raw['nearest'], stable, raw['observed'], raw.get('kick_accel'))


def decode_segments(probability, segments, prior, temperature, onsets=None, refine=True):
    """Decode each [start, stop) segment independently; frames outside them stay X (3)."""
    codes = np.full(len(probability), 3, dtype=np.int8)
    refined = 0
    for start, stop in segments:
        path = decode(probability[start:stop], prior, temperature)
        if refine and onsets is not None:
            path, changes = snap_control_boundaries(path, onsets, start)
            refined += len(changes)
        codes[start:stop] = path
    return codes, refined
