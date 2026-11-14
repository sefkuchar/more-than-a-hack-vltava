"""Globally decode team/dead evidence; transition costs suppress unsupported flicker."""

import numpy as np


def viterbi(log_evidence, switch_cost=8., dead_cost=None):
    evidence = np.asarray(log_evidence, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != 3 or not np.isfinite(evidence).all():
        raise ValueError('Expected finite n x 3 H/A/D evidence')
    if not len(evidence):
        return np.empty(0, dtype=np.int8)
    cost = np.full((3, 3), float(switch_cost))
    if dead_cost is not None:
        cost[2, :] = dead_cost
        cost[:, 2] = dead_cost
    np.fill_diagonal(cost, 0)
    back = np.zeros((len(evidence), 3), dtype=np.int8)
    score = evidence[0].copy()
    states = np.arange(3)
    for i in range(1, len(evidence)):
        transitions = score[:, None] - cost
        previous = transitions.argmax(axis=0)
        back[i] = previous
        score = evidence[i] + transitions[previous, states]
    result = np.empty(len(evidence), dtype=np.int8)
    result[-1] = score.argmax()
    for i in range(len(evidence)-1, 0, -1):
        result[i-1] = back[i, result[i]]
    return result
