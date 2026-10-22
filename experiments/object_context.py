"""Small object-context features inspired by Borghesi et al. (2023)."""

import numpy as np


def context_features(players, ball, teams):
    """Use observed objects only; pass each half separately for velocity features."""
    players, ball, teams = np.asarray(players), np.asarray(ball), np.asarray(teams)
    if (players.ndim != 3 or players.shape[2] != 3
            or ball.shape != (len(players), 2) or teams.shape != (players.shape[1],)
            or not np.isin(teams, (0, 1)).all()):
        raise ValueError('Expected players [frames, tracks, 3], ball [frames, 2] and H/A teams')
    valid = np.isfinite(players).all(axis=2) & (players[:, :, 2] >= 0)
    observed = np.isfinite(ball).all(axis=1)
    xy = players[:, :, :2]
    outside = np.any((xy < 0) | (xy > [105.3, 68]), axis=2)
    distance = np.linalg.norm(xy - ball[:, None, :], axis=2)
    distance = np.where(valid & observed[:, None], distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :5]
    nearest_valid = np.isfinite(np.take_along_axis(distance, nearest, axis=1))
    columns, names = [], []
    for name in ('visible_players', 'outside_fraction', 'nearest5_fraction'):
        for team, suffix in ((0, 'H'), (1, 'A')):
            mask = valid & (teams == team)
            count = mask.sum(axis=1)
            if name == 'visible_players':
                values = count
            elif name == 'outside_fraction':
                values = (mask & outside).sum(axis=1) / np.maximum(count, 1)
            else:
                values = (nearest_valid & (teams[nearest] == team)).sum(axis=1)
                values = values / np.maximum(nearest_valid.sum(axis=1), 1)
            columns.append(values)
            names.append('context_' + name + '_' + suffix)
    supported = np.zeros(len(ball), dtype=bool)
    if len(ball) >= 7:
        supported[3:-3] = np.lib.stride_tricks.sliding_window_view(observed, 7).all(axis=1)
    velocity = np.zeros((len(ball), 2))
    indices = np.flatnonzero(supported)
    velocity[indices] = (ball[indices + 3] - ball[indices - 3]) * (25 / 6)
    features = np.column_stack([*columns, velocity, supported]).astype(np.float32)
    names += ['context_ball_velocity_x', 'context_ball_velocity_y', 'context_velocity_supported']
    assert np.isfinite(features).all()
    return features, names


def selfcheck():
    from possession.model import swap_teams

    for frames in (0, 1, 6):
        features, names = context_features(np.empty((frames, 0, 3)), np.full((frames, 2), np.nan), [])
        assert features.shape == (frames, 9) and features.dtype == np.float32
        assert not features.any()
    players = np.tile([[1., 1., 1.], [2., 1., 2.], [106., 1., 1.],
                       [1., 1., np.nan]], (10, 1, 1))
    ball = np.column_stack([np.arange(10) / 25, np.ones(10)])
    teams = np.array([0, 1, 0, 1])
    features, names = context_features(players, ball, teams)
    assert np.array_equal(features[0, :6], [2, 1, .5, 0, np.float32(2 / 3), np.float32(1 / 3)])
    assert np.allclose(features[3:7, 6:8], [1, 0])
    assert np.array_equal(features[:, 8], [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    mirrored, _ = context_features(players, ball, 1 - teams)
    assert np.array_equal(mirrored, swap_teams(features, names))
    ball[4] = np.nan
    missing, _ = context_features(players, ball, teams)
    assert not missing[:, 6:].any() and not missing[4, 4:6].any()
    print('Object-context missingness, velocity and team-swap checks passed')


if __name__ == '__main__':
    selfcheck()
