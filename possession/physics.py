"""Tracking-only control evidence from Link & Hoernig (2017), pages 5–7.

Derivatives use observed runs, not interpolated gaps. These soft features cannot
establish live/dead status or prove control from a touch; the paper supplies a
running flag. Its thresholds need validation on this noisy tracking feed.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


def ball_motion(ball):
    """SG7/degree2 derivatives on observed runs; unsupported motion is zero."""
    ball = np.asarray(ball)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError('Expected ball positions [frames, 2]')
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    velocity = np.zeros((n, 2))
    acceleration = np.zeros((n, 2))
    support = np.zeros(n, bool)
    edges = np.flatnonzero(np.diff(np.r_[False, observed, False].astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 7:
            velocity[a:b] = savgol_filter(ball[a:b], 7, 2, deriv=1, delta=1/25, axis=0)
            acceleration[a:b] = savgol_filter(ball[a:b], 7, 2, deriv=2, delta=1/25, axis=0)
            support[a:b] = True
    speed = np.linalg.norm(velocity, axis=1)
    accel = np.linalg.norm(acceleration, axis=1)
    support &= np.isfinite(speed) & np.isfinite(accel) & (speed <= 60) & (accel <= 150)
    return np.where(support[:, None], velocity, 0), np.where(support, accel, 0), support


def paper_features(players, ball, teams):
    """Return finite physical features and team-symmetric names at 25 Hz."""
    players, ball, teams = np.asarray(players), np.asarray(ball), np.asarray(teams)
    if (players.ndim != 3 or players.shape[2] != 3
            or ball.shape != (len(players), 2) or teams.shape != (players.shape[1],)
            or not np.isin(teams, (0, 1)).all()):
        raise ValueError('Expected players[n,p,3], ball[n,2] and H/A teams[p]')
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    velocity, accel, support = ball_motion(ball)
    speed = np.linalg.norm(velocity, axis=1)
    speed, accel = np.where(support, speed, 0), np.where(support, accel, 0)
    mass = uniform_filter1d(support.astype(float), 25, mode='constant') if n else np.empty(0)

    def mean(values):
        if not n:
            return np.empty(0)
        total = uniform_filter1d(np.where(support, values, 0), 25, mode='constant')
        return np.divide(total, mass, out=np.zeros(n), where=mass > 0)

    columns, names = [], []

    def add(name, values):
        values = np.asarray(values)
        columns.append(values[:, None] if values.ndim == 1 else values)
        names.extend([name] if values.ndim == 1 else [name + '_H', name + '_A'])

    add('kick_support', support)
    add('kick_speed', speed)
    add('kick_accel', accel)
    add('kick_speed_variance', np.maximum(mean(speed * speed) - mean(speed)**2, 0))
    add('kick_accel_variance', np.maximum(mean(accel * accel) - mean(accel)**2, 0))

    xy = players[:, :, :2]
    valid = np.isfinite(players).all(axis=2) & (players[:, :, 2] >= 0)
    player_velocity = np.zeros(xy.shape)
    player_support = np.zeros(valid.shape, bool)
    if n >= 7:
        player_velocity[3:-3] = (xy[6:] - xy[:-6]) * (25/6)
        player_support[3:-3] = (valid[3:-3] & valid[:-6] & valid[6:]
            & (np.linalg.norm(player_velocity[3:-3], axis=2) <= 12))
    weighted, predicted, pressure = [], [], {r: [] for r in (1, 3, 5)}
    rows = np.arange(n)
    for team in (0, 1):
        tracks = np.flatnonzero(teams == team)
        if not len(tracks):
            weighted.append(np.zeros(n))
            predicted.append(np.full(n, 30.))
            for r in pressure:
                pressure[r].append(np.zeros(n))
            continue
        distance = np.linalg.norm(xy[:, tracks] - ball[:, None, :], axis=2)
        distance = np.where(valid[:, tracks] & observed[:, None], distance, np.inf)
        nearest = np.argmin(distance, axis=1)
        candidate = tracks[nearest]
        near = distance[rows, nearest]
        available = np.isfinite(near)
        weighted.append(np.exp(-.5 * (near / 1.5)**2) * accel)
        separation = np.linalg.norm(ball + velocity - xy[rows, candidate]
                                    - player_velocity[rows, candidate], axis=1)
        usable = available & support & player_support[rows, candidate]
        predicted.append(np.where(usable, np.minimum(separation, 30), 30))
        opponents = teams != team
        opponent_distance = np.linalg.norm(xy[:, opponents] - xy[rows, candidate, None], axis=2)
        for r in pressure:
            pressure[r].append((valid[:, opponents] & available[:, None]
                                & (opponent_distance < r)).sum(axis=1))
    add('kick_contact_accel', np.column_stack(weighted))
    add('kick_contact_accel_mean25', np.column_stack([mean(v) for v in weighted]))
    add('control_predicted_separation', np.column_stack(predicted))
    for r, values in pressure.items():
        add('control_opponents_' + str(r) + 'm', np.column_stack(values))
    features = np.concatenate(columns, axis=1).astype(np.float32)
    assert np.isfinite(features).all(), 'Nonfinite paper features'
    return features, names


def selfcheck():
    t = np.arange(41) / 25
    ball = np.column_stack([3*t, np.zeros(len(t))])
    players = np.zeros((len(t), 3, 3))
    players[:, :, 0] = 3*t[:, None]
    players[:, 1, 0] += 2
    players[:, :, 2] = 3
    players[:, 2, 2] = np.nan
    teams = np.array([0, 1, 1])
    x, names = paper_features(players, ball, teams)
    col = lambda name: x[:, names.index(name)]
    assert x.dtype == np.float32 and x.shape == (len(t), 17)
    assert np.all(col('kick_support') == 1)
    assert np.allclose(col('kick_speed'), 3)
    assert np.allclose(col('kick_accel'), 0, atol=1e-8)
    assert np.allclose(col('kick_speed_variance'), 0, atol=1e-8)
    assert np.allclose(col('control_predicted_separation_H')[3:-3], 0)
    assert np.all(col('control_predicted_separation_H')[[0, 1, 2, -3, -2, -1]] == 30)
    assert np.all(col('control_opponents_1m_H') == 0)
    assert np.all(col('control_opponents_3m_H') == 1)
    swapped, swapped_names = paper_features(players, ball, 1-teams)
    assert swapped_names == names
    for j, name in enumerate(names):
        other = name[:-2] + ('_A' if name.endswith('_H') else '_H') if name.endswith(('_H', '_A')) else name
        assert np.allclose(swapped[:, j], x[:, names.index(other)])
    gap_ball = ball.copy()
    gap_ball[15:20] = np.nan
    gap_ball[20:, 0] += 20
    gap, _ = paper_features(players, gap_ball, teams)
    assert np.all(gap[15:20, names.index('kick_support')] == 0)
    assert np.allclose(gap[:, names.index('kick_accel')], 0, atol=1e-8)
    assert np.allclose(gap[np.r_[0:15, 20:41], names.index('kick_speed')], 3)
    fast, _ = paper_features(players, ball * 30, teams)
    assert not fast[:, names.index('kick_support')].any()
    for length in (0, 1, 6):
        tiny, tiny_names = paper_features(players[:length], ball[:length], teams)
        assert tiny.shape == (length, len(names)) and tiny_names == names
        assert not tiny[:, names.index('kick_support')].any()
    missing, _ = paper_features(np.full_like(players, np.nan), np.full_like(ball, np.nan), teams)
    assert np.isfinite(missing).all() and not missing[:, names.index('kick_support')].any()
    absent, _ = paper_features(players[:, :1], ball, teams[:1])
    assert np.all(absent[:, names.index('control_predicted_separation_A')] == 30)
    assert not absent[:, names.index('control_opponents_5m_A')].any()


if __name__ == '__main__':
    selfcheck()
    print('Paper physics selfcheck passed')
