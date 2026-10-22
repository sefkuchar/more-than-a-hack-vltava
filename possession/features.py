"""Physical and temporal evidence, with no ground-truth or frame-number features."""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d, minimum_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags


def tracking(data_dir=None):
    meta, raw = require(['atd_meta', 'atd_raw'], data_dir=data_dir)
    teams = team_of_each_track(meta)
    cache = Path(raw).parent / 'cache_tracking.npz'
    if cache.exists() and cache.stat().st_mtime >= max(Path(raw).stat().st_mtime, Path(meta).stat().st_mtime):
        with np.load(cache) as z:
            return z['players'], z['ball'], teams
    n = source_frames()
    players = np.full((n, len(teams), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    with open(raw) as handle:
        for i, line in enumerate(handle):
            fields = line.strip().split(':')
            if len(fields) != 3 or int(fields[0]) != i + 1 or i >= n:
                raise ValueError('Non-contiguous or malformed tracking at line {}'.format(i + 1))
            values = np.fromstring(fields[1].replace(';', ','), sep=',', dtype=np.float32)
            if values.size != len(teams) * 3:
                raise ValueError('Wrong player count at line {}'.format(i + 1))
            players[i] = values.reshape(-1, 3)
            ball[i] = np.fromstring(fields[2], sep=',', dtype=np.float32)[:2]
    if i + 1 != n:
        raise ValueError('Incomplete tracking')
    scale = np.array(GAMES[GAME]['pitch'], dtype=np.float32)
    players[:, :, :2] *= scale
    ball *= scale
    np.savez_compressed(cache, players=players, ball=ball)
    return players, ball, teams


def rolling(values, radius):
    return uniform_filter1d(np.asarray(values, dtype=np.float64), 2 * radius + 1, axis=0, mode='nearest')


def interpolate(values):
    """Interpolation is an input feature, never a claim that the ball was observed."""
    idx = np.arange(len(values))
    out = np.array(values, copy=True)
    for j in range(out.shape[1]):
        ok = np.isfinite(out[:, j])
        if np.any(ok):
            out[:, j] = np.interp(idx, idx[ok], out[ok, j])
    return out


def history(values, window):
    count=np.minimum(np.arange(len(values))+1, window)
    sums=np.concatenate([np.zeros((1,values.shape[1])), np.cumsum(values,axis=0)])
    i=np.arange(len(values))
    return (sums[i+1]-sums[np.maximum(0,i+1-window)])/count[:,None]


def build_features(data_dir=None, arrays=None, use_tags=True, broadcast=False, history_features=False):
    players, ball, teams = tracking(data_dir) if arrays is None else arrays
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    idx = np.arange(n)
    previous = np.maximum.accumulate(np.where(observed, idx, -n))
    following = np.minimum.accumulate(np.where(observed, idx, 2*n)[::-1])[::-1]
    age = np.minimum(idx - previous, following - idx) / 25
    interpolated = interpolate(ball)
    distance = np.linalg.norm(players[:, :, :2] - interpolated[:, None, :], axis=2)
    distance = np.where(np.isfinite(distance), distance, 100)
    distance[~np.isfinite(players[:, :, 2])] = 100
    nearest = np.stack([distance[:, teams == k].min(axis=1) for k in (0, 1)], axis=1)
    velocity = (np.roll(interpolated, -3, axis=0) - np.roll(interpolated, 3, axis=0)) * (25/6)
    ball_speed = np.linalg.norm(velocity, axis=1)
    player_velocity = (np.roll(players[:, :, :2], -3, axis=0)
                       - np.roll(players[:, :, :2], 3, axis=0)) * (25/6)
    relative_speed = np.linalg.norm(player_velocity-velocity[:, None, :], axis=2)
    co_motion = np.nan_to_num(np.exp(-0.5*(relative_speed/6)**2), nan=0)
    stable = np.exp(-0.5*(distance/1.5)**2)*co_motion*observed[:, None]
    stable = np.stack([stable[:, teams == k].max(axis=1) for k in (0,1)], axis=1)
    before = interpolated-np.roll(interpolated, 5, axis=0)
    after = np.roll(interpolated, -5, axis=0)-interpolated
    turn = np.sum(before*after, axis=1)/(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)+1e-5)
    player_speed = np.nan_to_num(players[:, :, 2], nan=-1)
    # Fixed dummy goalkeeper tracklets have NaN speed and must not count as stationary players.
    valid_player = player_speed >= 0
    moving = np.sum(player_speed > 2, axis=1) / np.maximum(valid_player.sum(axis=1), 1)
    mean_speed = np.maximum(player_speed, 0).sum(axis=1) / np.maximum(valid_player.sum(axis=1), 1)
    boundary = np.minimum.reduce([interpolated[:, 0], 105.3-interpolated[:, 0],
                                  interpolated[:, 1], 68-interpolated[:, 1]])
    contact = np.exp(-0.5 * (nearest / 1.5)**2) * observed[:, None]
    tags = np.zeros((n, 2))
    restart = np.zeros(n)
    names = {'Barcelona': 0, 'Atletic Club': 1}
    for e in (load_tags(data_dir=data_dir) if use_tags else []):
        a, b = max(0, e['start_frame']), min(n, e['end_frame'])
        if b <= a:
            continue
        code = e['code'] or ''
        if code == 'SET PIECES':
            restart[a:b] = 1
        elif code in ('PROGRESSION', 'BUILD UP', 'ATTACKING TRANSITION', 'CREATING CHANCES',
                      'LONG BALL', 'DEFENDING IN MIDDLE THIRD', 'DEFENDING IN ATTACKING THIRD',
                      'DEFENDING IN DEFENSIVE THIRD', 'DEFENSIVE TRANSITION'):
            team = names.get(e['tags'].get('Team'))
            if team is not None:
                if code.startswith('DEFEND') or code == 'DEFENSIVE TRANSITION':
                    team = 1 - team
                tags[a:b, team] = 1
    columns, names_out = [], []
    def add(name, values):
        a = np.asarray(values)
        if a.ndim == 1:
            a = a[:, None]
        columns.append(a)
        names_out.extend([name] if a.shape[1] == 1 else [name+'_H', name+'_A'])
    add('observed', observed)
    add('observation_age', np.minimum(age, 30))
    add('distance', np.minimum(nearest, 30))
    add('distance_margin', np.clip(nearest[:, 1]-nearest[:, 0], -20, 20))
    add('ball_speed', np.minimum(ball_speed, 50))
    add('boundary', np.clip(boundary, -10, 25))
    add('moving_players', moving)
    add('mean_player_speed', mean_speed)
    add('restart', restart)
    add('phase', tags)
    add('contact', contact)
    add('stable_contact', stable)
    add('trajectory_turn', turn)
    for r in (5, 12, 25, 50, 125):
        add('contact_mean_'+str(r), rolling(contact, r))
        add('visibility_'+str(r), rolling(observed, r))
        add('motion_'+str(r), rolling(moving, r))
        add('ball_speed_mean_'+str(r), rolling(np.minimum(ball_speed, 50), r))
        add('boundary_min_'+str(r), minimum_filter1d(boundary, 2*r+1))
        add('stationary_'+str(r), rolling(ball_speed < 1.5, r))
        add('stable_contact_mean_'+str(r), rolling(stable, r))
    # A pass retains its previous controller; symmetric means lose that ordering.
    for w in ((25,75) if history_features else []):
        add('past_contact_'+str(w),history(contact,w))
        add('future_contact_'+str(w),history(contact[::-1],w)[::-1])
        add('past_stable_'+str(w),history(stable,w))
        add('future_stable_'+str(w),history(stable[::-1],w)[::-1])
    if broadcast:
        from experiments.align_broadcast import aligned_broadcast
        br = aligned_broadcast(data_dir=data_dir)
        seen = np.isfinite(br['nearest']).all(axis=1)
        bc = np.nan_to_num(np.exp(-0.5*(br['nearest']/1.5)**2)) * seen[:, None]
        add('broadcast_visible', seen)
        add('broadcast_distance', np.nan_to_num(br['nearest'], nan=30).clip(0, 30))
        add('broadcast_contact', bc)
        add('broadcast_quality', br['confidence'])
        add('broadcast_ball_error', np.nan_to_num(np.linalg.norm(br['ball']-ball, axis=1), nan=30).clip(0,30))
        for r in (5, 12, 25, 50):
            add('broadcast_contact_mean_'+str(r), rolling(bc, r))
    features = np.concatenate(columns, axis=1).astype(np.float32)
    assert np.isfinite(features).all(), 'Nonfinite features'
    return features, names_out, dict(ball=ball, nearest=nearest, contact=contact,
                                    moving=moving, mean_speed=mean_speed, tags=tags,
                                    observed=observed, boundary=boundary, ball_speed=ball_speed)
