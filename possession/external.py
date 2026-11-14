"""External supervision: public IDSSE / DFL-Sportec matches (other games, CC-BY-4.0).

No label from the provided BAR-ATH match is read anywhere in this module. The only
thing taken from BAR-ATH is the unlabelled ball-visibility pattern of its automatic
tracking, used to corrupt the external ball track so training sees realistic gaps.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from experiments import load_idsse
from games import GAME, half_video_blocks
from possession.features import build_features
from possession.physics import ball_motion

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / 'data/external'
TRAIN_MATCHES = ('J03WOH', 'J03WQQ')
VALIDATION_MATCH = 'J03WOY'
TARGET_PITCH = np.array([105.3, 68.0])
CODES = {'H': 0, 'A': 1, 'D': 2, 'X': 3}


def _find(directory, pattern, fallback):
    hits = sorted(directory.glob(pattern))
    if hits:
        return hits[0]
    path = directory / fallback
    if not path.exists():
        raise FileNotFoundError('Missing external file {} (see Vltava.md, Reproduction)'.format(pattern))
    return path


def load_match(match, directory=EXTERNAL_DIR):
    """Arrays for one IDSSE match on the BAR-ATH pitch scale; labels are targets only."""
    directory = Path(directory)
    work = directory / 'cache'
    work.mkdir(parents=True, exist_ok=True)
    tracking = _find(directory, 'DFL_04_03_positions_raw_observed_*{}.xml'.format(match), match + '_tracking.xml')
    meta = _find(directory, 'DFL_02_01_matchinformation_*{}.xml'.format(match), match + '_meta.xml')
    for source, name in ((tracking, '_tracking.xml'), (meta, '_meta.xml')):
        link = work / (match + name)
        if not link.exists():
            try:
                link.symlink_to(source.resolve())
            except OSError:  # Windows without symlink rights
                import shutil
                shutil.copyfile(source, link)
    d = load_idsse.load(work, match)
    scale = TARGET_PITCH / d['pitch']
    players = d['players'].copy()
    players[:, :, :2] *= scale
    ball = d['ball'] * scale
    labels = np.array([CODES[s] for s in d['labels']], dtype=np.int8)
    return dict(players=players, ball=ball.astype(np.float32), teams=d['teams'], half=d['half'], labels=labels)


def target_visibility(target_ball):
    """Unlabelled ball-observation flags of the provided game's live halves."""
    parts = [np.isfinite(target_ball[a:b + 1]).all(axis=1) for _, a, b in half_video_blocks(GAME)]
    return np.concatenate(parts)


def corrupt_ball(ball, visibility, seed):
    """Impose the target's gap pattern and smooth 1 m positional noise (deterministic)."""
    rng = np.random.RandomState(seed)
    offset = rng.randint(len(visibility))
    mask = np.resize(np.roll(visibility, -offset), len(ball))
    noise = gaussian_filter1d(rng.normal(size=ball.shape), 3, axis=0)
    noise /= noise.std()
    out = ball + noise.astype(np.float32)
    out[~mask] = np.nan
    return out


def kick_acceleration(ball, half):
    accel = np.zeros(len(ball), dtype=np.float32)
    for h in np.unique(half):
        ids = np.flatnonzero(half == h)
        _, a, _ = ball_motion(ball[ids])
        accel[ids] = a
    return accel


def match_features(m, ball, names=None):
    x, got, raw = build_features(arrays=(m['players'], ball, m['teams']), use_tags=False)
    if names is not None and got != names:
        raise ValueError('External feature schema differs from target schema')
    raw['kick_accel'] = kick_acceleration(ball, m['half'])
    return x, got, raw


def segments(half):
    edges = np.flatnonzero(np.diff(half)) + 1
    bounds = np.r_[0, edges, len(half)]
    return list(zip(bounds[:-1], bounds[1:]))
