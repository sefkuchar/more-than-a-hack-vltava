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


# CHALLENGE.md: possession changes only on *controlled* possession ("a player keeps the ball
# for a few seconds"); touches, deflections and duels without control do not count, and while
# the ball is loose possession stays with the last team that had it. The provider's
# BallPossession flags every touch, so live spells shorter than this are given back to the team
# in control before them. The value comes from that sentence, not from any score.
CONTROL_SECONDS = 2.0


def controlled_labels(labels, half, seconds=CONTROL_SECONDS, fps=25):
    """Merge uncontrolled live spells (< seconds) into the surrounding controlling team."""
    out = np.array(labels, dtype=np.int8, copy=True)
    limit = int(round(seconds * fps))
    for start, stop in segments(half):
        seq = out[start:stop]
        while True:
            edges = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
            runs = list(zip(edges[:-1], edges[1:]))
            changed = False
            for i, (a, b) in enumerate(runs):
                if seq[a] > 1 or b - a >= limit:
                    continue
                before = seq[runs[i - 1][0]] if i > 0 else 3
                after = seq[runs[i + 1][0]] if i + 1 < len(runs) else 3
                target = before if before < 2 else after if after < 2 else None
                if target is not None and target != seq[a]:
                    seq[a:b] = target
                    changed = True
                    break
            if not changed:
                break
        out[start:stop] = seq
    return out


def fit_duration_prior(label_arrays, half_arrays, fps=25):
    """Two-component log-normal mixtures of complete spell durations (half edges censored)."""
    from sklearn.mixture import GaussianMixture
    samples = {'possession': [], 'dead': []}
    for labels, half in zip(label_arrays, half_arrays):
        for start, stop in segments(half):
            seq = labels[start:stop]
            edges = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
            for a, b in zip(edges[1:-2], edges[2:-1]):
                if seq[a] < 3:
                    samples['dead' if seq[a] == 2 else 'possession'].append(int(b - a))
    groups = []
    for group in ('possession', 'dead'):
        d = np.array(samples[group])
        model = GaussianMixture(n_components=2, reg_covar=.25, random_state=0).fit(np.log(d).reshape(-1, 1))
        groups.append(dict(group=group, count=int(len(d)), mean_duration_seconds=float(d.mean() / fps),
                           median_duration_seconds=float(np.median(d) / fps),
                           weights=model.weights_.tolist(), log_frame_means=model.means_[:, 0].tolist(),
                           log_frame_sigmas=np.sqrt(model.covariances_[:, 0, 0]).tolist()))
    return {'schema_version': 1, 'fps': fps, 'groups': groups,
            'source': 'IDSSE training matches, controlled-possession labels',
            'attribution': 'Bassek et al.; DFL/Sportec IDSSE, CC-BY-4.0.'}
