"""Load external DFL XML without treating unknown possession as away ownership.

Usage: python experiments/load_idsse.py [directory]
Coordinates become metres from the lower left; player speeds become m/s.
BallStatus / BallPossession are labels and must never enter model features.
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PERIODS = {'firstHalf': 1, 'secondHalf': 2}


def frames(path):
    """Stream object-major frames, excluding unsupported match periods."""
    section = team = person = None
    for event, element in ET.iterparse(path, events=('start', 'end')):
        if event == 'start' and element.tag == 'FrameSet':
            section = PERIODS.get(element.get('GameSection'))
            team, person = element.get('TeamId'), element.get('PersonId')
        elif event == 'end' and element.tag == 'Frame':
            if section is not None:
                yield section, team, person, element.attrib
            element.clear()
        elif event == 'end' and element.tag == 'FrameSet':
            element.clear()


def load(directory, match='J03WOH'):
    directory = Path(directory)
    tracking = directory / f'{match}_tracking.xml'
    metadata = directory / f'{match}_meta.xml'
    cache = directory / f'{match}_arrays.npz'
    if cache.exists() and cache.stat().st_mtime >= max(tracking.stat().st_mtime, metadata.stat().st_mtime):
        with np.load(cache) as data:
            return dict(data)
    root = ET.parse(metadata).getroot()
    environment = root.find('.//Environment')
    pitch = np.array([float(environment.get('PitchX')), float(environment.get('PitchY'))])
    roster, teams = [], []
    for team in root.findall('.//Teams/Team'):
        for player in team.findall('./Players/Player'):
            roster.append(player.get('PersonId'))
            teams.append(0 if team.get('Role') == 'home' else 1)
    player_index = {person: j for j, person in enumerate(roster)}
    balls = []
    for half, team, person, row in frames(tracking):
        if team == 'BALL':
            balls.append((half, int(row['N']), float(row.get('X', 'nan')),
                          float(row.get('Y', 'nan')), float(row.get('Z', 'nan')),
                          int(row.get('BallStatus', '-1')), int(row.get('BallPossession', '-1'))))
    balls.sort(key=lambda row: row[:2])
    assert balls, 'No ball frames in external data'
    half = np.array([row[0] for row in balls], dtype=np.int8)
    frame_id = np.array([row[1] for row in balls], dtype=np.int32)
    index = {(row[0], row[1]): i for i, row in enumerate(balls)}
    assert len(index) == len(balls), 'Duplicate external ball frames'
    players = np.full((len(balls), len(roster), 3), np.nan, dtype=np.float32)
    for period, team, person, row in frames(tracking):
        j = player_index.get(person)
        i = index.get((period, int(row['N'])))
        if j is not None and i is not None:
            players[i, j] = (float(row.get('X', 'nan')), float(row.get('Y', 'nan')),
                             float(row.get('S', 'nan')) / 3.6)
    players[:, :, :2] += pitch / 2
    ball = np.array([row[2:4] for row in balls], dtype=np.float32) + pitch / 2
    status = np.array([row[5] for row in balls], dtype=np.int8)
    possession = np.array([row[6] for row in balls], dtype=np.int8)
    labels = np.full(len(balls), 'X', dtype='<U1')
    labels[status == 0] = 'D'
    labels[(status == 1) & (possession == 1)] = 'H'
    labels[(status == 1) & (possession == 2)] = 'A'
    result = dict(players=players, ball=ball.astype(np.float32), teams=np.array(teams),
                  half=half, frame_id=frame_id, status=status, possession=possession,
                  labels=labels, ball_z=np.array([row[4] for row in balls], dtype=np.float32),
                  pitch=pitch, player_ids=np.array(roster), fps=np.array(25))
    assert players.shape[0] == len(ball) == len(labels)
    np.savez_compressed(cache, **result)
    return result


if __name__ == '__main__':
    data = load(sys.argv[1] if len(sys.argv) > 1 else 'data/external/idsse')
    for name in ('status', 'possession', 'labels', 'half'):
        values, counts = np.unique(data[name], return_counts=True)
        print(name, dict(zip(values.tolist(), counts.tolist())))
    print('players', data['players'].shape, 'ball', data['ball'].shape)
