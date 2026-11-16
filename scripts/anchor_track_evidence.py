"""Stable tracked-candidate interaction evidence on measured and missing frames.

The model sees only raw tracking features. It follows candidate slots across
time before pooling, unlike framewise nearest-player reselection. Proximity is
one uncertain feature, never a label or controlled-possession gate.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from second_zero.contextual import velocity
from second_zero.common import sha256, json_write
from second_zero.evaluate import official_metrics
from scripts.train_contextual import corruption
from scripts import sprint_decoder_lattice as lattice
from scripts import sprint_decoder_segments_geometry_v2 as graph

DATA = ROOT / 'private/improvement/external/prepared'
RUN = ROOT / 'runs/anchor-refinement/anchor-track-v1'
REPORT = ROOT / 'reports/anchor-refinement/anchor-track-v1'
OFFSETS = [-60, -30, -15, -5, 0, 5, 15, 30, 60]
TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY']


def guard(stage):
    def check(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if 'J03WR9' in p or '/evaluation/gt/' in p:
                raise PermissionError('No reserved final or competition references')
            if stage != 'evaluate' and p.endswith('.truth.npy') and 'J03WQQ' in p:
                raise PermissionError('No WQQ truth before frozen evaluation')
    return check


class TrackedEvidence:
    def __init__(self, raw):
        self.ball = np.asarray(raw['ball'], np.float32)
        self.players = np.asarray(raw['players'], np.float32)
        self.team = np.asarray(raw['team'])
        self.n = len(self.ball)
        ix = np.arange(self.n)
        self.seen = np.isfinite(self.ball).all(1)
        prev = np.maximum.accumulate(np.where(self.seen, ix, -self.n))
        nxt = np.minimum.accumulate(np.where(self.seen, ix, 2 * self.n)[::-1])[::-1]
        left = ix - prev <= nxt - ix
        reference = np.where(left, prev, nxt)
        self.eligible = (reference >= 0) & (reference < self.n) & (abs(reference - ix) <= 60)
        self.reference = np.clip(reference, 0, self.n - 1)
        self.pv = velocity(self.players)
        self.bv = velocity(self.ball)
        self.ps = np.linalg.norm(self.pv, axis=2)
        self.bs = np.linalg.norm(self.bv, axis=1)
        self.pv[self.ps > 15] = np.nan
        self.bv[self.bs > 60] = np.nan
        self.seen_fraction = uniform_filter1d(self.seen.astype(np.float32), 31, mode='nearest')

    def features(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        ref = self.reference[rows]
        selected = []
        for t in (0, 1):
            slots = np.flatnonzero(self.team == t)
            distances = np.linalg.norm(self.players[ref][:, slots] - self.ball[ref, None], axis=2)
            selected.append(slots[np.argsort(np.nan_to_num(distances, nan=np.inf), axis=1)[:, :3]])
        candidate = []
        globals_ = [(ref - rows)[:, None] / 60, self.seen[rows, None].astype(float), self.seen_fraction[rows, None]]
        for t in (0, 1):
            ids = selected[t]
            anchor_player = self.players[ref[:, None], ids]
            anchor_relative = anchor_player - self.ball[ref, None]
            pieces = []
            for offset in OFFSETS:
                requested = rows + offset
                at = np.clip(requested, 0, self.n - 1)
                pos = self.players[at[:, None], ids]
                motion = self.pv[at[:, None], ids]
                rel = pos - self.ball[at, None]
                rel_motion = motion - self.bv[at, None]
                distance = np.linalg.norm(rel, axis=2)
                elapsed = abs(at - ref) / 25
                displacement = pos - anchor_player
                speed_bound = np.linalg.norm(displacement, axis=2) / np.maximum(elapsed[:, None], .04)
                # Slot jumps break continuity evidence; they do not imply loss
                # of possession, and availability remains an explicit feature.
                continuous = ((speed_bound <= 15) | (at == ref)[:, None]) & (requested >= 0)[:, None] & (requested < self.n)[:, None]
                valid = continuous & np.isfinite(distance)
                values = [distance / 20,
                          np.linalg.norm(rel - anchor_relative, axis=2) / 20,
                          np.linalg.norm(rel_motion, axis=2) / 30,
                          np.sum(rel * rel_motion, axis=2) / np.maximum(distance, .1) / 30,
                          np.sum(motion * self.bv[at, None], axis=2) / np.maximum(np.linalg.norm(motion, axis=2) * self.bs[at, None], .1),
                          np.linalg.norm(motion, axis=2) / 12]
                values = np.stack(values, axis=-1)
                values[~valid] = np.nan
                availability = np.stack([valid, continuous & np.isfinite(motion).all(2)], axis=-1).astype(np.float32)
                pieces.append(np.concatenate([values, availability], axis=-1).reshape(len(rows), -1))
                if t == 0:
                    globals_.extend([self.seen[at, None].astype(float), np.minimum(self.bs[at, None], 60) / 40])
            candidate.append(np.concatenate(pieces, axis=1))
        g = np.concatenate(globals_, axis=1)
        output = np.stack([np.c_[candidate[t], candidate[1 - t], g] for t in (0, 1)], axis=1).astype(np.float32)
        output[~np.isfinite(output)] = np.nan
        return np.clip(output, -10, 10)


def load(key):
    name, condition = key.rsplit('-', 1)
    with np.load(DATA / (name + '.npz')) as a:
        raw = {k: a[k] for k in ['ball', 'players', 'team', 'pitch']}
    if condition == 'corrupted':
        raw = corruption(raw, 100)
    return TrackedEvidence(raw)


def predict(model, features):
    p = [model.predict_proba(features[:, t])[:, 1] for t in (0, 1)]
    return .5 * (logit(np.clip(p[1], 1e-6, 1 - 1e-6)) - logit(np.clip(p[0], 1e-6, 1 - 1e-6)))


def predict_key(model, key):
    dest = RUN / 'raw'; dest.mkdir(exist_ok=True)
    path = dest / (key + '.npz')
    if path.exists():
        with np.load(path) as a:
            return {k: a[k] for k in a.files}
    data = load(key)
    result = np.zeros(data.n, np.float32)
    rows = np.flatnonzero(data.eligible)
    for start in range(0, len(rows), 4096):
        take = rows[start:start + 4096]
        result[take] = predict(model, data.features(take))
    output = {'raw_logit': result, 'eligible': data.eligible, 'ball_seen': data.seen}
    np.savez_compressed(path, **output)
    print('predicted', key, 'eligible', int(data.eligible.sum()), flush=True)
    return output


def train():
    RUN.mkdir(parents=True, exist_ok=False); REPORT.mkdir(parents=True, exist_ok=True)
    parameters = dict(max_iter=180, max_leaf_nodes=15, max_depth=5, min_samples_leaf=80,
                      l2_regularization=10., learning_rate=.06, early_stopping=False, random_state=20261024)
    protocol = {'train_matches': TRAIN, 'calibration': 'WPY temperature and probability blend 0,.05,.15',
                'candidate_selection': 'Top3 of each team at current measured frame or nearest measured endpoint within60f; no absolute player identity input',
                'history_offsets': OFFSETS, 'feature_definition': 'Same candidate slots across offsets: distance, relative-displacement change, relative speed, radial approach, co-motion cosine, player speed, observation and continuity flags; >15m/s slot jumps mark history unavailable',
                'sampling': 'Every12th eligible H/A frame in each training clean/corrupted view, both perspectives',
                'learned_owner_as_features': False, 'model': parameters,
                'source_sha256': sha256(__file__), 'no_proximity_or_contact_as_control_gate': True}
    json_write(RUN / 'protocol.json', protocol)
    xx, yy = [], []; started = time.monotonic()
    for key in graph.keys(TRAIN):
        data = load(key); y = graph.truth(key)
        rows = np.flatnonzero(data.eligible & (y < 2) & (np.arange(len(y)) % 12 == 0))
        f = data.features(rows)
        xx.extend([f[:, t] for t in (0, 1)]); yy.extend([y[rows] == t for t in (0, 1)])
        print('training_features', key, f.shape, flush=True)
    x, y = np.concatenate(xx), np.concatenate(yy); del xx, yy, f
    print('fit', x.shape, flush=True)
    model = HistGradientBoostingClassifier(**parameters).fit(x, y)
    with (RUN / 'model.pkl').open('wb') as f:
        pickle.dump(model, f)
    json_write(RUN / 'training.json', {'shape': list(x.shape), 'seconds': time.monotonic() - started})


def aggregate(views):
    return {condition: {m: float(np.mean([r[m] for k, r in views.items() if condition == 'overall' or k.endswith(condition)])) for m in ('spell', 'macro')}
            for condition in ('overall', 'clean', 'corrupted')}


def candidate(key, raw, temperature, weight, quality_model):
    tag = 'WPY' if key.startswith('J03WPY-') else 'reliability'
    with np.load(lattice.RUN / 'cache' / tag / '16' / (key + '.npz')) as a:
        cached = {k: a[k] for k in a.files}
    z = cached['logit'].copy()
    if not weight:
        return graph.graph_decode(z, cached['base'], cached['segments'], cached['quality'], .55, .01), z
    mask = raw['eligible']
    p = expit(z)
    p[mask] = (1 - weight) * p[mask] + weight * expit(raw['raw_logit'][mask] / temperature)
    z = logit(np.clip(p, 1e-7, 1 - 1e-7))
    x = graph.signals(key); x[:, 0] = z
    pool = graph.proposals(z, cached['base'], x[:, 4], x[:, 7:10])
    pool = lattice.proposals(z, cached['base'], pool, 16)
    q = np.clip(quality_model.predict(graph.segment_features(x, pool)), 0, 1)
    pred = graph.graph_decode(z, cached['base'], pool, q, .55, .01)
    return pred, z


def calibrate():
    with (RUN / 'model.pkl').open('rb') as f:
        model = pickle.load(f)
    quality_model = graph.get_model(); pm = official_metrics()
    cache, zs, ys = {}, [], []
    for key in graph.keys(['J03WPY']):
        raw = predict_key(model, key); cache[key] = raw; y = graph.truth(key)
        mask = raw['eligible'] & (y < 2); zs.append(raw['raw_logit'][mask]); ys.append(y[mask])
    z, y = np.concatenate(zs), np.concatenate(ys)
    fit = minimize_scalar(lambda lt: np.mean(np.logaddexp(0, z / np.exp(lt)) - y * z / np.exp(lt)), bounds=(-2.3, 2.3), method='bounded')
    temperature = float(np.exp(fit.x)); rows = []
    for weight in (0., .05, .15):
        dest = RUN / 'WPY' / str(weight); dest.mkdir(parents=True, exist_ok=True); views = {}
        for key, raw in cache.items():
            pred, owner = candidate(key, raw, temperature, weight, quality_model)
            np.savez_compressed(dest / (key + '.npz'), predicted=pred, logit=owner)
            views[key] = graph.metric(pm, graph.truth(key), pred)
        rows.append({'weight': weight, **aggregate(views), 'views': views})
        print('WPY', weight, rows[-1]['overall'], rows[-1]['clean'], rows[-1]['corrupted'], flush=True)
    selected = max(rows, key=lambda r: (r['overall']['spell'], r['overall']['macro']))
    json_write(RUN / 'freeze.json', {'temperature': temperature, 'weight': selected['weight'],
               'selected_WPY': selected['overall'], 'model_sha256': sha256(RUN / 'model.pkl'),
               'source_sha256': sha256(__file__), 'WQQ_labels_accessed': False})
    json_write(REPORT / 'calibration.json', {'rows': rows, 'selected': selected, 'temperature': temperature})


def evaluate():
    freeze = json.loads((RUN / 'freeze.json').read_text())
    if not freeze['weight']:
        print('WPY did not select new evidence; no WQQ evaluation.', flush=True)
        return
    with (RUN / 'model.pkl').open('rb') as f:
        model = pickle.load(f)
    quality_model = graph.get_model(); views = {}; dest = RUN / 'WQQ'; dest.mkdir(exist_ok=False)
    # Produce and freeze all candidate predictions before opening WQQ truth.
    for key in graph.keys(['J03WQQ']):
        raw = predict_key(model, key)
        pred, z = candidate(key, raw, freeze['temperature'], freeze['weight'], quality_model)
        np.savez_compressed(dest / (key + '.npz'), predicted=pred, logit=z)
    json_write(dest / 'prediction_freeze.json', {p.name: sha256(p) for p in dest.glob('*.npz')})
    pm = official_metrics()
    for key in graph.keys(['J03WQQ']):
        with np.load(dest / (key + '.npz')) as a:
            views[key] = graph.metric(pm, graph.truth(key), a['predicted'])
    result = {**aggregate(views), 'views': views, 'config': freeze,
              'scope': 'Complete fixed external pipeline on reused WQQ development; raw new head trained only on four other matches, WPY calibration.'}
    json_write(REPORT / 'evaluation.json', result)
    print('WQQ', result['overall'], result['clean'], result['corrupted'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['train', 'calibrate', 'evaluate'])
    args = parser.parse_args(); threadpool_limits(2); sys.addaudithook(guard(args.stage))
    globals()[args.stage]()
