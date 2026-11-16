"""Small frozen-winner comparison of soft anchors and directional boundaries.

Calibration reads WPY only. Evaluation must follow a saved freeze and reads WQQ.
No training-match complete-pipeline score is represented as held out, and the
reserved match and competition references are prohibited in both stages.
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
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from scripts import sprint_decoder_lattice as lattice
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts import sprint_decoder_boundary as directional
from second_zero.anchor_refinement import (AnchorConfig, stable_anchors,
    anchored_candidates, boundary_search_segments, maxima, connect_nodes,
    transition_quality)
from second_zero.common import sha256, json_write
from second_zero.evaluate import official_metrics

RUN = ROOT / 'runs/anchor-refinement/anchor-v1'
REPORT = ROOT / 'reports/anchor-refinement/anchor-v1'


def guard(stage):
    def check(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            if 'J03WR9' in path or '/evaluation/gt/' in path:
                raise PermissionError('Reserved final and public references excluded')
            if stage == 'calibrate' and path.endswith('.truth.npy') and 'J03WPY-' not in path:
                raise PermissionError('Anchor selection may read WPY truth only')
    return check


def directional_scores(key):
    path = directional.RUN / 'scores' / (key + '.npz')
    if path.exists():
        with np.load(path) as a:
            q = a['score']
    else:
        x = directional.signals(key)
        model = directional.getmodel()
        q = np.stack([model.predict_proba(directional.features(x, t))[:, 1] for t in (0, 1)], axis=1)
    calibration = json.loads((directional.RUN / 'calibration.json').read_text())
    return expit(calibration['slope'] * logit(np.clip(q, 1e-8, 1 - 1e-8)) + calibration['intercept'])


def refine_evidence(x, base, p_dead, pool, quality, quality_model,
                    config=AnchorConfig(), boundary=None):
    """Reusable reference-free API; all supplied arrays must be split-safe."""
    z = x[:, 0]
    winner = graph.graph_decode(z, base, pool, quality, .55, .01)
    anchors = stable_anchors(z, x[:, 1], x[:, 3], x[:, 4], p_dead, base, config)
    nodes, local, brackets = anchored_candidates(z, base, anchors, config, boundary)
    groups = boundary_search_segments(winner, anchors, config)
    chosen, joint_proposals = [], []
    prefix = np.r_[0., np.cumsum(z, dtype=float)]
    for points, left, right in groups:
        both = np.r_[left, right]
        q = np.clip(quality_model.predict(graph.segment_features(x, both)), 0, 1)
        frames = (2 * both[:, 2] - 1) * (prefix[both[:, 1]] - prefix[both[:, 0]])
        gains = q + .01 * frames
        scores = gains[:len(points)] + gains[len(points):]
        if boundary is not None:
            scores = scores + .1 * boundary[points, right[:, 2]]
        picks = maxima(points, scores, config.choices, config.separation)
        chosen.extend(picks)
        keep = np.isin(points, picks)
        joint_proposals.extend(left[keep].tolist())
        joint_proposals.extend(right[keep].tolist())
    if joint_proposals:
        local = np.r_[local, np.asarray(joint_proposals, dtype=np.int64)]
    nodes = np.unique(np.r_[nodes, np.asarray(chosen, dtype=np.int64), local[:, :2].ravel()])
    result = connect_nodes(pool, nodes, local, base, config.endpoint_radius)
    # Reconnect the new frame locations using exactly the winner's established
    # short-arc policy, retaining all original and multi-change alternatives.
    result = lattice.proposals(z, base, result, 16)
    q = np.clip(quality_model.predict(graph.segment_features(x, result)), 0, 1)
    metadata = {'anchors': len(anchors), 'anchor_frames': int(np.sum(anchors[:, 1] - anchors[:, 0])),
                'brackets': len(brackets), 'same_team_brackets': sum(a[2] == a[3] for a in brackets),
                'joint_boundary_searches': len(groups), 'new_boundary_nodes': int(len(np.setdiff1d(nodes, np.unique(pool[:, :2])))),
                'original_segments': len(pool), 'refined_segments': len(result)}
    return {'logit': z, 'base': base, 'segments': result, 'quality': q,
            'boundary': boundary if boundary is not None else np.zeros((len(z), 2)),
            'winner': winner, 'anchors': anchors}, metadata


def prepare(key, model, mode):
    dest = RUN / 'cache' / mode
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / (key + '.npz')
    if path.exists():
        with np.load(path) as a:
            data = {k: a[k] for k in a.files}
        return data, json.loads(path.with_suffix('.json').read_text())
    tag = 'WPY' if key.startswith('J03WPY-') else 'reliability'
    with np.load(lattice.RUN / 'cache' / tag / '16' / (key + '.npz')) as a:
        cached = {k: a[k] for k in a.files}
    x = graph.signals(key)
    x[:, 0] = cached['logit']
    with np.load(lattice.STATUS / (key + '.npz')) as a:
        p_dead = a['p_dead']
        assert np.array_equal(a['predicted'], cached['base'])
    boundary = directional_scores(key) if mode == 'anchor-boundary' else None
    data, metadata = refine_evidence(x, cached['base'], p_dead, cached['segments'], cached['quality'], model, boundary=boundary)
    np.savez_compressed(path, **data)
    json_write(path.with_suffix('.json'), metadata)
    print('prepared', key, mode, metadata, flush=True)
    return data, metadata


def summarize(views):
    return {'overall': {m: float(np.mean([r[m] for r in views.values()])) for m in ('spell', 'macro')},
            'conditions': {v: {m: float(np.mean([r[m] for k, r in views.items() if k.endswith(v)])) for m in ('spell', 'macro')} for v in ('clean', 'corrupted')},
            'views': views}


def run(stage):
    threadpool_limits(2)
    sys.addaudithook(guard(stage))
    RUN.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = graph.get_model()
    pm = official_metrics()
    matches = ['J03WPY'] if stage == 'calibrate' else ['J03WQQ']
    if stage == 'calibrate':
        variants = [('baseline', 0.), ('anchor', 0.), ('anchor-boundary', .05), ('anchor-boundary', .15)]
        plan = {'anchor_config': AnchorConfig().to_dict(), 'variants': variants,
                'quality_model_sha256': sha256(graph.RUN / 'model.pkl'),
                'directional_model_sha256': sha256(directional.RUN / 'model.pkl'),
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / 'second_zero/anchor_refinement.py']},
                'baseline': 'actual sprint winner .55 penalty/.01 frame weight/16 neighbors',
                'selection': 'WPY only, two additional-directional-evidence weights; no anchor threshold sweep',
                'split_limit': 'All frozen components trained on four training matches; status and multiple calibrations fit WPY. WQQ is reused external development, not untouched generalization.'}
        json_write(RUN / 'protocol.json', plan)
    else:
        freeze = json.loads((RUN / 'freeze.json').read_text())
        variants = [('baseline', 0.), ('anchor', 0.), ('anchor-boundary', freeze['boundary_weight'])]
    data_cache, baseline = {}, {}
    for key in graph.keys(matches):
        for mode in ('anchor', 'anchor-boundary'):
            data_cache[(key, mode)] = prepare(key, model, mode)[0]
        baseline[key] = data_cache[(key, 'anchor')]['winner']
    rows = []
    for mode, weight in variants:
        views = {}
        label = mode if mode != 'anchor-boundary' else mode + '-' + str(weight)
        folder = RUN / ('WPY' if stage == 'calibrate' else 'WQQ') / label
        folder.mkdir(parents=True, exist_ok=True)
        for key in graph.keys(matches):
            a = data_cache[(key, 'anchor' if mode == 'baseline' else mode)]
            pred = baseline[key] if mode == 'baseline' else graph.graph_decode(a['logit'], a['base'], a['segments'], a['quality'] + transition_quality(a['segments'], a['boundary'], a['base'], weight), .55, .01)
            if stage == 'evaluate' and mode == 'baseline':
                with np.load(ROOT / 'runs/two-hour-sprint/final' / (key + '.npz')) as saved:
                    assert np.array_equal(pred, saved['predicted']), 'Baseline must exactly replay selected winner'
            np.savez_compressed(folder / (key + '.npz'), predicted=pred, logit=a['logit'], status_mask=a['base'])
            views[key] = graph.metric(pm, graph.truth(key), pred)
            views[key]['changed_frames'] = int(np.sum(pred != baseline[key]))
        row = {'mode': mode, 'boundary_weight': weight, 'candidate': str(folder.relative_to(ROOT)), **summarize(views)}
        rows.append(row)
        print(stage, label, row['overall'], row['conditions'], flush=True)
    if stage == 'calibrate':
        eligible = [r for r in rows if r['mode'] == 'anchor-boundary']
        best = max(eligible, key=lambda r: (r['overall']['spell'], r['overall']['macro']))
        json_write(RUN / 'freeze.json', {'boundary_weight': best['boundary_weight'], 'selection': 'WPY only before WQQ scores',
                   'protocol_sha256': sha256(RUN / 'protocol.json'), 'anchor_config': AnchorConfig().to_dict(),
                   'quality_model_sha256': sha256(graph.RUN / 'model.pkl'), 'directional_model_sha256': sha256(directional.RUN / 'model.pkl')})
    json_write(REPORT / (stage + '.json'), {'rows': rows, 'seconds': time.monotonic() - started,
               'interpretation': 'Complete frozen-pipeline conditional decoder comparison on reused external development. Clean/corrupted views are paired same-match data.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['calibrate', 'evaluate'])
    run(parser.parse_args().stage)
