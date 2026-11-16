"""Frozen attribution control, real-view equivariance, and anchor diagnostics."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import anchor_refine_experiment as experiment
from scripts import sprint_decoder_lattice as lattice
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts.sprint_evaluate import score, guard, write_new
from second_zero.anchor_refinement import transition_quality
from second_zero.common import sha256
from second_zero.evaluate import official_metrics


def event_change(before, after):
    eb, ea = before['events'], after['events']
    gained, lost = [], []
    for old, new in zip(eb['events'], ea['events']):
        assert (old['start'], old['stop'], old['label']) == (new['start'], new['stop'], new['label'])
        if old['matched']['0.5'] == new['matched']['0.5']:
            continue
        item = {'start': old['start'], 'stop': old['stop'], 'label': old['label'], 'frames': old['frames'],
                'before_iou': old['best_same_team_iou'], 'after_iou': new['best_same_team_iou']}
        (gained if new['matched']['0.5'] else lost).append(item)
    return {'recovered': len(gained), 'lost': len(lost),
            'short_recovered': sum(r['frames'] <= 125 for r in gained),
            'short_lost': sum(r['frames'] <= 125 for r in lost),
            'gained_examples': gained, 'lost_examples': lost,
            'before_predicted_spells': eb['predicted_spells'], 'after_predicted_spells': ea['predicted_spells'],
            'before_short_matches': eb['short_matched_iou05'], 'after_short_matches': ea['short_matched_iou05'],
            'before_fragmentation': before['metrics']['fragmentation'],
            'after_fragmentation': after['metrics']['fragmentation'],
            'before_per_class': before['metrics']['per_class'], 'after_per_class': after['metrics']['per_class']}


def main():
    threadpool_limits(2)
    sys.addaudithook(guard)
    out = experiment.REPORT / 'attribution'
    out.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((experiment.RUN / 'freeze.json').read_text())
    weight = freeze['boundary_weight']
    control = experiment.RUN / 'boundary-only'
    control.mkdir(exist_ok=False)
    # This attribution control uses the same previously frozen weight. It does
    # not select a parameter after seeing WQQ, or change any model/checkpoint.
    write_new(out / 'control_protocol.json', {'boundary_weight': weight, 'freeze_sha256': sha256(experiment.RUN / 'freeze.json'),
              'scope': 'Attribution: existing lattice plus previously frozen directional score, no anchor proposals. No further selection.'})
    for key in graph.keys(['J03WPY', 'J03WQQ']):
        tag = 'WPY' if key.startswith('J03WPY-') else 'reliability'
        with np.load(lattice.RUN / 'cache' / tag / '16' / (key + '.npz')) as a:
            data = {k: a[k] for k in a.files}
        boundary = experiment.directional_scores(key)
        adjusted = data['quality'] + transition_quality(data['segments'], boundary, data['base'], weight)
        pred = graph.graph_decode(data['logit'], data['base'], data['segments'], adjusted, .55, .01)
        np.savez_compressed(control / (key + '.npz'), predicted=pred, logit=data['logit'])
    write_new(out / 'prediction_freeze.json', {p.name: sha256(p) for p in control.glob('*.npz')})

    # Fresh full proposal/scoring/decoding replay in both team orientations on
    # clean and corrupted real views, before loading any evaluation references.
    replay = {}
    model = graph.get_model()
    for key in ['J03WQQ-firstHalf-clean', 'J03WQQ-firstHalf-corrupted']:
        x = graph.signals(key)
        with np.load(lattice.RUN / 'cache/reliability/16' / (key + '.npz')) as a:
            data = {k: a[k] for k in a.files}
        x[:, 0] = data['logit']
        with np.load(lattice.STATUS / (key + '.npz')) as a:
            p_dead = a['p_dead']
        qboundary = experiment.directional_scores(key)
        original, metadata = experiment.refine_evidence(x, data['base'], p_dead, data['segments'], data['quality'], model, boundary=qboundary)
        flipped = x.copy(); flipped[:, [0, 1, 3, 7, 8, 9]] *= -1
        pool = data['segments'].copy(); pool[:, 2] = 1 - pool[:, 2]
        mask = np.where(data['base'] < 2, 1 - data['base'], data['base'])
        reverse, _ = experiment.refine_evidence(flipped, mask, p_dead, pool, data['quality'], model, boundary=qboundary[:, ::-1])
        def predict(a):
            q = a['quality'] + transition_quality(a['segments'], a['boundary'], a['base'], weight)
            return graph.graph_decode(a['logit'], a['base'], a['segments'], q, .55, .01)
        normal, exchanged = predict(original), predict(reverse)
        with np.load(experiment.RUN / 'WQQ' / ('anchor-boundary-' + str(weight)) / (key + '.npz')) as a:
            expected = a['predicted']
        changed = int(np.sum(normal != expected))
        discrepancies = int(np.sum(exchanged != np.where(normal < 2, 1 - normal, normal)))
        assert changed == discrepancies == 0
        replay[key] = {'exact_replay_changed_frames': changed, 'team_exchange_discrepancies': discrepancies,
                       'joint_before_after_searches': metadata['joint_boundary_searches'],
                       'fixed_status_preserved': bool(np.array_equal(normal[data['base'] >= 2], data['base'][data['base'] >= 2]))}
    write_new(out / 'reference_free_replay.json', {'views': replay, 'labels_accessed': False,
             'scope': 'Fresh anchors, both-side split quality, proposal connections, quality scoring and final decode from validated frozen upstream caches. Upstream models were not rerun.'})

    pm = official_metrics()
    rows, reliability = {}, {}
    for key in graph.keys(['J03WPY', 'J03WQQ']):
        y = graph.truth(key)
        name = 'WQQ' if key.startswith('J03WQQ') else 'WPY'
        with np.load(experiment.RUN / name / 'baseline' / (key + '.npz')) as a:
            baseline = score(pm, y, a['predicted'], True)
        candidates = {'boundary-only': control, 'anchor': experiment.RUN / name / 'anchor',
                      'anchor-boundary': experiment.RUN / name / ('anchor-boundary-' + str(weight))}
        rows[key] = {}
        for label, folder in candidates.items():
            with np.load(folder / (key + '.npz')) as a:
                scored = score(pm, y, a['predicted'], True)
            rows[key][label] = {'metrics': {m: scored['metrics'][m] for m in ['spell_f1_mean', 'macro_f1']},
                                'event_change': event_change(baseline, scored)}
        with np.load(experiment.RUN / 'cache/anchor' / (key + '.npz')) as a:
            anchors = a['anchors']
        purities, frames, correct, wrong, dead = [], 0, 0, 0, 0
        for s, e, t in anchors:
            values = y[s:e]; valid = values < 3
            frames += int(valid.sum()); correct += int((values == t).sum())
            wrong += int((values == 1 - t).sum()); dead += int((values == 2).sum())
            if valid.any():
                purities.append(float((values[valid] == t).mean()))
        reliability[key] = {'anchor_stretches': len(anchors), 'graded_anchor_frames': frames,
                            'correct_owner_frames': correct, 'wrong_owner_frames': wrong, 'true_dead_frames': dead,
                            'frame_precision': correct / max(1, frames),
                            'perfect_stretches': sum(p == 1 for p in purities),
                            'at_least95pct_correct_stretches': sum(p >= .95 for p in purities),
                            'scope': 'Reference-assisted diagnostic of local anchor reliability only; never inference features.'}
    aggregates = {}
    for match in ['J03WPY', 'J03WQQ']:
        aggregates[match] = {}
        for condition in ['clean', 'corrupted']:
            selected = [r for k, r in rows.items() if k.startswith(match) and k.endswith(condition)]
            aggregates[match][condition] = {}
            for candidate in selected[0]:
                aggregates[match][condition][candidate] = {'metrics': {m: float(np.mean([r[candidate]['metrics'][m] for r in selected])) for m in ['spell_f1_mean', 'macro_f1']},
                    'events': {k: sum(r[candidate]['event_change'][k] for r in selected) for k in ['recovered', 'lost', 'short_recovered', 'short_lost', 'before_predicted_spells', 'after_predicted_spells', 'before_short_matches', 'after_short_matches']},
                    'fragmentation': {side: float(np.mean([r[candidate]['event_change'][side + '_fragmentation'] for r in selected])) for side in ['before', 'after']}}
    write_new(out / 'results.json', {'aggregates': aggregates, 'anchor_reliability_diagnostic': reliability, 'views': rows,
              'reserved_final_accessed': False, 'new_model_fitting': False})
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
