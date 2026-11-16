"""Matched current-observation versus explicit spatial-memory ownership test.

Four external matches fit both heads with identical samples and spell/frame
weights. WPY calibrates temperature and blend; frozen WQQ is a separate stage.
No macro guard, minimum possession duration, or reference-based inference.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from second_zero.common import sha256, json_write
from second_zero.evaluate import official_metrics
from second_zero import vision_context_memory as memory
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts import sprint_decoder_lattice as lattice
from scripts.anchor_matched_quality import array, evidence, pool, HASHES
from scripts.train_contextual import corruption
from scripts.evaluate_temporal_ownership import fixed_state_owner

TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY']
READS = set()
PARAMETERS = dict(max_iter=180, max_leaf_nodes=15, max_depth=5,
                  min_samples_leaf=80, l2_regularization=10., learning_rate=.05,
                  random_state=20261028, early_stopping=False)
WEIGHTS = [0., .1, .25, .5, 1.]


def guard(stage):
    def check(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = os.fsdecode(args[0])
        if 'J03WR9' in p or '/evaluation/gt/' in p:
            raise PermissionError('Public and reserved references excluded')
        if stage in ['fit', 'calibrate'] and 'J03WQQ' in p:
            raise PermissionError('WQQ inputs are not needed before frozen selection')
        if p.endswith('.truth.npy'):
            match = Path(p).name.split('-')[0]
            allowed = TRAIN if stage == 'fit' else ['J03WPY'] if stage == 'calibrate' else ['J03WQQ'] if stage == 'audit' else []
            if match not in allowed:
                raise PermissionError('Reference forbidden in this stage: ' + p)
            READS.add(p)
    return check


def aggregate(views):
    return {condition: {m: float(np.mean([r[m] for k, r in views.items()
             if condition == 'overall' or k.endswith(condition)])) for m in ['spell', 'macro']}
            for condition in ['overall', 'clean', 'corrupted']}


def sample_spells(y, stride=8, spell_mass=None):
    """Every eighth frame plus one per spell, exact sampled spell/frame mass."""
    if not len(y):
        return np.empty(0, np.int64), np.empty(0, np.float64)
    bounds = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    spells = [(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if y[lo] < 2]
    live = sum(hi - lo for lo, hi in spells)
    if spell_mass is None:
        spell_mass = live / max(len(spells), 1)
    indices, weights = [], []
    for lo, hi in spells:
        take = np.unique(np.r_[np.arange(((lo + stride - 1) // stride) * stride, hi, stride), (lo + hi - 1) // 2])
        mass = .5 * (hi - lo) + .5 * spell_mass
        indices.extend(take.tolist()); weights.extend([mass / len(take)] * len(take))
    indices = np.asarray(indices, np.int64); weights = np.asarray(weights, np.float64)
    # Preserve relative masses across halves; normalize only after concatenation.
    return indices, weights


def inputs(key):
    name, condition = key.rsplit('-', 1)
    path = ROOT / 'private/improvement/external/prepared' / (name + '.npz')
    HASHES[str(path.relative_to(ROOT))] = sha256(path)
    with np.load(path) as a:
        raw = {k: a[k] for k in ['ball', 'players', 'team', 'pitch']}
    if condition == 'corrupted':
        raw = corruption(raw, 100)
    match = name.split('-')[0]
    if match in TRAIN:
        fold = 0 if match in TRAIN[:2] else 1
        for family in ['interp_learned_swap', 'long']:
            protocol_path = ROOT / f'runs/two-hour-sprint/model/{family}-{fold}/protocol.json'
            protocol = json.loads(protocol_path.read_text())
            assert match not in protocol['train_matches']
            assert match in protocol['eval_matches']
            assert match not in protocol.get('imputer_train_matches', [])
            HASHES[str(protocol_path.relative_to(ROOT))] = sha256(protocol_path)
            for filename in ['model.pt', 'freeze.json', 'calibration.json']:
                asset = protocol_path.parent / filename
                if asset.exists():
                    HASHES[str(asset.relative_to(ROOT))] = sha256(asset)
        status_protocol = ROOT / 'runs/two-hour-sprint/status-crossfit-v2/protocol.json'
        folds = json.loads(status_protocol.read_text())['folds']
        assert match not in folds[fold][0] and match in folds[fold][1]
        HASHES[str(status_protocol.relative_to(ROOT))] = sha256(status_protocol)
        learned = array(f'runs/two-hour-sprint/model/interp_learned_swap-{fold}', key)
        longer = array(f'runs/two-hour-sprint/model/long-{fold}', key)
        z = logit(np.clip(.5 * expit(learned) + .5 * expit(longer), 1e-8, 1 - 1e-8))
        status = array('runs/two-hour-sprint/status-crossfit-v2/cost50', key, 'predicted')
    else:
        z = array('runs/two-hour-sprint/carrier-mixture-v1', key)
        status = array('runs/two-hour-sprint/status-mixture-v1/reliability', key, 'predicted')
    return raw, z, status


def predict(model, features):
    scores = []
    for t in (0, 1):
        p = model.predict_proba(features[:, t])[:, 1]
        scores.append(logit(np.clip(p, 1e-6, 1 - 1e-6)))
    return ((scores[1] - scores[0]) * .5).astype(np.float32)


def source_hashes():
    paths = [Path(__file__), Path(memory.__file__), Path(graph.__file__),
             Path(lattice.__file__), ROOT / 'scripts/train_contextual.py',
             ROOT / 'scripts/anchor_matched_quality.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def verify(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    assert source_hashes() == protocol['source_sha256'], 'Sources changed after fitting'
    return protocol


def fit(args):
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes(); (out / 'sources').mkdir()
    for p in hashes:
        shutil.copyfile(ROOT / p, out / 'sources' / Path(p).name)
    json_write(out / 'protocol.json', {
        'training_matches': TRAIN, 'calibration': 'WPY', 'development': 'WQQ reused',
        'models': PARAMETERS, 'sample_stride': 8,
        'objective': 'Equal mixture of frame and full-native-spell mass; identical samples for current-only and memory arms',
        'training_evidence': 'Opposite-two-match learned-fill/long probability mean and status-crossfit-v2/cost50. No four-match held-out claim for fitted heads.',
        'inference_evidence': 'Unchanged sprint winner ownership and reliability status',
        'selection': 'Fixed180iterations. WPY temperature per arm; blends0,.1,.25,.5,1. Rank overall SpellF1, macroF1 only as exact tie breaker. No macro or condition guard.',
        'memory_retention': 'Memory adds value only if it strictly improves current-only and unchanged winner SpellF1. The best positive arm, including current-only if it wins, can advance to frozen WQQ.',
        'additional_comparison': 'Pure expert ownership-only with shared status and fixed transitioncost12, not used to select decoder settings',
        'minimum_duration': None, 'source_sha256': hashes,
        'target_references_used': False, 'reserved_match_used': False})
    start = time.monotonic(); xx, yy, ww, counts = [], [], [], {}
    truths = {key: graph.truth(key) for key in graph.keys(TRAIN)}
    live_count = sum(int((y < 2).sum()) for y in truths.values())
    spell_count = sum(int(((y < 2) & np.r_[True, y[1:] != y[:-1]]).sum()) for y in truths.values())
    spell_mass = live_count / spell_count
    c0 = None
    for key in graph.keys(TRAIN):
        raw, z, base = inputs(key)
        current, past, meta = memory.features(raw, z, base)
        y = truths[key]; ix, w = sample_spells(y, spell_mass=spell_mass)
        if c0 is None:
            c0 = current.shape[-1]; json_write(out / 'feature_schema.json', meta)
        assert c0 == current.shape[-1]
        x = np.concatenate([current[ix], past[ix]], axis=-1)
        assert np.isfinite(x).all()
        for t in (0, 1):
            xx.append(x[:, t]); yy.append((y[ix] == t).astype(np.int8)); ww.append(w)
        counts[key] = {'sampled_frames': len(ix), 'features': list(x.shape)}
        print('prepared', key, counts[key], 'seconds', round(time.monotonic() - start, 1), flush=True)
        del raw, current, past, x
    x, y, w = np.concatenate(xx), np.concatenate(yy), np.concatenate(ww)
    del xx, yy, ww
    w /= w.mean()
    for arm, count in [('current', c0), ('memory', x.shape[1])]:
        model = HistGradientBoostingClassifier(**PARAMETERS).fit(x[:, :count], y, sample_weight=w)
        folder = out / arm; folder.mkdir()
        with (folder / 'model.pkl').open('xb') as f:
            pickle.dump(model, f)
        print('fit', arm, 'seconds', round(time.monotonic() - start, 1), flush=True)
    json_write(out / 'training.json', {'shape': list(x.shape), 'static_channels': c0,
        'counts': counts, 'seconds': time.monotonic() - start, 'input_sha256': HASHES,
        'global_training_live_frames': live_count, 'global_training_spells': spell_count,
        'equal_spell_component_mass': spell_mass,
        'reference_reads': sorted(READS), 'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'models': {a: sha256(out / a / 'model.pkl') for a in ['current', 'memory']}})


def infer_pair(out, match):
    training = json.loads((out / 'training.json').read_text()); models = {}
    for arm in ['current', 'memory']:
        path = out / arm / 'model.pkl'; assert sha256(path) == training['models'][arm]
        with path.open('rb') as f:
            models[arm] = pickle.load(f)
    for key in graph.keys([match]):
        raw, z, base = inputs(key); current, past, _ = memory.features(raw, z, base)
        for arm in ['current', 'memory']:
            x = current if arm == 'current' else np.concatenate([current, past], axis=-1)
            dest = out / arm / 'raw'; dest.mkdir(exist_ok=True)
            np.savez_compressed(dest / (key + '.npz'), raw_logit=predict(models[arm], x))
        print('inferred pair', key, flush=True)


def decode(key, raw, temperature, weight, quality):
    x, base = evidence(key)
    if weight:
        p = (1 - weight) * expit(x[:, 0]) + weight * expit(raw / temperature)
        x[:, 0] = logit(np.clip(p, 1e-8, 1 - 1e-8))
    arcs = pool(x, base)
    q = np.clip(quality.predict(graph.segment_features(x, arcs)), 0, 1)
    return graph.graph_decode(x[:, 0], base, arcs, q, .55, .01), x[:, 0], base


def calibrate(args):
    out = args.out; verify(out); infer_pair(out, 'J03WPY')
    pm, quality = official_metrics(), graph.get_model(); results = {}
    truth = {k: graph.truth(k) for k in graph.keys(['J03WPY'])}
    for arm in ['current', 'memory']:
        folder = out / arm
        raw = {k: np.load(folder / 'raw' / (k + '.npz'))['raw_logit'] for k in truth}
        z = np.concatenate([raw[k][y < 2] for k, y in truth.items()])
        y = np.concatenate([y[y < 2] for y in truth.values()])
        fitted = minimize_scalar(lambda lt: np.mean(np.logaddexp(0, z / np.exp(lt)) - y * z / np.exp(lt)), bounds=(-2.3, 2.3), method='bounded')
        temperature = float(np.exp(fitted.x)); rows = []
        for weight in WEIGHTS:
            dest = folder / 'WPY' / str(weight); dest.mkdir(parents=True, exist_ok=False); views = {}
            for key in truth:
                pred, combined, base = decode(key, raw[key], temperature, weight, quality)
                if weight == 0:
                    path = lattice.RUN / 'cache' / 'WPY' / '16' / (key + '.npz')
                    with np.load(path) as a:
                        before = graph.graph_decode(a['logit'], a['base'], a['segments'], a['quality'], .55, .01)
                    assert np.array_equal(pred, before), 'Winner replay must be exact'
                np.savez_compressed(dest / (key + '.npz'), predicted=pred, logit=combined)
                views[key] = graph.metric(pm, truth[key], pred)
            row = {'weight': weight, **aggregate(views), 'views': views}; rows.append(row)
            print(arm, 'WPY', weight, row['overall'], flush=True)
        own, own_baseline = {}, {}
        for key in truth:
            _, old, base = inputs(key)
            own[key] = graph.metric(pm, truth[key], fixed_state_owner(base, raw[key] / temperature))
            own_baseline[key] = graph.metric(pm, truth[key], fixed_state_owner(base, old))
        best = max(rows, key=lambda r: (r['overall']['spell'], r['overall']['macro']))
        result = {'temperature': temperature, 'rows': rows, 'selected': best,
                  'ownership_only_cost12': {**aggregate(own), 'views': own},
                  'ownership_only_baseline_cost12': {**aggregate(own_baseline), 'views': own_baseline}}
        results[arm] = result; json_write(folder / 'calibration.json', result)
    chosen = max(results, key=lambda a: (results[a]['selected']['overall']['spell'], results[a]['selected']['overall']['macro']))
    selected = results[chosen]['selected']
    freeze = {'arm': chosen, 'weight': selected['weight'], 'temperature': results[chosen]['temperature'],
        'WPY_selected': selected, 'both_arms': results,
        'memory_adds_value': results['memory']['selected']['overall']['spell'] > results['current']['selected']['overall']['spell'],
        'model_sha256': sha256(out / chosen / 'model.pkl'), 'source_sha256': source_hashes(),
        'protocol_sha256': sha256(out / 'protocol.json'),
        'training_sha256': sha256(out / 'training.json'), 'input_sha256': HASHES,
        'reference_reads': sorted(READS), 'WQQ_truth_read': False}
    json_write(out / 'freeze.json', freeze)
    print('selected', chosen, selected['weight'], selected['overall'], flush=True)


def predict_frozen(args):
    out = args.out; verify(out); frozen = json.loads((out / 'freeze.json').read_text())
    assert sha256(out / 'training.json') == frozen['training_sha256']
    assert sha256(out / 'protocol.json') == frozen['protocol_sha256']
    assert source_hashes() == frozen['source_sha256']
    if frozen['weight'] == 0:
        print('WPY selects unchanged winner; no WQQ inference.', flush=True); return
    assert sha256(out / frozen['arm'] / 'model.pkl') == frozen['model_sha256']
    infer_pair(out, 'J03WQQ'); quality = graph.get_model(); dest = out / 'WQQ'; dest.mkdir(exist_ok=False)
    for key in graph.keys(['J03WQQ']):
        with np.load(out / frozen['arm'] / 'raw' / (key + '.npz')) as a:
            pred, z, base = decode(key, a['raw_logit'], frozen['temperature'], frozen['weight'], quality)
        np.savez_compressed(dest / (key + '.npz'), predicted=pred, logit=z, status_mask=base)
    json_write(dest / 'prediction_freeze.json', {p.name: sha256(p) for p in dest.glob('*.npz')})
    json_write(out / 'inference.json', {'input_sha256': HASHES, 'reference_reads': sorted(READS),
                                      'WQQ_truth_read': False, 'freeze_sha256': sha256(out / 'freeze.json')})


def audit(args):
    out = args.out; frozen = json.loads((out / 'freeze.json').read_text()); pm = official_metrics()
    inference = json.loads((out / 'inference.json').read_text())
    assert sha256(out / 'freeze.json') == inference['freeze_sha256']
    pinned = json.loads((out / 'WQQ/prediction_freeze.json').read_text()); views = {}
    for key in graph.keys(['J03WQQ']):
        path = out / 'WQQ' / (key + '.npz'); assert sha256(path) == pinned[path.name]
        with np.load(path) as a:
            views[key] = graph.metric(pm, graph.truth(key), a['predicted'])
    result = {**aggregate(views), 'views': views, 'freeze': frozen,
              'scope': 'Complete pipeline on reused WQQ; clean/corrupted paired, not independent matches.'}
    json_write(out / 'evaluation.json', result); print(json.dumps(aggregate(views), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['fit', 'calibrate', 'predict', 'audit'])
    p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    sys.addaudithook(guard(args.stage)); threadpool_limits(2)
    {'fit': fit, 'calibrate': calibrate, 'predict': predict_frozen, 'audit': audit}[args.stage](args)
