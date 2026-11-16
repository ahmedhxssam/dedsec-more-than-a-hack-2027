"""Condition-correct, held-match-excluded parents for the WQQ campaign.

Inference only. This program cannot open any reference array. Historical
calibration may include in-cohort TRAIN labels; every scored held match is
excluded. Clean WPY parents use the original explicit complete pipeline.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'private/improvement/runtime'), str(ROOT)]
sys.path.append(str(ROOT / 'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from second_zero.common import sha256

TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY']
HALVES = ['firstHalf', 'secondHalf']
CONDITIONS = ['clean', 'corrupted']
OLD = ROOT / 'runs/gap-bridge-v20'
BASE = ROOT / 'runs/full-baseline-gap-v1'
CAMPAIGN = ROOT / 'runs/wqq-090-20261031'
_FULL_ALLOWED = False


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def read(path):
    return json.loads(Path(path).read_text())


def checked(item):
    path = ROOT / item['path']
    if sha256(path) != item['sha256']:
        raise ValueError('Changed parent dependency: ' + item['path'])
    return path


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    if any(s in path for s in ('.truth.npy', 'J03WR9', '/evaluation/gt/', '/runs/main/')):
        raise PermissionError('Parent construction forbids reference/reserved assets')
    if 'J03WQQ' in path and not _FULL_ALLOWED:
        raise PermissionError('WQQ parent requires a verified nomination')


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    os.replace(tmp, path)


def _validate_key(which, match, half, condition):
    if which not in TRAIN + ['full'] or half not in HALVES or condition not in CONDITIONS:
        raise ValueError('Invalid parent request')
    if match not in TRAIN + ['J03WPY', 'J03WQQ']:
        raise ValueError('Reserved or unknown match')
    if match in TRAIN and which != match:
        raise ValueError('OOF parent must use its corresponding held-match model')
    if match == 'J03WQQ' and which != 'full':
        raise ValueError('WQQ requires full-four model')


def _cohort(which):
    return [m for m in TRAIN if m != which]


@lru_cache(maxsize=1)
def validate_baseline_ancestry():
    from scripts import full_baseline_evaluate as evaluation
    path = BASE / 'baseline-ancestry.json'
    doc = read(path)
    result = evaluation.validate_ancestry(doc, sha256(BASE / 'producer-protocol.json'), [])
    return {'manifest': pin(path), 'protocol': pin(BASE / 'producer-protocol.json'), 'result': result}


def _spec(which):
    if which == 'full':
        from scripts.wqq_campaign_baseline_ancestry import validate_full_baseline
        proof = validate_full_baseline()
        path = BASE / 'full-reference-spec.json'
        # Exact original complete model, already replayed against its loader.
        replay = read(BASE / 'loader-parity.json')
        if not replay['all_outputs_bit_exact'] or replay['spec'] != pin(path):
            raise ValueError('Full clean-parent loader parity missing')
        return read(path), [pin(path), pin(BASE / 'loader-parity.json'), *proof['evidence']], TRAIN
    proof = validate_baseline_ancestry()
    fold = 0 if which in TRAIN[:2] else 1
    path = BASE / f'specs/fold{fold}.json'; spec = read(path)
    if which in spec['train_matches'] or which not in spec['held_matches']:
        raise ValueError('Clean calibration parent includes held match')
    return spec, [pin(path), proof['manifest'], proof['protocol']], spec['train_matches']


def _verify_nomination(path):
    if path is None:
        raise PermissionError('WQQ parent requires explicit frozen nomination')
    d = read(path)
    if d.get('kind') != 'wqq-campaign-nomination-v1' or d.get('eligible') is not True:
        raise PermissionError('Invalid full nomination')
    checked(d['protocol']); audit = read(checked(d['oof_audit']))
    recipe_hash = hashlib.sha256(json.dumps(d['selected_recipe'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if recipe_hash != d['recipe_sha256'] or audit.get('run_id') != d['candidate_id'] or audit.get('eligible') is not True or audit.get('stage') != 'oof' or audit.get('protocol') != d['protocol']:
        raise PermissionError('Nomination is not bound to eligible candidate evidence')
    return pin(path)


def get_parent(which, match, half, condition, nomination=None):
    """Return (int8 labels, hash-bound ancestry manifest), without reference IO."""
    global _FULL_ALLOWED
    _validate_key(which, match, half, condition)
    nomination_pin = _verify_nomination(nomination) if match == 'J03WQQ' else None
    previous = _FULL_ALLOWED
    _FULL_ALLOWED = nomination_pin is not None
    try:
        return _get_parent(which, match, half, condition, nomination_pin)
    finally:
        _FULL_ALLOWED = previous


def _get_parent(which, match, half, condition, nomination_pin):
    # v1 pilot remains preserved; revised source produces a separate cache.
    folder = CAMPAIGN / 'parents-v2' / which
    dest = folder / f'{match}-{half}-{condition}.npz'
    manifest_path = dest.with_suffix('.json')
    if manifest_path.exists():
        meta = read(manifest_path)
        if any(meta.get(k) != v for k, v in {'which': which, 'match': match, 'half': half, 'condition': condition, 'nomination': nomination_pin}.items()):
            raise ValueError('Cached parent request identity mismatch')
        if meta['source'] != pin(Path(__file__)):
            raise ValueError('Existing parent producer source changed')
        for item in meta['upstream_evidence']: checked(item)
        checked(meta['artifact'])
        with np.load(dest, allow_pickle=False) as z: out = z['parent'].copy()
        if which in meta['fit_matches'] or which in meta['calibration_matches']:
            raise ValueError('Held ancestor in cached parent')
        return out, {**meta, 'manifest': pin(manifest_path)}
    if dest.exists():
        raise ValueError('Unsealed parent output exists; inspect before reuse')
    evidence = []
    started = time.monotonic()
    if match == 'J03WPY' and condition == 'clean':
        from second_zero.full_baseline_pipeline import ExplicitRawPipeline, raw_half
        spec, evidence, fit_matches = _spec(which)
        raw = raw_half(f'{match}-{half}-clean')
        if not np.isfinite(raw['ball']).all():
            raise ValueError('Native clean-parent equivalence assumes fully observed ball')
        prediction = ExplicitRawPipeline(spec).predict(raw)
        for key, value in prediction.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in 'fciu' and not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite clean-parent intermediate: ' + key)
        out = prediction['predicted'].astype(np.int8)
        evidence.append(pin(ROOT / 'private/improvement/external/prepared' / f'{match}-{half}.npz'))
        calibration_matches = ['J03WPY']
        method = 'complete-explicit-pipeline-on-actual-clean-input'
    elif match == 'J03WPY':
        cp = OLD / f'models/{which}/calibration.json'
        calibration = read(cp); cfg = calibration['selected']
        if which in calibration['calibration_matches']:
            raise ValueError('Held match used in historical calibration')
        grid = read(OLD / 'protocol.json')['parameter_grid']
        i = next(i for i, c in enumerate(grid) if all(c[k] == cfg[k] for k in ('owner_alpha', 'beta', 'temperature')))
        body = 'full' if which == 'full' else 'inner' + str(0 if which in TRAIN[:2] else 1)
        source = OLD / f'calibration-cases/{body}/{match}-{half}.npz'
        input_pin = next(v for v in calibration['inputs'] if v['path'] == str(source.relative_to(ROOT)))
        checked(input_pin)
        with np.load(source, allow_pickle=False) as z: out = z[f'param_{i}'].copy()
        fp = OLD / f'models/{body}/freeze.json'
        frozen = read(fp); fit_matches = frozen['residual_fit_matches']
        if which in fit_matches: raise ValueError('Held match in nested parent')
        evidence = [pin(cp), pin(fp), frozen['model'], input_pin, pin(OLD / 'protocol.json')]
        if which == 'full':
            from scripts.wqq_campaign_baseline_ancestry import validate_full_baseline
            proof = validate_full_baseline()
            evidence += proof['evidence']
        else:
            proof = validate_baseline_ancestry()
            evidence += [proof['manifest'], proof['protocol']]
        for item in evidence: checked(item)
        calibration_matches = calibration['calibration_matches']
        method = 'exact-frozen-nested-corrupted-parent-policy'
    else:
        full = match == 'J03WQQ'
        seal_path = OLD / ('full-seal.json' if full else 'oof-seal.json')
        source = OLD / ('full-predictions' if full else 'oof') / f'{match}-{half}-{condition}.npz'
        seal = read(seal_path)
        expected = seal['outputs'][str(source.relative_to(ROOT))]
        checked({'path': str(source.relative_to(ROOT)), 'sha256': expected})
        with np.load(source, allow_pickle=False) as z: out = z['temporal_context'].copy()
        fp = OLD / f'models/{which}/freeze.json'; cp = OLD / f'models/{which}/calibration.json'
        frozen, calibration = read(fp), read(cp)
        fit_matches = frozen['residual_fit_matches']
        calibration_matches = calibration['calibration_matches']
        if match in set(fit_matches) | set(calibration_matches):
            raise ValueError('Prediction match in historical parent ancestry')
        evidence = [pin(source), pin(seal_path), pin(fp), frozen['model'], pin(cp)]
        if not full:
            proof = validate_baseline_ancestry()
            evidence += [proof['manifest'], proof['protocol']]
        else:
            from scripts.wqq_campaign_baseline_ancestry import validate_full_baseline
            proof = validate_full_baseline()
            evidence += proof['evidence']
        for item in evidence: checked(item)
        method = 'exact-sealed-v20-parent'
    if out.ndim != 1 or not len(out) or not np.isin(out, [0, 1, 2, 3]).all():
        raise ValueError('Invalid parent labels')
    if not set(fit_matches) <= set(_cohort(which)):
        raise ValueError('Parent cohort exceeds current fitting role')
    folder.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.npz.tmp')
    with tmp.open('xb') as f: np.savez_compressed(f, parent=out.astype(np.int8))
    os.replace(tmp, dest)
    meta = {'kind': 'wqq-campaign-parent-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'which': which, 'match': match, 'half': half, 'condition': condition,
            'method': method, 'fit_matches': fit_matches, 'calibration_matches': calibration_matches,
            'upstream_evidence': evidence, 'source': pin(Path(__file__)), 'artifact': pin(dest),
            'frames': len(out), 'seconds': time.monotonic()-started, 'reference_arrays_opened': [],
            'ancestry_verified': True, 'nomination': nomination_pin}
    _write_json(manifest_path, meta)
    return out, {**meta, 'manifest': pin(manifest_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', choices=TRAIN + ['full'], required=True)
    parser.add_argument('--match', choices=TRAIN + ['J03WPY', 'J03WQQ'], required=True)
    parser.add_argument('--half', choices=HALVES)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--nomination')
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); threadpool_limits(1)
    sys.addaudithook(guard)
    for half in ([args.half] if args.half else HALVES):
        for condition in ([args.condition] if args.condition else CONDITIONS):
            _, meta = get_parent(args.which, args.match, half, condition, args.nomination)
            print(json.dumps({k: meta[k] for k in ['which', 'match', 'half', 'condition', 'method', 'frames', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
