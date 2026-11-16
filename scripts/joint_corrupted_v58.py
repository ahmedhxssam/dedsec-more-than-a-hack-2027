"""Prospectively frozen all-corrupted ownership/status/boundary study."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'private/improvement/runtime')]
sys.path.append(str(ROOT / 'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from second_zero import joint_corrupted_recovery as api
from second_zero.raw_context_owner import inputs
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from scripts.train_contextual import corruption
from scripts.full_baseline_evaluate import join_halves, score_full

RUN = ROOT / 'runs/joint-corrupted-v58'
REPORT = ROOT / 'reports/joint-corrupted-v58'
OLD = ROOT / 'runs/gap-bridge-v20'
DATA = ROOT / 'private/improvement/external/prepared'
TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY']
HALVES = ['firstHalf', 'secondHalf']
FITS = {**{m: [v for v in TRAIN if v != m] for m in TRAIN}, 'full': TRAIN}
POLICIES = [{'alpha': 0., 'change_offset': 0.}] + [
    {'alpha': a, 'change_offset': 0.} for a in [.5, .75, 1.]]
ALLOWED = set()
READS = []
FULL = False
HASH_ONLY = False
STAGE = 'prepare'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def pin(p):
    return {'path': rel(p), 'sha256': sha256(p)}


def checked(meta):
    p = ROOT / meta['path']
    if sha256(p) != meta['sha256']:
        raise ValueError('Changed immutable artifact: ' + meta['path'])
    return p


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = os.fsdecode(args[0])
    if any(v in p for v in ['J03WR9', '/evaluation/gt/', '/runs/main/']):
        raise PermissionError('Reserved references remain closed')
    if 'J03WQQ' in p and not FULL and not HASH_ONLY:
        raise PermissionError('WQQ closed until successful OOF gate and frozen policy')
    if p.endswith('.truth.npy') and not HASH_ONLY:
        match = Path(p).name.split('-')[0]
        if match not in ALLOWED:
            raise PermissionError('Reference outside current declared role')
        READS.append({'stage': STAGE, 'match': match, 'allowed': sorted(ALLOWED), 'path': rel(p)})
    if not HASH_ONLY and p.endswith('.npz') and any(v in p for v in ['/raw-context-owner-v17/training/', '/gap-bridge-v20/training/', '/joint-corrupted-v58/failure/']):
        if Path(p).parent.name not in ALLOWED:
            raise PermissionError('Training cache outside current fitting cohort')


def verify():
    global HASH_ONLY
    p = read(RUN / 'protocol.json')
    HASH_ONLY = True
    try:
        for path, h in {**p['sources'], **p['inputs']}.items():
            assert sha256(ROOT / path) == h, path
    finally:
        HASH_ONLY = False
    return p


def raw(match, half, condition='clean'):
    with np.load(DATA / f'{match}-{half}.npz', allow_pickle=False) as z:
        v = {k: z[k] for k in ['ball', 'players', 'team', 'pitch']}
    return corruption(v, 100) if condition == 'corrupted' else v


def truth(match, half):
    return np.load(DATA / f'{match}-{half}.truth.npy', allow_pickle=False)


def incumbent(match, half, condition):
    folder = 'full-predictions' if match == 'J03WQQ' else 'oof'
    p = OLD / folder / f'{match}-{half}-{condition}.npz'
    with np.load(p, allow_pickle=False) as z:
        return z['temporal_context'].copy(), pin(p)


def calibration_parent(which, half):
    cfg = read(OLD / f'models/{which}/calibration.json')['selected']
    grid = read(OLD / 'protocol.json')['parameter_grid']
    i = next(i for i, c in enumerate(grid) if all(c[k] == cfg[k] for k in ['owner_alpha', 'beta', 'temperature']))
    body = 'full' if which == 'full' else 'inner' + str(0 if which in TRAIN[:2] else 1)
    p = OLD / f'calibration-cases/{body}/J03WPY-{half}.npz'
    with np.load(p, allow_pickle=False) as z:
        value = z[f'param_{i}'].copy()
    return value, {'cache': pin(p), 'historical_policy': pin(OLD / f'models/{which}/calibration.json'), 'source_body': body, 'parameter_index': i}


def prepare():
    native = read(OLD / 'training-seal.json')['rows']
    assert len(native) == 32 and all(not r['primary_matches'] for r in native.values())
    files = [OLD / 'training-seal.json', OLD / 'protocol.json', OLD / 'oof-seal.json', OLD / 'full-seal.json', OLD / 'full-audit.json', ROOT / 'reports/gap-bridge-v20/artifact-freeze.json', REPORT / 'preflight.json']
    files += [ROOT / r['cache']['path'] for r in native.values()]
    for m in TRAIN + ['J03WPY']:
        for h in HALVES:
            files.append(DATA / f'{m}-{h}.npz')
            if m in TRAIN:
                files.append(DATA / f'{m}-{h}.truth.npy')
    for m in TRAIN + ['full']:
        files.append(OLD / f'models/{m}/calibration.json')
    files += list((OLD / 'oof').glob('*.npz'))
    # Preservation hashes only; no WQQ prediction or reference arrays are parsed.
    files += list((OLD / 'full-predictions').glob('*.npz'))
    files += list((OLD / 'calibration-cases').glob('*/J03WPY-*.npz'))
    sources = [Path(__file__), ROOT / 'scripts/run_joint_corrupted_v58.py', ROOT / 'tests/test_joint_corrupted_recovery.py', ROOT / 'tests/test_joint_corrupted_v58.py']
    sources += [Path(m.__file__) for name, m in sorted(sys.modules.items()) if name.startswith(('scripts.', 'second_zero.')) and getattr(m, '__file__', None)]
    write(RUN / 'protocol.json', {
        'kind': 'all-corrupted-joint-ownership-status-boundary-v58',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'target': {'overall_WQQ': .9, 'clean_predictions_identical': 142345, 'preserved_best': .778977925988195},
        'fits': FITS, 'policies': POLICIES,
        'architecture': 'Raw414, LN/linear48/GELU, one bidirectional GRU48, owner/dead/change heads. Fresh random weights; no source/public pretrained checkpoint or learned feature ancestors.',
        'features': 'Reuse all32 original v20 raw-only feature halves (100/271/phase0/phase275), plus8 new training-match-only failure views. No parent predictions or contact labels in training inputs. Same float16 quantization at fitting and inference. Input trajectory features already clip/interpolate missing ball, preserving explicit missingness.',
        'augmentation': 'Additional fixed seed20261118+10*matchindex+halfindex. Ball holes50..400/500frames or stale ball, .75m smoothed noise; player holes or stale coordinates25..200/2000; same-team slot switches6000frame spacing. Label-independent. Simulated plausible failures, not empirically validated realistic failure law; no target-match failure templates.',
        'supervision': 'Native controlled-possession H/A plus D. X ungraded, not a supervised fourth physical state. Include all known H/A/D, including observed-ball and dead periods. Equal mixture unweighted3stateNLL and sqrt125/spelllength clipped.25..4 durationweightedNLL; .1 exact changeBCE positiveweight5; .1 exact3state conditionalpathNLL resettingatX/unscoredcrop; .0001 logitL2. Statuschange and ownershipchange share learned hazard; subtractlog5 positiveweight correction for path/decoder.',
        'training': {'seed': 20261118, 'steps_per_physical_match': 500, 'batch': 6, 'length': 256, 'central': 128, 'stride': 5, 'optimizer': 'AdamW', 'lr': .0006, 'wd': .01, 'schedule': 'cosine1to.1', 'clip': 5, 'cpu_threads': 1, 'checkpoint': 'final-only', 'sampling': 'half all-known frames, half anyH/A/D change; random team swap'},
        'decode': 'Antisymmetric owner and symmetric dead/change dual perspectives; linearly interpolate to native25Hz. 3stateViterbi logemission/5; staylogsigmoid(-hazard), alternate logSigmoid(hazard)-log2. No minimumduration or gapgraph. All corruptedframes editable, formerD/Xincluded. Xparent prioruniform; knownparentprior .98/.01/.01.',
        'calibration': 'OnlyWPY pooled official full-matchSpellF1 chooses alpha0/.5/.75/1; hazardoffset0fixed, no parameter search onWQQ. Eachoutermodelother3 excludesheld; WPYparent exactold20 policy-specific complementaryinner2cachedarray. FullWPYparent exactold20fullcachedarray. Parent2->4 shift explicitly remains. All4calibrationarrays sealed before WPY truth read; alpha0 ties. No weight gradients on WPY.',
        'gate': 'All16 clean/corrupted physical OOF outputs sealed before one audit. Actualv20 overallmean strictly increases and bothproducerpairmeans nonnegative; then onefull4fit/WPYpolicy/WQQcandidate. No fallback or WQQsearch. Clean bypass exact savedv20array in every stage; no CSV/export or promotion.',
        'limits': 'Only4 independent native trainingmatches; no additional dense controlled-possession supervision verified. New model is a small feasibility study, not a demonstrated .9recipe. 5frame inputstride cannot represent sub-stride evidence. Adaptive reused development, not private/untouched challenge accuracy.',
        'sources': {rel(p): sha256(p) for p in sources},
        'inputs': {rel(p): sha256(p) for p in files},
    })
    print('V58 PROTOCOL', sha256(RUN / 'protocol.json'), flush=True)


def cache_failure():
    verify()
    rows = {}
    for mi, m in enumerate(TRAIN):
        ALLOWED.clear()
        ALLOWED.add(m)
        for hi, h in enumerate(HALVES):
            v = api.failure_view(raw(m, h), 20261118 + 10 * mi + hi)
            x, _, _ = inputs(v)
            dest = RUN / f'failure/{m}/{h}.npz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                np.savez_compressed(f, x=x.astype(np.float16))
            rows[f'{m}-{h}'] = {'cache': pin(dest), 'match': m, 'half': h, 'view': 'failure', 'primary_matches': [], 'shape': list(x.shape)}
            print('V58 FAILURE', m, h, x.shape, flush=True)
    assert not READS
    ALLOWED.clear()
    write(RUN / 'failure-seal.json', {'rows': rows, 'reference_reads': [], 'protocol': pin(RUN / 'protocol.json')})


def fit(which):
    verify()
    if which == 'full':
        assert read(RUN / 'oof-audit.json')['passed']
    cohort = FITS[which]
    ALLOWED.update(cohort)
    rows = {**read(OLD / 'training-seal.json')['rows'], **read(RUN / 'failure-seal.json')['rows']}
    dataset, meta, targets = [], [], {}
    for row in rows.values():
        if row['match'] not in cohort:
            continue
        assert not row['primary_matches']
        key = (row['match'], row['half'])
        if key not in targets:
            targets[key] = api.targets(truth(*key))
        with np.load(checked(row['cache']), allow_pickle=False) as z:
            # Old caches contain ownership targets; ONLY the raw x field is read.
            x = z['x'].astype(np.float16)
        assert x.shape == (2, 414, len(targets[key]['y']))
        dataset.append({'x': x, **targets[key]})
        meta.append(row)
    torch.manual_seed(20261118)
    rng = np.random.default_rng(20261118)
    model = api.JointRecovery().train()
    opt = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.01)
    count = 500 * len(cohort)
    history, started = [], time.monotonic()
    for step in range(count):
        for group in opt.param_groups:
            group['lr'] = .0006 * (.1 + .9 * .5 * (1 + np.cos(np.pi * step / count)))
        v = api.batch(dataset, rng)
        opt.zero_grad(set_to_none=True)
        error = api.loss(model, v)
        assert torch.isfinite(error)
        error.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if step % 300 == 0 or step == count - 1:
            row = {'step': step + 1, 'loss': float(error.detach()), 'seconds': time.monotonic() - started}
            history.append(row)
            print('V58 FIT', which, row, flush=True)
    dest = RUN / f'models/{which}/model.pt'
    dest.parent.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), dest)
    write(dest.parent / 'freeze.json', {'fit_matches': cohort, 'raw_feature_learned_ancestors': [], 'model': pin(dest), 'rows': meta, 'steps': count, 'history': history, 'reference_reads': READS.copy(), 'protocol': pin(RUN / 'protocol.json')})
    ALLOWED.clear()


def predict_logits(which, match, half):
    f = read(RUN / f'models/{which}/freeze.json')
    assert match not in f['fit_matches'] and f['fit_matches'] == FITS[which]
    model = api.JointRecovery()
    model.load_state_dict(torch.load(checked(f['model']), weights_only=True, map_location='cpu'))
    v = raw(match, half, 'corrupted')
    x, _, _ = inputs(v)
    return api.predict(model, x.astype(np.float16), len(v['ball'])), f


def calibrate(which):
    verify()
    ALLOWED.clear()
    outputs, parents = [], []
    for h in HALVES:
        logits, f = predict_logits(which, 'J03WPY', h)
        parent, pm = calibration_parent(which, h)
        dest = RUN / f'calibration-cases/{which}/{h}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f'param_{i}': api.compose(parent, logits, **cfg) for i, cfg in enumerate(POLICIES)}
        np.testing.assert_array_equal(arrays['param_0'], parent)
        with dest.open('xb') as out:
            np.savez_compressed(out, baseline=parent, logits=logits, **arrays)
        outputs.append(pin(dest))
        parents.append(pm)
    assert not READS
    write(RUN / f'calibration-cases/{which}/seal.json', {'outputs': outputs, 'parents': parents, 'model': f['model'], 'reference_reads': []})
    ALLOWED.add('J03WPY')
    y = join_halves([truth('J03WPY', h) for h in HALVES])
    candidates = []
    loaded = []
    for p in outputs:
        with np.load(checked(p), allow_pickle=False) as z:
            loaded.append({k: z[k].copy() for k in z.files if k != 'logits'})
    for i, cfg in enumerate(POLICIES):
        pred = join_halves([v[f'param_{i}'] for v in loaded])
        score = score_full(official_metrics(), y, pred)[0]
        candidates.append({**cfg, 'spell_f1_mean': score['spell_f1_mean'], 'changed': sum(int((v[f'param_{i}'] != v['baseline']).sum()) for v in loaded)})
    selected = max(enumerate(candidates), key=lambda v: (v[1]['spell_f1_mean'], -v[1]['changed'], -v[0]))[1]
    write(RUN / f'models/{which}/calibration.json', {'selected': selected, 'rows': candidates, 'calibration_matches': ['J03WPY'], 'reference_reads': READS.copy(), 'seal': pin(RUN / f'calibration-cases/{which}/seal.json')})
    ALLOWED.clear()
    print('V58 CALIBRATION', which, selected, flush=True)


def predict_all(full=False):
    global FULL
    verify()
    FULL = full
    ALLOWED.clear()
    if full:
        assert read(RUN / 'oof-audit.json')['passed']
    matches = ['J03WQQ'] if full else TRAIN
    outputs, views = {}, {}
    for m in matches:
        which = 'full' if full else m
        cfg = read(RUN / f'models/{which}/calibration.json')['selected']
        for h in HALVES:
            logits, f = predict_logits(which, m, h)
            for c in ['clean', 'corrupted']:
                parent, pm = incumbent(m, h, c)
                candidate = parent.copy() if c == 'clean' else api.compose(parent, logits, cfg['alpha'], cfg['change_offset'])
                key = f'{m}-{h}-{c}'
                dest = RUN / ('full-predictions' if full else 'oof') / f'{key}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as out:
                    np.savez_compressed(out, baseline=parent, joint=candidate)
                outputs[rel(dest)] = sha256(dest)
                views[key] = {'parent': pm, 'model': f['model'], 'fit_matches': f['fit_matches'], 'policy': pin(RUN / f'models/{which}/calibration.json'), 'changed': int((candidate != parent).sum()), 'changed_former_D_X': int(((candidate != parent) & (parent >= 2)).sum())}
                print('V58 PREDICT', key, views[key]['changed'], flush=True)
    assert not READS
    write(RUN / ('full-seal.json' if full else 'oof-seal.json'), {'outputs': outputs, 'views': views, 'reference_reads': [], 'protocol': pin(RUN / 'protocol.json')})


def audit(full=False):
    global FULL
    FULL = full
    verify()
    name = 'full' if full else 'oof'
    seal = read(RUN / f'{name}-seal.json')
    assert not seal['reference_reads']
    checked(seal['protocol'])
    loaded, clean_frames = {}, 0
    for p, digest in seal['outputs'].items():
        checked({'path': p, 'sha256': digest})
        key = Path(p).stem
        m, h, c = key.split('-')
        with np.load(ROOT / p, allow_pickle=False) as z:
            v = {k: z[k].copy() for k in z.files}
        parent, _ = incumbent(m, h, c)
        np.testing.assert_array_equal(v['baseline'], parent)
        view = seal['views'][key]
        assert m not in view['fit_matches']
        checked(view['model'])
        checked(view['policy'])
        assert np.isin(v['joint'], [0, 1, 2, 3]).all()
        if c == 'clean':
            np.testing.assert_array_equal(v['joint'], parent)
            clean_frames += len(parent)
        loaded[key] = v
    matches = ['J03WQQ'] if full else TRAIN
    assert len(loaded) == 4 * len(matches)
    if full:
        assert clean_frames == 142345
    ALLOWED.update(matches)
    details = {}
    for m in matches:
        y = join_halves([truth(m, h) for h in HALVES])
        details[m] = {c: {a: score_full(official_metrics(), y, join_halves([loaded[f'{m}-{h}-{c}'][a] for h in HALVES]))[0] for a in ['baseline', 'joint']} for c in ['clean', 'corrupted']}
    def mean(ms, cs):
        return {a: float(np.mean([details[m][c][a]['spell_f1_mean'] for m in ms for c in cs])) for a in ['baseline', 'joint']}
    overall = mean(matches, ['clean', 'corrupted'])
    if full:
        assert abs(overall['baseline'] - .778977925988195) < 1e-12
        assert abs(mean(matches, ['clean'])['baseline'] - .8055625790139065) < 1e-12
        assert abs(mean(matches, ['corrupted'])['baseline'] - .7523932729624837) < 1e-12
    pairs = {} if full else {str(i): mean(TRAIN[i*2:i*2+2], ['clean', 'corrupted']) for i in [0, 1]}
    passed = overall['joint'] > overall['baseline'] and all(v['joint'] >= v['baseline'] for v in pairs.values())
    result = {'overall': overall, 'conditions': {c: mean(matches, [c]) for c in ['clean', 'corrupted']}, 'per_match': {m: mean([m], ['clean', 'corrupted']) for m in matches}, 'per_producer_pair': pairs, 'passed': passed, 'clean_exact': True, 'clean_frames_verified': clean_frames, 'D_X_locked': False, 'details': details, 'prediction_seal': pin(RUN / f'{name}-seal.json'), 'reference_reads': READS.copy(), 'interpretation': 'Official full-match5IoUSpellF1, both halves pooled withXseparator; equalclean/corruptedmean. Adaptively reused development.'}
    write(RUN / f'{name}-audit.json', result)
    ALLOWED.clear()
    print('V58 AUDIT', name, {k: result[k] for k in ['overall', 'conditions', 'per_producer_pair', 'passed']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'cache_failure', 'fit', 'calibrate', 'predict_oof', 'audit_oof', 'predict_full', 'audit_full'])
    parser.add_argument('--held', choices=list(FITS))
    args = parser.parse_args()
    STAGE = args.stage
    threadpool_limits(1)
    torch.set_num_threads(1)
    if STAGE == 'prepare':
        prepare()
    else:
        sys.addaudithook(guard)
        if STAGE == 'cache_failure':
            cache_failure()
        elif STAGE in ['fit', 'calibrate']:
            assert args.held
            globals()[STAGE](args.held)
        elif STAGE.startswith('audit_'):
            audit(STAGE == 'audit_full')
        else:
            predict_all(STAGE == 'predict_full')
