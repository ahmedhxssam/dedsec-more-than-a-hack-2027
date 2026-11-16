"""Native-rate joint recovery, with frozen cohort-matched v20 ownership context."""
from __future__ import annotations
import argparse
import copy
import datetime
import importlib.util
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scripts._native_joint_v65_helpers', ROOT / 'scripts/joint_corrupted_v58.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
from second_zero import fast_native_joint_recovery as api
from second_zero import gap_bridge_training as owner_api
from second_zero.raw_context_owner import inputs as coarse_inputs
RUN = ROOT / 'runs/native-joint-v65'
REPORT = ROOT / 'reports/native-joint-v65'
OLD = base.OLD
TRAIN, HALVES, FITS = base.TRAIN, base.HALVES, base.FITS
POLICIES = [{'alpha': a, 'change_offset': 0.} for a in (0., .25, .5, .75, 1.)]
base.RUN, base.REPORT, base.api, base.POLICIES = RUN, REPORT, api, POLICIES
np, torch = base.np, base.torch
read, write, pin, checked, rel = base.read, base.write, base.pin, base.checked, base.rel
SEED = 20261121
RAW_NATIVE_SOURCE = ROOT / 'runs/native-joint-v62'


def guard(event, args):
    base.guard(event, args)
    if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
        return
    p = Path(base.os.fsdecode(args[0]))
    if base.STAGE in ('cache_native', 'bind_cache') and not base.HASH_ONLY and str(p).endswith('.truth.npy'):
        raise PermissionError('Native raw-only feature preparation forbids every reference')
    if any(str(folder) in str(p) for folder in [RUN / 'native', RAW_NATIVE_SOURCE / 'native']) and p.suffix == '.npz' and not base.HASH_ONLY:
        if p.parent.name not in base.ALLOWED:
            raise PermissionError('Native feature cache outside current fitting cohort')


def training_rows():
    return {**read(OLD / 'training-seal.json')['rows'], **read(ROOT / 'runs/joint-corrupted-v58/failure-seal.json')['rows']}


def prepare():
    rows = training_rows()
    assert len(rows) == 40 and all(not v['primary_matches'] for v in rows.values())
    files = [OLD / 'training-seal.json', OLD / 'protocol.json', OLD / 'oof-seal.json', OLD / 'full-seal.json', OLD / 'full-audit.json',
             ROOT / 'runs/joint-corrupted-v58/failure-seal.json', ROOT / 'reports/strong-joint-push-20261030/preservation-start.json', REPORT / 'preflight.json']
    files += [checked(v['cache']) for v in rows.values()]
    for m in TRAIN + ['J03WPY']:
        for h in HALVES:
            files.append(base.DATA / f'{m}-{h}.npz')
            if m in TRAIN:
                files.append(base.DATA / f'{m}-{h}.truth.npy')
    for which in TRAIN + ['full']:
        f = read(OLD / f'models/{which}/freeze.json')
        assert f['residual_fit_matches'] == FITS[which]
        files.extend([OLD / f'models/{which}/freeze.json', checked(f['model']), OLD / f'models/{which}/calibration.json'])
    files += list((OLD / 'oof').glob('*.npz')) + list((OLD / 'full-predictions').glob('*.npz'))
    files += list((OLD / 'calibration-cases').glob('*/J03WPY-*.npz'))
    assert not (RAW_NATIVE_SOURCE / 'models').exists() and not (RAW_NATIVE_SOURCE / 'oof-audit.json').exists()
    files += [RAW_NATIVE_SOURCE / 'protocol.json', RAW_NATIVE_SOURCE / 'native-seal.json', ROOT / 'reports/strong-joint-push-20261030/NATIVE_KERNEL_BENCHMARK.json', ROOT / 'reports/strong-joint-push-20261030/STRUCTURED_LOSS_BENCHMARK.json']
    files += [checked(v['cache']) for v in read(RAW_NATIVE_SOURCE / 'native-seal.json')['rows'].values()]
    sources = [Path(__file__), ROOT / 'scripts/run_native_joint_v65.py', ROOT / 'tests/test_native_joint_recovery.py', ROOT / 'tests/test_native_joint_v65.py', ROOT / 'tests/test_fast_native_joint_recovery.py', ROOT / 'tests/test_parallel_path_likelihood.py']
    sources += [Path(m.__file__) for name, m in sorted(sys.modules.items()) if name.startswith(('scripts.', 'second_zero.')) and getattr(m, '__file__', None)]
    write(RUN / 'protocol.json', {
        'kind': 'native25Hz-joint-equivalent-fast-kernels-v65', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'target': {'overall_WQQ': .9, 'clean_predictions_identical': 142345, 'preserved_best': .778977925988195},
        'fits': FITS, 'policies': POLICIES,
        'architecture': 'Native25Hz raw98 team-perspective nearest-neighbor kinematics plus raw40 team-neutral live/dead features, plus two frozen v20 ownership/boundary context logits. LN140/linear64/GELU, nine residual depthwise5/pointwise64 temporal blocks with dilations1..256, dropout.1, zero-initialized three-output residual head. 2045native-frame receptive field (81.8s) plus v20 ownership context. Owner starts exactly at frozen v20; dead starts0; hazard starts v20 boundary minuslog2. All three predictions can adapt throughout every corrupted frame, D/X included.',
        'implementation': 'Same native architecture, initialization, objective, seeds, example order, update budget, EMA and decoder as the untrained v62 registration. Replace each grouped five-tap depthwise convolution by its exact sum of shifted channel-wise products, and serial conditional likelihood by associative log-matrix products. Double precision values/gradients and full float32 model gradients checked. No label file used by the benchmark; measured 1.89x full-step speedup. Floating-point reduction order differs, so trained weights/predictions are not claimed bit-identical. V62/V63/V64 queued before fitting are superseded and preserved; no duplicate outer fits.',
        'teacher': 'Each fixed v20 checkpoint has exactly the same three native fitting matches as the outer head and excludes its held match; full teacher has all four TRAIN matches. Raw414 teacher inputs are unchanged. Frozen context is a constituent supervised network, not an out-of-cohort feature producer. No held/native reserved/source/WPY label ancestor. No v60 model or prediction dependency.',
        'features': 'Reuse the already sealed forty v62 native feature views; no repeated feature extraction or new augmentation. Bind the exact existing cache hashes before fitting, with zero reference-array reads. All forty sealed raw-only v20/v58 training views reconstructed with exact old seeds. Compute native98+40 before any rate reduction or time smoothing. Compare re-extracted coarse414 float16 array FRAME FOR FRAME with each old raw-only cached x; abort on any mismatch. Store native x only, no labels. Views100/271,phase0seed731/phase275seed991,failureseed20261118+10*matchindex+halfindex. No interpolation of missing raw input beyond existing documented feature functions; no clean target coordinate teacher.',
        'training': {'seed': SEED, 'steps_per_physical_match': 2000, 'outer_head_steps': 6000, 'full_head_steps': 8000, 'batch': 2, 'length_native_frames': 3072, 'central_native_frames': 1024,
                     'native_rate_Hz': 25, 'window_seconds': 122.88, 'optimizer': 'AdamW', 'lr': .0004, 'weight_decay': .01, 'clip': 5, 'schedule': 'cosine1to.1', 'cpu_threads_per_fit': 1,
                     'checkpoint': 'final EMA, decay.995, final raw weights preserved; no checkpoint selection', 'sampling': 'half uniform known H/A/D frames, half exact native state changes, random team perspective'},
        'supervision': 'Native25Hz H/A/D NLL, equal unweighted and sqrt125/spelllength durationweighted clipped.25..4. Exact25Hz changeBCE posweight25; no softened/contact boundary labels. .1 conditionalpath NLL at every fifth native crop sample; frame/edge losses and outputs remain25Hz, including short spells. .0001 owner/dead logitL2. X ungraded/pathreset. Native boundary positiveweight25 approximately corresponds to5*five-Hz hazard; subtractlog5 before fixed old25Hz Viterbi whose emissions are divided5. This hazard scaling is a modeling assumption tested only through WPY and physical OOF.',
        'calibration': 'OnlyWPY official full-match SpellF1 selects alpha0/.25/.5/.75/1, hazardoffset0. Exactold20 policy-specific target-excluded innertwo parent for outer, exactold20full parent for full. All candidate arrays sealed before WPY truth. Alpha0 preferred on ties; no gradients onWPY and noWQQ calibration.',
        'decode': 'Native output with exact overlap-save1022frame halo; dual-perspective antisymmetric owner and symmetric dead/change. Same fixed three-state v58Viterbi. No input/output rate reduction, parentgap restriction, minimumduration or D/X lock. Clean frame arrays copied exactly from preservedv20.',
        'gate': 'Seal all16selected physicalOOF arrays then audit official both-half full-match five-IoU SpellF1; strictly higher overall than actualv20 and nonnegative bothproducerpairs. Onlyifpass one full4head fit, oneWPYpolicy, oneWQQcandidate. No fallback/parametersearch onWQQ, noexport/promotion.',
        'execution': 'Bind only existing raw-only feature caches. Head fitting waits for v60driver completion only for CPU/RAM limits; there is no scientific result dependency. Two independent CPU head fits may run concurrently. Legacy pinned v58 score/role helpers retain V58 logprefix but operate in v65namespace.',
        'limits': 'Only four independent native training matches. Native resolution removes one representational limitation but does not imply recovered missing evidence or a .9score. Provider-source trials and all baselines preserved. Adaptive reused development, not private/untouched accuracy.',
        'sources': {rel(p): base.sha256(p) for p in sources}, 'inputs': {rel(p): base.sha256(p) for p in files}})
    print('V65 PROTOCOL', base.sha256(RUN / 'protocol.json'), flush=True)


def raw_view(row):
    m, h, view = row['match'], row['half'], row['view']
    v = base.raw(m, h)
    if view in ('100', '271', 100, 271):
        return base.corruption(v, int(view))
    if view in ('phase0', 'phase275'):
        return owner_api.phase_corruption(v, 0 if view == 'phase0' else 275, 731 if view == 'phase0' else 991)
    if view == 'failure':
        return base.api.old.failure_view(v, 20261118 + 10*TRAIN.index(m) + HALVES.index(h))
    raise ValueError('Unregistered corruption view')


def cache_native():
    base.verify()
    rows = {}
    for key, row in training_rows().items():
        base.ALLOWED.clear()
        base.ALLOWED.add(row['match'])
        raw = raw_view(row)
        coarse, _, _ = coarse_inputs(raw)
        with np.load(checked(row['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(coarse.astype(np.float16), z['x'])
        native = api.inputs(raw)
        dest = RUN / f"native/{row['match']}/{row['half']}-{row['view']}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            np.savez_compressed(f, x=native)
        rows[key] = {'cache': pin(dest), 'coarse': row['cache'], 'match': row['match'], 'half': row['half'], 'view': row['view'],
                     'shape': list(native.shape), 'primary_matches': [], 'coarse_raw_feature_exact': True}
        print('V65 CACHE', key, native.shape, flush=True)
    assert not base.READS
    base.ALLOWED.clear()
    write(RUN / 'native-seal.json', {'rows': rows, 'reference_reads': [], 'protocol': pin(RUN / 'protocol.json')})


def bind_cache():
    base.verify()
    original = read(RAW_NATIVE_SOURCE / 'native-seal.json')
    assert not original['reference_reads'] and len(original['rows']) == 40
    base.ALLOWED.update(TRAIN)
    for row in original['rows'].values():
        assert row['coarse_raw_feature_exact'] and not row['primary_matches']
        checked(row['cache'])
        checked(row['coarse'])
    assert not base.READS
    base.ALLOWED.clear()
    write(RUN / 'native-seal.json', {'rows': original['rows'], 'reference_reads': [],
        'source_seal': pin(RAW_NATIVE_SOURCE / 'native-seal.json'), 'protocol': pin(RUN / 'protocol.json')})
    print('V65 BOUND existing40nativeviews; no extraction or references', flush=True)


def teacher(which):
    f = read(OLD / f'models/{which}/freeze.json')
    assert f['residual_fit_matches'] == FITS[which]
    model = owner_api.GapBridge().eval()
    model.load_state_dict(torch.load(checked(f['model']), map_location='cpu', weights_only=True))
    return model, f


def context(model, x, n):
    owner, boundary = owner_api.predict_both(model, x.astype(np.float32), np.zeros(x.shape[-1], np.float32), n)
    return np.c_[owner, boundary - np.log(2.)].astype(np.float32)


def fit(which):
    base.verify()
    if which == 'full':
        assert read(RUN / 'oof-audit.json')['passed']
    cohort = FITS[which]
    base.ALLOWED.update(cohort)
    coarse_model, parent = teacher(which)
    dataset, metadata, target = [], [], {}
    for row in read(RUN / 'native-seal.json')['rows'].values():
        if row['match'] not in cohort:
            continue
        assert row['coarse_raw_feature_exact'] and not row['primary_matches']
        key = (row['match'], row['half'])
        if key not in target:
            target[key] = api.targets(base.truth(*key))
        with np.load(checked(row['cache']), allow_pickle=False) as z:
            x = z['x'].astype(np.float16)
        with np.load(checked(row['coarse']), allow_pickle=False) as z:
            coarse = context(coarse_model, z['x'], len(target[key]['y']))
        assert x.shape == (2, 138, len(target[key]['y']))
        dataset.append({'x': x, 'coarse': coarse, **target[key]})
        metadata.append(row)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = api.JointRecovery().train()
    opt = torch.optim.AdamW(model.parameters(), lr=.0004, weight_decay=.01)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    count = api.steps_per_match * len(cohort)
    history, started, loss_sum = [], time.monotonic(), 0.
    for step in range(count):
        for group in opt.param_groups:
            group['lr'] = .0004 * (.1 + .9*.5*(1+np.cos(np.pi*step/count)))
        v = api.batch(dataset, rng)
        opt.zero_grad(set_to_none=True)
        error = api.loss(model, v)
        assert torch.isfinite(error)
        error.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        assert torch.isfinite(norm)
        opt.step()
        with torch.no_grad():
            for avg, current in zip(ema.parameters(), model.parameters()):
                avg.mul_(.995).add_(current, alpha=.005)
        loss_sum += float(error.detach())
        if (step+1) % 500 == 0 or step == 0 or step == count-1:
            span = 1 if step == 0 else (499 if step == 499 else 500)
            row = {'step': step+1, 'loss_window_mean': loss_sum/span, 'seconds': time.monotonic()-started}
            history.append(row)
            loss_sum = 0.
            print('V65 FIT', which, row, flush=True)
    dest = RUN / f'models/{which}/model.pt'
    dest.parent.mkdir(parents=True, exist_ok=False)
    torch.save(ema.state_dict(), dest)
    raw_dest = dest.parent / 'final-raw-model.pt'
    torch.save(model.state_dict(), raw_dest)
    write(dest.parent / 'freeze.json', {'fit_matches': cohort, 'held': which, 'model': pin(dest), 'final_raw_model': pin(raw_dest),
        'context_teacher': {'freeze': pin(OLD / f'models/{which}/freeze.json'), 'model': parent['model'], 'fit_matches': cohort},
        'native_feature_learned_ancestors': [], 'EMA_decay': .995, 'rows': metadata, 'steps': count, 'history': history,
        'reference_reads': base.READS.copy(), 'protocol': pin(RUN / 'protocol.json')})
    base.ALLOWED.clear()


def predict_logits(which, match, half):
    f = read(RUN / f'models/{which}/freeze.json')
    assert match not in f['fit_matches'] and f['fit_matches'] == FITS[which]
    coarse_model, parent = teacher(which)
    assert parent['model'] == f['context_teacher']['model']
    raw = base.raw(match, half, 'corrupted')
    x, _, _ = coarse_inputs(raw)
    coarse = context(coarse_model, x, len(raw['ball']))
    model = api.JointRecovery().eval()
    model.load_state_dict(torch.load(checked(f['model']), weights_only=True, map_location='cpu'))
    return api.predict(model, api.inputs(raw), coarse), f


base.predict_logits = predict_logits
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'cache_native', 'bind_cache', 'fit', 'calibrate', 'predict_oof', 'audit_oof', 'predict_full', 'audit_full'])
    parser.add_argument('--held', choices=list(FITS))
    args = parser.parse_args()
    base.STAGE = args.stage
    base.threadpool_limits(1)
    torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare()
    else:
        sys.addaudithook(guard)
        if args.stage == 'bind_cache':
            bind_cache()
        elif args.stage == 'cache_native':
            cache_native()
        elif args.stage == 'fit':
            fit(args.held)
        elif args.stage == 'calibrate':
            base.calibrate(args.held)
        elif args.stage.startswith('audit_'):
            base.audit(args.stage == 'audit_full')
        else:
            base.predict_all(args.stage == 'predict_full')
