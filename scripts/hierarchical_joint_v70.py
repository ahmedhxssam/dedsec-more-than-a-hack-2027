"""Additional native joint fitting with a matched training/decoding path model."""
from __future__ import annotations
import argparse
import copy
import datetime
import importlib.util
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scripts._hierarchical_joint_v70_helpers', ROOT / 'scripts/joint_corrupted_v58.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base; spec.loader.exec_module(base)
from scripts import native_joint_v65 as source
from second_zero import hierarchical_joint_recovery as api
from second_zero.raw_context_owner import inputs as coarse_inputs
RUN = ROOT / 'runs/hierarchical-joint-v70'
REPORT = ROOT / 'reports/hierarchical-joint-v70'
TRAIN, HALVES, FITS = base.TRAIN, base.HALVES, base.FITS
POLICIES = [{'alpha': a, 'change_offset': 0.} for a in (0., .25, .5, .75, 1.)]
SEED, STEPS_PER_MATCH = 20261130, 1000
base.RUN, base.REPORT, base.api, base.POLICIES = RUN, REPORT, api, POLICIES
np, torch = base.np, base.torch
read, write, pin, checked, rel = base.read, base.write, base.pin, base.checked, base.rel


def guard(event, args):
    base.guard(event, args)
    if event == 'open' and isinstance(args[0], (str, bytes, Path)) and not base.HASH_ONLY:
        p = Path(base.os.fsdecode(args[0]))
        if '/native-joint-v62/native/' in str(p) and p.suffix == '.npz' and p.parent.name not in base.ALLOWED:
            raise PermissionError('Native feature cache outside current fitting cohort')


def prepare():
    inherited = read(source.RUN / 'protocol.json')
    extra = [source.RUN / 'protocol.json', source.RUN / 'native-seal.json', source.REPORT / 'artifact-freeze.json',
             ROOT / 'runs/native-nonnegative-v68/protocol.json', ROOT / 'runs/native-nonnegative-v68/oof-audit.json',
             ROOT / 'runs/capped-fusion-v69/protocol.json', ROOT / 'runs/capped-fusion-v69/oof-audit.json']
    for which in TRAIN:
        p = source.RUN / f'models/{which}/freeze.json'; extra.extend([p, checked(read(p)['model'])])
    sources = [Path(__file__), ROOT / 'scripts/run_hierarchical_joint_v70.py', ROOT / 'tests/test_hierarchical_joint_recovery.py', ROOT / 'tests/test_hierarchical_joint_v70.py']
    sources += [Path(m.__file__) for name, m in sorted(sys.modules.items()) if name.startswith(('scripts.', 'second_zero.')) and getattr(m, '__file__', None)]
    write(RUN / 'protocol.json', {
        'kind': 'native-joint-three-nonnegative-transition-costs-v70', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'target': inherited['target'], 'fits': FITS, 'policies': POLICIES,
        'architecture': 'Initialize exact native65 140input/64hidden/nineTCN native25Hz model, same target-excluded old20 frozen context. Retain owner/dead output initialization exactly. Add zero-weight three-output transition head, one cost each for live ownership switch, live-to-dead and dead-to-live. All emission/backbone parameters may adapt after250freeze updates. Original unused single-change head weights preserved as inert state.',
        'objective': 'Native frame H/A/D loss as65 plus .1mean of three conditionally masked exact-boundary BCEs with positiveweight25; .25 exact native25Hz conditional-path NLL, same emission/5 and transition scores as decoder, same X/ungraded resets. No five-frame subsampling in this path loss. .0001 owner/deadL2. Native central1024frames, full3072frame input, batch2. Explicit nonnegative switch costs softplus(-h); H/Aflip cost1, liveD cost2, Dlive cost3+log2 for each of two destinations, diagonal0. Unbiased h=raw-log25 in both loss and decoding. No negative switch credits, tuned cap, minimum duration or state lock.',
        'hazard_initialization': 'Training physical target frames counted once (not once per corruption view), Beta1,1 conditional event-rate priors, logit(p)+log25 raw bias. Every count strictly from current cohort; X resets, no WPY/WQQ counts.',
        'training': {'seed': SEED, 'additional_steps_per_physical_match': STEPS_PER_MATCH, 'additional_outer_steps': 3000, 'additional_full_steps': 4000,
            'inherited_native65_steps_outer': 6000, 'inherited_native65_steps_full': 8000, 'window_frames': 3072, 'central_frames': 1024, 'native_Hz': 25, 'batch': 2,
            'optimizer': 'AdamW', 'new_head_lr': .0006, 'inherited_lr': .0001, 'inherited_frozen_first_updates': 250, 'cosine_floor': .1, 'weight_decay': .01, 'clip': 5,
            'checkpoint': 'final EMA .995 only; final raw weights also preserved; no checkpoint selection', 'cpu_threads': 1},
        'data': 'Bind exact same40native65 raw-only cached views; no additional source or regenerated corruption. Same native65 event/uniform sampling, team swap and old20 coarse context. Each outer source65 checkpoint and old20 context has identical other3 physical matches; no held-target ancestor. Contact/source auxiliary labels never substituted for native continuous ownership truth.',
        'calibration': inherited['calibration'],
        'decode': 'Five logit channels owner/dead/flip/stop/start, both team perspectives; exact native overlap-save radius1022; three distinct nonnegative transition costs. Parent prior and alpha grid same65. All corrupted frames editable; exact saved clean bypass. X ungraded in training, uniform parent prior when decoded.',
        'gate': inherited['gate'],
        'full_source': 'Own successful OOF gate required. Mechanically reuse exact native65 full4 intermediate from fusion66, only after66fullaudit exists; never inspect its WQQ score to choose weights or recipe. Full70 adds4000updates to this source in own namespace. No duplicate source full fit or intermediate evaluation.',
        'execution': 'One additional70fit at a time, at most two fitting workers including existing sequential66component fit. Existing40cache reused. No change to any registered earlier scientific source/protocol. Calibration and OOF infer in this family once; full policy/inference waits67/68/69 completion to bound RAM. No WQQ tuning, export or promotion.',
        'motivation_and_limits': 'Native68 and cappedfusion69 improved allfour OOF matches after a WPY-only boundary-credit diagnosis. This motivates distinct transition types and training/decoding alignment, not WQQ feedback. Four native training matches remain; further gain and .9 unproven. Adaptive reused development.',
        'sources': {rel(p): base.sha256(p) for p in sources},
        'inputs': {**inherited['inputs'], **{rel(p): base.sha256(p) for p in extra}}})
    write(RUN / 'native-seal.json', {'rows': read(source.RUN / 'native-seal.json')['rows'], 'reference_reads': [], 'source_seal': pin(source.RUN / 'native-seal.json'), 'protocol': pin(RUN / 'protocol.json')})
    print('V70 PROTOCOL', base.sha256(RUN / 'protocol.json'), flush=True)


def initial_source(which):
    root = source.RUN
    if which == 'full':
        prior = ROOT / 'runs/joint-fusion-v66'
        assert (prior / 'full-audit.json').exists()
        seal = read(prior / 'full-component-seal.json'); checked(seal['model'])
        root = ROOT / seal['component_roots']['native']
    p = root / f'models/{which}/freeze.json'; f = read(p)
    assert f['fit_matches'] == FITS[which] and which not in f['fit_matches']
    assert f['steps'] == (8000 if which == 'full' else 6000)
    checked(f['model']); checked(f['context_teacher']['model'])
    return f, p


def fit(which):
    base.verify()
    if which == 'full': assert read(RUN / 'oof-audit.json')['passed']
    cohort = FITS[which]; base.ALLOWED.update(cohort)
    initial, initial_path = initial_source(which)
    coarse_model, teacher = source.teacher(which)
    assert teacher['model'] == initial['context_teacher']['model']
    dataset, metadata, targets = [], [], {}
    for row in read(RUN / 'native-seal.json')['rows'].values():
        if row['match'] not in cohort: continue
        assert row['coarse_raw_feature_exact'] and not row['primary_matches']
        key = row['match'], row['half']
        if key not in targets: targets[key] = api.targets(base.truth(*key))
        with np.load(checked(row['cache']), allow_pickle=False) as z: x = z['x'].astype(np.float16)
        with np.load(checked(row['coarse']), allow_pickle=False) as z: context = source.context(coarse_model, z['x'], len(targets[key]['y']))
        dataset.append({'x': x, 'coarse': context, **targets[key]}); metadata.append(row)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model = api.JointRecovery().train()
    missing, unexpected = model.load_state_dict(torch.load(checked(initial['model']), map_location='cpu', weights_only=True), strict=False)
    assert set(missing) == {'transitions.weight', 'transitions.bias'} and not unexpected
    bias, counts = api.initial_hazards(targets.values())
    with torch.no_grad(): model.transitions.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    new = list(model.transitions.parameters()); new_ids = {id(p) for p in new}
    old = [p for p in model.parameters() if id(p) not in new_ids]
    opt = torch.optim.AdamW([{'params': old, 'lr': .0001}, {'params': new, 'lr': .0006}], weight_decay=.01)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    count = STEPS_PER_MATCH*len(cohort); started = time.monotonic(); history = []; error_sum = 0.
    for step in range(count):
        scale = .1+.9*.5*(1+np.cos(np.pi*step/count))
        opt.param_groups[0]['lr'] = 0. if step < 250 else .0001*scale
        opt.param_groups[1]['lr'] = .0006*scale
        v = api.batch(dataset, rng); opt.zero_grad(set_to_none=True)
        error = api.loss(model, v); assert torch.isfinite(error)
        error.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); assert torch.isfinite(norm)
        opt.step()
        with torch.no_grad():
            for avg, current in zip(ema.parameters(), model.parameters()): avg.mul_(.995).add_(current, alpha=.005)
        error_sum += float(error.detach())
        if step == 0 or (step+1) % 500 == 0 or step == count-1:
            span = 1 if step == 0 else (499 if step == 499 else 500)
            row = {'step': step+1, 'loss_window_mean': error_sum/span, 'seconds': time.monotonic()-started}
            history.append(row); error_sum = 0.; print('V70 FIT', which, row, flush=True)
    dest = RUN / f'models/{which}/model.pt'; dest.parent.mkdir(parents=True, exist_ok=False)
    torch.save(ema.state_dict(), dest); raw = dest.parent / 'final-raw-model.pt'; torch.save(model.state_dict(), raw)
    write(dest.parent / 'freeze.json', {'fit_matches': cohort, 'held': which, 'model': pin(dest), 'final_raw_model': pin(raw),
        'context_teacher': {'freeze': pin(source.OLD / f'models/{which}/freeze.json'), 'model': teacher['model'], 'fit_matches': cohort},
        'weight_initialization': {'freeze': pin(initial_path), 'model': initial['model'], 'fit_matches': cohort, 'inherited_steps': initial['steps']},
        'steps': count, 'history': history, 'EMA_decay': .995, 'hazard_prior_training_counts': counts,
        'rows': metadata, 'reference_reads': base.READS.copy(), 'protocol': pin(RUN / 'protocol.json')})
    base.ALLOWED.clear()


def predict_logits(which, match, half):
    f = read(RUN / f'models/{which}/freeze.json')
    assert match not in f['fit_matches'] and f['fit_matches'] == FITS[which]
    coarse_model, parent = source.teacher(which); assert parent['model'] == f['context_teacher']['model']
    raw = base.raw(match, half, 'corrupted'); x, _, _ = coarse_inputs(raw)
    coarse = source.context(coarse_model, x, len(raw['ball']))
    model = api.JointRecovery().eval(); model.load_state_dict(torch.load(checked(f['model']), map_location='cpu', weights_only=True))
    return api.predict(model, api.inputs(raw), coarse), f


base.predict_logits = predict_logits
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'fit', 'calibrate', 'predict_oof', 'audit_oof', 'predict_full', 'audit_full']); p.add_argument('--held', choices=list(FITS))
    args = p.parse_args(); base.STAGE = args.stage; base.threadpool_limits(1); torch.set_num_threads(1)
    if args.stage == 'prepare': prepare()
    else:
        sys.addaudithook(guard)
        if args.stage == 'fit': fit(args.held)
        elif args.stage == 'calibrate': base.calibrate(args.held)
        elif args.stage.startswith('audit_'): base.audit(args.stage == 'audit_full')
        else: base.predict_all(args.stage == 'predict_full')
