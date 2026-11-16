"""Split-safe transfer audit for a frozen joint controller/ball recipe.

Neural training is a separate stage. This script rejects checkpoints that did
not train on exactly the current fold's side. It reuses verified fold-specific
upstream components, calibrates temperature and composition on WPY, and freezes
both folds' predictions before opening any held-out reference.
"""
from __future__ import annotations
import argparse, json, os, pickle, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from second_zero.joint_controller_ball import JointControllerBall, infer_raw
from second_zero.joint_controller_ball_global import GlobalJointControllerBall
from second_zero.sprint_decoder import decode
from scripts import anchor_export_reduced_crossfit as cf
from scripts.joint_controller_ball_audit import subset_events
from scripts.sprint_evaluate import score
from scripts.anchor_refine_audit import event_change
from scripts.joint_controller_ball_compose import load_interval, composed_prediction

PRIOR = ROOT/'runs/anchor-refinement/reduced-crossfit-v1'


def prepare(args):
    """Cache only split-safe upstream signals; never read held-out truth."""
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    for fold, (train, heldout) in enumerate(cf.FOLDS):
        cf.STATE.update(fold=fold, phase='prepare-signals', allowed=[])
        folder = out/f'fold{fold}'; folder.mkdir()
        prior = PRIOR/f'fold{fold}'; frozen = cf.meta(prior/'freeze.json')
        protocol = cf.meta(prior/'protocol.json')
        assert protocol['train_matches'] == train and protocol['heldout_matches'] == heldout
        assert sha256(cf.pin(prior/'model.pkl')) == frozen['model_sha256']
        with (prior/'model.pkl').open('rb') as handle: quality = pickle.load(handle)
        engine = cf.FoldEvidence(fold, train, heldout)
        for key in cf.graph.keys(heldout):
            x, base, _ = engine.signals(key); seg = cf.pool(x, base)
            q = np.clip(quality.predict(cf.graph.segment_features(x, seg)), 0, 1)
            predicted = cf.graph_prediction(x, base, seg, q, frozen['graph_penalty'])
            with np.load(cf.pin(prior/'quality'/f'{key}.npz')) as a:
                np.testing.assert_array_equal(predicted, a['predicted'])
            np.savez_compressed(folder/f'{key}.npz', x=x, base=base)
            print('Prepared exact-replayed signals', fold, key, flush=True)
        cf.write(folder/'freeze.json', {'train_matches': train, 'heldout_matches': heldout,
            'outputs': {p.name: sha256(p) for p in folder.glob('*.npz')},
            'input_sha256': cf.HASHES.copy(), 'reference_reads': cf.STATE['reads'].copy(),
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(cf.__file__), Path(cf.graph.__file__), Path(cf.lattice.__file__)]},
            'all_baselines_exact_replayed': True, 'status_replay': engine.replay})


def load_model(folder, train, recipe):
    freeze = cf.meta(folder/'freeze-fit.json')
    assert set(freeze['train_matches']) == set(train), 'Wrong training split'
    assert sha256(cf.pin(folder/'model.pt')) == freeze['model_sha256']
    ck = torch.load(folder/'model.pt', weights_only=True, map_location='cpu')
    if 'global_context' in ck:
        protocol = cf.meta(folder/'protocol.json')
        assert sha256(folder/'protocol.json') == freeze['protocol_sha256']
        assert recipe['family'] == 'global'
        assert ck['global_context'] == recipe['global_context']
        assert freeze['step'] == recipe['continuation_steps']
        assert protocol['steps'] == recipe['continuation_steps']
        warmstart = Path(protocol['warmstart'])
        if not warmstart.is_absolute(): warmstart = ROOT/warmstart
        warm = cf.meta(warmstart/'freeze-fit.json')
        assert set(warm['train_matches']) == set(train)
        assert warm['step'] == recipe['base_steps']
        assert sha256(cf.pin(warmstart/'model.pt')) == protocol['warmstart_model_sha256'] == warm['model_sha256']
        base_protocol = cf.meta(warmstart.parent/'protocol.json')
        assert sha256(warmstart.parent/'protocol.json') == warm['protocol_sha256']
        model = GlobalJointControllerBall(ck['width'], ck['global_context'])
    else:
        assert recipe['family'] == 'base'
        assert freeze['step'] == recipe['base_steps']
        base_protocol = cf.meta(folder.parent/'protocol.json')
        assert sha256(folder.parent/'protocol.json') == freeze['protocol_sha256']
        model = JointControllerBall(ck['width'], ck['feedback'])
        protocol = base_protocol
    for training_protocol in [protocol, base_protocol]:
        assert set(training_protocol['train_matches']) == set(train)
        for source, digest in training_protocol['sources'].items():
            assert sha256(cf.pin(ROOT/source)) == digest
    assert {k:base_protocol[k] for k in recipe['base_configuration']} == recipe['base_configuration']
    model.load_state_dict(ck['state_dict']); model.eval()
    return model


def blend_evidence(x, neural, raw, temperature, weight, scope):
    result = x.copy()
    if weight == 0: return result
    if scope == 'all': gate = 1.
    else:
        assert scope == 'missing25'
        gate = uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64), 25, mode='nearest')
    w = weight*gate
    result[:, 0] = logit(np.clip((1-w)*expit(x[:, 0])+w*expit(neural/temperature), 1e-7, 1-1e-7))
    return result


def calibrate_fold(folder, engine, quality, interval, temperature, penalty):
    """Select this fold's blend using only this fold's models and WPY."""
    pm = official_metrics(); cache = {}
    for key in cf.graph.keys(['J03WPY']):
        x, base, _ = engine.signals(key)
        with np.load(folder/'raw'/f'{key}.npz') as a: neural = a['owner'].copy()
        cache[key] = (x, base, neural, cf.raw_data(key), cf.truth(key))
    def aggregate(views):
        return {condition: {m: float(np.mean([v[m] for k, v in views.items()
                    if condition == 'overall' or k.endswith(condition)])) for m in ['spell', 'macro']}
                for condition in ['overall', 'clean', 'corrupted']}
    rows = []
    for scope, weights in [('all', [0., .1, .25, .5, 1.]), ('missing25', [.25, .5, 1.])]:
        for weight in weights:
            views = {}
            for key, (x, base, neural, raw, y) in cache.items():
                nx = blend_evidence(x, neural, raw, temperature, weight, scope)
                seg = cf.pool(nx, base)
                q = np.clip(quality.predict(cf.graph.segment_features(nx, seg)), 0, 1)
                pred = cf.graph_prediction(nx, base, seg, q, penalty)
                views[key] = cf.graph.metric(pm, y, pred)
            row = {'scope': scope, 'weight': weight, **aggregate(views), 'views': views}
            rows.append(row)
            print('Fold WPY blend', engine.fold, scope, weight, row['overall'], flush=True)
    selected = max(rows, key=lambda r: (r['overall']['spell'], r['overall']['macro']))
    compositions = [{'interval_weight': 0., **{k: selected[k] for k in ['overall', 'clean', 'corrupted', 'views']}}]
    if interval:
        views = {}
        for key, (x, base, neural, raw, y) in cache.items():
            nx = blend_evidence(x, neural, raw, temperature, selected['weight'], selected['scope'])
            pred = composed_prediction(key, nx, base, quality, *interval, penalty=penalty)
            views[key] = cf.graph.metric(pm, y, pred)
        compositions.append({'interval_weight': .25, **aggregate(views), 'views': views})
    composition = max(compositions, key=lambda r: (r['overall']['spell'], r['overall']['macro']))
    cf.write(folder/'WPY-calibration.json', {'rows': rows, 'selected_blend': selected,
        'compositions': compositions, 'selected_composition': composition,
        'rule': 'Same eight ownership settings as the full WPY experiment; then ROI off versus .25 only at the selected setting. All fitted components use this fold training side. SpellF1 first, macroF1 only breaks ties.'})
    return selected['weight'], selected['scope'], composition['interval_weight'], composition['overall']


def predict_fold(args):
    out = args.out.resolve(); recipe = json.loads(args.recipe.read_text())
    if not out.exists():
        out.mkdir(parents=True)
        cf.write(out/'recipe.json', recipe)
        sources = [Path(__file__), Path(cf.__file__), Path(cf.graph.__file__),
                   Path(cf.lattice.__file__), ROOT/'second_zero/joint_controller_ball.py',
                   ROOT/'second_zero/joint_controller_ball_global.py',
                   ROOT/'scripts/joint_controller_ball_compose.py',
                   ROOT/'scripts/vision_interval_train.py', ROOT/'second_zero/vision_interval.py']
        for source in sources:
            dest = out/'sources'/source.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        cf.write(out/'protocol.json', {
            'folds': cf.FOLDS, 'recipe_sha256': sha256(args.recipe),
            'calibration_amendment_sha256': sha256(ROOT/'runs/joint-controller-ball/transfer-calibration-amendment.json'),
            'scope': 'Reduced complete pipeline, not a crossfit reproduction of the full sprint ensemble.',
            'calibration': 'Per-fold temperature, eight ownership settings and ROI off/.25 are selected on WPY using only same-fold models. The prospective transfer-calibration amendment supersedes the recipe rule that transferred full-model blend settings. Family and training steps remain fixed research choices; these matches have been used in prior development.',
            'upstream': 'Same-fold ownership, imputation, status and quality. Existing quality training uses in-sample upstream signals, never an opposite-fold model trained on the evaluation side.',
            'neutral_control': 'At the same fold-selected scope/weight, replace the added neural ownership probability by .5 and leave ROI off. Compare with joint_only to distinguish confidence softening from new ownership evidence. No additional selection.',
            'heldout_reference_policy': 'Both folds predictions frozen before any held-out scoring; WR9/WQQ/competition inputs forbidden.',
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources}})
    else:
        assert cf.meta(out/'recipe.json') == recipe
        for path, digest in cf.meta(out/'protocol.json')['source_sha256'].items():
            assert sha256(ROOT/path) == digest, 'Transfer sources changed between folds'
    folder = out/f'fold{args.fold}'; folder.mkdir(exist_ok=False)
    train, heldout = cf.FOLDS[args.fold]
    cf.STATE.update(fold=args.fold, phase='load', allowed=[])
    model = load_model(args.model.resolve(), train, recipe)
    prior = PRIOR/f'fold{args.fold}'
    p = cf.meta(prior/'protocol.json'); frozen = cf.meta(prior/'freeze.json')
    assert p['train_matches'] == train and p['heldout_matches'] == heldout
    assert sha256(cf.pin(prior/'model.pkl')) == frozen['model_sha256']
    with (prior/'model.pkl').open('rb') as handle: quality = pickle.load(handle)
    interval = None
    if recipe.get('interval_weight', 0):
        assert recipe['interval_weight'] == .25
        interval_folder = ROOT/f'runs/vision-push/interval-crossfit-v1/fold{args.fold}'
        for name in ['model.pt', 'normalizer.npz', 'freeze.json']: cf.pin(interval_folder/name)
        interval = load_interval(interval_folder, train)
    evidence = args.evidence.resolve()/f'fold{args.fold}' if args.evidence else None
    if evidence:
        evidence_freeze = cf.meta(evidence/'freeze.json')
        assert evidence_freeze['train_matches'] == train and evidence_freeze['heldout_matches'] == heldout
        for path, digest in evidence_freeze['input_sha256'].items(): assert sha256(cf.pin(path)) == digest
        for path, digest in evidence_freeze['source_sha256'].items(): assert sha256(ROOT/path) == digest
    engine = cf.FoldEvidence(args.fold, train, heldout)
    # Persist WPY raw logits before permitting temperature-label reads.
    raw_folder = folder/'raw'; raw_folder.mkdir()
    for key in cf.graph.keys(['J03WPY']):
        result = infer_raw(model, cf.raw_data(key))
        np.savez_compressed(raw_folder/f'{key}.npz', **result)
        print('Crossfit raw WPY', args.fold, key, flush=True)
    cf.write(folder/'WPY-raw-freeze.json', {p.name: sha256(p) for p in raw_folder.glob('*.npz')})
    cf.STATE.update(phase='temperature', allowed=['J03WPY'])
    zs, ys = [], []
    for key in cf.graph.keys(['J03WPY']):
        with np.load(raw_folder/f'{key}.npz') as a: z = a['owner']
        y = cf.truth(key); zs.append(z[y<2]); ys.append(y[y<2])
    z, y = np.concatenate(zs), np.concatenate(ys)
    fitted = minimize_scalar(lambda t: np.mean(np.logaddexp(0, z/np.exp(t))-y*z/np.exp(t)),
                             bounds=(-2.3, 2.3), method='bounded')
    temperature = float(np.exp(fitted.x))
    cf.STATE.update(phase='WPY-calibration', allowed=['J03WPY'])
    weight, scope, interval_weight, selected_score = calibrate_fold(
        folder, engine, quality, interval, temperature, frozen['graph_penalty'])
    cf.write(folder/'freeze.json', {'train_matches': train, 'heldout_matches': heldout,
        'temperature': temperature, 'temperature_WPY_NLL': float(fitted.fun),
        'weight': weight, 'scope': scope, 'interval_weight': interval_weight,
        'WPY_selected': selected_score, 'calibration_sha256': sha256(folder/'WPY-calibration.json'),
        'recipe_sha256': sha256(out/'recipe.json'),
        'model_sha256': sha256(args.model/'model.pt'), 'model_path': str(args.model.resolve()),
        'base_quality_sha256': frozen['model_sha256'], 'graph_penalty': frozen['graph_penalty'],
        'heldout_truth_read': False, 'fold_specific_blend_selection': True})
    cf.STATE.update(phase='inference', allowed=[])
    arms = ['baseline', 'neutral', 'joint_only', 'candidate', 'owner_baseline', 'owner_joint']
    for arm in arms: (folder/arm).mkdir()
    replay = {}
    for key in cf.graph.keys(heldout):
        raw = cf.raw_data(key); neural = infer_raw(model, raw)
        np.savez_compressed(raw_folder/f'{key}.npz', **neural)
        if evidence:
            path = evidence/f'{key}.npz'; assert sha256(cf.pin(path)) == evidence_freeze['outputs'][path.name]
            with np.load(path) as a: x, base = a['x'], a['base']
        else: x, base, _ = engine.signals(key)
        segments = cf.pool(x, base)
        q = np.clip(quality.predict(cf.graph.segment_features(x, segments)), 0, 1)
        baseline = cf.graph_prediction(x, base, segments, q, frozen['graph_penalty'])
        with np.load(cf.pin(prior/'quality'/f'{key}.npz')) as a:
            np.testing.assert_array_equal(baseline, a['predicted'])
        nx = blend_evidence(x, neural['owner'], raw, temperature, weight, scope)
        seg = cf.pool(nx, base)
        nq = np.clip(quality.predict(cf.graph.segment_features(nx, seg)), 0, 1)
        joint_only = cf.graph_prediction(nx, base, seg, nq, frozen['graph_penalty'])
        ux = blend_evidence(x, np.zeros(len(base)), raw, temperature, weight, scope)
        useg = cf.pool(ux, base)
        uq = np.clip(quality.predict(cf.graph.segment_features(ux, useg)), 0, 1)
        neutral = cf.graph_prediction(ux, base, useg, uq, frozen['graph_penalty'])
        candidate = (composed_prediction(key, nx, base, quality, *interval, penalty=frozen['graph_penalty'])
                     if interval_weight else joint_only)
        predictions = {'baseline': baseline, 'neutral': neutral, 'joint_only': joint_only, 'candidate': candidate,
                       'owner_baseline': decode(x[:, 0], base, cost=12.),
                       'owner_joint': decode(neural['owner']/temperature, base, cost=12.)}
        for arm, pred in predictions.items():
            np.testing.assert_array_equal(pred[base>=2], base[base>=2])
            np.savez_compressed(folder/arm/f'{key}.npz', predicted=pred, status_mask=base)
        replay[key] = {'baseline_exact_replay': True, 'changed_frames': int((baseline!=candidate).sum())}
        print('Crossfit inferred', args.fold, key, replay[key], flush=True)
    cf.write(folder/'inference.json', {
        'outputs': {str(p.relative_to(folder)): sha256(p) for arm in arms for p in (folder/arm).glob('*.npz')},
        'raw_outputs': {p.name: sha256(p) for p in raw_folder.glob('*.npz')},
        'baseline_replay': replay, 'status_replay': evidence_freeze['status_replay'] if evidence else engine.replay,
        'input_sha256': cf.HASHES, 'reference_reads': cf.STATE['reads'], 'heldout_truth_read': False})


def audit(args):
    out = args.out.resolve(); outputs = {}
    for fold in [0, 1]:
        folder = out/f'fold{fold}'; frozen = cf.meta(folder/'inference.json')
        for path, digest in frozen['outputs'].items():
            assert sha256(folder/path) == digest
            outputs[f'fold{fold}/{path}'] = digest
    cf.write(out/'prediction-freeze.json', {'outputs': outputs, 'heldout_truth_read': False})
    cf.STATE.update(fold='paired-audit', phase='audit', allowed=[m for _, held in cf.FOLDS for m in held])
    pm = official_metrics(); views = {}
    for fold, (_, heldout) in enumerate(cf.FOLDS):
        for key in cf.graph.keys(heldout):
            y = cf.truth(key); raw = cf.raw_data(key); scored = {}; subsets = {}
            for arm in ['baseline', 'neutral', 'joint_only', 'candidate', 'owner_baseline', 'owner_joint']:
                with np.load(out/f'fold{fold}'/arm/f'{key}.npz') as a: pred = a['predicted']
                scored[arm], subsets[arm] = subset_events(pm, y, pred, raw)
            views[key] = {'metrics': {arm: s['metrics'] for arm, s in scored.items()},
                'event_change': event_change(scored['baseline'], scored['candidate']),
                'neutral_to_joint_event_change': event_change(scored['neutral'], scored['joint_only']),
                'owner_event_change': event_change(scored['owner_baseline'], scored['owner_joint']), 'subsets': subsets}
    groups = {}
    for condition in ['overall', 'clean', 'corrupted']:
        rows = [v for k, v in views.items() if condition=='overall' or k.endswith(condition)]
        g = {arm: {m: float(np.mean([r['metrics'][arm][m] for r in rows]))
                   for m in ['spell_f1_mean', 'macro_f1', 'fragmentation']}
             for arm in ['baseline', 'neutral', 'joint_only', 'candidate', 'owner_baseline', 'owner_joint']}
        g['events'] = {k: sum(r['event_change'][k] for r in rows) for k in
            ['recovered', 'lost', 'short_recovered', 'short_lost', 'before_short_matches', 'after_short_matches',
             'before_predicted_spells', 'after_predicted_spells']}
        g['subsets'] = {arm: {name: {k: sum(r['subsets'][arm][name][k] for r in rows)
                         for k in ['references', 'matched_iou05']}
                         for name in ['short', 'very_short_25', 'gap_overlap', 'short_gap_overlap']}
                       for arm in ['baseline', 'candidate']}
        g['per_class_F1'] = {arm: {label: float(np.mean([r['metrics'][arm]['per_class'][label]['f1'] for r in rows]))
                                 for label in ['H', 'A', 'D']} for arm in ['baseline', 'candidate']}
        groups[condition] = g
    matches = sorted({k.split('-')[0] for k in views})
    per_match = {match: {arm: {m: float(np.mean([v['metrics'][arm][m] for k,v in views.items() if k.startswith(match)]))
                              for m in ['spell_f1_mean', 'macro_f1']} for arm in ['baseline', 'candidate']}
                 for match in matches}
    cf.write(out/'audit.json', {'groups': groups, 'per_match': per_match, 'views': views,
        'reference_reads': cf.STATE['reads'],
        'scope': 'Four external matches, paired clean/corrupted views; reduced complete pipeline. Pure ownership cost12 reported separately. Fixed family and training steps, with same-fold WPY calibration of blend/scope/ROI. All fitted components exclude each held-out match. Reused development, not untouched evaluation. No audit-driven tuning.'})
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'predict', 'audit'])
    p.add_argument('--out', type=Path, required=True); p.add_argument('--recipe', type=Path)
    p.add_argument('--fold', type=int, choices=[0, 1]); p.add_argument('--model', type=Path)
    p.add_argument('--evidence', type=Path)
    args = p.parse_args(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True); threadpool_limits(2)
    sys.addaudithook(cf.guard)
    if args.stage == 'prepare': prepare(args)
    elif args.stage == 'predict': predict_fold(args)
    else: audit(args)
