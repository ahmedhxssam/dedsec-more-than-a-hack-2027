"""One fixed composition of WPY-selected joint evidence and retained ROI quality."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from second_zero.common import json_write, sha256
from second_zero.evaluate import official_metrics
from second_zero.vision_interval import IntervalMotion
from scripts import joint_controller_ball as joint
from scripts import joint_controller_ball_scope as scope
from scripts import sprint_decoder_lattice as lattice
from scripts.vision_interval_train import infer

INTERVAL = ROOT/'runs/vision-push/interval-independent-v1'
WEIGHT = .25


def load_interval(folder=INTERVAL, train=None):
    frozen = json.loads((folder/'freeze.json').read_text())
    for name, field in [('model.pt', 'model_sha256'), ('normalizer.npz', 'normalizer_sha256')]:
        assert sha256(folder/name) == frozen[field]
    ck = torch.load(folder/'model.pt', weights_only=True, map_location='cpu')
    if train is not None:
        assert set(frozen['train_matches']) == set(train)
        assert ck['independent']
        ck = ck['state_dict']
    else:
        p = json.loads((folder/'protocol.json').read_text())
        assert set(p['training_matches']) == set(joint.old.TRAIN)
        assert p['independent_quality']
        assert frozen['selected']['weight'] == WEIGHT
    model = IntervalMotion(independent=True); model.load_state_dict(ck); model.eval()
    with np.load(folder/'normalizer.npz') as z: mean, std = z['mean'], z['std']
    return model, mean, std


def composed_prediction(key, x, base, quality, model, mean, std, penalty=.55):
    segments = lattice.proposals(x[:, 0], base, joint.graph.proposals(x[:, 0], base, x[:, 4], x[:, 7:10]), 16)
    features = joint.graph.segment_features(x, segments)
    q = np.clip(quality.predict(features), 0, 1).astype(np.float32)
    native_path = ROOT/'runs/ownership-temporal-v1/features'/f'{key}.npy'
    row = {'segments': segments, 'features': features, 'quality': q,
           'native': np.load(native_path, mmap_mode='r')}
    roi = infer(model, row, mean, std)
    pred = joint.graph.graph_decode(x[:, 0], base, segments, (1-WEIGHT)*q+WEIGHT*roi, penalty, .01)
    return pred


def build(args, match):
    frozen = json.loads((args.run/'scope-freeze.json').read_text())
    assert sha256(args.run/'model.pt') == frozen['model_sha256']
    assert sha256(args.run/'scope-calibration.json') == frozen['calibration_sha256']
    raw_freeze = json.loads((args.run/f'raw-freeze-{match}.json').read_text())
    model, mean, std = load_interval(); quality = joint.graph.get_model()
    folder = args.out/match; folder.mkdir(parents=True, exist_ok=False)
    for key in joint.graph.keys([match]):
        path = args.run/'raw'/f'{key}.npz'; assert sha256(path) == raw_freeze['outputs'][path.name]
        with np.load(path) as a: z = a['owner']
        original, logit = scope.scoped_candidate(key, z, frozen['temperature'], frozen['weight'], frozen['scope'], quality)
        tag = 'WPY' if match=='J03WPY' else 'reliability'
        with np.load(lattice.RUN/'cache'/tag/'16'/f'{key}.npz') as a: base = a['base']
        x = joint.graph.signals(key); x[:, 0] = logit
        combined = composed_prediction(key, x, base, quality, model, mean, std)
        np.testing.assert_array_equal(combined[base>=2], base[base>=2])
        np.savez_compressed(folder/f'{key}.npz', joint=original, predicted=combined, logit=logit, status_mask=base)
        print('Composition frozen', key, flush=True)
    json_write(folder/'prediction-freeze.json', {p.name: sha256(p) for p in folder.glob('*.npz')})
    joint.ALLOWED.add(match); pm = official_metrics(); views = {'joint': {}, 'combined': {}}
    for key in joint.graph.keys([match]):
        y = joint.graph.truth(key)
        with np.load(folder/f'{key}.npz') as a:
            views['joint'][key] = joint.graph.metric(pm, y, a['joint'])
            views['combined'][key] = joint.graph.metric(pm, y, a['predicted'])
    result = {arm: {**joint.old.aggregate(v), 'views': v} for arm, v in views.items()}
    json_write(args.out/f'{match}-scores.json', result)
    print(json.dumps(result), flush=True)
    return result


def main(args):
    args.run = args.run.resolve(); args.out = args.out.resolve()
    if args.stage == 'calibrate':
        args.out.mkdir(parents=True, exist_ok=False)
        result = build(args, 'J03WPY')
        selected = max(result, key=lambda arm: (result[arm]['overall']['spell'], result[arm]['overall']['macro']))
        json_write(args.out/'freeze.json', {'joint_run': str(args.run), 'joint_scope_sha256': sha256(args.run/'scope-freeze.json'),
            'interval_model_sha256': sha256(INTERVAL/'model.pt'), 'normalizer_sha256': sha256(INTERVAL/'normalizer.npz'),
            'interval_weight': WEIGHT if selected=='combined' else 0., 'selected': selected,
            'WPY_selected': result[selected]['overall'], 'source_sha256': sha256(__file__),
            'selection': 'One fixed previously retained 25% ROI composition versus the single WPY-selected joint recipe. SpellF1 then macroF1; no WQQ tuning.'})
    else:
        frozen = json.loads((args.out/'freeze.json').read_text())
        assert frozen['source_sha256'] == sha256(__file__)
        assert frozen['joint_scope_sha256'] == sha256(args.run/'scope-freeze.json')
        assert frozen['interval_model_sha256'] == sha256(INTERVAL/'model.pt')
        assert frozen['normalizer_sha256'] == sha256(INTERVAL/'normalizer.npz')
        assert frozen['selected'] == 'combined', 'Composition rejected on WPY'
        build(args, 'J03WQQ')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['calibrate', 'evaluate'])
    p.add_argument('--run', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(2); threadpool_limits(2); sys.addaudithook(joint.guard)
    main(args)
