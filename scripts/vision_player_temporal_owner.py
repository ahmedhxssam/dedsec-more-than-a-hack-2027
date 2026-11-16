"""Train and calibrate a compact identity-temporal ownership network.

Four whole matches train the model; WPY selects its checkpoint/temperature/blend.
WQQ is accessed only by the explicit frozen evaluation stage. No feature tensors
are written, and only the best compact checkpoint is retained.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from second_zero.vision_player_temporal import PlayerTemporalOwner, crop_features, infer_raw, FEATURES, VERSION, RADIUS
from second_zero.common import sha256, json_write
from second_zero.evaluate import official_metrics
from scripts.train_contextual import corruption
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts.anchor_track_evidence import candidate as decode_candidate

DATA = ROOT / 'private/improvement/external/prepared'
TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY']
DEFAULT_RUN = ROOT / 'runs/vision-push/player-temporal-v1'
READS = set()


def guard(stage):
    def check(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if 'J03WR9' in p or '/evaluation/gt/' in p:
                raise PermissionError('Reserved and competition references excluded')
            if p.endswith('.truth.npy'):
                READS.add(p)
                if stage != 'evaluate' and 'J03WQQ' in p:
                    raise PermissionError('WQQ references prohibited before frozen evaluation')
    return check


def load_raw(key):
    name, condition = key.rsplit('-', 1)
    with np.load(DATA / (name + '.npz')) as a:
        raw = {k: a[k] for k in ['ball', 'players', 'team', 'pitch']}
    return corruption(raw, 100) if condition == 'corrupted' else raw


def boundary_targets(y, radius=3):
    valid = np.r_[False, (y[1:] < 2) & (y[:-1] < 2)]
    edges = np.flatnonzero(valid & np.r_[False, y[1:] != y[:-1]])
    target = np.zeros((len(y), 2), np.float32)
    for edge in edges:
        target[max(0, edge - radius):min(len(y), edge + radius + 1), y[edge]] = 1
    target[~valid] = 0
    return target, valid, edges


def training_records():
    records = []
    for key in graph.keys(TRAIN):
        raw = load_raw(key); y = graph.truth(key)
        boundary, boundary_valid, edges = boundary_targets(y)
        records.append({'key': key, 'raw': raw, 'truth': y, 'boundary': boundary,
                        'boundary_valid': boundary_valid, 'edges': edges})
    return records


def batch(records, rng, batch_size=6, target=192):
    xx, yy, bb, vv = [], [], [], []
    for _ in range(batch_size):
        row = records[int(rng.integers(len(records)))]; y = row['truth']; n = len(y)
        start = int(rng.integers(0, n - target + 1))
        if len(row['edges']) and rng.random() < .2:
            anchor = int(row['edges'][int(rng.integers(len(row['edges'])))])
            start = int(np.clip(anchor - int(rng.integers(target)), 0, n - target))
        stop = start + target
        x = crop_features(row['raw'], start - RADIUS, stop + RADIUS).astype(np.float32)
        if rng.random() < .5:
            x[..., [0, 4, 8, 23, 26]] *= -1
        if rng.random() < .5:
            x[..., [1, 5, 9, 24, 27]] *= -1
        xx.append(x); yy.append(y[start:stop]); bb.append(row['boundary'][start:stop]); vv.append(row['boundary_valid'][start:stop])
    max_players = max(x.shape[2] for x in xx)
    xx = [np.pad(x, ((0, 0), (0, 0), (0, max_players - x.shape[2]), (0, 0))) for x in xx]
    return tuple(torch.from_numpy(np.stack(arrays)) for arrays in [xx, yy, bb, vv])


def probe_records():
    result = []
    for key in graph.keys(['J03WPY']):
        raw, y = load_raw(key), graph.truth(key)
        for start in np.linspace(0, len(y) - 256, 12).astype(int):
            x = crop_features(raw, start - RADIUS, start + 256 + RADIUS)
            result.append((x, y[start:start + 256]))
    return result


def probe_quality(model, records):
    model.eval(); total, count = 0., 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for x, y in records:
            z = model(torch.from_numpy(x[None].astype(np.float32)).to(device))[0, RADIUS:RADIUS + len(y)]
            valid = torch.from_numpy(y < 2).to(device); target = torch.from_numpy(y.astype(np.float32)).to(device)
            total += (F.binary_cross_entropy_with_logits(z, target, reduction='none') * valid).sum().item()
            count += int(valid.sum())
    return total / max(1, count)


def masked_bce(logits, targets, mask, pos_weight=None):
    """Fixed-shaped masked reduction avoids per-mask MPS graph compilation."""
    losses = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def memory_mib():
    # Darwin reports bytes; Linux reports kilobytes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def profile(args):
    started = time.monotonic(); records = training_records()
    torch.manual_seed(args.seed); model = PlayerTemporalOwner(player_width=args.width).eval()
    x, y, _, _ = batch(records, np.random.default_rng(20261024), args.batch)
    with torch.no_grad():
        cpu_output = model(x)
    model = model.to(args.device); x = x.to(args.device); y = y.to(args.device)
    timings = []
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        if args.device == 'mps':
            torch.mps.synchronize()
        before = time.monotonic(); full = model(x); out = full[:, RADIUS:RADIUS + y.shape[1]]
        loss = F.binary_cross_entropy_with_logits(out[y < 2], y[y < 2].float()); loss.backward()
        if args.device == 'mps':
            torch.mps.synchronize()
        timings.append(time.monotonic() - before)
    with torch.no_grad():
        repeat = model(x).cpu(); actual = full.detach().cpu()
    row = {'parameters': sum(p.numel() for p in model.parameters()),
           'raw_array_MiB': sum(v.nbytes for r in records for v in r['raw'].values()) / 1024 ** 2,
           'batch_shape': list(x.shape), 'peak_RSS_MiB': memory_mib(),
           'device': args.device, 'forward_backward_seconds': timings,
           'CPU_device_max_absolute_difference': float((cpu_output - actual).abs().max()),
           'repeat_max_absolute_difference': float((repeat - actual).abs().max()),
           'total_profile_seconds': time.monotonic() - started, 'feature_cache_bytes': 0,
           'model_parameter_MiB': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2}
    print(json.dumps(row, indent=2), flush=True)


def save_model(model, path, args):
    torch.save({'state_dict': model.state_dict(), 'player_width': args.width,
                'channels': len(FEATURES), 'team_width': 32,
                'mask_missing_positions': model.mask_missing_positions}, path)


def get_model(out):
    checkpoint = torch.load(out / 'model.pt', weights_only=True, map_location='cpu')
    model = PlayerTemporalOwner(channels=checkpoint['channels'], player_width=checkpoint['player_width'], team_width=checkpoint['team_width'],
                                mask_missing_positions=checkpoint.get('mask_missing_positions', False))
    model.load_state_dict(checkpoint['state_dict']); model.eval(); return model


def train(args):
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    plan = {'version': VERSION, 'training_matches': TRAIN, 'calibration_match': 'J03WPY',
            'evaluation_match': 'J03WQQ', 'checkpoint_selection': 'WPY fixed12uniform256f windows per view, live-frame NLL; no WQQ labels',
            'architecture': 'same-slot temporal dilation1/2/4, relative-team social attention, same-slot temporal8/16/32/64, then masked team pools and shared team temporal1/2/4/8/16',
            'features': FEATURES, 'player_width': args.width, 'team_width': 32, 'model_radius_frames': RADIUS,
            'training': {'steps': args.steps, 'batch': args.batch, 'target_frames': 192, 'lr': .001,
                         'sampling': '80% uniform windows,20% training H/A boundaries; original25Hz targets',
                         'boundary_auxiliary_weight': .05, 'boundary_positive_weight': 20.,
                         'loss': 'ownership BCE plus directional boundary±3frame BCE; no temporal smoothing or minimum duration'},
            'feature_cache': False, 'pretrained_external_weights': False, 'seed': args.seed,
            'sources': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / 'second_zero/vision_player_temporal.py']}}
    json_write(out / 'protocol.json', plan)
    snapshot = out / 'sources'; snapshot.mkdir()
    for source in [Path(__file__), ROOT / 'second_zero/vision_player_temporal.py']:
        shutil.copyfile(source, snapshot / source.name)
    records = training_records(); probes = probe_records()
    model = PlayerTemporalOwner(player_width=args.width).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    started = time.monotonic(); best = float('inf'); history = []
    for step in range(1, args.steps + 1):
        model.train(); x, y, boundary, boundary_valid = (t.to(args.device) for t in batch(records, rng, args.batch))
        owner, directional = model(x, return_boundary=True)
        owner = owner[:, RADIUS:RADIUS + y.shape[1]]
        directional = directional[:, RADIUS:RADIUS + y.shape[1]]
        valid = y < 2
        if not valid.any():
            continue
        main = masked_bce(owner, y.float(), valid)
        aux = masked_bce(directional, boundary, boundary_valid[..., None],
                         pos_weight=torch.tensor(20., device=args.device)) / 2
        loss = main + .05 * aux
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
        if step % 100 == 0:
            print('step', step, 'owner_loss', main.item(), 'boundary_loss', aux.item(), 'seconds', round(time.monotonic() - started, 1), 'RSS_MiB', round(memory_mib()), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            nll = probe_quality(model, probes)
            row = {'step': step, 'WPY_probe_NLL': nll, 'seconds': time.monotonic() - started}
            if nll < best:
                best = nll; save_model(model, out / 'model.pt', args); row['selected'] = True
            history.append(row); json_write(out / 'training.json', {'history': history, 'best_WPY_probe_NLL': best,
                       'peak_RSS_MiB': memory_mib(), 'seconds': time.monotonic() - started,
                       'parameters': sum(p.numel() for p in model.parameters()), 'reference_reads': sorted(READS)})
            print('checkpoint', json.dumps(row), flush=True)


def infer_and_save(out, model, key):
    folder = out / 'raw'; folder.mkdir(exist_ok=True)
    path = folder / (key + '.npz')
    if path.exists():
        with np.load(path) as a:
            return a['raw_logit']
    z, boundary, control = infer_raw(model, load_raw(key), return_control=True)
    np.savez_compressed(path, raw_logit=z, boundary_logit=boundary, control_logit=control)
    print('inferred', key, flush=True); return z


def aggregate(views):
    return {condition: {metric: float(np.mean([r[metric] for key, r in views.items() if condition == 'overall' or key.endswith(condition)])) for metric in ['spell', 'macro']}
            for condition in ['overall', 'clean', 'corrupted']}


def calibrate(args):
    out = args.out; model = get_model(out).to(args.device); cache, zs, ys = {}, [], []
    for key in graph.keys(['J03WPY']):
        z = infer_and_save(out, model, key); cache[key] = z; y = graph.truth(key)
        zs.append(z[y < 2]); ys.append(y[y < 2])
    z, y = np.concatenate(zs), np.concatenate(ys)
    raw_nll = float(np.mean(np.logaddexp(0, z) - y * z))
    fit = minimize_scalar(lambda lt: np.mean(np.logaddexp(0, z / np.exp(lt)) - y * z / np.exp(lt)), bounds=(-2.3, 2.3), method='bounded')
    temperature = float(np.exp(fit.x)); quality_model = graph.get_model(); pm = official_metrics(); results = []
    for weight in [0., .1, .25, .5]:
        folder = out / 'WPY' / str(weight); folder.mkdir(parents=True, exist_ok=True); views = {}
        for key, z in cache.items():
            pred, owner = decode_candidate(key, {'raw_logit': z, 'eligible': np.ones(len(z), bool)}, temperature, weight, quality_model)
            np.savez_compressed(folder / (key + '.npz'), predicted=pred, logit=owner)
            views[key] = graph.metric(pm, graph.truth(key), pred)
        row = {'weight': weight, **aggregate(views), 'views': views}; results.append(row)
        print('WPY', weight, row['overall'], row['clean'], row['corrupted'], flush=True)
    best = max(results, key=lambda r: (r['overall']['spell'], r['overall']['macro']))
    json_write(out / 'calibration.json', {'temperature': temperature, 'raw_NLL': raw_nll,
             'calibrated_NLL': float(fit.fun), 'rows': results, 'selected': best})
    json_write(out / 'freeze.json', {'temperature': temperature, 'weight': best['weight'], 'WPY_selected': best['overall'],
               'model_sha256': sha256(out / 'model.pt'), 'protocol_sha256': sha256(out / 'protocol.json'),
               'calibration_sha256': sha256(out / 'calibration.json'), 'WQQ_reference_accessed': False})


def evaluate(args):
    out = args.out; freeze = json.loads((out / 'freeze.json').read_text())
    if freeze['weight'] == 0:
        print('WPY selected unchanged winner; no WQQ inference/evaluation.', flush=True); return
    assert sha256(out / 'model.pt') == freeze['model_sha256']
    model, quality_model = get_model(out).to(args.device), graph.get_model(); dest = out / 'WQQ'; dest.mkdir(exist_ok=False)
    for key in graph.keys(['J03WQQ']):
        raw = infer_and_save(out, model, key)
        pred, z = decode_candidate(key, {'raw_logit': raw, 'eligible': np.ones(len(raw), bool)}, freeze['temperature'], freeze['weight'], quality_model)
        np.savez_compressed(dest / (key + '.npz'), predicted=pred, logit=z)
    json_write(dest / 'prediction_freeze.json', {p.name: sha256(p) for p in dest.glob('*.npz')})
    pm = official_metrics(); views = {}
    for key in graph.keys(['J03WQQ']):
        with np.load(dest / (key + '.npz')) as a:
            views[key] = graph.metric(pm, graph.truth(key), a['predicted'])
    result = {**aggregate(views), 'views': views, 'freeze': freeze, 'interpretation': 'Frozen complete-pipeline comparison on repeatedly reused WQQ development, paired clean/corrupted views.'}
    json_write(out / 'evaluation.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['profile', 'train', 'calibrate', 'evaluate'])
    p.add_argument('--out', type=Path, default=DEFAULT_RUN); p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--batch', type=int, default=4); p.add_argument('--width', type=int, default=24)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--seed', type=int, default=20261024); p.add_argument('--checkpoint-every', type=int, default=400)
    args = p.parse_args(); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    sys.addaudithook(guard(args.stage)); globals()[args.stage](args)
