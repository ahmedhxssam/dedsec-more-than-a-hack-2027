"""Explicit artifact loading for the unchanged complete raw baseline.

The inherited prediction method is the retained pipeline itself. This module
changes only where fold-specific fitted artifacts come from, never its math.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.special import expit, logit
from second_zero.common import sha256
from scripts import aggressive_unseen_pipeline as retained

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY']
FOLDS = [(TRAIN[2:], TRAIN[:2]), (TRAIN[:2], TRAIN[2:])]
HALVES = ['firstHalf', 'secondHalf']
CONDITIONS = ['clean', 'corrupted']
RUN = ROOT/'runs/full-baseline-gap-v1'


def keys(matches):
    return [f'{m}-{h}-{c}' for m in matches for h in HALVES for c in CONDITIONS]


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path, **metadata):
    path = Path(path)
    return {'path': rel(path), 'sha256': sha256(path), **metadata}


def checked(item):
    path = ROOT/item['path']
    if sha256(path) != item['sha256']:
        raise ValueError('Artifact changed: '+item['path'])
    return path


def raw_half(key):
    match, half, condition = key.split('-')
    if match not in TRAIN+['J03WPY', 'J03WQQ'] or half not in HALVES or condition not in CONDITIONS:
        raise ValueError('Invalid raw half key')
    with np.load(retained.RAW_DATA/f'{match}-{half}.npz') as archive:
        raw = {name: archive[name] for name in ['ball', 'players', 'team', 'pitch']}
    if condition == 'corrupted':
        raw = retained.corruption(raw, 100)
    return raw


def full_reference_spec():
    """Resolve all original artifacts without opening observation or GT files."""
    s = retained.SPRINT
    owners = []
    for folder in retained.UNIFORM+[retained.SHORT]:
        f = read(folder/'freeze.json')
        owners.append(artifact(folder/'model.pt', temperature=f['temperature']))
        assert owners[-1]['sha256'] == f['model_sha256']
    interp = s/'model/interp-full'
    memory = s/'memory/gru-interp-v3'
    gp, gf = read(memory/'protocol.json'), read(memory/'freeze.json')
    carrier = s/'carrier-specialist-v1'
    cf = read(carrier/'freeze.json')
    temporal = ROOT/'runs/live-dead-temporal-v1'
    tf = read(temporal/'freeze.json')
    context = s/'status-context-v1'
    reliability = read(s/'status-mixture-v1/reliability/calibration.json')
    boundary = ROOT/'runs/overnight-boundary-v1'
    bf = read(boundary/'freeze.json')
    joint = read(retained.JOINT/'scope-freeze.json')
    return {
        'kind': 'complete-retained-raw-baseline-artifacts-v1',
        'prediction_source': artifact(Path(retained.__file__)),
        'owners': owners,
        'imputer': artifact(s/'ball-imputer-v1/models.pkl'),
        'interpolation_owner': artifact(interp/'model.pt', temperature=read(interp/'calibration.json')['temperature']),
        'memory': artifact(memory/'model.pt', temperature=gf['temperature'], channels=gp['channels'], width=gp['width'], stride=gp['stride'], context=gp['context']),
        'carrier': artifact(carrier/'model.pkl', temperature=cf['temperature'], weight=read(s/'carrier-mixture-v1/freeze.json')['WPY_weight']),
        'legacy': artifact(ROOT/'runs/contextual-external-v1/models.pkl'),
        'temporal_status': artifact(temporal/'model.pt', temperature=tf['temperature']),
        'context_status': artifact(context/'model.pkl', calibration=read(context/'calibration.json')),
        'reliability': {'fit': reliability['fit'], 'selection': reliability['selected']},
        'boundary': artifact(boundary/'models.pkl', config=bf['config']),
        'quality': artifact(retained.graph.RUN/'model.pkl'),
        'joint': artifact(retained.JOINT/'model.pt', temperature=joint['temperature'], weight=joint['weight'], scope=joint['scope']),
        'contract': {'observed_channels': 98, 'interpolation_channels': 24, 'status_channels': 40,
                     'precision': 'float32', 'quality_channels': 175, 'penalty': .55,
                     'frame_weight': .01, 'neighbors': 16, 'learned_residual': .75},
    }


class ExplicitRawPipeline(retained.FrozenRawPipeline):
    """Same raw API and predict implementation; explicit hashed component spec."""
    def __init__(self, spec):
        checked(spec['prediction_source'])
        self.inventory = spec
        if len(spec['owners']) != 4:
            raise ValueError('Complete baseline needs uniform3 plus short')
        self.owners = []
        for item in spec['owners']:
            ck = torch.load(checked(item), weights_only=True, map_location='cpu')
            model = retained.TemporalOwner(ck['channels'], ck['width'])
            model.load_state_dict(ck['state_dict']); model.eval()
            self.owners.append((model, item['temperature']))
        def fitted(name):
            with checked(spec[name]).open('rb') as handle:
                return pickle.load(handle)
        self.imputers = fitted('imputer')
        item = spec['interpolation_owner']
        ck = torch.load(checked(item), weights_only=True, map_location='cpu')
        self.sprint_owner = retained.SprintOwner(**ck['architecture'])
        self.sprint_owner.load_state_dict(ck['state_dict']); self.sprint_owner.eval()
        self.sprint_temperature = item['temperature']
        item = spec['memory']
        if (item['channels'], item['stride'], item['context']) != (103, 5, 64):
            raise ValueError('Full memory contract mismatch')
        self.memory = retained.ControlMemory(item['channels'], item['width'])
        self.memory.load_state_dict(torch.load(checked(item), weights_only=True, map_location='cpu')); self.memory.eval()
        self.memory_temperature = item['temperature']
        self.carrier = fitted('carrier')
        self.carrier_freeze = {'temperature': spec['carrier']['temperature']}
        self.carrier_weight = spec['carrier']['weight']
        self.legacy = fitted('legacy')
        item = spec['temporal_status']
        ck = torch.load(checked(item), weights_only=True, map_location='cpu')
        self.temporal_status = retained.TemporalLiveDead(ck['channels'], ck['width'])
        self.temporal_status.load_state_dict(ck['state_dict']); self.temporal_status.eval()
        self.status_temperature = item['temperature']
        self.context = fitted('context_status')
        self.context_calibration = spec['context_status']['calibration']
        self.status_fit = spec['reliability']['fit']
        self.status_selection = spec['reliability']['selection']
        if self.status_selection['dead_cost'] != 50 or not self.status_selection['B_endpoints']:
            raise ValueError('Full reliability/endpoints contract mismatch')
        self.boundary_models = fitted('boundary')
        self.boundary_config = spec['boundary']['config']
        self.quality = fitted('quality')
        if self.quality.n_features_in_ != 175:
            raise ValueError('Full graph quality must have 175 features')
        item = spec['joint']
        if (item['weight'], item['scope']) != (.25, 'missing25'):
            raise ValueError('Full joint composition contract mismatch')
        ck = torch.load(checked(item), weights_only=True, map_location='cpu')
        self.joint = retained.JointControllerBall(ck['width'], ck['feedback'])
        self.joint.load_state_dict(ck['state_dict']); self.joint.eval()
        self.joint_freeze = item


def gap_candidate(raw, baseline, expert_logit, quality):
    """One fixed gap blend and a final fence preserving every other frame."""
    missing = ~np.isfinite(raw['ball']).all(1)
    support = uniform_filter1d(missing.astype(np.float64), 25, mode='nearest')
    eligible = (support > 0) & (baseline['predicted'] < 2)
    # This path is mathematically identity for clean inputs and bypasses a
    # second graph traversal, guaranteeing exact original predictions.
    if not missing.any():
        return {'predicted': baseline['predicted'].copy(), 'logit': baseline['logit'].copy(),
                'eligible_mask': eligible, 'gap_weight': support*.5}
    signals = baseline['signals'].copy()
    weight = .5*support
    take = weight > 0
    probability = expit(signals[:, 0])
    probability[take] = (1-weight[take])*probability[take]+weight[take]*expit(expert_logit[take])
    signals[take, 0] = logit(np.clip(probability[take], 1e-7, 1-1e-7))
    decoded, arcs, q = retained.decode_graph(signals, baseline['status_mask'], quality)
    predicted = baseline['predicted'].copy()
    predicted[eligible] = decoded[eligible]
    assert np.array_equal(predicted[~eligible], baseline['predicted'][~eligible])
    return {'predicted': predicted, 'logit': signals[:, 0], 'signals': signals,
            'segments': arcs, 'quality': q, 'eligible_mask': eligible, 'gap_weight': weight}
