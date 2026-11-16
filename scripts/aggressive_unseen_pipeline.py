"""Raw-half frozen baseline inference, with reference-free WQQ replay checks.

Raw inference has no match key and cannot consume saved prediction/feature
caches. WQQ outputs are opened only after fresh inference, as verification
targets. All reference arrays and every reserved-match path remain blocked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'private/improvement/runtime'), str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from second_zero.common import json_write, sha256
from second_zero.contextual import features as legacy_features, binary_decode
from second_zero.temporal_ownership import TemporalOwner, trajectory_features, infer_logits as temporal_owner_infer
from second_zero.temporal_live_dead import TemporalLiveDead, trajectory_features as status_features, infer_logits as temporal_status_infer
from second_zero.sprint_model_owner import SprintOwner, infer_logits as sprint_owner_infer
from second_zero.sprint_model_interpolation import interpolated_features
from second_zero.overnight_boundaries import propose_moves, apply_moves
from second_zero.joint_controller_ball import JointControllerBall, infer_raw as joint_infer
from scripts.sprint_memory import ControlMemory, pack as memory_pack, infer as memory_infer
from scripts.sprint_ball_imputer import predict as impute_ball
from scripts.sprint_carrier_specialist import carrier_features
from scripts.sprint_gap_specialist import predict as carrier_infer
from scripts.sprint_status_context import infer as context_status_infer
from scripts.evaluate_temporal_ownership import fixed_state_owner
from scripts.train_contextual import corruption
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts import sprint_decoder_lattice as lattice

SPRINT = ROOT/'runs/two-hour-sprint'
RAW_DATA = ROOT/'private/improvement/external/prepared'
UNIFORM = [ROOT/'runs/ownership-temporal-v1', ROOT/'runs/next-evidence-seeds/20261024-uniform',
           ROOT/'runs/next-evidence-seeds/20261025-uniform']
SHORT = ROOT/'runs/extended-short-v2'
JOINT = ROOT/'runs/joint-controller-ball/joint-v1/step4000'
PHASE = 'inference'
PINNED = {}
OPENED = set()
FRESH_OUTPUTS = set()
REPLAY_TARGETS = set()
RESERVED_AUTHORIZED = False


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    if '.truth.npy' in path or '/evaluation/gt/' in path or ('J03WR9' in path and not RESERVED_AUTHORIZED):
        raise PermissionError('Raw pipeline cannot open references or reserved-match paths')
    if PHASE == 'inference' and 'J03WQQ' in path and path.endswith(('.npz', '.npy')):
        if str(RAW_DATA) not in path and str(Path(path).resolve()) not in FRESH_OUTPUTS:
            raise PermissionError('Fresh inference may not consume WQQ prediction/feature caches')
    try:
        p = Path(path).resolve()
        if p.is_relative_to(ROOT) and p.is_file():
            OPENED.add(p)
    except (OSError, ValueError):
        pass


def pin(path):
    path = Path(path).resolve()
    PINNED[str(path.relative_to(ROOT))] = sha256(path)
    return path


def meta(path):
    return json.loads(pin(path).read_text())


def fitted_pickle(folder, filename='model.pkl', hash_key='model_sha256'):
    frozen = meta(folder/'freeze.json')
    path = pin(folder/filename)
    assert sha256(path) == frozen[hash_key]
    if frozen.get('protocol_sha256'):
        assert sha256(pin(folder/'protocol.json')) == frozen['protocol_sha256']
    with path.open('rb') as handle:
        return pickle.load(handle), frozen


def compare(actual, expected):
    assert actual.shape == expected.shape
    delta = np.abs(np.asarray(actual, np.float64)-np.asarray(expected, np.float64))
    return {'bit_exact':bool(np.array_equal(actual, expected)),
            'different_values':int(np.count_nonzero(actual != expected)),
            'max_abs_error':float(np.max(delta)), 'mean_abs_error':float(np.mean(delta))}


def physical_geometry(raw, observed, imputers):
    filled, _ = impute_ball(raw, imputers)
    linear = interpolated_features(raw['ball'], raw['players'], raw['team'], raw['pitch'])
    learned = interpolated_features(filled, raw['players'], raw['team'], raw['pitch'])
    unaries, reliability = [], []
    for x, measured in ((observed, True), (linear, False), (learned, False)):
        distance, velocity = x[:,:,0]*20, x[:,:,3]*30
        both = (x[:,:,49 if measured else 18] < .5).all(1)
        confidence = np.where(both, np.exp(-np.min(distance, axis=1)/10), 0)
        unary = np.clip((distance[:,0]-distance[:,1])/3+.25*(velocity[:,0]-velocity[:,1])/10, -4, 4)*confidence
        unary[~both] = 0
        unaries.append(unary); reliability.append(confidence)
    return np.c_[*unaries, *reliability].astype(np.float32)


def decode_graph(signals, base, quality):
    z = signals[:,0]
    arcs = graph.proposals(z, base, signals[:,4], signals[:,7:10])
    arcs = lattice.proposals(z, base, arcs, 16)
    q = np.clip(quality.predict(graph.segment_features(signals, arcs)), 0, 1)
    predicted = graph.graph_decode(z, base, arcs, q, .55, .01)
    assert np.array_equal(predicted[base >= 2], base[base >= 2])
    return predicted, arcs, q


class FrozenRawPipeline:
    """Pure raw-observation API; fitted artifacts load once, no match IDs."""
    def __init__(self):
        self.owners = []
        self.inventory = {}
        for name, folder in [(f'uniform{i}', p) for i,p in enumerate(UNIFORM)] + [('short', SHORT)]:
            frozen = meta(folder/'freeze.json')
            assert sha256(pin(folder/'model.pt')) == frozen['model_sha256']
            assert sha256(pin(folder/'protocol.json')) == frozen['protocol_sha256']
            checkpoint = torch.load(folder/'model.pt', weights_only=True, map_location='cpu')
            model = TemporalOwner(checkpoint['channels'], checkpoint['width'])
            model.load_state_dict(checkpoint['state_dict']); model.eval()
            self.owners.append((model, frozen['temperature']))
            self.inventory[name] = {'folder':str(folder.relative_to(ROOT)), 'temperature':frozen['temperature'], 'model_sha256':frozen['model_sha256']}
        self.imputers, imputer_freeze = fitted_pickle(SPRINT/'ball-imputer-v1', 'models.pkl', 'models_sha256')
        owner_folder = SPRINT/'model/interp-full'
        complete = meta(owner_folder/'complete.json')
        assert sha256(pin(owner_folder/'model.pt')) == complete['model_sha256']
        self.sprint_temperature = meta(owner_folder/'calibration.json')['temperature']
        ck = torch.load(owner_folder/'model.pt', weights_only=True, map_location='cpu')
        self.sprint_owner = SprintOwner(**ck['architecture'])
        self.sprint_owner.load_state_dict(ck['state_dict']); self.sprint_owner.eval()
        learned = SPRINT/'model/interp_learned_swap-full'
        learned_meta = meta(learned/'complete.json')
        assert sha256(pin(learned/'model.pt')) == complete['model_sha256'] == learned_meta['model_sha256']
        assert meta(learned/'calibration.json')['temperature'] == self.sprint_temperature
        assert learned_meta['imputer_sha256'] == imputer_freeze['models_sha256']
        self.inventory['linear_and_learned_owner'] = {'model_sha256':complete['model_sha256'], 'temperature':self.sprint_temperature,
             'geometry_rules':['bounded linear', 'learned imputer residual75percent; original gap flags retained']}
        gru = SPRINT/'memory/gru-interp-v3'
        gp, gf = meta(gru/'protocol.json'), meta(gru/'freeze.json')
        assert gp['stride'] == 5 and gp['context'] == 64 and gp['channels'] == 103
        assert sha256(pin(gru/'model.pt')) == gf['model_sha256']
        assert sha256(gru/'protocol.json') == gf['protocol_sha256']
        self.memory = ControlMemory(gp['channels'], gp['width'])
        self.memory.load_state_dict(torch.load(gru/'model.pt', weights_only=True, map_location='cpu')); self.memory.eval()
        self.memory_temperature = gf['temperature']
        self.carrier, self.carrier_freeze = fitted_pickle(SPRINT/'carrier-specialist-v1')
        mixture = meta(SPRINT/'carrier-mixture-v1/freeze.json')
        assert sha256(SPRINT/'carrier-specialist-v1/freeze.json') == mixture['carrier_freeze_sha256']
        self.carrier_weight = mixture['WPY_weight']
        self.legacy, _ = fitted_pickle(ROOT/'runs/contextual-external-v1', 'models.pkl', 'models_sha256')
        status = ROOT/'runs/live-dead-temporal-v1'
        sf = meta(status/'freeze.json')
        assert sha256(pin(status/'model.pt')) == sf['model_sha256']
        ck = torch.load(status/'model.pt', weights_only=True, map_location='cpu')
        self.temporal_status = TemporalLiveDead(ck['channels'], ck['width'])
        self.temporal_status.load_state_dict(ck['state_dict']); self.temporal_status.eval()
        self.status_temperature = sf['temperature']
        self.context, context_freeze = fitted_pickle(SPRINT/'status-context-v1')
        self.context_calibration = meta(SPRINT/'status-context-v1/calibration.json')
        assert sha256(SPRINT/'status-context-v1/calibration.json') == context_freeze['calibration_sha256']
        self.status_fit = meta(SPRINT/'status-mixture-v1/mixture_fit.json')
        mixture_calibration = meta(SPRINT/'status-mixture-v1/reliability/calibration.json')
        mixture_freeze = meta(SPRINT/'status-mixture-v1/reliability/freeze.json')
        assert sha256(SPRINT/'status-mixture-v1/reliability/calibration.json') == mixture_freeze['calibration_sha256']
        assert self.status_fit == mixture_calibration['fit']
        self.status_selection = mixture_calibration['selected']
        self.boundary_models, bf = fitted_pickle(ROOT/'runs/overnight-boundary-v1', 'models.pkl', 'models_sha256')
        self.boundary_config = bf['config']
        self.quality, _ = fitted_pickle(graph.RUN)
        self.joint_freeze = meta(JOINT/'scope-freeze.json')
        fit = meta(JOINT/'freeze-fit.json')
        assert sha256(pin(JOINT/'model.pt')) == fit['model_sha256'] == self.joint_freeze['model_sha256']
        ck = torch.load(JOINT/'model.pt', weights_only=True, map_location='cpu')
        self.joint = JointControllerBall(ck['width'], ck['feedback'])
        self.joint.load_state_dict(ck['state_dict']); self.joint.eval()
        self.inventory.update({'gru':{'model_sha256':gf['model_sha256'], 'temperature':self.memory_temperature, 'experts':'uniform0/1/2,short,linear in that order'},
             'imputer':imputer_freeze, 'carrier':{'weight':self.carrier_weight, **self.carrier_freeze},
             'status':{'selection':self.status_selection, 'source':'legacy/context/temporal reliability mixture, then frozen onset/restart models'},
             'joint':self.joint_freeze, 'quality_model':str((graph.RUN/'model.pkl').relative_to(ROOT)),
             'frozen_runtime_threads':{'uniform_short_linear_learned_owners':3, 'temporal_status':3,
                                      'GRU_and_joint':2, 'HGB_BLAS':2}})

    def predict(self, raw, progress=None):
        # Deliberately accept observations only, discarding no label-bearing
        # fields silently: callers must provide exactly this four-key contract.
        if set(raw) != {'ball','players','team','pitch'}:
            raise ValueError('Raw half must contain exactly ball,players,team,pitch')
        if set(np.unique(raw['team'])) != {0,1}:
            raise ValueError('Both explicit mapped teams are required')
        tick = progress or (lambda name:None)
        n = len(raw['ball'])
        context = legacy_features(raw['ball'], raw['players'], raw['team'], raw['pitch'])
        observed = trajectory_features(raw['ball'], raw['players'], raw['team'], raw['pitch'], contextual=context)
        status_x = status_features(raw['ball'], raw['players'], raw['team'], raw['pitch'])
        result = {}
        z = []
        # Original owners were inferred with three Torch threads. Matching the
        # recorded runtime avoids tiny convolution reductions changing discrete
        # downstream graph paths. GRU/joint used two threads in their drivers.
        torch.set_num_threads(3)
        for i, (model, temperature) in enumerate(self.owners):
            native = temporal_owner_infer(model, observed)
            value = native/temperature
            result[f'owner_raw_{i}'] = native
            result[f'owner_{i}'] = value
            z.append(value.astype(np.float32)); tick('owner'+str(i))
        uniform = np.stack(z[:3], axis=1)
        linear = interpolated_features(raw['ball'], raw['players'], raw['team'], raw['pitch'])
        linear_z = sprint_owner_infer(self.sprint_owner, np.concatenate((observed, linear), -1))/self.sprint_temperature
        result['linear_owner'] = linear_z
        filled75, _ = impute_ball(raw, self.imputers, blend=.75)
        learned = interpolated_features(filled75, raw['players'], raw['team'], raw['pitch'])
        learned[:,:,12:17] = linear[:,:,12:17]
        learned_z = sprint_owner_infer(self.sprint_owner, np.concatenate((observed, learned), -1))/self.sprint_temperature
        result['learned_owner'] = learned_z; tick('imputed_owner')
        torch.set_num_threads(2)
        experts = np.column_stack((*z, linear_z)).astype(np.float32)
        gru_raw = memory_infer(self.memory, memory_pack(observed, experts))
        gru_z = gru_raw/self.memory_temperature
        result['gru_raw'], result['gru_owner'] = gru_raw, gru_z
        pair = logit(np.clip(.5*expit(np.asarray(gru_z, np.float32))+.5*expit(np.asarray(learned_z, np.float32)), 1e-8, 1-1e-8))
        carrier_x, eligible = carrier_features(raw)
        carrier_raw = np.zeros(n, np.float32)
        if carrier_x is not None:
            carrier_raw[eligible] = carrier_infer(self.carrier, carrier_x)
        probability = expit(pair)
        probability[eligible] = ((1-self.carrier_weight)*probability[eligible]
                                 +self.carrier_weight*expit(carrier_raw[eligible]/self.carrier_freeze['temperature']))
        carrier_z = logit(np.clip(probability, 1e-8, 1-1e-8))
        result['carrier_owner'], result['carrier_raw'], result['carrier_eligible'] = carrier_z, carrier_raw, eligible
        tick('GRU_carrier')
        legacy = np.where(context['ball_seen'], self.legacy['dead'].predict_proba(context['dead'])[:,1],
                          self.legacy['player'].predict_proba(context['player'])[:,1])
        legacy_logit = logit(np.clip(legacy, .01, .99))
        torch.set_num_threads(3)
        temporal = temporal_status_infer(self.temporal_status, status_x)/self.status_temperature
        torch.set_num_threads(2)
        context_raw = context_status_infer(self.context, status_x)
        context_z = context_raw*self.context_calibration['slope']+self.context_calibration['intercept']
        sources = np.c_[context_z, legacy_logit, temporal]
        seen = status_x[:,36] > 0
        dead = np.empty(n)
        for label, mask in [('observed', seen), ('missing', ~seen)]:
            weights = np.array(self.status_fit[label]['weights'])
            dead[mask] = sources[mask]@weights[:3]+weights[3]
        state = np.where(binary_decode(np.clip(dead, -np.log(99), np.log(99)), self.status_selection['dead_cost']),2,0).astype(np.int8)
        if self.status_selection['B_endpoints']:
            moves = propose_moves(self.boundary_models, status_x, state, self.boundary_config['radii'])
            state, _ = apply_moves(state, moves, self.boundary_config['thresholds'])
        # This old owner is used only to reproduce saved status H/A labels;
        # graph inference consumes its D/X mask, not these live labels.
        average64 = np.mean([expit(np.asarray(v, np.float64)) for v in z[:3]], axis=0, dtype=np.float64)
        safe64 = np.clip(average64, 1e-12, 1-1e-12)
        status_owner = np.log(safe64)-np.log1p(-safe64)
        status = fixed_state_owner(state, status_owner)
        result.update(legacy_p_dead=legacy, temporal_dead_logit=temporal, context_raw_dead_logit=context_raw,
                      reliability_dead_logit=dead, status_mask=status, status_owner=status_owner)
        tick('status')
        physical = physical_geometry(raw, observed, self.imputers)
        x = np.c_[carrier_z, z[3], uniform.std(1), np.mean(np.sign(uniform),axis=1),
                  status_x[:,[36,38,39]], physical].astype(np.float32)
        sprint, arcs, quality = decode_graph(x, status, self.quality)
        result.update(sprint_signals=x.copy(), sprint_predicted=sprint,
                      sprint_segments=arcs, sprint_quality=quality, physical_geometry=physical)
        tick('sprint_graph')
        joint = joint_infer(self.joint, raw)
        result['joint_raw_owner'] = joint['owner']
        old = x[:,0].copy(); p = expit(old)
        gate = uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64), 25, mode='nearest')
        incoming = logit(np.clip((1-gate)*p+gate*expit(joint['owner']/self.joint_freeze['temperature']),1e-7,1-1e-7))
        mask = gate > 0
        p[mask] = ((1-self.joint_freeze['weight'])*p[mask]
                   +self.joint_freeze['weight']*expit(incoming[mask]))
        x[:,0] = logit(np.clip(p,1e-7,1-1e-7))
        parent, arcs, quality = decode_graph(x, status, self.quality)
        result.update(predicted=parent, logit=x[:,0], signals=x,
                      segments=arcs, quality=quality)
        tick('joint_graph')
        return result


def replay(key, result):
    global PHASE
    PHASE = 'replay'
    targets = []
    for i,folder in enumerate(UNIFORM+[SHORT]):
        targets += [(f'owner_raw_{i}', folder/(key+'.npz'), 'raw_logit'),
                    (f'owner_{i}', folder/(key+'.npz'), 'logit')]
    targets += [('linear_owner', SPRINT/'model/interp-full'/(key+'.npz'),'logit'),
                ('learned_owner', SPRINT/'model/interp_learned_swap-full'/(key+'.npz'),'logit'),
                ('gru_raw', SPRINT/'memory/gru-interp-v3'/(key+'.npz'),'raw_logit'),
                ('carrier_owner', SPRINT/'carrier-mixture-v1'/(key+'.npz'),'logit'),
                ('temporal_dead_logit', ROOT/'runs/live-dead-temporal-v1'/(key+'.npz'),'dead_logit'),
                ('context_raw_dead_logit', SPRINT/'status-context-v1'/(key+'.npz'),'raw_dead_logit'),
                ('legacy_p_dead', ROOT/'runs/overnight-boundary-v1'/(key+'.npz'),'p_dead'),
                ('reliability_dead_logit', SPRINT/'status-mixture-v1/reliability'/(key+'.npz'),'dead_logit'),
                ('status_mask', SPRINT/'status-mixture-v1/reliability'/(key+'.npz'),'predicted'),
                ('sprint_predicted', SPRINT/'final'/(key+'.npz'),'predicted'),
                ('joint_raw_owner', JOINT/'raw'/(key+'.npz'),'owner'),
                ('logit', JOINT/'WQQ-scope'/(key+'.npz'),'logit'),
                ('predicted', JOINT/'WQQ-scope'/(key+'.npz'),'predicted')]
    checks = {}
    for name,path,field in targets:
        REPLAY_TARGETS.add(str(path.resolve().relative_to(ROOT)))
        with np.load(pin(path)) as data:
            checks[name] = compare(result[name], data[field])
    p = SPRINT/'decoder/physical-geometry-cache'/(key+'.npy')
    REPLAY_TARGETS.add(str(p.resolve().relative_to(ROOT)))
    checks['physical_geometry'] = compare(result['physical_geometry'], np.load(pin(p)))
    PHASE = 'inference'
    return checks


def main(args):
    global PHASE, RESERVED_AUTHORIZED
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pipeline = FrozenRawPipeline()
    source_paths = [Path(__file__), ROOT/'second_zero/temporal_ownership.py', ROOT/'second_zero/temporal_live_dead.py',
                    ROOT/'second_zero/contextual.py', ROOT/'second_zero/sprint_model_owner.py',
                    ROOT/'second_zero/sprint_model_interpolation.py', ROOT/'second_zero/joint_controller_ball.py',
                    ROOT/'second_zero/overnight_boundaries.py', ROOT/'scripts/sprint_memory.py',
                    ROOT/'scripts/sprint_status_context.py', ROOT/'scripts/sprint_carrier_specialist.py',
                    ROOT/'scripts/sprint_ball_imputer.py', Path(graph.__file__), Path(lattice.__file__)]
    for path in source_paths:
        pin(path)
    json_write(out/'inventory.json', {'components':pipeline.inventory, 'fitted_inputs_sha256':PINNED.copy(),
               'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in source_paths},
               'missing_weights':[], 'requires_retraining':False, 'raw_contract':'ball[N,2],players[N,P,2]metres;team[P]0/1;pitch[2];25Hzsinglephysicalhalf',
               'reserved_access':False, 'reference_access':False})
    if args.inventory_only:
        print('Frozen raw pipeline inventory complete; no match input opened', flush=True)
        return
    if args.reserved_inference_authorization is not None:
        # This route is dormant unless a later explicit root authorization
        # supplies a receipt linking the already selected candidate and this
        # exact model/source inventory. Reference arrays remain prohibited.
        authorization = meta(args.reserved_inference_authorization)
        assert authorization['reserved_inputs_authorized'] is True
        assert authorization['pipeline_inventory_sha256'] == sha256(out/'inventory.json')
        selection_path = ROOT/authorization['selection_freeze']
        assert sha256(pin(selection_path)) == authorization['selection_freeze_sha256']
        replay_path = ROOT/authorization['reference_free_replay_completion']
        assert sha256(pin(replay_path)) == authorization['reference_free_replay_completion_sha256']
        replay_metadata = json.loads(replay_path.read_text())
        assert replay_metadata['pipeline_inventory_sha256'] == sha256(out/'inventory.json')
        assert replay_metadata['replay_final_labels_exact'] and len(replay_metadata['replay']) == 4
        RESERVED_AUTHORIZED = True
    keys = args.replay_keys or [None]
    all_checks, elapsed = {}, {}
    for key in keys:
        started = time.monotonic()
        PHASE = 'inference'
        if key is not None:
            assert key.startswith('J03WQQ-') and key.endswith(('-clean','-corrupted'))
            name, condition = key.rsplit('-',1)
            path = RAW_DATA/(name+'.npz')
        else:
            if args.input is None:
                raise ValueError('Provide --input for a new raw half')
            path, condition = args.input, 'clean'
        with np.load(pin(path)) as data:
            raw = {k:data[k] for k in ('ball','players','team','pitch')}
        if condition == 'corrupted':
            raw = corruption(raw,100)
        label = key or 'new-half'
        def progress(stage):
            print('RAW PIPELINE', label, stage, round(time.monotonic()-started,1), 'seconds', flush=True)
        result = pipeline.predict(raw, progress)
        # Freeze fresh output BEFORE any old cache is read for replay comparison.
        dest = out/(label+'.npz')
        FRESH_OUTPUTS.add(str(dest))
        # Temporarily allow this newly computed output filename; its bytes are
        # written, never consumed as a source of its own predictions.
        PHASE = 'write'
        np.savez_compressed(dest, **result)
        PHASE = 'inference'
        json_write(out/(label+'-prediction-freeze.json'), {'output_sha256':sha256(dest),
                   'raw_input_sha256':sha256(path), 'reference_reads':[], 'fresh_raw_inference':True})
        if key is not None:
            all_checks[key] = replay(key, result)
            for field in ('sprint_predicted', 'predicted', 'status_mask'):
                assert all_checks[key][field]['bit_exact'], 'Frozen raw replay failed: '+key+'/'+field
        elapsed[label] = time.monotonic()-started
    PHASE = 'verification'
    for path,expected in PINNED.items():
        assert sha256(ROOT/path) == expected
    json_write(out/'completion.json', {'replay':all_checks,
               'inference_input_sha256':{p:h for p,h in PINNED.items() if p not in REPLAY_TARGETS},
               'verification_target_sha256':{p:h for p,h in PINNED.items() if p in REPLAY_TARGETS},
               'outputs_sha256':{p.name:sha256(p) for p in out.glob('*.npz')},
               'pipeline_inventory_sha256':sha256(out/'inventory.json'),
               'replay_final_labels_exact':bool(all_checks) and all(v[n]['bit_exact'] for v in all_checks.values()
                                      for n in ('sprint_predicted','predicted','status_mask')),
               'seconds':elapsed, 'reference_arrays_opened':[], 'reserved_inputs_opened':RESERVED_AUTHORIZED,
               'training_performed':False, 'metrics_computed':False,
               'scope':'Every baseline upstream model/geometry/status/boundary/quality component ran from raw tracking. Saved WQQ arrays served only as after-inference verification targets.'})
    print('RAW PIPELINE COMPLETE', json.dumps({k:{n:v[n] for n in ('sprint_predicted','predicted','status_mask')} for k,v in all_checks.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'runs/aggressive-20261029/unseen-pipeline-WQQ')
    p.add_argument('--input', type=Path)
    p.add_argument('--inventory-only', action='store_true')
    p.add_argument('--replay-keys', nargs='+')
    p.add_argument('--reserved-inference-authorization', type=Path)
    args = p.parse_args()
    sys.addaudithook(guard); torch.set_num_threads(2); threadpool_limits(2)
    main(args)
