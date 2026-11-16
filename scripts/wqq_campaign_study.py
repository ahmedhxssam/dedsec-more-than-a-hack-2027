"""Registered, restartable native-model studies with sealed calibration and OOF.

Prediction and reference-reading phases are disjoint. The benchmark condition
only constructs observations. Every physical producer selects one WPY policy.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'private/improvement/runtime')]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts import wqq_campaign_model as model
from scripts import wqq_campaign_parents as parents
from scripts import wqq_campaign_evaluate as evaluation

CAMPAIGN = ROOT/'runs/wqq-090-20261031'
_TRUTH = set()
_FULL = False


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    if any(s in path for s in ('J03WR9', '/evaluation/gt/', '/runs/main/', '.actor.npy')):
        raise PermissionError('Reserved references remain closed')
    if 'J03WQQ' in path and not _FULL:
        raise PermissionError('WQQ requires a frozen nomination')
    if '.truth.npy' in path:
        match = Path(path).name.split('-')[0]
        if match not in _TRUTH:
            raise PermissionError('Reference opened outside its sealed scoring phase')


@contextlib.contextmanager
def references(matches):
    global _TRUTH
    previous = _TRUTH
    _TRUTH = set(matches)
    try:
        yield
    finally:
        _TRUTH = previous


def pin(path):
    return model.pin(path)


def read(path):
    return model.read_json(path)


def emit(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def source_pins():
    paths = [Path(__file__), Path(model.__file__), Path(parents.__file__), Path(evaluation.__file__)]
    return {model.relative(p): model.sha256(p) for p in paths}


def register(study, family):
    study = Path(study).resolve()
    study.relative_to(CAMPAIGN/'studies')
    model_protocol = model.DEFAULT_WORK/'protocol.json'
    model.verify_protocol()
    policies = [{**p, 'id':p['policy_id']} for p in model.policy_grid()]
    recipe = {'model_family':family, 'model_protocol':pin(model_protocol),
              'parent':'condition-correct frozen v20 with nested calibration ancestry',
              'policies':policies, 'selection':'one equal-condition WPY policy per producer',
              'transition':'unchanged v70 typed hazards and nonnegative costs',
              'conditions':['clean','corrupted'], 'corruption_seed':100}
    protocol = {'kind':'wqq-campaign-study-v1', 'candidate_id':study.name,
                'recipe':recipe, 'source_hashes':source_pins(),
                'train_matches':list(model.TRAIN), 'calibration_matches':['J03WPY'],
                'target_match':'J03WQQ', 'target_role':'one audit after OOF nomination only',
                'controls':['v70','v69'], 'primary_metric':'equal clean/corrupted official five-IoU full-match SpellF1',
                'cohort_selection':'physical OOF mean; pair results diagnostic',
                'no_human_challenge_annotations':True}
    model.atomic_json(study/'protocol.json', protocol, immutable=True)
    return protocol


def protocol_for(study):
    protocol = read(study/'protocol.json')
    model.verify_sources(protocol['source_hashes'])
    model.checked(protocol['recipe']['model_protocol'])
    if protocol['candidate_id'] != study.name:
        raise ValueError('Study identity changed')
    return protocol


def add_nodes(target, additions):
    for name, node in additions.items():
        if name in target and target[name] != node:
            raise ValueError('Conflicting ancestry node identity')
        target[name] = node


def weight_nodes(freeze_pin, nodes):
    """Bind every learned initialization/context ancestor to its own role receipt."""
    path = model.checked(freeze_pin)
    name = 'weight-'+freeze_pin['sha256']
    if name in nodes:
        return name
    freeze = read(path)
    field = 'fit_matches' if 'fit_matches' in freeze else 'residual_fit_matches'
    fit = freeze[field]
    dependencies = []
    for key in ('weight_initialization', 'context_teacher'):
        ancestor = freeze.get(key)
        if ancestor:
            if 'freeze' not in ancestor:
                raise ValueError('Learned ancestor lacks a complete role receipt')
            dependencies.append(weight_nodes(ancestor['freeze'], nodes))
    artifacts = [freeze_pin, freeze['model']]
    for key in ('protocol','training_seal'):
        if key in freeze:
            model.checked(freeze[key]); artifacts.append(freeze[key])
    nodes[name] = {'kind':'weights','fit_matches':fit,'calibration_matches':[],
                   'dependencies':dependencies, 'artifacts':artifacts,
                   'role_receipt':{**freeze_pin,'fit_field':field},
                   'artifact_bindings':[{'field':'model.sha256','sha256':freeze['model']['sha256']}]}
    return name


def checkpoint(study, marker):
    path = study/'progress.json'
    model.atomic_json(path, {'kind':'wqq-campaign-inference-progress-v1', 'marker':marker,
                             'protocol':pin(study/'protocol.json')})
    if os.environ.get('WQQ_CAMPAIGN_CHECKPOINT_FILE'):
        model.atomic_json(Path(os.environ['WQQ_CAMPAIGN_CHECKPOINT_FILE']),
                          {'checkpoint_path':model.relative(path),'sha256':model.sha256(path)})
    if model.stop_requested():
        raise model.CheckpointedStop('Inference stopped between sealed view artifacts')


def produce_view(study, which, match, half, condition, policies, permit=None):
    protocol = protocol_for(study)
    family = protocol['recipe']['model_family']
    directory = study/'views'/which
    destination = directory/f'{match}-{half}-{condition}.npz'
    metadata_path = destination.with_suffix('.json')
    identity = {'which':which,'match':match,'half':half,'condition':condition,
                'protocol':pin(study/'protocol.json'),'policies':policies}
    if metadata_path.exists():
        meta = read(metadata_path)
        if meta['identity'] != identity:
            raise ValueError('View resume identity mismatch')
        model.checked(meta['artifact'])
        for item in meta['inputs']: model.checked(item)
        return meta
    if destination.exists():
        raise FileExistsError('Unsealed composed output requires inspection: '+str(destination))
    logits_path = study/'logits'/which/f'{match}-{half}-{condition}.npz'
    torch.set_num_threads(1)
    logits_meta = model.predict_logits(family, which, match, half, condition, logits_path, permit=permit)
    parent, parent_meta = parents.get_parent(which, match, half, condition, nomination=permit)
    with np.load(logits_path, allow_pickle=False) as z:
        logits = z['logits'].copy()
    arrays = {p['id']:model.compose(parent, logits, p['alpha'], p['parent_confidence']) for p in policies}
    model.atomic_npz(destination, **arrays)
    nodes = {}
    native_roots = [weight_nodes(a['freeze'], nodes) for a in logits_meta['model_bundle']['learned_ancestors']]
    parent_name = 'parent-'+parent_meta['manifest']['sha256']
    nodes[parent_name] = {'kind':'historical_parent','fit_matches':parent_meta['fit_matches'],
                          'calibration_matches':parent_meta['calibration_matches'], 'dependencies':[],
                          'artifacts':[parent_meta['manifest'],parent_meta['artifact'],parent_meta['source']],
                          'role_receipt':parent_meta['manifest']}
    native_name = 'logits-'+logits_meta['output']['sha256']
    native_receipt = pin(logits_path.with_suffix('.json'))
    nodes[native_name] = {'kind':'pipeline','fit_matches':[],'calibration_matches':[],
                          'dependencies':native_roots, 'artifacts':[native_receipt,logits_meta['output'],logits_meta['raw_input'],
                           *[{'path':p,'sha256':s} for p,s in logits_meta['sources'].items()]]}
    output_name = 'composition-'+pin(destination)['sha256']+'-'+which+'-'+match+'-'+half+'-'+condition
    nodes[output_name] = {'kind':'pipeline','fit_matches':[],'calibration_matches':[],
                          'dependencies':[parent_name,native_name], 'artifacts':[pin(destination),pin(study/'protocol.json')]}
    meta = {'kind':'wqq-campaign-composition-v1','identity':identity,'artifact':pin(destination),
            'inputs':[native_receipt, parent_meta['manifest'], pin(study/'protocol.json')],
            'ancestry_nodes':nodes,'ancestry_roots':[output_name], 'reference_reads':[],
            'arrays':{p['id']:p['id'] for p in policies}, 'frames':len(parent)}
    model.atomic_json(metadata_path, meta, immutable=True)
    emit(stage='prediction', study=study.name, which=which, match=match, half=half, condition=condition, frames=len(parent))
    checkpoint(study, model.relative(metadata_path))
    return meta


def row_for(meta, selection=None):
    row = {k:meta['identity'][k] for k in ('match','half','condition')}
    row.update(artifact=meta['artifact'],arrays=meta['arrays'],ancestry_roots=meta['ancestry_roots'])
    if selection: row['policy_selection'] = selection
    return row


def calibrate(study, which, permit=None):
    protocol = protocol_for(study)
    folder = study/'calibration'/which
    seal, selection = folder/'seal.json', folder/'selection.json'
    if seal.exists():
        evaluation.load_seal(seal, stage='calibration')
    else:
        rows, nodes = [], {}
        for half in model.HALVES:
            for condition in ('clean','corrupted'):
                meta = produce_view(study, which, 'J03WPY', half, condition, protocol['recipe']['policies'], permit)
                rows.append(row_for(meta)); add_nodes(nodes,meta['ancestry_nodes'])
        evaluation.create_seal(seal, run_id=study.name, stage='calibration', rows=rows,
                               policies=protocol['recipe']['policies'], protocol=pin(study/'protocol.json'),
                               ancestry_nodes=nodes, producer_held=which)
    if selection.exists():
        result = read(selection)
        if result['prediction_seal'] != pin(seal) or result['producer_held'] != which:
            raise ValueError('Calibration receipt belongs to a different producer')
    else:
        with references(['J03WPY']):
            result = evaluation.select_policy_from_seal(seal, out=selection)
    emit(stage='calibration',study=study.name,which=which,selected=result['selected_id'],score=result['selected_score'])
    return result, pin(selection), read(seal)


def produce_scored_match(study, which, match, permit=None):
    result, selection, cal_seal = calibrate(study,which,permit)
    policy = {**result['selected'],'id':'candidate'}
    nodes = dict(cal_seal['ancestry_nodes'])
    calibration_name = 'calibration-'+selection['sha256']
    roots = list(dict.fromkeys(n for row in cal_seal['rows'] for n in row['ancestry_roots']))
    nodes[calibration_name] = {'kind':'calibration','fit_matches':[],'calibration_matches':['J03WPY'],
                               'dependencies':roots, 'artifacts':[selection,result['prediction_seal']],
                               'role_receipt':selection}
    rows = []
    for half in model.HALVES:
        for condition in ('clean','corrupted'):
            meta = produce_view(study,which,match,half,condition,[policy],permit)
            add_nodes(nodes,meta['ancestry_nodes'])
            row = row_for(meta,selection)
            row['ancestry_roots'] = row['ancestry_roots']+[calibration_name]
            rows.append(row)
    inventory = {'rows':rows,'ancestry_nodes':nodes,'selection':selection,
                 'protocol':pin(study/'protocol.json'),'which':which}
    path = study/'inventories'/f'{which}.json'
    model.atomic_json(path,inventory,immutable=True)
    return inventory


def audit_oof(study):
    protocol_for(study)
    path, output = study/'oof-seal.json', study/'oof-audit.json'
    if output.exists():
        result = read(output)
        if result['prediction_seal'] != pin(path): raise ValueError('Changed OOF audit evidence')
        return result
    if not path.exists():
        rows,nodes = [],{}
        for which in model.TRAIN:
            inventory=read(study/'inventories'/f'{which}.json')
            if inventory['protocol'] != pin(study/'protocol.json'): raise ValueError('Mixed study inventories')
            rows.extend(inventory['rows']); add_nodes(nodes,inventory['ancestry_nodes'])
        evaluation.create_seal(path,run_id=study.name,stage='oof',rows=rows,policies=[{'id':'candidate'}],
                               protocol=pin(study/'protocol.json'),ancestry_nodes=nodes)
    with references(model.TRAIN):
        result=evaluation.audit_predictions(path,out=output)
    emit(stage='oof-audit',study=study.name,overall=result['overall'],eligible=result['eligible'])
    return result


def full_audit(study,permit):
    global _FULL
    protocol=protocol_for(study)
    model.validate_nomination(permit,protocol['recipe']['model_family'])
    _FULL=True
    try:
        inventory=produce_scored_match(study,'full','J03WQQ',permit)
        path=study/'full-seal.json'
        if not path.exists():
            evaluation.create_seal(path,run_id=study.name,stage='full',rows=inventory['rows'],policies=[{'id':'candidate'}],
                                   protocol=pin(study/'protocol.json'),ancestry_nodes=inventory['ancestry_nodes'],nomination=pin(permit))
        with references(['J03WQQ']):
            result=evaluation.audit_predictions(path,out=study/'full-audit.json')
        emit(stage='full-audit',study=study.name,overall=result['overall'],eligible=result['eligible'])
        return result
    finally:
        _FULL=False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','outer','oof-audit','full'])
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--family',choices=['v70','control','balanced'])
    p.add_argument('--held',choices=model.TRAIN)
    p.add_argument('--permit',type=Path)
    args=p.parse_args(); study=args.study.resolve(); study.relative_to(CAMPAIGN/'studies')
    threadpool_limits(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    # Registration hashes TRAIN references in the already sealed model protocol;
    # inference and evaluation afterwards use the stricter phased IO guard.
    if args.command=='register':
        if not args.family: p.error('register requires --family')
        register(study,args.family); emit(stage='registered',study=study.name); return
    sys.addaudithook(guard)
    try:
        if args.command=='outer':
            if not args.held: p.error('outer requires --held')
            produce_scored_match(study,args.held,args.held)
        elif args.command=='oof-audit': audit_oof(study)
        else:
            if not args.permit: p.error('full requires --permit')
            full_audit(study,args.permit)
    except model.CheckpointedStop as exc:
        emit(stage='checkpointed',message=str(exc)); raise SystemExit(75) from exc


if __name__=='__main__':
    main()
