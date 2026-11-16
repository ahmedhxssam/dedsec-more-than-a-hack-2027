"""Prospective seals and full-match audit of the actual retained baseline.

No fitting or inference lives here. prepare/seal cannot open references. audit
validates and retains every prediction in memory before opening TRAIN4 truth.
Functional ancestry is checked against hash-bound producer fit receipts, including
strict inner crossfit features. Source review must still establish DAG completeness.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import string
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'private/improvement/runtime'), str(ROOT)]
import numpy as np
from scipy.ndimage import uniform_filter1d
from second_zero.common import sha256

DATA = ROOT/'private/improvement/external/prepared'
FOLDS = [(['J03WOH','J03WOY'], ['J03WMX','J03WN1']),
         (['J03WMX','J03WN1'], ['J03WOH','J03WOY'])]
HALVES = ('firstHalf', 'secondHalf')
CONDITIONS = ('clean', 'corrupted')
TRAIN4 = set(m for _, held in FOLDS for m in held)
WPY = 'J03WPY'
IOUS = (.3, .4, .5, .6, .7)
COMPONENTS = ('uniform23','uniform24','uniform25','short25','interp_owner',
              'imputer','gru','carrier','legacy','context_status','temporal_status',
              'reliability','boundary_onset','boundary_restart','quality','joint')
# Required semantic contract; extra declarative fields are permitted.
FULL_CONTRACT = {
    'native_channels':98, 'native_dtype':'float32',
    'interpolation_channels':24, 'interpolation_dtype':'float32',
    'status_channels':40, 'status_dtype':'float32',
    'owner_order':['uniform20261023','uniform20261024','uniform20261025','short25'],
    'memory_channels':103, 'memory_width':48, 'memory_stride':5, 'memory_context':64,
    'x1':'short25', 'x2':'uniform3_std', 'x3':'uniform3_mean_sign',
    'quality_channels':175, 'segment_penalty':.55, 'frame_weight':.01, 'neighbors':16,
    'joint_weight':.25, 'joint_scope':'missing25', 'dead_cost':50, 'endpoints':True,
    'learned_residual':.75, 'strict_inner_crossfit':True,
    'calibration_matches':[WPY],
}
CANDIDATE_CONTRACT = {
    'expert':'raw290_HGB_TRAIN2_only_reused', 'probability_weight':.5,
    'mask_window':25, 'mask_mode':'nearest', 'mask_dtype':'float64',
    'postdecode_fence':True, 'lock_status':True, 'whole_clean_exact':True,
    'eligibility':'(uniform_filter1d(raw_ball_missing.float64,25,nearest)>0)&(baseline<2)',
    'new_wide':False, 'new_source_teacher':False, 'learned_gate':False,
}
STAGE = 'prepare'
TRUTH_OPEN = False
READS = []


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
    path = os.fsdecode(args[0])
    if any(s in path for s in ('J03WQQ','J03WR9','/evaluation/gt/','/runs/main/')):
        raise PermissionError('Reserved assets are closed in full-baseline OOF evaluator')
    if path.endswith('.truth.npy'):
        match = Path(path).name.split('-')[0]
        if not (STAGE == 'audit' and TRUTH_OPEN and match in TRAIN4):
            raise PermissionError('References closed until complete prediction validation')
        READS.append(path)


def absolute(path, *, reference=False):
    p = Path(path)
    p = (p if p.is_absolute() else ROOT/p).resolve()
    p.relative_to(ROOT)
    name = str(p)
    if any(s in name for s in ('J03WQQ','J03WR9','/evaluation/gt/','/runs/main/')):
        raise ValueError('Reserved asset path prohibited')
    if name.endswith('.truth.npy') and not reference:
        raise ValueError('Reference payload cannot be an input/source/artifact pin')
    return p


def rel(path): return str(absolute(path).relative_to(ROOT))

def pin(path, digest=None):
    p = absolute(path)
    actual = sha256(p)
    if digest is not None and actual != digest: raise ValueError('Changed artifact: '+rel(p))
    return p


def read(path, digest=None): return json.loads(pin(path, digest).read_text())

def write(path, value):
    p = absolute(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as h:
        json.dump(value, h, indent=2, allow_nan=False); h.write('\n')


def keys():
    return [(fold,m,h,c,f'{m}-{h}-{c}') for fold,(_,held) in enumerate(FOLDS)
            for m in held for h in HALVES for c in CONDITIONS]


def assert_contract(actual, expected, name):
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f'{name} mismatch for {key}: expected {value!r}')


def dotted(obj, field):
    for item in field.split('.'):
        obj = obj[int(item)] if isinstance(obj, list) else obj[item]
    return obj


def pins(mapping):
    if not isinstance(mapping, dict): raise ValueError('Expected path-to-SHA mapping')
    for path, digest in mapping.items():
        if not isinstance(digest,str) or len(digest)!=64: raise ValueError('Invalid SHA256 pin')
        pin(path, digest)


def fold_roles(folds):
    if set(folds) != {'0','1'}: raise ValueError('Exactly two outer folds required')
    for fold, (train, held) in enumerate(FOLDS):
        item = folds[str(fold)]
        if set(item['train_matches']) != set(train) or set(item['heldout_matches']) != set(held):
            raise ValueError('Incorrect outer train/held partition')


def validate_ancestry(doc, protocol_sha, names):
    """Check declared functional ancestry plus immutable artifact/receipt evidence."""
    if doc.get('kind') != 'full-baseline-ancestry-v1' or doc['producer_protocol_sha256'] != protocol_sha:
        raise ValueError('Ancestry/protocol identity mismatch')
    fold_roles(doc['folds']); pins(doc['sources']); pins(doc.get('inputs', {}))
    nodes = doc['nodes']
    if not nodes: raise ValueError('Empty fitted ancestry')
    visiting=set(); closures={}
    def closure(node_id):
        if node_id in closures: return closures[node_id]
        if node_id not in nodes: raise ValueError('Missing dependency: '+node_id)
        if node_id in visiting: raise ValueError('Ancestry cycle')
        visiting.add(node_id); result={node_id}
        for dep in nodes[node_id]['dependencies']: result |= closure(dep)
        visiting.remove(node_id); closures[node_id]=result
        return result
    for node_id,n in nodes.items():
        closure(node_id)
        if n['kind'] not in ('weights','calibration','training_features','observations','pipeline'):
            raise ValueError('Unknown ancestry node kind')
        fold=n['outer_fold']
        if type(fold) is not int or fold not in (0,1): raise ValueError('Missing outer fold on node')
        labels=n['label_matches']
        if not isinstance(labels,list) or len(set(labels)) != len(labels): raise ValueError('Invalid label matches')
        allowed = set(FOLDS[fold][0]) if n['kind']=='weights' else {WPY} if n['kind']=='calibration' else set()
        if not set(labels) <= allowed: raise ValueError('Illegal label ancestry: '+node_id)
        if n['kind'] in ('weights','calibration'):
            if not labels or not n.get('artifacts'): raise ValueError('Fitted node needs labels and bound artifacts')
            receipt=n['receipt']; r=read(receipt['path'], receipt['sha256'])
            recorded=dotted(r,receipt['matches_field']); recorded=[recorded] if isinstance(recorded,str) else recorded
            if not isinstance(recorded,list) or set(recorded)!=set(labels): raise ValueError('Receipt label-role mismatch')
            bindings=receipt['bindings']
            if set(bindings)!=set(n['artifacts']): raise ValueError('Every fitted artifact must bind to fit receipt')
            for path,field in bindings.items():
                if dotted(r,field)!=n['artifacts'][path]: raise ValueError('Receipt/model SHA mismatch')
        pins(n.get('artifacts',{}))
        pins(n.get('provenance',{}))
        if n['kind']=='training_features':
            predicted=set(n.get('prediction_matches',[]))
            if n.get('crossfit') is not True or not predicted or not predicted<=set(FOLDS[fold][0]):
                raise ValueError('Strict inner crossfit declaration required')
            for ancestor in closure(node_id)-{node_id}:
                if predicted & set(nodes[ancestor]['label_matches']):
                    raise ValueError('Training feature learned from its prediction match: '+node_id)
        for ancestor in closure(node_id):
            a=nodes[ancestor]
            if a['outer_fold'] != fold: raise ValueError('Cross-outer-fold dependency')
            if set(a['label_matches']) & set(FOLDS[fold][1]): raise ValueError('Held-label ancestor')
    for fold in (0,1):
        f=doc['folds'][str(fold)]; components=f['components']
        if set(components)!=set(COMPONENTS): raise ValueError('Incomplete actual full baseline component inventory')
        baseline=f['baseline_root']; baseline_closure=closure(baseline)
        if nodes[baseline]['outer_fold']!=fold: raise ValueError('Baseline root wrong fold')
        if not set(components.values()) <= baseline_closure: raise ValueError('Baseline does not depend on required component')
        for component,node_id in components.items():
            if nodes[node_id]['kind'] not in ('weights','calibration','pipeline'):
                raise ValueError('Component is not fitted/bundled: '+component)
        # A flag on the outer recipe alone cannot establish nested stacking.
        for component in ('gru','quality','boundary_onset','boundary_restart'):
            produced=set()
            for node_id in closure(components[component]):
                if nodes[node_id]['kind']=='training_features':
                    produced.update(nodes[node_id]['prediction_matches'])
            if produced!=set(FOLDS[fold][0]):
                raise ValueError('Incomplete strict inner feature coverage for '+component)
        if 'prediction_spec' not in f:
            raise ValueError('Actual loader prediction specification required')
        if 'prediction_spec' in f:
            item=f['prediction_spec']; explicit=read(item['path'],item['sha256'])
            declared={path:digest for node_id in baseline_closure for path,digest in nodes[node_id].get('artifacts',{}).items()}
            declared.update(doc['sources'])
            if declared.get(rel(item['path']))!=item['sha256']:
                raise ValueError('Actual loader specification is not pinned by baseline root')
            def check_spec(value):
                if isinstance(value,dict):
                    if 'path' in value and 'sha256' in value:
                        path=rel(value['path'])
                        if declared.get(path)!=value['sha256']:
                            raise ValueError('Loader specification artifact missing from baseline ancestry: '+path)
                    for child in value.values(): check_spec(child)
                elif isinstance(value,list):
                    for child in value: check_spec(child)
            check_spec(explicit)
        if set(f.get('candidate_roots',{}))!=set(names): raise ValueError('Candidate ancestry inventory mismatch')
        for name,node_id in f.get('candidate_roots',{}).items():
            if nodes[node_id]['outer_fold']!=fold or baseline not in closure(node_id):
                raise ValueError('Candidate root must depend on same actual baseline')
    return {'nodes':len(nodes), 'outer_held_labels_excluded':True,
            'strict_training_feature_crossfit_verified':True,
            'limitation':'Hash-bound declared functional DAG; source review must establish no undeclared dependencies.'}


def source_paths():
    return [Path(__file__), ROOT/'second_zero/common.py', ROOT/'second_zero/evaluate.py',
            ROOT/'scripts/train_contextual.py', ROOT/'second_zero/contextual.py',
            ROOT/'vendor/mtah/evaluation/possession_metrics.py']


def validate_descriptor(d):
    if d['role'] not in ('baseline','candidate'): raise ValueError('Unknown prediction role')
    if not d['id'] or not all(c.isalnum() or c in '_-' for c in d['id']): raise ValueError('Invalid prediction id')
    fields={f for _,f,_,_ in string.Formatter().parse(d['path_template']) if f}
    if not fields <= {'fold','key'} or 'key' not in fields: raise ValueError('Invalid prediction template')
    for fold,_,_,_,key in keys(): absolute(d['path_template'].format(fold=fold,key=key))
    if not isinstance(d.get('label_array'),str): raise ValueError('Missing labels array')
    if d['role']=='baseline' and not d.get('status_array'): raise ValueError('Baseline requires status mask')
    if d['role']=='candidate' and not d.get('eligibility_array'): raise ValueError('Candidate requires eligible mask')


def require_baseline_anchor(descriptors,anchor):
    if any(d['role']=='candidate' for d in descriptors) and not anchor:
        raise ValueError('Candidate requires independently sealed baseline-only anchor')


def prepare(args):
    p=read(args.producer_protocol); fold_roles(p['folds'])
    assert_contract(p['full_contract'],FULL_CONTRACT,'Full baseline contract')
    pins(p['sources']); pins(p.get('inputs',{}))
    inventory=read(args.inventory)
    descriptors=inventory['predictions']
    if len({d['id'] for d in descriptors})!=len(descriptors): raise ValueError('Duplicate prediction id')
    if sum(d['role']=='baseline' for d in descriptors)!=1 or sum(d['role']=='candidate' for d in descriptors)>1:
        raise ValueError('Exactly one baseline; at most one prospective candidate')
    for d in descriptors: validate_descriptor(d)
    require_baseline_anchor(descriptors,args.baseline_seal)
    candidates=[d['id'] for d in descriptors if d['role']=='candidate']
    if candidates:
        assert_contract(p['candidate_contract'],CANDIDATE_CONTRACT,'Gap candidate contract')
        if not p.get('adaptivity_disclosure'): raise ValueError('Disclose reused development/expert selection')
    anchor=None
    if args.baseline_seal:
        anchor=read(args.baseline_seal)
        if anchor.get('kind')!='actual-full-baseline-prediction-seal-v1' or anchor.get('held_truth_read') or anchor.get('reference_reads') or len(anchor['rows'])!=16:
            raise ValueError('Invalid baseline anchor seal')
        if {r['key'] for r in anchor['rows']}!={r[-1] for r in keys()}: raise ValueError('Incomplete baseline anchor inventory')
        # A baseline-only seal made before the baseline-only audit is the immutable anchor.
        if any(len(r['predictions'])!=1 for r in anchor['rows']): raise ValueError('Baseline-only anchor required')
    spec={'kind':'actual-full-baseline-evaluation-spec-v1','created_utc':now(),
          'producer_protocol':rel(args.producer_protocol),'producer_protocol_sha256':sha256(absolute(args.producer_protocol)),
          'inventory':rel(args.inventory),'inventory_sha256':sha256(absolute(args.inventory)),
          'predictions':descriptors,'ancestry_path':rel(args.ancestry),
          'source_sha256':{rel(s):sha256(s) for s in source_paths()},
          'baseline_anchor':rel(args.baseline_seal) if anchor else None,
          'baseline_anchor_sha256':sha256(absolute(args.baseline_seal)) if anchor else None,
          'held_truth_read':False,'reference_reads':[],
          'selection':'Strict overall and corrupted improvement; whole-clean predictions exact. No tie promotion.',
          'scoring':'Official once per physical match/condition, both halves pooled with ungraded X; equal four matches and two conditions; five IoUs.'}
    write(args.out,spec)
    print(json.dumps({'spec':rel(args.out),'candidate_ids':candidates,'held_truth_read':False}),flush=True)


def validate_spec(path,digest=None):
    s=read(path,digest)
    if s['kind']!='actual-full-baseline-evaluation-spec-v1' or s['held_truth_read'] or s['reference_reads']:
        raise ValueError('Invalid evaluation spec')
    pins(s['source_sha256']); p=read(s['producer_protocol'],s['producer_protocol_sha256'])
    fold_roles(p['folds']); assert_contract(p['full_contract'],FULL_CONTRACT,'Full baseline contract')
    pins(p['sources']); pins(p.get('inputs',{}))
    inventory=read(s['inventory'],s['inventory_sha256'])
    if inventory['predictions']!=s['predictions']: raise ValueError('Changed prediction descriptors')
    for d in s['predictions']: validate_descriptor(d)
    require_baseline_anchor(s['predictions'],s.get('baseline_anchor'))
    if any(d['role']=='candidate' for d in s['predictions']):
        assert_contract(p['candidate_contract'],CANDIDATE_CONTRACT,'Candidate contract')
    return s


def labels(value,name='prediction'):
    a=np.asarray(value)
    if a.ndim!=1 or a.size==0 or not np.issubdtype(a.dtype,np.integer) or not np.isin(a,[0,1,2,3]).all():
        raise ValueError('Expected canonical integer H/A/D/X '+name)
    return a.astype(np.int8)


def observational_support(match,half,condition):
    from scripts.train_contextual import corruption
    path=absolute(DATA/f'{match}-{half}.npz')
    with np.load(path,allow_pickle=False) as z:
        raw={k:np.array(z[k]) for k in ('ball','players','team','pitch')}
    if condition=='corrupted': raw=corruption(raw,100)
    missing=~np.isfinite(raw['ball']).all(axis=1)
    return uniform_filter1d(missing.astype(np.float64),25,mode='nearest'), {rel(path):sha256(path)}


def validate_arrays(arrays,descriptors,n,condition,support):
    desc={d['id']:d for d in descriptors}; baseline=next(d['id'] for d in descriptors if d['role']=='baseline')
    b=labels(arrays[baseline][desc[baseline]['label_array']])
    status=labels(arrays[baseline][desc[baseline]['status_array']],'status')
    if b.shape!=(n,) or status.shape!=b.shape: raise ValueError('Baseline/input/status shape mismatch')
    if not np.array_equal(b>=2,status>=2) or not np.array_equal(b[b>=2],status[b>=2]):
        raise ValueError('Baseline/status D/X mismatch')
    if support.shape!=b.shape or not np.isfinite(support).all(): raise ValueError('Invalid observational support')
    expected=(support>0)&(b<2)
    output={baseline:b}
    for name,d in desc.items():
        if name==baseline: continue
        a=labels(arrays[name][d['label_array']])
        if a.shape!=b.shape: raise ValueError('Candidate shape mismatch')
        mask=np.asarray(arrays[name][d['eligibility_array']])
        if mask.dtype!=np.bool_ or mask.shape!=b.shape or not np.array_equal(mask,expected):
            raise ValueError('Candidate mask differs from independent raw-input eligibility')
        if not np.array_equal(a>=2,b>=2) or not np.array_equal(a[b>=2],b[b>=2]):
            raise ValueError('Candidate changed D/X status')
        if not np.array_equal(a[~expected],b[~expected]): raise ValueError('Candidate crossed final input fence')
        if condition=='clean' and not np.array_equal(a,b): raise ValueError('Whole clean prediction changed')
        output[name]=a
    return output


def collect_predictions(spec,sealed=None):
    """Validate all bytes/arrays/masks into memory while references remain closed."""
    if TRUTH_OPEN: raise ValueError('Prediction preflight must precede reference opening')
    expected=keys(); rows=[]; loaded={}; files={}; observations={}
    old={r['key']:r for r in sealed['rows']} if sealed else {}
    if sealed and (len(sealed['rows'])!=16 or set(old)!={r[-1] for r in expected}): raise ValueError('Incomplete sealed view inventory')
    anchor=read(spec['baseline_anchor'],spec['baseline_anchor_sha256']) if spec.get('baseline_anchor') else None
    anchorrows={r['key']:r for r in anchor['rows']} if anchor else {}
    for fold,match,half,condition,key in expected:
        arrays={}; records={}
        for d in spec['predictions']:
            path=absolute(d['path_template'].format(fold=fold,key=key)); digest=sha256(path)
            record={'path':rel(path),'sha256':digest,'label_array':d['label_array']}
            if sealed:
                prior=old[key]
                if (prior['fold'],prior['match'],prior['half'],prior['condition'])!=(fold,match,half,condition): raise ValueError('Misindexed sealed view')
                if set(prior['predictions'])!={x['id'] for x in spec['predictions']}: raise ValueError('Sealed candidate inventory changed')
                if prior['predictions'][d['id']]!=record: raise ValueError('Sealed prediction bytes/descriptor changed')
            if anchor and d['role']=='baseline':
                only=list(anchorrows[key]['predictions'].values())
                if len(only)!=1 or only[0]!=record: raise ValueError('Candidate baseline differs from sealed baseline-only bytes')
            with np.load(path,allow_pickle=False) as z:
                names=[d['label_array'],d['status_array']] if d['role']=='baseline' else [d['label_array'],d['eligibility_array']]
                arrays[d['id']]={name:np.array(z[name]) for name in names}
            records[d['id']]=record; files[rel(path)]=digest
        support,pinned=observational_support(match,half,condition)
        for path,digest in pinned.items():
            if path in observations and observations[path]!=digest:
                raise ValueError('Observational input changed between paired views')
            observations[path]=digest
        values=validate_arrays(arrays,spec['predictions'],len(support),condition,support)
        row={'fold':fold,'match':match,'half':half,'condition':condition,'key':key,'frames':len(support),'predictions':records}
        if sealed and row!=old[key]: raise ValueError('Changed sealed row metadata')
        rows.append(row); loaded[key]=values
    if sealed and (files!=sealed['prediction_files'] or observations!=sealed['observational_inputs']): raise ValueError('Changed prediction/input inventory')
    return rows,loaded,files,observations


def seal(args):
    spec=validate_spec(args.spec)
    ancestry=read(spec['ancestry_path'])
    names=[d['id'] for d in spec['predictions'] if d['role']=='candidate']
    review=validate_ancestry(ancestry,spec['producer_protocol_sha256'],names)
    rows,_,files,observations=collect_predictions(spec)
    if READS: raise ValueError('Reference-contaminated seal')
    out={'kind':'actual-full-baseline-prediction-seal-v1','created_utc':now(),
         'spec':rel(args.spec),'spec_sha256':sha256(absolute(args.spec)),
         'ancestry':spec['ancestry_path'],'ancestry_sha256':sha256(absolute(spec['ancestry_path'])),
         'ancestry_review':review,'rows':rows,'prediction_files':files,'observational_inputs':observations,
         'held_truth_read':False,'reference_reads':[]}
    write(args.out,out)
    print(json.dumps({'seal':rel(args.out),'views':len(rows),'prediction_files':len(files),'held_truth_read':False}),flush=True)


def join_halves(parts):
    if len(parts)!=2: raise ValueError('Exactly two halves required')
    return np.concatenate([np.asarray(parts[0]),np.array([3],np.int8),np.asarray(parts[1])])


def score_full(pm,y,p):
    if y.shape!=p.shape: raise ValueError('Full-match shape mismatch')
    m=pm.evaluate(y,p,bootstrap_n=0)
    if tuple(pm.SPELL_IOUS)!=IOUS: raise ValueError('Official IoU inventory changed')
    runs=pm.valid_runs(y); refs=pm.possession_spells(y,runs)
    preds=pm.possession_spells(pm.fill_abstentions(p,runs),runs)
    overlaps=pm.spell_overlaps(refs,preds)
    flags={str(t):pm.matched_gt_flags(overlaps,len(refs),t) for t in IOUS}
    durations=np.array([b-a for a,b,_ in refs]); short=durations<=125; tiny=durations<=25
    result={k:m[k] for k in ('spell_f1_mean','macro_f1','spell_f1','n_gt_spells','n_pred_spells','per_class','fragmentation','alive_dead_f1','confusion','n_abstain_on_graded')}
    result['short']={'references':int(short.sum()),'very_short_references':int(tiny.sum()),
                     'matched':{t:int((f&short).sum()) for t,f in flags.items()},
                     'very_short_matched':{t:int((f&tiny).sum()) for t,f in flags.items()}}
    return result,(refs,durations,flags)


def changes(before,after):
    refs,durations,b=before; ar,ad,a=after
    if refs!=ar or not np.array_equal(durations,ad): raise ValueError('Mismatched interval reference inventory')
    result={}
    for t in b:
        gained=a[t]&~b[t]; lost=b[t]&~a[t]; short=durations<=125; tiny=durations<=25
        result[t]={'recovered':int(gained.sum()),'lost':int(lost.sum()),
                   'short_recovered':int((gained&short).sum()),'short_lost':int((lost&short).sum()),
                   'very_short_recovered':int((gained&tiny).sum()),'very_short_lost':int((lost&tiny).sum())}
    return result


def aggregate(matches,names):
    def avg(items):
        return {name:{metric:float(np.mean([r[name][metric] for r in items]))
                      for metric in ('spell_f1_mean','macro_f1','alive_dead_f1')} for name in names}
    conditions={c:avg([matches[m][c] for m in sorted(TRAIN4)]) for c in CONDITIONS}
    overall=avg([matches[m][c] for m in sorted(TRAIN4) for c in CONDITIONS])
    per_match={m:avg([matches[m][c] for c in CONDITIONS]) for m in sorted(TRAIN4)}
    per_fold={str(i):avg([matches[m][c] for m in held for c in CONDITIONS]) for i,(_,held) in enumerate(FOLDS)}
    return overall,conditions,per_match,per_fold


def audit(args):
    global TRUTH_OPEN
    if absolute(args.out).exists(): raise ValueError('Audit output already exists; refusing repeated audit')
    sealed=read(args.seal)
    if sealed['kind']!='actual-full-baseline-prediction-seal-v1' or sealed['held_truth_read'] or sealed['reference_reads']:
        raise ValueError('Invalid prediction seal')
    spec=validate_spec(sealed['spec'],sealed['spec_sha256'])
    names=[d['id'] for d in spec['predictions']]; baseline=next(d['id'] for d in spec['predictions'] if d['role']=='baseline')
    if len(names)!=2 or len(sealed['prediction_files'])!=32:
        raise ValueError('Focused audit requires baseline plus one candidate, all32 sealed; baseline-only scoring prohibited')
    validate_ancestry(read(sealed['ancestry'],sealed['ancestry_sha256']),spec['producer_protocol_sha256'],[n for n in names if n!=baseline])
    _,loaded,_,_=collect_predictions(spec,sealed)
    # No prediction/status/mask file opens below this gate. They are already validated in memory.
    TRUTH_OPEN=True
    from second_zero.evaluate import official_metrics
    pm=official_metrics(); matches={}; events={}
    for m in sorted(TRAIN4):
        y=join_halves([labels(np.load(absolute(DATA/f'{m}-{h}.truth.npy',reference=True),allow_pickle=False),'reference') for h in HALVES])
        matches[m]={}; events[m]={}
        for c in CONDITIONS:
            metrics={}; details={}
            for name in names:
                pred=join_halves([loaded[f'{m}-{h}-{c}'][name] for h in HALVES])
                metrics[name],details[name]=score_full(pm,y,pred)
            matches[m][c]=metrics
            events[m][c]={n:changes(details[baseline],details[n]) for n in names if n!=baseline}
    overall,conditions,per_match,per_fold=aggregate(matches,names)
    five_iou={c:{n:{str(t):float(np.mean([matches[m][c][n]['spell_f1'].get(t,matches[m][c][n]['spell_f1'].get(str(t)))['f1'] for m in sorted(TRAIN4)])) for t in IOUS} for n in names} for c in CONDITIONS}
    status_exact=all(matches[m][c][n]['alive_dead_f1']==matches[m][c][baseline]['alive_dead_f1'] for m in TRAIN4 for c in CONDITIONS for n in names)
    advanced=[n for n in names if n!=baseline and overall[n]['spell_f1_mean']>overall[baseline]['spell_f1_mean'] and conditions['corrupted'][n]['spell_f1_mean']>conditions['corrupted'][baseline]['spell_f1_mean']]
    if not status_exact: raise ValueError('Scored status changed despite label fence')
    result={'kind':'actual-full-baseline-full-match-OOF-audit-v1','created_utc':now(),
            'prediction_seal':rel(args.seal),'prediction_seal_sha256':sha256(absolute(args.seal)),
            'spec_sha256':sealed['spec_sha256'],'ancestry_sha256':sealed['ancestry_sha256'],
            'baseline':baseline,'overall':overall,'conditions':conditions,'per_match':per_match,'per_fold':per_fold,
            'five_iou':five_iou,'full_match_details':matches,'interval_change_vs_baseline':events,
            'status_exact':True,'whole_clean_prediction_exact':True,
            'selected':advanced[0] if advanced else baseline,'candidate_passed_advancement':bool(advanced),
            'reference_reads':READS.copy(),'WQQ_access':False,'WR9_access':False,'competition_GT_access':False,
            'interpretation':'Four adaptively reused physical matches; paired conditions correlated. Strict improvement gate permits one later sealed full transfer, never sibling fallback.'}
    write(args.out,result)
    print(json.dumps({'overall':overall,'conditions':conditions,'selected':result['selected'],'candidate_passed_advancement':bool(advanced)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--producer-protocol',required=True); p.add_argument('--inventory',required=True); p.add_argument('--ancestry',required=True); p.add_argument('--baseline-seal'); p.add_argument('--out',required=True)
    p=sub.add_parser('seal'); p.add_argument('--spec',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('audit'); p.add_argument('--seal',required=True); p.add_argument('--out',required=True)
    args=parser.parse_args(); STAGE=args.stage; sys.addaudithook(guard); globals()[args.stage](args)
