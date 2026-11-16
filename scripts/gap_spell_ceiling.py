"""Prospective, selected-parent-only ceiling diagnostics; never exports oracles."""
from __future__ import annotations
import argparse,datetime,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
import numpy as np
from second_zero.common import sha256
from second_zero.evaluate import official_metrics
from second_zero import gap_spell as gs

OLD=ROOT/'runs/full-baseline-gap-v1';RUN=ROOT/'runs/gap-spell-v2';REPORT=ROOT/'reports/gap-spell-v2'
TRAIN=['J03WMX','J03WN1','J03WOH','J03WOY'];READS=[];STAGE='prepare'


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def read(p):return json.loads(Path(p).read_text())
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    if any(s in p for s in ['J03WR9','/evaluation/gt/','/runs/main/']):raise PermissionError('Reserved assets closed')
    if p.endswith('.truth.npy'):
        if STAGE!='ceiling' or Path(p).name.split('-')[0] not in TRAIN+['J03WQQ']:raise PermissionError('Ceiling references closed')
        READS.append(rel(p))


def prepare():
    p=read(OLD/'verdict.json');assert p['selected']=='gap_recovery' and p['full_transfer_improved']
    full=read(OLD/'full-transfer/audit.json');assert full['overall']['gap_recovery']['spell_f1_mean']==.769357376463475
    pins={}
    for seal,field in [(OLD/'prediction-seal.json','prediction_files'),(OLD/'full-transfer/prediction-seal.json','outputs')]:
        pins.update(read(seal)[field]);pins[rel(seal)]=sha256(seal)
    sources=[Path(__file__),Path(gs.__file__),ROOT/'scripts/sprint_decoder_segments_geometry_v2.py',ROOT/'second_zero/sprint_decoder.py',ROOT/'vendor/mtah/evaluation/possession_metrics.py']
    preservation=read(ROOT/'reports/full-baseline-gap-v1/preservation-after.json')['hashes']
    for group in preservation.values():pins.update(group)
    for path,digest in pins.items():
        if sha256(ROOT/path)!=digest:raise ValueError('Preserved baseline changed: '+path)
    write(RUN/'ceiling-protocol.json',{'kind':'gap-spell-ceiling-v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline':.769357376463475,
        'scope':'Same exact eligible mask, all D/X and outside frames locked, whole clean unchanged',
        'diagnostics':['GT reference fill on eligible live frames (feasible mask oracle, not exact maximum)','Existing graph plus exact baseline spells, feasible locked arcs only; exact independent-credit fractional upper bound and official score of its feasible path'],
        'cohorts':TRAIN+['J03WQQ'],'metric':'Full match with X between halves; equal conditions and IoUs .3/.4/.5/.6/.7',
        'gate':'Train a gap-only spell scorer if a feasible WQQ oracle reaches .8 overall and OOF constrained proposal headroom is positive on every match. WQQ labels never enter learned features or fitting.',
        'no_oracle_arrays_exported':True,'sources':{rel(q):sha256(q) for q in sources},'inputs':pins,'preservation':preservation})
    print('CEILING FROZEN',sha256(RUN/'ceiling-protocol.json'),flush=True)


def current(match,half,condition):
    if match=='J03WQQ':base=OLD/'full-transfer/full_baseline';parent=OLD/'full-transfer/gap_recovery'
    else:
        fold=0 if match in TRAIN[:2] else 1;base=OLD/f'baseline/fold{fold}';parent=OLD/f'gap_recovery/fold{fold}'
    key=f'{match}-{half}-{condition}'
    with np.load(base/(key+'.npz'),allow_pickle=False) as z:value={k:z[k] for k in ['predicted','signals','segments','quality','status_mask']}
    with np.load(parent/(key+'.npz'),allow_pickle=False) as z:value.update({k:z[k] for k in z.files if k in ['predicted','signals','segments','quality','status_mask','eligible_mask','global_logit']})
    if condition=='clean' and value['eligible_mask'].any():raise AssertionError('Clean must have zero mutable frames')
    return value


def ceiling():
    p=read(RUN/'ceiling-protocol.json')
    for name,digest in {**p['sources'],**p['inputs']}.items():
        if sha256(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    pm=official_metrics();rows={};join=lambda v:np.r_[v[0],np.int8(3),v[1]]
    for match in p['cohorts']:
        halves=[]
        for half in ['firstHalf','secondHalf']:
            row=current(match,half,'corrupted');row['truth']=np.load(ROOT/f'private/improvement/external/prepared/{match}-{half}.truth.npy');halves.append(row)
        truth=join([h['truth'] for h in halves]);baseline=pm.evaluate(truth,join([h['predicted'] for h in halves]),bootstrap_n=0)
        clean=pm.evaluate(truth,join([current(match,h,'clean')['predicted'] for h in ['firstHalf','secondHalf']]),bootstrap_n=0)
        filled=[]
        for h in halves:
            pred=h['predicted'].copy();mask=h['eligible_mask']&(h['truth']<2);pred[mask]=h['truth'][mask];gs.check_lock(pred,h['predicted'],h['eligible_mask']);filled.append(pred)
        fill=pm.evaluate(truth,join(filled),bootstrap_n=0)
        oracle,_=gs.fractional_oracle(halves,pm)
        row={'clean':float(clean['spell_f1_mean']),'corrupted':float(baseline['spell_f1_mean']),'overall':float((clean['spell_f1_mean']+baseline['spell_f1_mean'])/2),
            'mask_reference_fill_corrupted':float(fill['spell_f1_mean']),'mask_reference_fill_overall':float((clean['spell_f1_mean']+fill['spell_f1_mean'])/2),
            'constrained_graph':oracle,'constrained_graph_overall_feasible':float((clean['spell_f1_mean']+oracle['official_feasible_spell_f1'])/2),'constrained_graph_overall_upper_bound':float((clean['spell_f1_mean']+oracle['independent_credit_upper_bound'])/2)}
        rows[match]=row;print('CEILING',match,json.dumps(row),flush=True)
    wqq=rows['J03WQQ'];passed=wqq['constrained_graph_overall_feasible']>=.8 and all(rows[m]['constrained_graph']['official_feasible_spell_f1']>rows[m]['corrupted'] for m in TRAIN)
    result={'kind':'gap-spell-ceiling-v2','protocol_sha256':sha256(RUN/'ceiling-protocol.json'),'rows':rows,'sufficient_existing_graph_headroom':bool(passed),'no_oracle_arrays_exported':True,
        'interpretation':'Reference-assisted diagnostic. Upper bound ignores one-to-one conflicts; feasible path official score is a realizable graph/mask oracle, not learned performance. Clean is fixed exactly.', 'reference_reads':READS}
    write(REPORT/'ceiling.json',result);print('HEADROOM',passed,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','ceiling']);args=a.parse_args();STAGE=args.stage
    sys.addaudithook(guard);globals()[args.stage]()
