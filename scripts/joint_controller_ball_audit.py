"""Fixed, reference-separated event and missing-ball audit for joint candidates.

No fitting or parameter selection. All label-dependent diagnostics follow the
prediction freeze. Short spells include one-frame references; D/X split them.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.special import expit
from second_zero.common import sha256,json_write
from second_zero.evaluate import official_metrics
from second_zero.sprint_decoder import decode
from scripts import sprint_decoder_lattice as lattice
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts import joint_controller_ball as run
from scripts.sprint_evaluate import score
from scripts.anchor_refine_audit import event_change


def observation_bins(raw):
    seen=np.isfinite(raw['ball']).all(1); ix=np.arange(len(seen))
    prev=np.maximum.accumulate(np.where(seen,ix,-100000));age=ix-prev
    return {'observed':seen,'missing_1_25':~seen&(age<=25),
            'missing_26_125':~seen&(age>25)&(age<=125),'missing_over125':~seen&(age>125)}


def subset_events(pm,y,pred,raw):
    audit=score(pm,y,pred,True); seen=np.isfinite(raw['ball']).all(1)
    result={}
    for name,test in [('short',lambda e:e['frames']<=125),('very_short_25',lambda e:e['frames']<=25),
                      ('gap_overlap',lambda e:not seen[e['start']:e['stop']].all()),
                      ('short_gap_overlap',lambda e:e['frames']<=125 and not seen[e['start']:e['stop']].all())]:
        events=[e for e in audit['events']['events'] if test(e)]
        result[name]={'references':len(events),'matched_iou05':sum(e['matched']['0.5'] for e in events)}
    return audit,result


def main(args):
    args.run=args.run.resolve();args.out=args.out.resolve()
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    freeze_path=args.run/('scope-freeze.json' if args.scope else 'freeze.json')
    frozen=json.loads(freeze_path.read_text())
    assert sha256(args.run/'model.pt')==frozen['model_sha256']
    assert sha256(args.run/('scope-calibration.json' if args.scope else 'calibration.json'))==frozen['calibration_sha256']
    assert frozen['weight']>0,'No changed model selected on calibration'
    if args.match=='J03WQQ':folder=args.run/('WQQ-scope' if args.scope else 'WQQ')
    else:folder=args.run/('WPY-scope' if args.scope and frozen['scope']=='missing25' else 'WPY')/str(frozen['weight'])
    # All raw and decoded arrays pinned before opening evaluation labels.
    outputs={};prepared={}
    for key in graph.keys([args.match]):
        tag='WPY' if args.match=='J03WPY' else 'reliability'
        path=lattice.RUN/'cache'/tag/'16'/f'{key}.npz'
        with np.load(path) as z:baseline={k:z[k] for k in z.files}
        before=graph.graph_decode(baseline['logit'],baseline['base'],baseline['segments'],baseline['quality'],.55,.01)
        path=folder/f'{key}.npz'; outputs[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(path) as z:after=z['predicted'].copy()
        path=args.run/'raw'/f'{key}.npz';outputs[str(path.relative_to(ROOT))]=sha256(path)
        with np.load(path) as z:neural={k:z[k] for k in z.files}
        raw=run.old.load_raw(key)
        owner_pair={'winner':decode(baseline['logit'],baseline['base'],cost=12.),
                    'joint':decode(neural['owner']/frozen['temperature'],baseline['base'],cost=12.)}
        prepared[key]={'before':before,'after':after,'raw':raw,'neural':neural,
                       'owner':owner_pair,'baseline':baseline}
        for label,p in owner_pair.items():outputs[key+'/'+label+'-cost12-array']=hashlib.sha256(p.tobytes()).hexdigest()
        np.testing.assert_array_equal(after[before>=2],before[before>=2])
    json_write(out/'pre-score.json',{'outputs':outputs,'cost':12.,'references_opened':False,
        'scope':'Full-pipeline selected blend plus explicit cost12 pure-ownership diagnostic.'})
    run.ALLOWED.add(args.match);pm=official_metrics();views={}
    for key,r in prepared.items():
        y=graph.truth(key);old,old_sub=subset_events(pm,y,r['before'],r['raw'])
        new,new_sub=subset_events(pm,y,r['after'],r['raw'])
        owner={k:score(pm,y,v,True) for k,v in r['owner'].items()}
        bins={}
        for name,mask in observation_bins(r['raw']).items():
            live=mask&(y<2);z=r['neural']['owner']/frozen['temperature'];base=r['baseline']['logit']
            bins[name]={'live_frames':int(live.sum())}
            for label,logit in [('winner',base),('joint',z)]:
                bins[name][label]={'correct_frames':int(((logit>0)==y)[live].sum()),
                    'NLL':float(np.mean((np.logaddexp(0,logit)-np.minimum(y,1)*logit)[live])) if live.any() else None}
        name,condition=key.rsplit('-',1)
        with np.load(run.DATA/f'{name}.npz') as z:clean_ball=z['ball'];pitch=z['pitch']
        missing=~np.isfinite(r['raw']['ball']).all(1)&np.isfinite(clean_ball).all(1)
        ball=r['neural']['ball']*10+pitch/2
        errors=np.linalg.norm(ball[missing]-clean_ball[missing],axis=1)
        views[key]={'before':old['metrics'],'after':new['metrics'],'event_change':event_change(old,new),
            'subsets':{'before':old_sub,'after':new_sub},'evidence_by_ball_availability':bins,
            'pure_ownership_cost12':{k:v['metrics'] for k,v in owner.items()},
            'pure_ownership_events':event_change(owner['winner'],owner['joint']),
            'coordinate_diagnostic':{'frames':int(missing.sum()),'mean_error_m':float(errors.mean()) if len(errors) else None,
                'median_error_m':float(np.median(errors)) if len(errors) else None},
            'D_X_unchanged':True,'changed_frames':int((r['before']!=r['after']).sum())}
    groups={}
    for condition in ['overall','clean','corrupted']:
        values=[r for key,r in views.items() if condition=='overall' or key.endswith(condition)]
        g={arm:{k:float(np.mean([r[arm][k] for r in values])) for k in ['spell_f1_mean','macro_f1','fragmentation']} for arm in ['before','after']}
        g['events']={k:sum(r['event_change'][k] for r in values) for k in ['recovered','lost','short_recovered','short_lost','before_short_matches','after_short_matches','before_predicted_spells','after_predicted_spells']}
        g['subsets']={arm:{name:{k:sum(r['subsets'][arm][name][k] for r in values) for k in ['references','matched_iou05']}
                              for name in ['short','very_short_25','gap_overlap','short_gap_overlap']} for arm in ['before','after']}
        g['pure_ownership_cost12']={arm:{k:float(np.mean([r['pure_ownership_cost12'][arm][k] for r in values])) for k in ['spell_f1_mean','macro_f1','fragmentation']} for arm in ['winner','joint']}
        g['per_class_F1']={arm:{label:float(np.mean([r[arm]['per_class'][label]['f1'] for r in values])) for label in ['H','A','D']} for arm in ['before','after']}
        groups[condition]=g
    json_write(out/'audit.json',{'match':args.match,'freeze_sha256':sha256(freeze_path),
        'groups':groups,'views':views,'interpretation':'Paired clean/corrupt views; reference-assisted diagnostics only. WQQ is reused development. Coordinate error is not a possession score.',
        'reference_reads':run.READS})
    print(json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--match',choices=['J03WPY','J03WQQ'],default='J03WQQ');p.add_argument('--scope',action='store_true');args=p.parse_args()
    sys.addaudithook(run.guard);main(args)
