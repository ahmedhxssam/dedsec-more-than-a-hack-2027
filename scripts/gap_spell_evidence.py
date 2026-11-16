"""Second prospective scorer: raw intervals plus strictly crossfit ownership.

Old stage-v2 code is imported unchanged. Explicit overrides add175 features
with target-excluded owner/imputer parents; all validation uses actual .7694.
"""
from __future__ import annotations
import argparse,datetime,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)];sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
import numpy as np
import torch
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingRegressor
from second_zero.common import sha256
from second_zero.temporal_ownership import TemporalOwner,infer_logits,trajectory_features
from second_zero.temporal_live_dead import trajectory_features as status_features
from scripts.aggressive_unseen_pipeline import physical_geometry
from scripts import gap_spell_train as base
from scripts import sprint_decoder_segments_geometry_v2 as graph
from second_zero.gap_spell_model import RawIntervals,training_proposals,MODEL

RAW_RUN=base.RUN;RUN=ROOT/'runs/gap-spell-v3-evidence';REPORT=ROOT/'reports/gap-spell-v3-evidence';OLD=base.OLD
base.RUN=RUN;base.REPORT=REPORT
PARENTS={}


class EvidenceIntervals(RawIntervals):
    def __init__(self,raw,signals):super().__init__(raw);self.signals=signals
    def features(self,arcs):return np.c_[super().features(arcs),graph.segment_features(self.signals,arcs)].astype(np.float32)


def prepare():
    original=base.read(RAW_RUN/'scorer-protocol.json');first=base.read(RAW_RUN/'oof-audit.json');assert first['passed']
    p={**original,'kind':'crossfit-evidence-gap-spell-v3','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'features':'747: original572 raw-only interval features plus175 canonical segment evidence features. x0 is calibrated uniform3 probability-mean logit, NOT final GRU/carrier/global logit, consistently in fitting/calibration/inference; x1 short, x2/3uniform dispersion/sign, raw status36/38/39, physical6.',
       'training_ancestry':'Outer TRAIN2 training samples use OTHER TRAIN1 uniform3/short/imputer, own WPY-only temperatures, excluding target and outerheld. Full TRAIN4 fitting uses original outer TRAIN2 producers for each opposite held target. GT used only for fitting interval sampling/targets. No fullTRAIN4 feature parent in outer fits.',
       'training':'Same24 raw-v2 fitting interval samples/targets. Reuse572 raw columns exactly, append175 correctly excluded evidence columns; raw-v2 model settings unchanged. Corruption271 new owner inference uses same target-excluded frozen parent weights.',
       'arms':{'proposal_control':original['arms']['proposal_control'],'mixed':'Same control plus analytic WPY-calibrated evidence-credit mixture, penalty.55/frame.01','raw_credit':'747-feature evidence-credit model alone, fixed penalty.4/frame0'},
       'stage_disclosure':'Added after raw-v2 OOF gain was small and mostly proposal control; before ANY new full WQQ model candidate. Both family outcomes reported. Global best OOF family alone can get one full transfer test.',
       'storage':'Shared immutable572 raw columns plus compressed175 evidence columns; concatenate only in memory. Storage-only repair before any evidence fit; old source/protocol and golden full-matrix digest preserved.',
       'sources':{**original['sources'],base.rel(Path(__file__)):sha256(Path(__file__))},
       'inputs':{**original['inputs'],base.rel(RAW_RUN/'scorer-protocol.json'):sha256(RAW_RUN/'scorer-protocol.json'),base.rel(RAW_RUN/'oof-audit.json'):sha256(RAW_RUN/'oof-audit.json'),base.rel(RAW_RUN/'training-cache-seal.json'):sha256(RAW_RUN/'training-cache-seal.json'),base.rel(OLD/'owner/complete.json'):sha256(OLD/'owner/complete.json')}}
    base.write(RUN/'scorer-protocol.json',p);print('EVIDENCE PROTOCOL',sha256(RUN/'scorer-protocol.json'),flush=True)


def producer(match,mode):
    if mode=='oof':
        f=0 if match in base.TRAIN[2:] else 1;train=base.FOLDS[f][0];other=next(m for m in train if m!=match);folder=OLD/f'owner/fold{f}/inner/fit-{other}';manifest=folder/'manifest.json'
    else:
        f=0 if match in base.TRAIN[:2] else 1;folder=OLD/f'owner/fold{f}';manifest=folder/'reuse-manifest.json'
    m=base.read(manifest);assert match not in m['train_matches']
    return folder,m['components'],m['train_matches'],manifest


def evidence(match,half,view,mode,raw):
    folder,parts,train,manifest=producer(match,mode);key=(mode,match)
    if key not in PARENTS:
        models={}
        for name in ['uniform0','uniform1','uniform2','short']:
            item=parts[name];path=ROOT/item['model'];assert sha256(path)==item['model_sha256'];ck=torch.load(path,map_location='cpu',weights_only=True);model=TemporalOwner(ck['channels'],ck['width']);model.load_state_dict(ck['state_dict']);model.eval();models[name]=model
        item=parts['imputer'];path=ROOT/item['model'];assert sha256(path)==item['model_sha256']
        with path.open('rb') as f:imputer=pickle.load(f)
        PARENTS[key]=(models,imputer)
    models,imputer=PARENTS[key];native=trajectory_features(raw['ball'],raw['players'],raw['team'],raw['pitch']);sx=status_features(raw['ball'],raw['players'],raw['team'],raw['pitch'])
    if view!='271':
        k=f'{match}-{half}-'+('clean' if view=='clean' else 'corrupted');path=folder/'experts'/(k+'.npz')
        with np.load(path,allow_pickle=False) as z:zs=np.column_stack([z[n] for n in ['uniform0','uniform1','uniform2','short']]).astype(np.float32)
    else:
        torch.set_num_threads(3);zs=np.column_stack([infer_logits(models[n],native)/parts[n]['temperature'] for n in ['uniform0','uniform1','uniform2','short']]).astype(np.float32);torch.set_num_threads(2)
    physical=physical_geometry(raw,native,imputer);mean=logit(np.clip(expit(zs[:,:3]).mean(1),1e-10,1-1e-10));x=np.c_[mean,zs[:,3],zs[:,:3].std(1),np.sign(zs[:,:3]).mean(1),sx[:,[36,38,39]],physical].astype(np.float32)
    return x,{'label_matches':train,'target_match':match,'manifest':base.pin(manifest),'weight_and_calibration_parts':parts}


def cache(mode):
    base.verify();pm=base.official_metrics();old=base.read(RAW_RUN/'training-cache-seal.json');rows={}
    for mi,match in enumerate(base.TRAIN):
        base.ALLOWED.clear();base.ALLOWED.add(match)
        for hi,half in enumerate(['firstHalf','secondHalf']):
            y=base.truth(match,half)
            for vi,view in enumerate(['clean','100','271']):
                start=time.monotonic();raw=base.raw(match,half,view);arcs=training_proposals(y,raw,pm,20261030+mi*100+hi*10+vi);stem=RAW_RUN/'train-cache'/match/f'{half}-{view}';meta=old['rows'][base.rel(stem)];xr=np.load(base.check(meta['x']));target=np.load(base.check(meta['y']));np.testing.assert_array_equal(base.gs.credit(y,arcs,pm).astype(np.float32),target)
                signals,ancestry=evidence(match,half,view,mode,raw);extra=graph.segment_features(signals,arcs);x=np.c_[xr,extra].astype(np.float32);assert x.shape==(20000,747)
                dest=RUN/'train-cache'/mode/match/f'{half}-{view}';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(str(dest)+'-evidence.npz',features=extra)
                if mode=='oof' and match=='J03WMX' and half=='firstHalf' and view=='clean':
                    golden=RUN/'repairs/storage-v1/golden-evidence.npz'
                    if golden.exists():
                        with np.load(golden,allow_pickle=False) as z:np.testing.assert_array_equal(extra,z['features'])
                        golden_record=base.read(RUN/'repairs/storage-v1/record.json');assert __import__('hashlib').sha256(x.tobytes()).hexdigest()==golden_record['old_complete_feature_bytes_sha256']
                rows[base.rel(dest)]={'match':match,'view':view,'mode':mode,'rows':len(arcs),'features':747,'raw_x':meta['x'],'evidence_x':base.pin(Path(str(dest)+'-evidence.npz')),'y':meta['y'],'raw_source':meta,'crossfit_ancestor':ancestry,'arc_sha256':__import__('hashlib').sha256(arcs.tobytes()).hexdigest(),'seconds':time.monotonic()-start}
                print('EVIDENCE CACHE',mode,match,half,view,x.shape,'seconds',time.monotonic()-start,flush=True);del x,xr,extra,signals,raw
    base.ALLOWED.clear();base.write(RUN/f'training-cache-seal-{mode}.json',{'protocol_sha256':sha256(RUN/'scorer-protocol.json'),'rows':rows,'references':base.READS})


def fit(which):
    base.verify();mode='full' if which=='full' else 'oof';matches=base.TRAIN if mode=='full' else base.FOLDS[int(which)][0];base.ALLOWED.update(matches);seal=RUN/f'training-cache-seal-{mode}.json';cache=base.read(seal);xx=[];yy=[];inputs={};ancestors={};start=time.monotonic()
    for row in cache['rows'].values():
        if row['match'] not in matches:continue
        labels=row['crossfit_ancestor']['label_matches'];assert row['match'] not in labels and set(labels)<=set(matches)
        raw_x=np.load(base.check(row['raw_x']))
        with np.load(base.check(row['evidence_x']),allow_pickle=False) as z:extra=z['features']
        xx.append(np.c_[raw_x,extra].astype(np.float32));yy.append(np.load(base.check(row['y'])));inputs[row['raw_x']['path']]=row['raw_x']['sha256'];inputs[row['evidence_x']['path']]=row['evidence_x']['sha256'];inputs[row['y']['path']]=row['y']['sha256'];ancestors[row['match']]=row['crossfit_ancestor']
    x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy;model=HistGradientBoostingRegressor(**MODEL).fit(x,y);folder=RUN/'models'/str(which);folder.mkdir(parents=True,exist_ok=False)
    with (folder/'model.pkl').open('xb') as f:pickle.dump(model,f)
    base.write(folder/'freeze.json',{'train_matches':matches,'calibration_matches':[],'raw_only_no_learned_ancestors':False,'strict_crossfit_feature_ancestors':ancestors,'shape':list(x.shape),'model':base.pin(folder/'model.pkl'),'protocol_sha256':sha256(RUN/'scorer-protocol.json'),'training_cache_seal':base.pin(seal),'training_inputs':inputs,'seconds':time.monotonic()-start});base.ALLOWED.clear();print('EVIDENCE FIT',which,x.shape,time.monotonic()-start,flush=True)


def component(signals,owners):
    x=signals.copy();z=np.column_stack([owners['owner_'+str(i)] for i in range(3)]).astype(np.float32);x[:,0]=logit(np.clip(expit(z).mean(1),1e-10,1-1e-10));return x


def parent_wpy(which):
    folder=RAW_RUN/'calibration-parent'/str(which)
    if not (folder/'seal.json').exists():
        # Full parent is computed once by the unchanged stage-v2 producer.
        folder=ORIGINAL_PARENT_WPY(which)
    for p,h in base.read(folder/'seal.json')['outputs'].items():
        assert sha256(ROOT/p)==h
    return folder


def calibrate(which):
    base.verify();model,freeze=base.fitted(which);folder=parent_wpy(which);qmodel=base.original_quality(which);pm=base.official_metrics();base.ALLOWED.clear();base.ALLOWED.add('J03WPY');numerator=0.;denominator=0.;views={};deps={}
    for half in ['firstHalf','secondHalf']:
        path=folder/f'J03WPY-{half}-corrupted.npz'
        with np.load(path,allow_pickle=False) as z:p={k:z[k] for k in z.files}
        raw=base.raw_half(f'J03WPY-{half}-corrupted');pack=EvidenceIntervals(raw,component(p['signals'],p));arcs=base.dense.proposals(p['predicted'],p['eligible_mask'],p['segments']);qr=pack.predict(model,arcs);qo=base.old_quality(qmodel,p['signals'],arcs);target=base.gs.credit(base.truth('J03WPY',half),arcs,pm);d=qr-qo;numerator+=float(np.mean(d*(target-qo)));denominator+=float(np.mean(d*d));deps[base.rel(path)]=sha256(path);views[half]={'proposals':len(arcs),'raw_MSE':float(np.mean((qr-target)**2)),'old_MSE':float(np.mean((qo-target)**2))}
    alpha=float(np.clip(numerator/max(denominator,1e-12),0,1));base.write(RUN/'models'/str(which)/'calibration.json',{'calibration_matches':['J03WPY'],'alpha':alpha,'model':freeze['model'],'dependencies':deps,'views':views,'objective':'equal-view squared independent IoU-credit error; analytic convex blend; no SpellF1 selection','reference_reads':base.READS});base.ALLOWED.clear();print('EVIDENCE ALPHA',which,alpha,flush=True)


def prediction(which,match,half,condition,arms):
    p=base.ceiling.current(match,half,condition);b=p['predicted'];mask=p['eligible_mask'];out={'baseline':b.copy(),'eligible_mask':mask}
    if not mask.any():out.update({a:b.copy() for a in arms});return out
    raw=base.raw_half(f'{match}-{half}-{condition}');arcs=base.dense.proposals(b,mask,p['segments']);qo=base.old_quality(base.original_quality(which),p['signals'],arcs)
    if any(a!='proposal_control' for a in arms):
        model,freeze=base.fitted(which);cal=base.read(RUN/'models'/str(which)/'calibration.json');assert cal['model']==freeze['model']
        folder=OLD/'full-transfer/full_baseline' if match=='J03WQQ' else OLD/f'baseline/fold{0 if match in base.TRAIN[:2] else 1}'
        with np.load(folder/f'{match}-{half}-{condition}.npz',allow_pickle=False) as z:owners={f'owner_{i}':z[f'owner_{i}'] for i in range(3)}
        qr=EvidenceIntervals(raw,component(p['signals'],owners)).predict(model,arcs);alpha=cal['alpha']
    for a in arms:
        q=qo if a=='proposal_control' else (1-alpha)*qo+alpha*qr if a=='mixed' else qr;penalty=.4 if a=='raw_credit' else .55;fw=0 if a=='raw_credit' else .01
        out[a]=base.gs.decode(b,mask,arcs,q,penalty,p['signals'][:,0],fw)
    return out


def validated_oof():
    base.verify();seal=base.read(RUN/'oof-prediction-seal.json');assert len(seal['outputs'])==16 and not seal['held_truth_reads'];loaded={}
    for f,(train,held) in enumerate(base.FOLDS):
        meta=seal['models'][str(f)];freeze=base.read(base.check(meta['freeze']));cal=base.read(base.check(meta['calibration']));base.check(freeze['model']);base.check(freeze['training_cache_seal']);assert freeze['train_matches']==train and not freeze['raw_only_no_learned_ancestors'];assert cal['calibration_matches']==['J03WPY'] and cal['model']==freeze['model']
        for target,a in freeze['strict_crossfit_feature_ancestors'].items():
            assert target in train and target not in a['label_matches'] and set(a['label_matches'])<=set(train);base.check(a['manifest'])
        for key in base.keys(held):
            path=RUN/f'oof/fold{f}'/(key+'.npz');assert sha256(path)==seal['outputs'][base.rel(path)]
            with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            match,half,c=key.split('-');old=base.ceiling.current(match,half,c);np.testing.assert_array_equal(v['baseline'],old['predicted']);np.testing.assert_array_equal(v['eligible_mask'],old['eligible_mask'])
            for a in base.ARMS:
                base.gs.check_lock(v[a],v['baseline'],v['eligible_mask'])
                if c=='clean':np.testing.assert_array_equal(v[a],v['baseline'])
            with np.load(RAW_RUN/f'oof/fold{f}'/(key+'.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(v['proposal_control'],z['proposal_control'])
            loaded[key]=v
    return seal,loaded


ORIGINAL_PARENT_WPY=base.parent_wpy
base.fit=fit;base.calibrate=calibrate;base.prediction=prediction;base.validated_oof=validated_oof;base.parent_wpy=parent_wpy


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','cache','fit','calibrate','predict_oof','audit_oof','fit_full','predict_full','audit_full']);parser.add_argument('--fold',choices=['0','1']);parser.add_argument('--mode',choices=['oof','full'],default='oof');args=parser.parse_args();base.STAGE=args.stage
    base.threadpool_limits(2);torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(base.guard)
    if args.stage=='prepare':prepare()
    elif args.stage=='cache':cache(args.mode)
    elif args.stage=='fit':fit(args.fold)
    elif args.stage=='calibrate':calibrate(args.fold)
    elif args.stage=='fit_full':
        selected=base.authorize_full()
        if selected!='proposal_control':cache('full')
        base.fit_full()
    else:getattr(base,args.stage)()
