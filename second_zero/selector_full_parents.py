"""Actual complete frozen baseline/challenger parents, no reference reads."""
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit,logit
from scripts import region_gap_recovery_v8 as region

b=region.b;e=region.e;s=region.s;gs=region.gs;ROOT=region.ROOT;OLD=region.OLD


def load_archive(path,names=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in (z.files if names is None else names)}


def context(which,match,half):
    key=f'{match}-{half}-corrupted';raw=b.raw_half(key)
    if match=='J03WPY':
        path=e.parent_wpy(which)/f'{key}.npz';p=load_archive(path);owners={f'owner_{i}':p[f'owner_{i}'] for i in range(3)}
    else:
        p=b.ceiling.current(match,half,'corrupted');folder=OLD/'full-transfer/full_baseline' if which=='full' else OLD/f'baseline/fold{which}';path=folder/f'{key}.npz';owners=load_archive(path,[f'owner_{i}' for i in range(3)])
    quality=b.original_quality(which);credit,_=b.fitted(which);cc=s.read(e.RUN/f'models/{which}/calibration.json');canonical=e.component(p['signals'],owners);mask=p['eligible_mask']
    arcs=b.dense.proposals(p['predicted'],mask,p['segments']);qo=b.old_quality(quality,p['signals'],arcs);qr=e.EvidenceIntervals(raw,canonical).predict(credit,arcs);baseline=gs.decode(p['predicted'],mask,arcs,(1-cc['alpha'])*qo+cc['alpha']*qr,.4,p['signals'][:,0],.01)
    if match!='J03WPY':np.testing.assert_array_equal(baseline,region.c.previous.incumbent(match,half,'corrupted'))
    model,freeze=s.fitted(which);cal=s.read(s.RUN/f'models/{which}/calibration.json');rc=s.read(region.RUN/f'calibration/{which}.json');assert cal['model']==rc['model']==freeze['model'];new=expit(s.predict_owner(model,s.owner.features(raw,canonical))/cal['temperature']);support=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest')
    alternatives=[];rich=None;all_arcs=[arcs];rich_quality=None
    for name in ['fixed025','calibrated','region']:
        weight=.25*support if name=='fixed025' else cal['alpha']*support if name=='calibrated' else cal['alpha']*support+rc['observed_alpha']*(1-support)
        signals=p['signals'].copy();signals[:,0]=logit(np.clip((1-weight)*expit(signals[:,0])+weight*new,1e-7,1-1e-7)).astype(np.float32);preliminary,aa,_=s.decode_graph(signals,p['status_mask'],quality)
        if name=='region' and rc['observed_alpha']>0:
            dense=b.dense.proposals(preliminary,mask,aa);aa=gs.pool(preliminary,preliminary<2,np.r_[aa,dense,gs.spells(baseline)]);mutable=preliminary<2
        else:
            preliminary[~mask]=baseline[~mask];aa=b.dense.proposals(preliminary,mask,aa);mutable=mask
        oq=b.old_quality(quality,signals,aa);nq=e.EvidenceIntervals(raw,e.component(signals,owners)).predict(credit,aa);pred=gs.decode(preliminary,mutable,aa,(1-cc['alpha'])*oq+cc['alpha']*nq,.4,signals[:,0],.01)
        if name in ['fixed025','calibrated'] and match in s.TRAIN:
            expected=load_archive(s.RUN/f'oof/fold{which}'/(key+'.npz'),[name])[name];np.testing.assert_array_equal(pred,expected)
        if name=='region' and match!='J03WPY':
            folder=region.RUN/'full-predictions' if which=='full' else region.RUN/f'oof/fold{which}';expected=load_archive(folder/(key+'.npz'),['region'])['region'];np.testing.assert_array_equal(pred,expected)
        alternatives.append(pred);all_arcs.append(aa)
        if name=='region':rich=signals;rich_quality=(aa,oq,nq)
    # Local alternatives are proposed against the actual .7708 parent, with
    # support from either complete decoded model's interval inventory.
    pool=gs.pool(baseline,baseline<2,np.concatenate(all_arcs));q=b.old_quality(quality,p['signals'],pool)
    return {'raw':raw,'baseline':baseline,'alternatives':alternatives,'baseline_signals':p['signals'],'rich_signals':rich,'pool':pool,'pool_quality':q,'quality_model':quality,'producer':str(which),'primary_parent':str(path.relative_to(ROOT)),'replayed_baseline':match!='J03WPY','replayed_complete_challengers':match!='J03WPY'}
