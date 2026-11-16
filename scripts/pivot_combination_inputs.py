"""Read-only wide50 parents and fixed independent-evidence composition."""
from pathlib import Path
import pickle
import numpy as np
from scipy.special import expit,logit
from scripts import pivot_wide_graph as wide
from scripts import source_teacher_graph_pass as source
from scripts import anchor_export_reduced_crossfit as cf
from second_zero import source_teacher_graph as adapter

ROOT=wide.ROOT


def load_heads(fold):
    with (source.joint.PRIOR/f'fold{fold}/model.pkl').open('rb')as h:q=pickle.load(h)
    with (source.RUN/'heads'/f'fold{fold}-teacher92.pkl').open('rb')as h:head=pickle.load(h)
    return q,head


def parent_view(fold,key,extra_owner=None,weight=.25,heads=None):
    qmodel,head=load_heads(fold)if heads is None else heads
    with np.load(source.RUN/'cache'/f'fold{fold}'/(key+'.npz'))as z:x=z['x'].copy();base=z['base']
    with np.load(wide.RUN/'wide50'/f'fold{fold}/predictions'/(key+'.npz'))as z:
        x[:,0]=z['logit'];expected=z['predicted']
    if extra_owner is not None:
        if len(extra_owner)!=len(x)or not np.isfinite(extra_owner).all():raise ValueError('Aligned finite evidence required')
        x[:,0]=logit(np.clip((1-weight)*expit(x[:,0])+weight*expit(extra_owner),1e-7,1-1e-7))
    with np.load(source.RUN/'teacher'/(key+'.npz'))as z:teacher={k:z[k]for k in z.files}
    old=cf.pool(x,base);oq=source.quality_predict(qmodel,x,old)
    before=cf.graph_prediction(x,base,old,oq,.45);new,origin=adapter.proposals(base,before,old,teacher)
    aq=source.quality_predict(qmodel,x,new);common,treatment=adapter.features(x,new,aq,teacher)
    quality=np.r_[oq,np.clip(head.predict(treatment),0,1)];pool=np.r_[old,new]
    pred=cf.graph_prediction(x,base,pool,quality,.45)
    np.testing.assert_array_equal(pred[base>=2],base[base>=2])
    if extra_owner is None:np.testing.assert_array_equal(pred,expected)
    return {'x':x,'base':base,'predicted':pred,'pool':pool,'quality':quality,'old_segments':old,
            'old_quality':oq,'new_segments':new,'teacher92':treatment,'common52':common,'origin':origin}
