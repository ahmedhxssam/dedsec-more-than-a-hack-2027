"""Match-local ownership learning from visible, confident physical control.

No primary labels enter. Student features contain observed players/team/pitch
only. The observed ball and frozen ownership score select pseudo anchors.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from second_zero import global_team_context as context

MODEL=dict(max_iter=180,max_leaf_nodes=15,max_depth=5,min_samples_leaf=80,
           learning_rate=.05,l2_regularization=10.,random_state=20261030,early_stopping=False)


def anchors(raw,teacher,baseline):
    if set(raw)!={'ball','players','team','pitch'}:raise ValueError('Raw observations only')
    ball=np.asarray(raw['ball']);players=np.asarray(raw['players']);team=np.asarray(raw['team']);n=len(ball)
    distance=np.linalg.norm(players-ball[:,None],axis=-1);distance[~np.isfinite(distance)]=np.inf
    near=np.stack([np.min(np.where((team==t)[None],distance,np.inf),axis=1) for t in [0,1]],1)
    owner=np.argmin(near,axis=1);closest=np.min(near,axis=1);opponent=np.max(near,axis=1)
    speed=np.full(n,np.inf);seen=np.isfinite(ball).all(1)
    if n>1:
        valid=seen[:-1]&seen[1:];motion=np.linalg.norm(np.diff(ball,axis=0),axis=1)*25
        speed[1:]=np.where(valid,motion,np.inf)
    margin=np.full(n,-np.inf);np.subtract(opponent,closest,out=margin,where=np.isfinite(opponent)&np.isfinite(closest))
    mask=seen&(np.asarray(baseline)<2)&(closest<2.)&(margin>2.)&(np.abs(teacher)>=1.5)&((teacher>0)==owner)&(speed<12.)
    indices=np.flatnonzero(mask & (np.arange(n)%5==0))
    return indices,owner[indices].astype(np.int8)


def rows(x,indices,owner):
    xx=np.concatenate([x[indices,0],x[indices,1]])
    yy=np.r_[owner==0,owner==1].astype(np.int8)
    return xx,yy


def owner_logit(model,x):
    logits=[]
    for t in [0,1]:
        p=model.predict_proba(x[:,t])[:,1];logits.append(logit(np.clip(p,1e-6,1-1e-6)))
    return .5*(logits[1]-logits[0])


def fit_predict(raw,teacher,baseline):
    indices,owner=anchors(raw,teacher,baseline)
    if len(indices)<200:return np.asarray(teacher).copy(),None,{'anchors':len(indices),'accepted':False,'reason':'fewer_than_200_visible_control_anchors'}
    x=context.features(raw['players'],raw['team'],raw['pitch']);frame=np.arange(len(x));held=(frame//500)%5==0;buffered=maximum_filter1d(held.astype(np.int8),501,mode='constant')>0
    train=~buffered[indices];valid=held[indices]
    if train.sum()<100 or valid.sum()<50:return np.asarray(teacher).copy(),None,{'anchors':len(indices),'accepted':False,'reason':'insufficient_buffered_block_validation'}
    xx,yy=rows(x,indices[train],owner[train]);cv=HistGradientBoostingClassifier(**MODEL).fit(xx,yy)
    vx,vy=rows(x,indices[valid],owner[valid]);prob=np.clip(cv.predict_proba(vx)[:,1],1e-8,1-1e-8);nll=float(np.mean(-vy*np.log(prob)-(1-vy)*np.log1p(-prob)));accepted=nll<np.log(2.)
    if not accepted:return np.asarray(teacher).copy(),None,{'anchors':len(indices),'accepted':False,'blocked_pseudo_NLL':nll,'reason':'not_better_than_balanced_chance'}
    xx,yy=rows(x,indices,owner);model=HistGradientBoostingClassifier(**MODEL).fit(xx,yy);predicted=owner_logit(model,x)
    return predicted,model,{'anchors':len(indices),'accepted':True,'blocked_pseudo_NLL':nll,'blocked_train_anchors':int(train.sum()),'blocked_validation_anchors':int(valid.sum()),'anchors_H':int((owner==0).sum()),'anchors_A':int((owner==1).sum()),'primary_references_used':False}


def blend(teacher,local,support,alpha):
    w=np.clip(alpha*np.asarray(support),0,1);p=(1-w)*expit(teacher)+w*expit(local)
    return logit(np.clip(p,1e-7,1-1e-7)).astype(np.float32)
