"""Ball-independent anonymous team shape and directional flow.

Only observed player coordinates and team codes enter. Positions are not filled,
stationary observations remain visible, and implausible adjacent-slot motion is
masked. Each perspective uses the sign of own-minus-opponent centroid per axis;
this is relative formation orientation, not an assumed attacking direction.
"""
from __future__ import annotations
import numpy as np

VERSION = 'global-team-context-v1'
CHANNELS = 290
CURRENT_CHANNELS = 122
WINDOWS = (25, 125, 250)
TEMPORAL_TEAM_INDICES = (0, 1, 2, 5, 6, 18, 19, 20, 21, 22, 23, 24, 25, 29)
TEAM_NAMES = ['observed_mass', 'velocity_coverage', 'coordinate_repeat_fraction',
    'absolute_center_x', 'absolute_center_y', 'spread_x', 'spread_y', 'covariance_xy',
    *['relative_x_q'+str(q) for q in (10,25,50,75,90)],
    *['relative_y_q'+str(q) for q in (10,25,50,75,90)],
    'mean_velocity_x', 'mean_velocity_y', 'median_velocity_x', 'median_velocity_y',
    'mean_speed', 'speed_q90', 'flow_coherence', 'radial_expansion',
    'opponent_distance_q10', 'opponent_distance_q50', 'opponent_distance_q90',
    'opponent_within3_fraction', 'teammate_distance_median',
    *[f'cell_{kind}_{x}_{y}' for kind in ('mass','vx','vy') for x in range(3) for y in range(3)]]
PAIR_NAMES = ['centroid_distance', 'centroid_dx', 'centroid_dy',
              'relative_velocity_x', 'relative_velocity_y', 'flow_alignment']


def names():
    current = ['own_'+s for s in TEAM_NAMES] + ['opponent_'+s for s in TEAM_NAMES] + PAIR_NAMES
    selected = [p+TEAM_NAMES[j] for p in ('own_', 'opponent_') for j in TEMPORAL_TEAM_INDICES]
    return current + [f'{direction}{window}_'+s for window in WINDOWS
                      for direction in ('past','future') for s in selected]


def _mean(value, weight):
    return (value * weight).sum(1) / np.maximum(weight.sum(1), 1e-8)


def _quantile(value, weight, qs):
    order = np.argsort(value, axis=1, kind='stable')
    x = np.take_along_axis(value, order, 1)
    w = np.take_along_axis(weight, order, 1)
    cumulative = np.cumsum(w, 1)
    total = cumulative[:, -1]
    result = []
    for q in qs:
        at = np.argmax(cumulative >= (total*q)[:, None], axis=1)
        result.append(np.where(total > 0, x[np.arange(len(x)), at], 0.))
    return np.stack(result, 1)


def motion(players, fps=25.):
    """Average adjacent valid velocities; never associate across missing frames."""
    p = np.asarray(players, np.float32)
    available = np.isfinite(p).all(-1)
    d = np.diff(p, axis=0) * fps
    valid = available[1:] & available[:-1] & (np.linalg.norm(d, axis=-1) <= 12.)
    d = np.where(valid[..., None], d, 0.)
    velocity = np.zeros_like(p); count = np.zeros(p.shape[:-1], np.float32)
    velocity[1:] += d; velocity[:-1] += d
    count[1:] += valid; count[:-1] += valid
    velocity /= np.maximum(count[..., None], 1.)
    repeated = np.zeros(p.shape[:-1], bool)
    repeat = valid & (np.linalg.norm(d, axis=-1) < 1e-5)
    repeated[1:] |= repeat; repeated[:-1] |= repeat
    return velocity, count > 0, repeated


def _chunk(players, team, velocity, velocity_valid, repeated, pitch):
    n, p = players.shape[:2]
    available = np.isfinite(players).all(-1)
    available &= (players[...,0]>=0)&(players[...,0]<=pitch[0])
    available &= (players[...,1]>=0)&(players[...,1]<=pitch[1])
    available &= (team>=0)[None]
    xy = np.where(available[...,None], players, 0.)
    # Exact duplicated coordinates get one total vote, not one vote per slot.
    difference = xy[:,:,None] - xy[:,None]
    distance = np.linalg.norm(difference, axis=-1)
    same = (team[:,None] == team[None,:])[None]
    duplicate = (distance<1e-4)&same&available[:,:,None]&available[:,None]
    weight = available / np.maximum(duplicate.sum(2),1)
    weights = [weight*(team==t)[None] for t in (0,1)]
    centers = np.stack([np.stack([_mean(xy[...,a],w) for a in (0,1)],1) for w in weights],1)
    velocity_valid = velocity_valid & available
    vv = np.where(velocity_valid[...,None], velocity, 0.)
    means = np.stack([np.stack([_mean(vv[...,a],w*velocity_valid) for a in (0,1)],1) for w in weights],1)
    current = np.zeros((n,2,CURRENT_CHANNELS),np.float32)
    for perspective in (0,1):
        direction = np.sign(centers[:,perspective]-centers[:,1-perspective])
        summaries = []
        for t in (perspective,1-perspective):
            w = weights[t]; vm = w*velocity_valid
            centered = (xy-centers[:,t,None])*direction[:,None]
            pv = vv*direction[:,None]
            speed = np.linalg.norm(vv,axis=-1)
            mean_speed = _mean(speed,vm)
            mean_v = np.stack([_mean(pv[...,a],vm) for a in (0,1)],1)
            radial = (vv*(xy-centers[:,t,None])).sum(-1)/np.maximum(np.linalg.norm(xy-centers[:,t,None],axis=-1),.1)
            other = available & (team==1-t)[None]
            opponent_distance = np.min(np.where(other[:,None],distance,np.inf),2)
            own = available & (team==t)[None]
            teammate_distance = np.min(np.where(own[:,None]&(distance>1e-4),distance,np.inf),2)
            opp_valid = w*np.isfinite(opponent_distance)
            own_valid = w*np.isfinite(teammate_distance)
            opponent_distance = np.nan_to_num(opponent_distance,posinf=0.)
            teammate_distance = np.nan_to_num(teammate_distance,posinf=0.)
            values = [w.sum(1)/11.,_mean(velocity_valid,w),_mean(repeated,w),
                abs(centers[:,t,0]-pitch[0]/2)/(pitch[0]/2),
                abs(centers[:,t,1]-pitch[1]/2)/(pitch[1]/2),
                np.sqrt(_mean(centered[...,0]**2,w))/pitch[0],
                np.sqrt(_mean(centered[...,1]**2,w))/pitch[1],
                _mean(centered[...,0]*centered[...,1],w)/np.prod(pitch)]
            values.extend(_quantile(centered[...,0]/pitch[0],w,(.1,.25,.5,.75,.9)).T)
            values.extend(_quantile(centered[...,1]/pitch[1],w,(.1,.25,.5,.75,.9)).T)
            values.extend([mean_v[:,0]/12,mean_v[:,1]/12,
                *_quantile(pv[...,0]/12,vm,(.5,)).T,*_quantile(pv[...,1]/12,vm,(.5,)).T,
                mean_speed/12,*_quantile(speed/12,vm,(.9,)).T,
                np.linalg.norm(mean_v,axis=1)/np.maximum(mean_speed,.1),_mean(radial,vm)/12,
                *_quantile(opponent_distance/50,opp_valid,(.1,.5,.9)).T,
                _mean(opponent_distance<3,opp_valid),*_quantile(teammate_distance/50,own_valid,(.5,)).T])
            # Coarse relative-position occupancy and its signed motion fields.
            gx=np.clip(np.floor((centered[...,0]+15)/10),0,2).astype(int)
            gy=np.clip(np.floor((centered[...,1]+12)/8),0,2).astype(int)
            for kind in ('mass','vx','vy'):
                for a in range(3):
                    for b in range(3):
                        cell=(gx==a)&(gy==b)
                        values.append((w*cell).sum(1)/11 if kind=='mass' else _mean(pv[...,0 if kind=='vx' else 1]/12,vm*cell))
            summary=np.stack(values,1)
            summary[w.sum(1)==0]=0
            summaries.append(summary)
        delta=centers[:,perspective]-centers[:,1-perspective]
        md=(means[:,perspective]-means[:,1-perspective])*direction
        alignment=(means[:,perspective]*means[:,1-perspective]).sum(1)/np.maximum(np.linalg.norm(means[:,perspective],axis=1)*np.linalg.norm(means[:,1-perspective],axis=1),.1)
        pair=np.c_[np.linalg.norm(delta,axis=1)/50,abs(delta[:,0])/pitch[0],abs(delta[:,1])/pitch[1],md/12,alignment]
        pair[(weights[0].sum(1)==0)|(weights[1].sum(1)==0)]=0
        current[:,perspective]=np.c_[summaries[0],summaries[1],pair]
    return current


def features(players, team, pitch=(105.3,68.), fps=25., chunk=1024):
    """Return finite native-frame [N,2,290] features, no fitted ancestors."""
    players=np.asarray(players,np.float32);team=np.asarray(team,np.int64);pitch=np.asarray(pitch,np.float32)
    if players.ndim!=3 or players.shape[-1]!=2 or team.shape!=(players.shape[1],):
        raise ValueError('Expected players[N,P,2] and fixed team[P]')
    if np.any((team<-1)|(team>1)) or pitch.shape!=(2,) or np.any(pitch<=0):
        raise ValueError('Invalid team codes or pitch')
    n=len(players)
    if n==0:return np.empty((0,2,CHANNELS),np.float32)
    if not players.shape[1]:return np.zeros((n,2,CHANNELS),np.float32)
    v,valid,repeated=motion(players,fps)
    current=np.empty((n,2,CURRENT_CHANNELS),np.float32)
    for start in range(0,n,chunk):
        sl=slice(start,min(n,start+chunk))
        current[sl]=_chunk(players[sl],team,v[sl],valid[sl],repeated[sl],pitch)
    indices=np.array([*TEMPORAL_TEAM_INDICES,*[58+i for i in TEMPORAL_TEAM_INDICES]])
    selected=current[:,:,indices]
    prefix=np.concatenate([np.zeros((1,2,len(indices))),np.cumsum(selected,axis=0,dtype=np.float64)],axis=0)
    frames=np.arange(n);context=[]
    for window in WINDOWS:
        lo=np.maximum(0,frames-window);hi=np.minimum(n,frames+window+1)
        context.append((prefix[frames]-prefix[lo])/np.maximum((frames-lo)[:,None,None],1))
        context.append((prefix[hi]-prefix[frames+1])/np.maximum((hi-frames-1)[:,None,None],1))
    result=np.concatenate([current,*context],axis=-1).astype(np.float32)
    if result.shape!=(n,2,CHANNELS) or not np.isfinite(result).all():raise ValueError('Feature failure')
    return result
