"""Observation-only, phase-aligned data for latent ownership learning."""
from __future__ import annotations
import collections,json
from pathlib import Path
import numpy as np
from .aggressive_controller import (_bounded,_velocity,crop_features,pack_values,
    held_coordinates,FEATURE_CONTEXT)
from .vision_spell_balance import full_spell_weights
from .common import sha256

ROOT=Path(__file__).resolve().parents[1]
TRAIN4=('J03WMX','J03WN1','J03WOH','J03WOY')
SOURCE=ROOT/'private/improvement/skillcorner-20261029'
PRIMARY=ROOT/'private/improvement/external/prepared'
_CACHE=collections.OrderedDict()
BALL_CHANNELS=('provided_x_centered_div50','provided_y_centered_div50',
    'bounded_x_centered_div50','bounded_y_centered_div50',
    'provided_vx_div40','provided_vy_div40','bounded_vx_div40','bounded_vy_div40',
    'provided_position_available','bounded_position_available',
    'previous_observed_age_div250','next_observed_age_div250')


def _safe(path):
    text=str(path)
    if any(x in text for x in ('J03WQQ','J03WR9','/evaluation/gt/','/runs/main/')):
        raise PermissionError('Reserved and competition data are excluded')
    return Path(path)


def valid_starts(mask,target=200,step=5):
    """Global multiples of step whose entire target interval is usable."""
    mask=np.asarray(mask,bool)
    starts=np.arange(0,max(0,len(mask)-target+1),step,dtype=np.int64)
    invalid=np.r_[0,np.cumsum(~mask)]
    return starts[invalid[starts+target]==invalid[starts]]


def make_record(key,raw=None,*,y=None,actor=None,source=False,path=None,condition='clean',valid=None):
    if y is None:
        if raw is None:raise ValueError('A raw record or labels must define its length')
        y=np.full(len(raw['ball']),3,np.int8)
    y=np.asarray(y,np.int8);n=len(y)
    if not n:raise ValueError('Empty halves cannot provide a training core')
    actor=np.full(n,-1,np.int64)if actor is None else np.asarray(actor,np.int64)
    if actor.shape!=(n,):raise ValueError('Actor labels must align exactly with frames')
    valid=np.ones(n,bool)if valid is None else np.asarray(valid,bool)
    if valid.shape!=(n,):raise ValueError('Temporal validity must align exactly with frames')
    primary=np.where((y==0)|(y==1),y,-1).astype(np.int64)
    cuts=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,n]
    short=[(int(a),int(b))for a,b in zip(cuts[:-1],cuts[1:])if y[a]in(0,1)and b-a<=125]
    acuts=np.r_[0,np.flatnonzero(actor[1:]!=actor[:-1])+1,n]
    intervals=[(int(a),int(b))for a,b in zip(acuts[:-1],acuts[1:])if actor[a]>=0]
    return {'key':key,'raw':raw,'path':Path(path)if path is not None else None,'condition':condition,
        'source':bool(source),'y':y,'truth':y,'actor':actor,'primary':primary,'valid':valid,
        'weights':np.zeros(n,np.float32)if source else full_spell_weights(y),'short':short,
        'actor_intervals':intervals,'starts':{}}


def raw_record(row):
    if row['raw']is not None:return row['raw']
    key=(str(row['path']),row['condition'])
    if key in _CACHE:_CACHE.move_to_end(key);return _CACHE[key]
    with np.load(_safe(row['path']))as z:
        raw={k:z[k]for k in ('ball','players','team','pitch')}
        if 'source_exact'in z:raw['source_exact']=z['source_exact']
    if row['condition']=='corrupted':
        from scripts.train_contextual import corruption
        raw=corruption(raw,100)
    _CACHE[key]=raw
    while len(_CACHE)>48:_CACHE.popitem(last=False)
    return raw


def load_records(primary_matches=TRAIN4,source_matches=None,include_corrupted=True,
                 primary_root=PRIMARY,source_root=SOURCE):
    """Return (primary,source,provenance); source CAL/heldout are never loaded."""
    if not set(primary_matches)<=set(TRAIN4):raise PermissionError('Fit primary subset must be contained in TRAIN4')
    primary_root=_safe(primary_root);source_root=_safe(source_root)
    ready_path=source_root/'READY.json';manifest_path=source_root/'manifest.json'
    ready=json.loads(ready_path.read_text());manifest=json.loads(manifest_path.read_text())
    assert ready['ready']and ready['manifest_sha256']==sha256(manifest_path)and ready['split']==manifest['split']
    allowed=set(ready['split']['train'])
    chosen=sorted(allowed)if source_matches is None else [int(str(x).removeprefix('SK'))for x in source_matches]
    if not set(chosen)<=allowed:raise PermissionError('Only the frozen sixteen source TRAIN matches may fit')
    primary=[];source=[];files={}
    primary_manifest=ROOT/'runs/aggressive-20261029/sequence/input-manifest.json'
    original=json.loads(primary_manifest.read_text())['files']
    for match in primary_matches:
        for half in ('firstHalf','secondHalf'):
            name=f'{match}-{half}';path=primary_root/(name+'.npz');truth=primary_root/(name+'.truth.npy')
            for p in (path,truth):
                digest=sha256(_safe(p));relative=str(p.relative_to(ROOT))
                assert relative in original,'Primary file is absent from the frozen input manifest: '+relative
                assert digest==original[relative]['sha256'],'Primary input/reference hash changed: '+relative
                files[str(p)]=digest
            y=np.load(truth)
            for condition in (('clean','corrupted')if include_corrupted else ('clean',)):
                primary.append(make_record(name+'-'+condition,y=y,path=path,condition=condition))
    for match in chosen:
        for half in ('firstHalf','secondHalf'):
            name=f'SK{match}-{half}';base=source_root/'prepared25';paths=[base/(name+s)for s in ('.npz','.truth.npy','.actor.npy')]
            for p in paths:
                digest=sha256(_safe(p));assert digest==ready['files'][str(p.relative_to(source_root))]
                files[str(p)]=digest
            source.append(make_record(name,source=True,path=paths[0],y=np.load(paths[1]),actor=np.load(paths[2])))
    return primary,source,{'files':files,'source_ready_sha256':sha256(ready_path),'source_manifest_sha256':sha256(manifest_path),
        'primary_input_manifest_sha256':sha256(primary_manifest),
        'primary_matches':list(primary_matches),'source_matches':chosen,'source_calibration_read':False,'WQQ_access':False,'WR9_access':False}


def ball_features(raw):
    """ProvidedXY, boundedXY, their velocities, flags and observation ages."""
    ball=np.asarray(raw['ball'],np.float32);estimated,seen,age=_bounded(ball,250)
    estimated,seen,age=estimated[:,0],seen[:,0],age[:,0]
    velocity,_=_velocity(ball,60.);bounded_velocity,_=_velocity(estimated,60.)
    center=np.asarray(raw['pitch'],np.float32)/2
    values=np.concatenate([(ball-center)/50,(estimated-center)/50,velocity/40,bounded_velocity/40,
        seen[:,None],np.isfinite(estimated).all(-1)[:,None],age],-1)
    return np.clip(np.nan_to_num(values,nan=0,posinf=0,neginf=0),-5,5).astype(np.float32)


def _slice_pad(values,start,stop):
    n=len(values)
    return np.pad(values[max(0,start):min(n,stop)],((max(0,-start),max(0,stop-n)),)+((0,0),)*(values.ndim-1))


def sample_record(row,start,*,context=360,target=200,rng=None,augment=True,slot_swap_probability=.25,primary_policy='fully-live'):
    """Materialize one crop. GT determines targets/sampling, never input masks."""
    if context%5 or target%5 or start%5:raise ValueError('Global crop/core boundaries must preserve the five-frame phase')
    if primary_policy not in('fully-live','valid-runs'):raise ValueError('Unknown primary sampling policy')
    n=len(row['y']);stop=start+target
    partial=not row['source']and primary_policy=='valid-runs';real_stop=min(n,stop)
    if start<0 or start>=n or(not partial and stop>n):raise ValueError('Target core must remain inside one half')
    usable=row['valid'][start:real_stop]&(row['primary'][start:real_stop]>=0)
    if partial:
        if not usable.any():raise ValueError('A valid-runs core needs at least one usable primary H/A frame')
    elif not row['valid'][start:stop].all():raise ValueError('Target core crosses an invalid temporal interval')
    elif not row['source']and not(row['primary'][start:stop]>=0).all():raise ValueError('Primary target core must be fully H/A')
    raw=raw_record(row);lo=max(0,start-context-FEATURE_CONTEXT);hi=min(n,stop+context+FEATURE_CONTEXT)
    local={k:(v[lo:hi].copy()if k in('ball','players','source_exact')else np.asarray(v).copy())for k,v in raw.items()}
    actors=np.full(target,-1,np.int64);actors[:real_stop-start]=row['actor'][start:real_stop]
    trace={'key':row['key'],'start':int(start),'gap':None,'hold':None,'swap':None,'reflection':[False,False]}
    forced=np.zeros(local['players'].shape[:2],bool)
    if augment:
        if rng is None:raise ValueError('Augmentation requires an explicit RNG')
        reflect=rng.random(2)<.5
        for axis in np.flatnonzero(reflect):
            local['ball'][:,axis]=local['pitch'][axis]-local['ball'][:,axis]
            local['players'][:,:,axis]=local['pitch'][axis]-local['players'][:,:,axis]
        trace['reflection']=reflect.tolist()
        if rng.random()<.5:
            a=int(rng.integers(max(0,start-lo-125),max(1,real_stop-lo)));length=int(rng.integers(25,251))
            local['ball'][a:a+length]=np.nan;trace['gap']=[a+lo,length]
        if rng.random()<.25:
            slots=np.flatnonzero(np.isfinite(local['players']).all(-1).any(0))
            if len(slots):
                j=int(rng.choice(slots));a=int(rng.integers(max(0,start-lo-500),max(1,real_stop-lo)));length=int(rng.integers(100,1001))
                local=held_coordinates(local,j,a,a+length);trace['hold']=[j,a+lo,length]
        if rng.random()<slot_swap_probability:
            candidates=[np.flatnonzero((local['team']==t)&np.isfinite(local['players']).all(-1).any(0))for t in (0,1)]
            candidates=[x for x in candidates if len(x)>=2]
            if candidates and len(local['ball'])>=2:
                a,b=map(int,rng.choice(candidates[int(rng.integers(len(candidates)))],2,replace=False))
                lower=max(1,min(start-lo,len(local['ball'])-1));upper=max(lower+1,min(real_stop-lo,len(local['ball'])))
                cut=int(rng.integers(lower,upper))
                local['players'][cut:,[a,b]]=local['players'][cut:,[b,a]]
                forced[cut,[a,b]]=True;after=np.arange(start,stop)>=lo+cut;old=actors.copy()
                actors[after&(old==a)]=b;actors[after&(old==b)]=a;trace['swap']=[a,b,cut+lo]
    a=start-context-lo;b=stop+context-lo
    values=crop_features(local,a,b);values,inverse=pack_values(values,minimum=28)
    jump=_slice_pad(forced,a,b);packed_jump=np.zeros(values['observed'].shape,bool)
    for original,mapped in enumerate(inverse):
        if mapped>=0:packed_jump[:,mapped]=jump[:,original]
    values['track_jumps']=(values['player'][...,-1]>.5)|packed_jump
    values['ball']=_slice_pad(ball_features(local),a,b)
    mapped=np.full(target,-1,np.int64);known=(actors>=0)&(actors<len(inverse));mapped[known]=inverse[actors[known]]
    available=values['position_available'][context:context+target]
    keep=(mapped>=0)&available[np.arange(target),np.maximum(mapped,0)];mapped[~keep]=-1
    primary=np.full(target,-1,np.int64);mask=np.zeros(target,bool);weights=np.zeros(target,np.float32)
    if row['source']:mask[:real_stop-start]=row['valid'][start:real_stop]
    else:
        primary[:real_stop-start]=row['primary'][start:real_stop]
        mask[:real_stop-start]=usable
        primary[~mask]=-1;weights[:real_stop-start]=row['weights'][start:real_stop];weights[~mask]=0
    group=np.where(row['y'][start:stop]<2,row['y'][start:stop],-1).astype(np.int64)if row['source']else np.full(target,-1,np.int64)
    return {'values':values,'primary_team':primary,'source_actor':mapped if row['source']else np.full(target,-1,np.int64),
        'source_group':group,'full_spell_weights':weights,'core_mask':mask,
        'is_source':row['source'],'trace':trace}


def _choose_start(row,rng,target,return_anchor=False,primary_policy='fully-live'):
    if primary_policy not in('fully-live','valid-runs'):raise ValueError('Unknown primary sampling policy')
    if not row['source']and primary_policy=='valid-runs':
        known=np.flatnonzero(row['valid']&(row['primary']>=0))
        if not len(known):raise ValueError('Record has no usable primary H/A frames')
        anchor=None
        if row['short']and rng.random()<.3:
            a,b=row['short'][int(rng.integers(len(row['short'])))];eligible=known[(known>=a)&(known<b)]
            if len(eligible):anchor=int(rng.choice(eligible))
        if anchor is None:anchor=int(rng.choice(known))
        first=max(0,((anchor-target+1+4)//5)*5);last=(anchor//5)*5
        start=int(rng.choice(np.arange(first,last+1,5,dtype=np.int64)))
        return(start,anchor)if return_anchor else start
    if target not in row['starts']:
        usable=row['valid']if row['source']else row['valid']&(row['primary']>=0)
        row['starts'][target]=valid_starts(usable,target)
    starts=row['starts'][target]
    if not len(starts):raise ValueError('Record has no fully usable aligned target cores: '+row['key'])
    intervals=row['actor_intervals']if row['source']else row['short']if rng.random()<.3 else []
    if intervals:
        a,b=intervals[int(rng.integers(len(intervals)))];anchor=int(rng.integers(a,b))
        eligible=starts[(starts<=anchor)&(starts+target>anchor)]
        if len(eligible):
            start=int(rng.choice(eligible));return(start,anchor)if return_anchor else start
    start=int(rng.choice(starts));return(start,None)if return_anchor else start


def batch(primary,source,rng,context=360,target=200,batchsize=4,device='cpu',augment=True,slot_swap_probability=.25,primary_policy='fully-live'):
    if batchsize<2 or batchsize%2:raise ValueError('A matched batch requires an even size of at least two')
    if context%5 or target%5 or target<=0:raise ValueError('Context and positive target must preserve the five-frame phase')
    if not primary or not source:raise ValueError('Both primary and frozen source TRAIN records are required')
    samples=[]
    for j in range(batchsize):
        collection=primary if j<batchsize//2 else source
        for _ in range(100):
            row=collection[int(rng.integers(len(collection)))]
            try:start,anchor=_choose_start(row,rng,target,return_anchor=True,primary_policy=primary_policy)
            except ValueError:continue
            s=sample_record(row,start,context=context,target=target,rng=rng,augment=augment,slot_swap_probability=slot_swap_probability,primary_policy=primary_policy)
            if row['source']and(anchor is None or s['source_actor'][anchor-start]<0):continue
            s['trace']['sampling_anchor']=anchor
            samples.append(s);break
        else:raise ValueError('Could not sample a feasible labelled core from the declared records')
    maximum=max(x['values']['player'].shape[1]for x in samples);values={}
    for key in samples[0]['values']:
        padded=[]
        for sample in samples:
            value=sample['values'][key];p=sample['values']['player'].shape[1]
            if key=='team':value=np.pad(value,(0,maximum-p),constant_values=-1)
            elif key!='ball':value=np.pad(value,((0,0),(0,maximum-p))+((0,0),)*(value.ndim-2))
            padded.append(value)
        values[key]=np.stack(padded)
    result={'values':values,**{key:np.stack([s[key]for s in samples])for key in ('primary_team','source_actor','source_group','full_spell_weights','core_mask','is_source')},'trace':[s['trace']for s in samples]}
    if device is not None:
        import torch
        result['values']={k:torch.from_numpy(v).to(device)for k,v in values.items()}
        for key in ('primary_team','source_actor','source_group','full_spell_weights','core_mask','is_source'):result[key]=torch.from_numpy(result[key]).to(device)
    return result
