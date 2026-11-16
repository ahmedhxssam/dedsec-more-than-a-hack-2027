"""Fit a joint controller/ball model; freeze predictions before reference audit."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'private/improvement/runtime'), str(ROOT)]
os.environ['OMP_NUM_THREADS']='2'; os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from second_zero.common import json_write, sha256
from second_zero.joint_controller_ball import JointControllerBall, crop_features, infer_raw, RADIUS, VERSION
from second_zero.vision_spell_balance import full_spell_weights, mixed_spell_bce
from scripts import vision_player_temporal_owner as old
from scripts.train_contextual import corruption
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts.anchor_track_evidence import candidate
from second_zero.evaluate import official_metrics

DATA = ROOT/'private/improvement/external/prepared'
METRICA = ROOT/'private/improvement/metrica-training'
EVENTS = ROOT/'runs/vision-push/player-supervision/elastic-events.json'
READS = []; ALLOWED = set()


def guard(event, args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
    p=os.fsdecode(args[0])
    if 'J03WR9' in p or '/evaluation/gt/' in p: raise PermissionError('Reserved/competition references excluded')
    if p.endswith('.truth.npy'):
        if Path(p).name.split('-')[0] not in ALLOWED: raise PermissionError('Reference outside current stage: '+p)
        READS.append(p)


def prepare_metrica(out):
    out.mkdir(parents=True,exist_ok=False); original=json.loads((METRICA/'provenance.json').read_text())
    records=[]; source_hashes={}; stats={}
    for game in (1,2):
        event_file=METRICA/f'Sample_Game_{game}_RawEventsData.csv'
        source_hashes[str(event_file.relative_to(ROOT))]=sha256(event_file)
        events=pd.read_csv(event_file); names=[]
        for side in ('Home','Away'):
            path=METRICA/f'Sample_Game_{game}_RawTrackingData_{side}_Team.csv'
            with path.open() as h:
                rows=csv.reader(h); next(rows); next(rows); names.extend(next(rows)[3:-2:2])
            source_hashes[str(path.relative_to(ROOT))]=sha256(path)
        people={name:i for i,name in enumerate(names)}
        for period,half in ((1,'firstHalf'),(2,'secondHalf')):
            key=f'METRICA{game}-{half}'; path=METRICA/'prepared'/f'{key}.npz'
            with np.load(path) as z: raw={k:z[k] for k in z.files}
            source_hashes[str(path.relative_to(ROOT))]=sha256(path)
            origin=next(h['source_frame_origin'] for h in original['halves'] if h['key']==key)
            proposed={}
            for idx,e in events.loc[(events.Period==period)&(events.Type=='PASS')&events.Subtype.isna()].iterrows():
                sender,receiver=people.get(e['From']),people.get(e['To']); t={'Home':0,'Away':1}[e.Team]
                if sender is None or receiver is None or raw['team'][sender]!=t or raw['team'][receiver]!=t: continue
                for role,slot,frame in [('sender',sender,int(e['Start Frame'])-origin),('receiver',receiver,int(e['End Frame'])-origin)]:
                    if 0<=frame<len(raw['ball']): proposed.setdefault((frame,slot),[]).append(f'{idx}-{role}')
            bs=np.linalg.norm(np.diff(raw['ball'],axis=0),axis=-1)*25
            ps=np.linalg.norm(np.diff(raw['players'],axis=0),axis=-1)*25
            counts={'proposed':len(proposed),'eligible':0,'excluded':0}
            frame_teams={}
            for f,s in proposed: frame_teams.setdefault(f,set()).add(int(raw['team'][s]))
            for (f,s),ids in sorted(proposed.items()):
                lo,hi=max(0,f-5),min(len(bs),f+5)
                valid=np.isfinite(raw['ball'][f]).all() and np.isfinite(raw['players'][f,s]).all()
                valid=valid and not (bs[lo:hi]>50).any() and not (ps[lo:hi,s]>15).any() and len(frame_teams[f])==1
                if not valid: counts['excluded']+=1; continue
                t=int(raw['team'][s]); counts['eligible']+=1
                records.append({'key':key,'match':f'METRICA{game}','frame':f,'team':t,'prepared_slot':s,
                    'team_slot':int((raw['team'][:s]==t).sum()),'tolerance_frames':5,'source_events':ids})
            stats[key]=counts
    manifest={'source':'https://github.com/metrica-sports/sample-data',
        'revision':'e706dd506b360d69d9d123d5b8026e7294b13996','attribution':'Metrica Sports',
        'usage':'Training-only sparse pass endpoints and observed coordinate reconstruction; no weak dense labels read.',
        'selection':'PASS with empty subtype; explicit same-team identities; finite observations; plausible local speeds; exclude simultaneous opposing endpoints. Distance never labels control.',
        'events':records,'statistics':stats,'input_sha256':source_hashes,'source_sha256':sha256(__file__)}
    json_write(out/'metrica-events.json',manifest); print('Metrica eligible',len(records),stats,flush=True)


def load_records(matches, metrica_events):
    allowed=set(matches); manifest=json.loads(EVENTS.read_text()); rows=[]
    extra=json.loads(metrica_events.read_text()) if metrica_events else {'events':[]}
    for match in matches:
        for half in ('firstHalf','secondHalf'):
            name=f'{match}-{half}'
            with np.load(DATA/f'{name}.npz') as z: clean={k:z[k] for k in ('ball','players','team','pitch')}
            y=np.load(DATA/f'{name}.truth.npy'); edges=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
            shorts=[(s,e) for s,e in zip(edges[:-1],edges[1:]) if y[s]<2 and e-s<=125]
            actors=[e for e in manifest['events'] if e['key']==name]
            for condition,raw in [('clean',clean),('corrupted',corruption(clean,100))]:
                rows.append({'key':name+'-'+condition,'raw':raw,'clean_ball':clean['ball'],'truth':y,
                    'weights':full_spell_weights(y),'shorts':shorts,'events':actors,'extra':False})
    if metrica_events:
        for key in sorted({e['key'] for e in extra['events']}):
            with np.load(METRICA/'prepared'/f'{key}.npz') as z: raw={k:z[k] for k in z.files}
            y=np.full(len(raw['ball']),3,np.int8)
            rows.append({'key':key,'raw':raw,'clean_ball':raw['ball'],'truth':y,'weights':np.zeros(len(y),np.float32),
                         'shorts':[],'events':[e for e in extra['events'] if e['key']==key],'extra':True})
    assert {r['key'].split('-')[0] for r in rows if not r['extra']}==allowed
    return rows


def augment_crop(raw, lo, hi, rng):
    n=len(raw['ball']); a,b=max(0,lo-250),min(n,hi+250)
    local={k:(v[a:b].copy() if k in ('ball','players') else v.copy()) for k,v in raw.items()}
    count=len(local['ball']); core_start=max(0,lo+RADIUS-a); core_stop=min(count,hi-RADIUS-a)
    # Randomized observational corruption is independent of possession labels.
    if rng.random()<.7:
        length=int(rng.choice([25,75,150,250])); center=int(rng.integers(core_start,max(core_start+1,core_stop)))
        local['ball'][max(0,center-length//2):min(count,center+(length+1)//2)]=np.nan
    if rng.random()<.25:
        # Approximate an incomplete camera footprint; center derives from a
        # randomly sampled measured location, never ownership or actor labels.
        known=np.flatnonzero(np.isfinite(local['ball']).all(1))
        if len(known):
            center=local['ball'][int(rng.choice(known))]
            visible=(np.abs(local['players'][...,0]-center[0])<rng.uniform(22,42)) & (np.abs(local['players'][...,1]-center[1])<rng.uniform(18,34))
            local['players'][~visible]=np.nan
    if rng.random()<.25:
        slot=int(rng.integers(local['players'].shape[1])); start=int(rng.integers(count))
        local['players'][start:start+int(rng.integers(50,201)),slot]=np.nan
    flips=[]
    for axis in (0,1):
        flip=bool(rng.random()<.5); flips.append(flip)
        if flip:
            local['ball'][:,axis]=local['pitch'][axis]-local['ball'][:,axis]
            local['players'][...,axis]=local['pitch'][axis]-local['players'][...,axis]
    return local,a,flips


def batch(rows,rng,count,padding,target=192):
    primary=[r for r in rows if not r['extra']]; extra=[r for r in rows if r['extra']]
    arrays={}; ys=[]; ws=[]; balls=[]; bm=[]; anchors=[]; trace=[]
    for sample in range(count):
        use_extra=bool(extra) and rng.random()<.2; choices=extra if use_extra else primary
        row=choices[int(rng.integers(len(choices)))]; n=len(row['truth'])
        start=int(rng.integers(n-target+1)); choice=rng.random()
        if (choice<.2 or use_extra) and row['events']:
            center=row['events'][int(rng.integers(len(row['events'])))]['frame']
            start=int(np.clip(center-target//2,0,n-target))
        elif choice<.5 and row['shorts']:
            s,e=row['shorts'][int(rng.integers(len(row['shorts'])))]; center=(s+e)//2
            start=int(np.clip(center-int(rng.integers(target//4,3*target//4)),0,n-target))
        stop=start+target; lo=start-RADIUS; hi=stop+RADIUS
        local,origin,flips=augment_crop(row['raw'],lo,hi,rng)
        values=crop_features(local,lo-origin,hi-origin)
        for k in ('player','position'):
            v=values[k]; values[k]=np.pad(v,((0,0),(0,0),(0,padding-v.shape[2]),(0,0)))
        clean=row['clean_ball'][start:stop].copy()
        for axis,flip in enumerate(flips):
            if flip: clean[:,axis]=row['raw']['pitch'][axis]-clean[:,axis]
        clean=(clean-row['raw']['pitch']/2)/10
        ballmask=np.isfinite(clean).all(1) & (values['ball_flags'][RADIUS:RADIUS+target,0]<.5)
        # Suppress discontinuous coordinate targets; this is observed quality,
        # never a controller-label assignment.
        speed=np.linalg.norm(np.diff(clean,axis=0),axis=1)*250
        bad=np.r_[False,speed>60]|np.r_[speed>60,False]; ballmask &= ~bad
        balls.append(np.nan_to_num(clean,nan=0).astype(np.float32)); bm.append(ballmask)
        for k,v in values.items(): arrays.setdefault(k,[]).append(v)
        ys.append(row['truth'][start:stop]); ws.append(row['weights'][start:stop])
        local_events=[e for e in row['events'] if start<=e['frame']<stop]
        if len(local_events)>4: local_events=[local_events[i] for i in rng.choice(len(local_events),4,replace=False)]
        for e in local_events:
            at=e['frame']-start; radius=e['tolerance_frames']
            anchors.append((sample,max(0,at-radius),min(target,at+radius+1),e['team'],e['team_slot'],use_extra))
        trace.append((row['key'],start,flips))
    return ({k:torch.from_numpy(np.stack(v)) for k,v in arrays.items()},torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(ws)),torch.from_numpy(np.stack(balls)),torch.from_numpy(np.stack(bm)),anchors,trace)


def losses(model,packed,device):
    values={k:v.to(device) for k,v in packed[0].items()}; y,w,ball,mask=[v.to(device) for v in packed[1:5]]
    result=model(values); core=slice(RADIUS,RADIUS+y.shape[1]); owner=result['owner'][:,core]
    main,frame,spell=mixed_spell_bce(owner,y,w,.3)
    actor,_,_=mixed_spell_bce(result['actor_owner'][:,core],y,w,.3)
    active=values['player'][:,core,...,-1]>.5; p=active.shape[-1]; times=torch.arange(y.shape[1],device=device)
    sparse=[]
    for sample,start,stop,team,slot,extra in packed[5]:
        valid=active[sample,:,team,slot] & (times>=start)&(times<stop)
        if not extra: valid &= y[sample]==team
        for k in ('actor_logp','initial_actor_logp'):
            lp=result[k][sample,core,team*p+slot]
            # Fixed-shaped log-sum-exp; exclude all-missing bags from numerator.
            count=valid.sum(); value=-(torch.logsumexp((4*lp).masked_fill(~valid,-1e4),0)-count.clamp_min(1).float().log())/4
            sparse.append(value*(count>0).to(value.dtype))
    sparse_loss=torch.stack(sparse).mean() if sparse else owner.sum()*0
    error=F.smooth_l1_loss(result['inferred_ball'][:,core],ball,reduction='none',beta=.5).sum(-1)
    scale=result['log_scale'][:,core]
    reconstruction=((error*torch.exp(-scale)+scale)*mask).sum()/mask.sum().clamp_min(1)
    loss=main+.2*actor+.1*sparse_loss+.1*reconstruction
    return loss,{'owner':main,'frame':frame,'spell':spell,'actor':actor,'sparse':sparse_loss,'ball':reconstruction,
                 'ball_error_m':(torch.linalg.vector_norm(result['inferred_ball'][:,core]-ball,dim=-1)*mask).sum()*10/mask.sum().clamp_min(1)}


def sources():
    return [Path(__file__),ROOT/'second_zero/joint_controller_ball.py',ROOT/'second_zero/vision_player_temporal.py',
            ROOT/'second_zero/temporal_ownership.py',ROOT/'second_zero/set_ownership.py',ROOT/'second_zero/contextual.py',
            ROOT/'second_zero/sprint_model_interpolation.py',ROOT/'second_zero/vision_spell_balance.py',Path(old.__file__)]


def fit(args):
    ALLOWED.update(args.train_matches); rows=load_records(args.train_matches,args.metrica_events)
    padding=max(int((r['raw']['team']==t).sum()) for r in rows for t in (0,1))
    torch.manual_seed(args.seed); model=JointControllerBall(args.width,args.feedback).to(args.device)
    rng=np.random.default_rng(args.seed); packed=batch(rows,rng,args.batch,padding)
    if args.stage=='profile':
        model.eval(); timings=[]
        for _ in range(3):
            start=time.monotonic(); model.zero_grad(); loss,details=losses(model,packed,args.device); loss.backward()
            if args.device=='mps': torch.mps.synchronize()
            timings.append(time.monotonic()-start)
        print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'radius':RADIUS,
            'batch_shape':list(packed[0]['player'].shape),'seconds':timings,'RSS_MiB':old.memory_mib(),
            'loss':{k:float(v.detach().cpu()) for k,v in details.items()}}),flush=True); return
    out=args.out; out.mkdir(parents=True,exist_ok=False)
    plan={'version':VERSION,'train_matches':args.train_matches,'steps':args.steps,'width':args.width,'batch':args.batch,
          'seed':args.seed,'feedback':args.feedback,'lr':args.lr,'radius_frames':RADIUS,
          'architecture':'Shared player temporal/social encoder; controller-conditioned ball distribution; explicit ball geometry/uncertainty feeds refined controller and native owner inference.',
          'loss':'0.7 frame+0.3 equal-spell ownership BCE;0.2 same actor team BCE;0.1 sparse actor MIL;0.1 robust heteroscedastic masked-ball reconstruction.',
          'augmentation':'70% crops extra25/75/150/250f ball gap;25% fixed partial field footprint;25% single player gap; independent XY reflection.',
          'sampling':'Primary50%uniform30%short-spell20%actor crops;20%extra-source actor crops if supplied. All native frames kept.',
          'metrica_events_sha256':sha256(args.metrica_events) if args.metrica_events else None,
          'external_actor_sha256':sha256(EVENTS),'sources':{str(p.relative_to(ROOT)):sha256(p) for p in sources()},
          'selection':'Only planned half-way/final checkpoints; WPY lexicographic full-pipeline SpellF1 then macro. No WQQ or final fitting.',
          'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for match in args.train_matches
              for half in ('firstHalf','secondHalf') for p in (DATA/f'{match}-{half}.npz',DATA/f'{match}-{half}.truth.npy')},
          'reference_reads':sorted(set(READS))}
    json_write(out/'protocol.json',plan)
    for path in sources():
        dest=out/'sources'/path.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    start=time.monotonic(); history=[]; trace=hashlib.sha256(); torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    for step in range(1,args.steps+1):
        model.train(); packed=batch(rows,rng,args.batch,padding); trace.update(json.dumps(packed[-1]).encode())
        loss,details=losses(model,packed,args.device); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step()
        if step%100==0:
            row={'step':step,'seconds':round(time.monotonic()-start,1),'loss':float(loss.detach().cpu()),
                 **{k:float(v.detach().cpu()) for k,v in details.items()},'RSS_MiB':old.memory_mib()}
            history.append(row); print(json.dumps(row),flush=True)
            json_write(out/'training.json',{'history':history,'sampling_sha256':trace.hexdigest(),'reference_reads':sorted(set(READS))})
        if step in (args.steps//2,args.steps):
            folder=out/f'step{step}'; folder.mkdir()
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'width':args.width,'feedback':args.feedback},folder/'model.pt')
            json_write(folder/'freeze-fit.json',{'step':step,'train_matches':args.train_matches,'model_sha256':sha256(folder/'model.pt'),
                'protocol_sha256':sha256(out/'protocol.json'),'sampling_sha256':trace.hexdigest(),'evaluation_references_read':False})


def get_model(out,device):
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    model=JointControllerBall(checkpoint['width'],checkpoint['feedback']); model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval(); return model


def predictions(args):
    out=args.out; frozen=json.loads((out/'freeze-fit.json').read_text())
    assert sha256(out/'model.pt')==frozen['model_sha256']
    assert not set(args.matches)&set(frozen['train_matches']), 'Model trained on requested evaluation match'
    model=get_model(out,args.device); folder=out/'raw'; folder.mkdir(exist_ok=True)
    for key in graph.keys(args.matches):
        path=folder/f'{key}.npz'
        if path.exists(): raise FileExistsError(path)
        result=infer_raw(model,old.load_raw(key)); np.savez_compressed(path,**result)
        print('inferred',key,flush=True)
    json_write(out/('raw-freeze-'+','.join(args.matches)+'.json'),{'model_sha256':sha256(out/'model.pt'),
        'outputs':{f'{key}.npz':sha256(folder/f'{key}.npz') for key in graph.keys(args.matches)},'reference_reads':READS})


def calibrate(args):
    ALLOWED.add('J03WPY'); out=args.out; cache={}; zs=[]; ys=[]
    freeze=json.loads((out/'raw-freeze-J03WPY.json').read_text())
    for key in graph.keys(['J03WPY']):
        path=out/'raw'/f'{key}.npz'; assert sha256(path)==freeze['outputs'][path.name]
        with np.load(path) as a: z=a['owner']
        cache[key]=z; y=graph.truth(key); zs.append(z[y<2]); ys.append(y[y<2])
    z,y=np.concatenate(zs),np.concatenate(ys)
    fitted=minimize_scalar(lambda lt:np.mean(np.logaddexp(0,z/np.exp(lt))-y*z/np.exp(lt)),bounds=(-2.3,2.3),method='bounded')
    temperature=float(np.exp(fitted.x)); q=graph.get_model(); pm=official_metrics(); rows=[]
    for weight in [0.,.1,.25,.5,1.]:
        dest=out/'WPY'/str(weight); dest.mkdir(parents=True,exist_ok=False); views={}
        for key,z in cache.items():
            pred,logit=candidate(key,{'raw_logit':z,'eligible':np.ones(len(z),bool)},temperature,weight,q)
            np.savez_compressed(dest/f'{key}.npz',predicted=pred,logit=logit)
            views[key]=graph.metric(pm,graph.truth(key),pred)
        row={'weight':weight,**old.aggregate(views),'views':views};rows.append(row)
        print('WPY',weight,row['overall'],row['clean'],row['corrupted'],flush=True)
    selected=max(rows,key=lambda x:(x['overall']['spell'],x['overall']['macro']))
    json_write(out/'calibration.json',{'temperature':temperature,'NLL':float(fitted.fun),'rows':rows,'selected':selected})
    json_write(out/'freeze.json',{'temperature':temperature,'weight':selected['weight'],'WPY_selected':selected['overall'],
        'model_sha256':sha256(out/'model.pt'),'calibration_sha256':sha256(out/'calibration.json'),'evaluation_reference_read':False})


def evaluate(args):
    out=args.out; freeze=json.loads((out/'freeze.json').read_text()); assert sha256(out/'model.pt')==freeze['model_sha256']
    if freeze['weight']==0: print('Unchanged winner selected; no evaluation.'); return
    q=graph.get_model(); dest=out/'WQQ';dest.mkdir(exist_ok=False)
    raw_freeze=json.loads((out/'raw-freeze-J03WQQ.json').read_text())
    for key in graph.keys(['J03WQQ']):
        path=out/'raw'/f'{key}.npz'; assert sha256(path)==raw_freeze['outputs'][path.name]
        with np.load(path) as a:z=a['owner']
        pred,logit=candidate(key,{'raw_logit':z,'eligible':np.ones(len(z),bool)},freeze['temperature'],freeze['weight'],q)
        np.savez_compressed(dest/f'{key}.npz',predicted=pred,logit=logit)
    json_write(dest/'prediction-freeze.json',{p.name:sha256(p) for p in dest.glob('*.npz')})
    ALLOWED.add('J03WQQ');pm=official_metrics();views={}
    for key in graph.keys(['J03WQQ']):
        with np.load(dest/f'{key}.npz') as a:pred=a['predicted']
        views[key]=graph.metric(pm,graph.truth(key),pred)
    result={**old.aggregate(views),'views':views,'scope':'Full pipeline; WQQ excluded from fitting but repeatedly reused development, not untouched test.'}
    json_write(out/'evaluation.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','profile','fit','predict','calibrate','evaluate'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--metrica-events',type=Path)
    p.add_argument('--train-matches',nargs='+',choices=old.TRAIN,default=old.TRAIN)
    p.add_argument('--matches',nargs='+',choices=old.TRAIN+['J03WPY','J03WQQ'],default=['J03WPY'])
    p.add_argument('--feedback',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch',type=int,default=4);p.add_argument('--width',type=int,default=32)
    p.add_argument('--lr',type=float,default=.0007);p.add_argument('--seed',type=int,default=20261025)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu');args=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.addaudithook(guard)
    if args.stage=='prepare':prepare_metrica(args.out)
    elif args.stage in ('fit','profile'):fit(args)
    elif args.stage=='predict':predictions(args)
    else:globals()[args.stage](args)
