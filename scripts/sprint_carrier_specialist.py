"""Track candidate carrier identities across missing-ball intervals.

Only measured endpoints and the same player's trajectory are used. No hidden
ball coordinates are read. Perspective-shared scoring preserves team symmetry.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
from second_zero.contextual import velocity
from second_zero.sprint_model_interpolation import interpolated_features
from second_zero.common import json_write,sha256
from scripts import sprint_gap_specialist as engine
from scripts.train_contextual import corruption

def carrier_features(a):
    ball=np.asarray(a['ball'],np.float32);players=np.asarray(a['players'],np.float32);team=np.asarray(a['team']);n=len(ball);ix=np.arange(n)
    seen=np.isfinite(ball).all(1);prev=np.maximum.accumulate(np.where(seen,ix,-n));nxt=np.minimum.accumulate(np.where(seen,ix,2*n)[::-1])[::-1]
    eligible=(~seen)&(prev>=0)&(nxt<n)&(nxt-prev<=250);rows=np.flatnonzero(eligible)
    if not len(rows):return None,eligible
    lo=prev[rows];hi=nxt[rows];phase=(rows-lo)/(hi-lo);seconds=(hi-lo)/25
    b0=ball[lo];b1=ball[hi];delta=b1-b0;linear=b0+phase[:,None]*delta;path_velocity=delta/seconds[:,None]
    pv=velocity(players);acc=velocity(pv);linear_extra=interpolated_features(ball,players,team,a['pitch'])
    output=[];rr=np.arange(len(rows))[:,None]
    for t in (0,1):
        parts=[linear_extra[rows,t],phase[:,None],seconds[:,None]/10,np.linalg.norm(path_velocity,axis=1)[:,None]/40]
        for target_team in (t,1-t):
            slots=np.flatnonzero(team==target_team)
            for anchor in (lo,hi):
                distances=np.linalg.norm(players[anchor][:,slots]-ball[anchor,None],axis=2)
                selected=slots[np.argsort(np.nan_to_num(distances,nan=np.inf),axis=1)[:,:3]]
                left=players[lo[:,None],selected];right=players[hi[:,None],selected];current=players[rows[:,None],selected]
                speed=pv[rows[:,None],selected];acceleration=acc[rows[:,None],selected]
                relative=current-linear[:,None];rv=speed-path_velocity[:,None]
                curvature=current-((1-phase[:,None,None])*left+phase[:,None,None]*right)
                values=[np.linalg.norm(left-b0[:,None],axis=2)/20,np.linalg.norm(right-b1[:,None],axis=2)/20,
                    np.linalg.norm(relative,axis=2)/20,np.linalg.norm(speed,axis=2)/12,np.linalg.norm(rv,axis=2)/30,
                    np.sum(relative*rv,axis=2)/np.maximum(np.linalg.norm(relative,axis=2),.1)/30,
                    np.linalg.norm(curvature,axis=2)/20,np.linalg.norm(acceleration,axis=2)/10,
                    np.linalg.norm(current-left,axis=2)/50,np.linalg.norm(right-current,axis=2)/50,
                    np.sum(speed*path_velocity[:,None],axis=2)/400,
                    np.sum(curvature*delta[:,None],axis=2)/400]
                parts.append(np.stack(values,axis=-1).reshape(len(rows),-1))
        result=np.column_stack(parts).astype(np.float32);result[np.isinf(result)]=np.nan;output.append(np.clip(result,-10,10))
    return np.stack(output,axis=1),eligible

def features(key,unused_imputers):
    name,condition=key.rsplit('-',1)
    with np.load(engine.DATA/(name+'.npz')) as z:a={k:z[k] for k in z.files}
    if condition=='corrupted':a=corruption(a,100)
    return carrier_features(a)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.addaudithook(engine.guard)
    engine.features=features;matches=engine.fit(a.out,'full')
    json_write(a.out/'carrier-protocol.json',{'features':'Stable candidate player slots selected at observed left/right endpoints; current trajectory curvature, speed, acceleration, endpoint proximity, and uncertain linear geometry','hidden_ball_positions_used':False,'source_sha256':sha256(Path(__file__))})
    (a.out/'carrier-source.py').write_text(Path(__file__).read_text());engine.evaluate(a.out,matches,'full')

if __name__=='__main__':main()
