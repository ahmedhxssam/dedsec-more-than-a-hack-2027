"""Stream the public Sportec XML into anonymous metre-space arrays and sealed truth."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from second_zero.common import json_write,sha256

def prepare(match,folder):
    source=next(folder.glob(f'*positions*{match}.xml'))
    meta=next(folder.glob(f'*matchinformation*{match}.xml'))
    root=ET.parse(meta).getroot();general=root.find('.//General')
    home=general.get('HomeTeamId');away=general.get('GuestTeamId')
    env=root.find('.//Environment');pitch=np.array([float(env.get('PitchX')),float(env.get('PitchY'))])
    groups=[]
    for event,e in ET.iterparse(source,events=['end']):
        if e.tag!='FrameSet':continue
        if e.get('TeamId') in [home,away,'BALL']:
            fields=['N','X','Y']+(['BallStatus','BallPossession'] if e.get('TeamId')=='BALL' else [])
            vals=np.array([[float(f.get(k,'nan')) for k in fields] for f in e],dtype=np.float64)
            groups.append((e.get('GameSection'),e.get('PersonId'),e.get('TeamId'),vals))
        e.clear()
    out=folder/'prepared';out.mkdir(exist_ok=True)
    provenance={'match':match,'raw_sha256':sha256(source),'metadata_sha256':sha256(meta),'pitch':pitch.tolist(),'halves':[]}
    for half in ['firstHalf','secondHalf']:
        g=[x for x in groups if x[0]==half];ballrow=next(x[3] for x in g if x[2]=='BALL')
        low=int(ballrow[:,0].min());high=int(ballrow[:,0].max())+1;n=high-low
        players=[x for x in g if x[2]!='BALL']
        xy=np.full((n,len(players),2),np.nan,dtype=np.float32);teams=[]
        for i,(_,_,t,data) in enumerate(players):
            index=data[:,0].astype(int)-low;valid=(index>=0)&(index<n)
            xy[index[valid],i]=data[valid,1:3]+pitch/2
            teams.append(0 if t==home else 1)
        ball=np.full((n,2),np.nan,dtype=np.float32);truth=np.full(n,3,dtype=np.int8)
        idx=ballrow[:,0].astype(int)-low
        if len(np.unique(idx))!=len(idx):raise ValueError('Duplicate ball frames')
        ball[idx]=ballrow[:,1:3]+pitch/2
        status=ballrow[:,3];owner=ballrow[:,4]
        # Provider flags are targets ONLY; never included in inference arrays.
        code=np.where(status==0,2,np.where((status==1)&(owner==1),0,np.where((status==1)&(owner==2),1,3)))
        truth[idx]=code
        name=f'{match}-{half}'
        np.savez_compressed(out/(name+'.npz'),ball=ball,players=xy,team=np.array(teams),pitch=pitch)
        np.save(out/(name+'.truth.npy'),truth)
        provenance['halves'].append({'name':name,'frames':n,'source_frame_origin':low,'input_sha256':sha256(out/(name+'.npz')),'truth_sha256':sha256(out/(name+'.truth.npy'))})
    json_write(out/(match+'.json'),provenance)
    return {'match':match,'halves':len(provenance['halves']),'frames':sum(h['frames'] for h in provenance['halves'])}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--match',nargs='*');a=p.parse_args()
    folder=ROOT/'private/improvement/external';split=json.loads((folder/'split.lock.json').read_text())['split']
    for match in a.match or sum(split.values(),[]):
        if (folder/'prepared'/f'{match}.json').exists():print('Already prepared',match,flush=True);continue
        print(json.dumps(prepare(match,folder)),flush=True)
