"""WPY-only scope ablation: joint evidence everywhere or near ball gaps."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
import numpy as np
from scipy.special import expit,logit
from scipy.ndimage import uniform_filter1d
from second_zero.common import sha256,json_write
from second_zero.evaluate import official_metrics
from scripts import joint_controller_ball as joint
from scripts import sprint_decoder_lattice as lattice
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts.anchor_track_evidence import candidate


def scoped_candidate(key,z,temperature,weight,scope,quality):
    if scope=='all':return candidate(key,{'raw_logit':z,'eligible':np.ones(len(z),bool)},temperature,weight,quality)
    assert scope=='missing25'
    raw=joint.old.load_raw(key)
    gate=uniform_filter1d((~np.isfinite(raw['ball']).all(1)).astype(np.float64),25,mode='nearest')
    tag='WPY' if key.startswith('J03WPY-') else 'reliability'
    with np.load(lattice.RUN/'cache'/tag/'16'/f'{key}.npz') as a:old_z=a['logit']
    mixed=logit(np.clip((1-gate)*expit(old_z)+gate*expit(z/temperature),1e-7,1-1e-7))
    return candidate(key,{'raw_logit':mixed,'eligible':gate>0},1.,weight,quality)


def calibrate(args):
    joint.ALLOWED.add('J03WPY');out=args.out
    original=json.loads((out/'calibration.json').read_text());temperature=original['temperature']
    frozen=json.loads((out/'freeze.json').read_text())
    assert sha256(out/'model.pt')==frozen['model_sha256']
    assert sha256(out/'calibration.json')==frozen['calibration_sha256']
    raw_frozen=json.loads((out/'raw-freeze-J03WPY.json').read_text())
    rows=[{**r,'scope':'all'} for r in original['rows']];cache={}
    for key in graph.keys(['J03WPY']):
        path=out/'raw'/f'{key}.npz';assert sha256(path)==raw_frozen['outputs'][path.name]
        with np.load(path) as a:cache[key]=a['owner']
    pm=official_metrics();quality=graph.get_model()
    for weight in [.25,.5,1.]:
        folder=out/'WPY-scope'/str(weight);folder.mkdir(parents=True,exist_ok=False);views={}
        for key,z in cache.items():
            pred,owner=scoped_candidate(key,z,temperature,weight,'missing25',quality)
            np.savez_compressed(folder/f'{key}.npz',predicted=pred,logit=owner)
            views[key]=graph.metric(pm,graph.truth(key),pred)
        row={'scope':'missing25','weight':weight,**joint.old.aggregate(views),'views':views};rows.append(row)
        print('WPY missing25',weight,row['overall'],row['clean'],row['corrupted'],flush=True)
    selected=max(rows,key=lambda r:(r['overall']['spell'],r['overall']['macro']))
    json_write(out/'scope-calibration.json',{'temperature':temperature,'rows':rows,'selected':selected,
        'adaptation':'Three missing-window blends added after preliminary WPY all-frame result showed clean/corrupt tradeoff; no WQQ scoring or fitting.'})
    json_write(out/'scope-freeze.json',{'temperature':temperature,'weight':selected['weight'],'scope':selected['scope'],
        'WPY_selected':selected['overall'],'model_sha256':sha256(out/'model.pt'),
        'calibration_sha256':sha256(out/'scope-calibration.json'),'reference_scope':'WPY only',
        'window':'Centered25-frame mean of raw ball-missing indicator; no fitted gate or minimum duration.',
        'source_sha256':sha256(__file__)})
    print('SELECTED',selected['scope'],selected['weight'],selected['overall'],flush=True)


def evaluate(args):
    out=args.out;freeze=json.loads((out/'scope-freeze.json').read_text())
    assert freeze['source_sha256']==sha256(__file__)
    assert sha256(out/'model.pt')==freeze['model_sha256']
    assert sha256(out/'scope-calibration.json')==freeze['calibration_sha256']
    assert freeze['weight']>0,'Winner unchanged on WPY'
    raw_freeze=json.loads((out/'raw-freeze-J03WQQ.json').read_text());dest=out/'WQQ-scope';dest.mkdir(exist_ok=False)
    quality=graph.get_model()
    for key in graph.keys(['J03WQQ']):
        path=out/'raw'/f'{key}.npz';assert sha256(path)==raw_freeze['outputs'][path.name]
        with np.load(path) as a:z=a['owner']
        pred,owner=scoped_candidate(key,z,freeze['temperature'],freeze['weight'],freeze['scope'],quality)
        np.savez_compressed(dest/f'{key}.npz',predicted=pred,logit=owner)
    json_write(dest/'prediction-freeze.json',{p.name:sha256(p) for p in dest.glob('*.npz')})
    joint.ALLOWED.add('J03WQQ');pm=official_metrics();views={}
    for key in graph.keys(['J03WQQ']):
        with np.load(dest/f'{key}.npz') as a:pred=a['predicted']
        views[key]=graph.metric(pm,graph.truth(key),pred)
    result={**joint.old.aggregate(views),'views':views,'freeze':freeze,
            'scope':'Complete pipeline, frozen before WQQ references; WQQ repeatedly reused development, not untouched test.'}
    json_write(out/'scope-evaluation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','evaluate']);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();sys.addaudithook(joint.guard);globals()[args.stage](args)
