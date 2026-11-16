"""Context-only absolute gap ownership, same frozen parents/data/budget."""
import argparse,datetime,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import temporal_context_recovery_v14 as r
from second_zero import masked_gap_training as masked

BASE_RUN=r.RUN;RUN=ROOT/'runs/masked-gap-context-v16';REPORT=ROOT/'reports/masked-gap-context-v16'
ORIGINAL_INPUT=r.input_case;ORIGINAL_GUARD=r.guard;ORIGINAL_PARENT=r.d.boundary_parent


def parent(which,match,half,condition='corrupted'):
    if condition=='clean' or match=='J03WPY':return ORIGINAL_PARENT(which,match,half,condition)
    path=BASE_RUN/f'full-predictions/{match}-{half}-{condition}.npz' if match=='J03WQQ' else BASE_RUN/f'oof/{match}-{half}-{condition}.npz'
    with np.load(path) as z:return z['temporal_context'].copy()


def input_case(which,match,half):
    v,meta,signals,x,prior,support=ORIGINAL_INPUT(which,match,half)
    extra=masked.raw_geometry(r.b.raw_half(f'{match}-{half}-corrupted'))
    return v,meta,signals,np.concatenate([x,extra],axis=1).astype(np.float16).astype(np.float32),prior,support


def guard(event,args):
    ORIGINAL_GUARD(event,args)
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0])
        if str(RUN/'training') in p and p.endswith('.npz') and not r.HASH_ONLY:
            if Path(p).parent.name not in r.ALLOWED:raise PermissionError('Masked training cache outside current fit')


def owner_case(which,match,half,alphas):
    producer='full' if which=='full' else r.prior.producer(match) if match!='J03WPY' else r.prior.producer(which)
    v,meta,signals,x,prior,support=input_case(producer,match,half)
    model,freeze=r.fitted(which);student=masked.predict(model,x,prior,len(v['baseline']))
    oldcal=r.read(BASE_RUN/f'models/{which}/calibration.json');oldalpha=oldcal['selected']['alpha'];oldowner=None
    if oldalpha:
        oldpath=BASE_RUN/f'owner/{which}/{match}-{half}.npz'
        with np.load(oldpath) as z:oldstudent=z['student']
        oldowner=r.pin(oldpath);oldlogit=r.local.local.blend(v['unary'],oldstudent,support,oldalpha)
    else:oldstudent=v['unary'];oldlogit=v['unary']
    base=r.local.predict(v,signals,oldstudent,support,oldalpha,producer) if match=='J03WPY' else parent(producer,match,half)
    composed={**v,'baseline':base,'unary':oldlogit};out={'baseline':base}
    for i,alpha in enumerate(alphas):out[f'alpha_{i}']=r.local.predict(composed,signals,student,support,alpha,producer)
    dest=RUN/f'owner/{which}/{match}-{half}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:np.savez_compressed(f,student=student,support=support)
    return out,{'owner':r.pin(dest),'source':meta['cache'],'model':freeze['model'],'parent_owner':oldowner,'parent_calibration':r.pin(BASE_RUN/f'models/{which}/calibration.json'),'alphas':alphas}


r.RUN=RUN;r.REPORT=REPORT;r.STEPS_PER_MATCH=4000
r.model_api.batch=masked.batch;r.model_api.predict=masked.predict;r.model_api.TemporalResidual=masked.RawGeometryTemporal;r.input_case=input_case;r.guard=guard;r.owner_case=owner_case;r.d.boundary_parent=parent


def prepare():
    p=r.read(BASE_RUN/'protocol.json');af=ROOT/'reports/temporal-context-recovery-v14/artifact-freeze.json';inputs={**p['inputs'],**p['sources'],r.rel(af):r.sha256(af)};inputs.update(r.read(af)['hashes'])
    p={**p,'kind':'masked-absolute-context-gap-v16','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_research_best':.773631491936103,
       'architecture':{**p['architecture'],'channels':429,'raw_observed_geometry':98,'raw_linear_geometry':24},
       'architecture_change':'Same32-width temporal blocks,429 channels. Append98 raw observed nearest-player/availability features and24 bounded raw linear ball interpolation/uncertainty features to original307. Inside raw-missing25 support ALL13 learned model-evidence channels plus parentH/A onehots are zeroed; ALL412 raw context/geometry channels and missing/support remain. Ownership evidence outside gaps supplies surrounding context. Output prior connection is ZERO: learn absolute missing-interval ownership instead of small corrections to a confident old guess. No original hidden ball coordinate is read.',
       'objective_change':'Same original v14 ownership/clean-teacher/boundary/residual terms, but weights and window centers include ONLY eligible gap-live frames. True changes outside gaps are not training centers. Clean teacher remains fitting target only. Random shared-team swap and symmetric absolute inference unchanged.',
       'fit':{**p['fit'],'steps_per_match':4000,'full_steps':16000,'checkpoint':'fixed final step; no early stop or model grid'},
       'composition':'Compose on FROZEN v14 .773631492 research best: its already calibrated teacher/student probability blend becomes prior, then blend absolute context student by WPY-selected alpha*missing25. Alpha0 copies exact v14 labels. Existing175/747 gap decoder and pool fixed. All clean,D/X/outside-gap labels exact. Legacy NPZ/audit field baseline_07714 now denotes the composed v14 parent (.773631492 full), not the old numeric score. Input/fitting parents remain the primary-only .771388 features: the old neural parent enters composition only, never new training features or targets. OOF old-neural parent excludes held and is independently hash frozen.',
       'comparison':'Same original raw corrupted100 source/targets/primary parents as v14, but fresh429-channel cache adds observed/interpolated geometry. v15 longer pointwise OOF supplies a step-budget reference, not an isolated architecture ablation. No WQQ comparison between alternate currentalpha values. One frozen calibrated full candidate; no fallback.',
       'sources':{**p['sources'],r.rel(Path(__file__)):r.sha256(Path(__file__)),r.rel(Path(masked.__file__)):r.sha256(Path(masked.__file__))},'inputs':inputs}
    r.write(RUN/'protocol.json',p);print('V16 PROTOCOL',r.sha256(RUN/'protocol.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cache','fit','calibrate','predict_oof','audit_oof','full_run','audit_full']);p.add_argument('--held',choices=r.TRAIN+['full']);args=p.parse_args();r.threadpool_limits(2);r.torch.set_num_threads(2);r.STAGE=args.stage
    if args.stage=='prepare':prepare()
    else:
        sys.addaudithook(r.guard)
        if args.stage in ['fit','calibrate']:assert args.held;getattr(r,args.stage)(args.held)
        elif args.stage.startswith('audit_'):r.audit(args.stage=='audit_full')
        else:getattr(r,args.stage)()
