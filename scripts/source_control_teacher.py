"""Source-TRAIN16 native-point teacher preparation, profile and authorized fit.

Raw tracking and eligible Dynamic Event actors remain separate conventions.
Documented PP-start is conditional on a known eligible DE actor; unknown
frames never become negatives. Sportec, source CAL and all ownership truth
arrays are prohibited. Existing prepared arrays/helpers are never modified.
"""
from __future__ import annotations
import argparse,copy,dataclasses,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'private/improvement/runtime'),str(ROOT)]
sys.path.append(str(ROOT/'private/improvement/vision-env/lib/python3.11/site-packages'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from second_zero.source_control_teacher import SourceControlTeacher,TeacherConfig,actor_loss
from second_zero.latent_owner_data import make_record,sample_record
from second_zero.common import json_write,sha256

SOURCE=ROOT/'private/improvement/skillcorner-20261029'
TRAIN=(1874553,1886347,1899585,1925299,1927964,1953632,1959846,1986691,
       1996435,1996436,2006229,2006363,2007448,2007721,2010085,2011166)
READS=set();PINNED={}


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    text=os.fsdecode(args[0]).replace('\\','/')
    if any(t in text for t in ('J03','/evaluation/gt/','.truth.npy','.phase_owner.npy','.dynamic_owner.npy')):
        raise PermissionError('Source teacher excludes all Sportec/ownership truth/phase arrays')
    p=Path(text)
    if any(t in text for t in ('.actor.npy','.dynamic_actor.npy','/prepared10/','/prepared25/','/raw/')) and str(SOURCE) in str(p):
        allowed=any(p.name.startswith('SK'+str(m)+'-') or '/raw/'+str(m)+'/' in text for m in TRAIN)
        if not allowed:raise PermissionError('Source CAL/heldout data excluded')
        READS.add(text)


def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def pin(path,expected=None):
    path=Path(path).resolve();guard('open',(path,'rb',0));digest=sha256(path)
    if expected is not None and digest!=expected:raise ValueError('Frozen source input changed: '+str(path))
    PINNED[rel(path)]=digest;return path
def meta(path,expected=None):return json.loads(pin(path,expected).read_text())


def sources():
    return [Path(__file__),ROOT/'second_zero/source_control_teacher.py',ROOT/'second_zero/latent_owner_data.py',
        ROOT/'second_zero/aggressive_controller.py',ROOT/'second_zero/vision_spell_balance.py',
        ROOT/'second_zero/common.py',ROOT/'scripts/train_contextual.py',ROOT/'second_zero/contextual.py',
        ROOT/'scripts/prepare_source_control_teacher.py']



def prepare(args):
    """Freeze a trainer recipe around the separate audited artifact builder."""
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=meta(args.data_manifest)
    assert tuple(manifest['source_train_matches'])==TRAIN
    expected={f'SK{m}-{h}'for m in TRAIN for h in ('firstHalf','secondHalf')}
    assert set(manifest['rows'])==expected
    for path,digest in manifest.get('source_sha256',{}).items():pin(ROOT/path,digest)
    for path,digest in manifest.get('input_sha256',{}).items():pin(ROOT/path,digest)
    for item in manifest['rows'].values():
        pin(ROOT/item['tracking_path'],item['tracking_sha256'])
        pin(ROOT/item['labels_path'],item['labels_sha256'])
        pin(ROOT/item['events_path'],item['events_sha256'])
    audits=ROOT/'reports/aggressive-20261029/source-semantics'
    for name in ('dynamic-actor-audit','actor-point-audit'):
        for extension in ('.json','.md','.py'):
            pin(audits/(name+extension))
    for path in sources():pin(path)
    recipe={'steps':600,'batch':8,'target':400,'context':200,'seed':20261104,'lr':.0007,'weight_decay':.001,
        'warmup_steps':30,'minimum_lr':.00007,'gradient_clip':2.,'device':'mps','precision':'bf16',
        'loss_weights':{'raw_actor':.5,'controlled_actor':.5,'pp_start':.25},'slot_swap_probability':.25,
        'sampler':'Half event-based eligibleone-touch,quarter other eligiblePP starts,quarter originalnative rawcontroller intervals; shuffled queues, one sharedcrop/start and restoredaugmentationRNG forbothheads.'}
    protocol={'kind':'source-only-native-control-teacher-v1','source_train_matches':list(TRAIN),
        'architecture':dataclasses.asdict(TeacherConfig()),'recipe':recipe,
        'source_sha256':{rel(p):sha256(p)for p in sources()},'input_sha256':dict(PINNED),
        'data_manifest_path':rel(args.data_manifest),'data_manifest_sha256':sha256(args.data_manifest),
        'rows':manifest['rows'],'data_totals':manifest.get('totals',{}),
        'point_policy':'Builderunique nearest25Hz nativepoints; no repeated25Hz supervision.',
        'cross_stream_policy':'Keep eligible raw/DE disagreements in separateheads; nevermergeorrelabel. Reportconflictexposures.',
        'pp_start_policy':'DocumentedeligiblePPfirstpoint1, knowneligibleDEnativecontinuation0, allotherpointsunknown; NOTprimaryacquisition.',
        'unknown_policy':'Unknownactors arenottransit/negatives; group/phase/Sportecownership neverloaded.',
        'tail_policy':'Padlabeldomainonlytothenext5frameboundarywithunknown/False (atmost4frames); rawtrackingunchanged andfrozenhelperzero-padsfeatures. No knownlabel orobservationcreated.',
        'fit_requires_profile_budget_receipt':True,'checkpoint_selection':'Final600only; noCAL/primaryselection.',
        'source_CAL_read':False,'Sportec_read':False,'training_performed':False}
    json_write(out/'protocol.json',protocol)
    print('SOURCE TEACHER TRAINER PREPARED',protocol['data_totals'],flush=True)


def make_teacher_row(key,tracking_path,arrays):
    """Keep original labels/masks, pad only unknown label-domain tail to phase5."""
    required={'raw_actor','de_actor','pp_first_touch','native_point','raw_de_conflict',
        'raw_loss_mask','de_loss_mask','one_touch_anchor','eligible_PP_start_anchor','raw_intervals'}
    if not required<=set(arrays):raise ValueError('Derivedteacherartifactfields missing:'+str(sorted(required-set(arrays))))
    n=len(arrays['raw_actor']);padded=((n+4)//5)*5;extra=padded-n
    raw=np.asarray(arrays['raw_actor'],np.int64).copy();de=np.asarray(arrays['de_actor'],np.int64).copy()
    pp=np.asarray(arrays['pp_first_touch'],np.float32).copy()
    raw[~arrays['raw_loss_mask'].astype(bool)]=-1;de[~arrays['de_loss_mask'].astype(bool)]=-1;pp[de<0]=-1
    native=np.asarray(arrays['native_point'],bool)
    assert native[raw>=0].all() and native[de>=0].all() and native[pp>=0].all()
    raw=np.pad(raw,(0,extra),constant_values=-1);de=np.pad(de,(0,extra),constant_values=-1);pp=np.pad(pp,(0,extra),constant_values=-1)
    unknown=np.full(padded,3,np.int8)
    one=np.flatnonzero(arrays['one_touch_anchor']&(arrays['de_actor']>=0)&arrays['de_loss_mask'])
    starts=np.flatnonzero(arrays['eligible_PP_start_anchor']&(arrays['de_actor']>=0)&arrays['de_loss_mask'])
    intervals=np.asarray(arrays['raw_intervals'],np.int64).reshape(-1,2)
    intervals=np.array([(a,b)for a,b in intervals if (raw[a:b]>=0).any()],np.int64).reshape(-1,2)
    return {'key':key,'raw':make_record(key,path=tracking_path,y=unknown,actor=raw,source=True),
        'DE':make_record(key,path=tracking_path,y=unknown,actor=de,source=True),
        'raw_actor':raw,'controlled_actor':de,'pp_start':pp,'native_point':np.pad(native,(0,extra)),
        'cross_stream_conflict':np.pad(np.asarray(arrays['raw_de_conflict'],bool),(0,extra)),
        'one_touch_anchors':one,'PP_anchors':np.setdiff1d(starts,one),'raw_intervals':intervals,
        'original_frames':n,'unknown_tail_padding':extra}


def load_rows(out):
    p=meta(out/'protocol.json');assert tuple(p['source_train_matches'])==TRAIN
    for path,digest in p['source_sha256'].items():pin(ROOT/path,digest)
    for path,digest in p['input_sha256'].items():pin(ROOT/path,digest)
    manifest=meta(ROOT/p['data_manifest_path'],p['data_manifest_sha256'])
    assert manifest['rows']==p['rows'] and tuple(manifest['source_train_matches'])==TRAIN
    rows=[]
    for key,item in p['rows'].items():
        path=pin(ROOT/item['labels_path'],item['labels_sha256']);tracking=pin(ROOT/item['tracking_path'],item['tracking_sha256'])
        with np.load(path,allow_pickle=False)as z:a={k:z[k]for k in z.files}
        rows.append(make_teacher_row(key,tracking,a))
    return p,rows


class Sampler:
    def __init__(self,rows,seed):
        self.rows=rows;self.rng=np.random.default_rng(seed);self.pools={'one':[],'PP':[],'raw':[]};self.orders={};self.cursor={}
        for i,row in enumerate(rows):
            self.pools['one'] += [(i,int(a))for a in row['one_touch_anchors']]
            self.pools['PP'] += [(i,int(a))for a in row['PP_anchors']]
            self.pools['raw'] += [(i,(int(a),int(b)))for a,b in row['raw_intervals']]
        if any(not values for values in self.pools.values()):raise ValueError('Every declared source sampler stratum needs support')
        for name,values in self.pools.items():self.orders[name]=self.rng.permutation(len(values));self.cursor[name]=0
    def state(self):return {'rng':copy.deepcopy(self.rng.bit_generator.state),'orders':copy.deepcopy(self.orders),'cursor':dict(self.cursor)}
    def restore(self,state):self.rng.bit_generator.state=state['rng'];self.orders=copy.deepcopy(state['orders']);self.cursor=dict(state['cursor'])
    def draw(self,name,target):
        if self.cursor[name]==len(self.orders[name]):self.orders[name]=self.rng.permutation(len(self.pools[name]));self.cursor[name]=0
        index=int(self.orders[name][self.cursor[name]]);self.cursor[name]+=1;i,anchor=self.pools[name][index];row=self.rows[i]
        if name=='raw':
            a,b=anchor;known=np.flatnonzero(row['raw_actor'][a:b]>=0)+a;anchor=int(self.rng.choice(known))
        n=len(row['raw_actor']);first=max(0,((anchor-target+1+4)//5)*5);last=min((anchor//5)*5,((n-target)//5)*5)
        if first>last:raise ValueError('Source anchor has no complete phasealigned core')
        return row,int(self.rng.choice(np.arange(first,last+1,5))),int(anchor),name


def paired_crop(row,start,rng,context,target,augment=True,slot_swap_probability=.25):
    before=copy.deepcopy(rng.bit_generator.state)
    raw=sample_record(row['raw'],start,context=context,target=target,rng=rng,augment=augment,slot_swap_probability=slot_swap_probability)
    after=copy.deepcopy(rng.bit_generator.state);rng.bit_generator.state=before
    de=sample_record(row['DE'],start,context=context,target=target,rng=rng,augment=augment,slot_swap_probability=slot_swap_probability)
    assert rng.bit_generator.state==after
    for name,value in raw['values'].items():np.testing.assert_array_equal(value,de['values'][name])
    stop=start+target;pp=row['pp_start'][start:stop].copy();pp[de['source_actor']<0]=-1
    native=row['native_point'][start:stop]
    assert native[raw['source_actor']>=0].all() and native[de['source_actor']>=0].all() and native[pp>=0].all()
    return {'values':raw['values'],'raw_actor':raw['source_actor'],'controlled_actor':de['source_actor'],'pp_start':pp,
        'trace':raw['trace'],'cross_stream_conflict':row['cross_stream_conflict'][start:stop].copy()}


def batch(sampler,recipe,device):
    size=recipe['batch']
    if size<4 or size%4:raise ValueError('Teacher batch must be a multipleof4 fordeclaredstrata')
    names=['one']*(size//2)+['PP']*(size//4)+['raw']*(size//4);samples=[]
    for name in names:
        row,start,anchor,stratum=sampler.draw(name,recipe['target'])
        s=paired_crop(row,start,sampler.rng,recipe['context'],recipe['target'],slot_swap_probability=recipe['slot_swap_probability'])
        s['trace'].update(stratum=stratum,sampling_anchor=anchor);samples.append(s)
    maximum=max(len(s['values']['team'])for s in samples);values={}
    for key in samples[0]['values']:
        padded=[]
        for sample in samples:
            v=sample['values'][key];p=len(sample['values']['team'])
            if key=='team':v=np.pad(v,(0,maximum-p),constant_values=-1)
            elif key!='ball':v=np.pad(v,((0,0),(0,maximum-p))+((0,0),)*(v.ndim-2))
            padded.append(v)
        values[key]=torch.from_numpy(np.stack(padded)).to(device)
    labels={key:torch.from_numpy(np.stack([s[key]for s in samples])).to(device)for key in ('raw_actor','controlled_actor','pp_start','cross_stream_conflict')}
    return {'values':values,**labels,'trace':[s['trace']for s in samples]}


def loss(output,sample,weights):
    raw,nraw=actor_loss(output['raw_actor_logits'].float(),sample['raw_actor'],output['active'])
    de,nde=actor_loss(output['controlled_actor_logits'].float(),sample['controlled_actor'],output['active'])
    index=sample['controlled_actor'].clamp_min(0).long();pp_logits=output['pp_start_logits'].float().gather(2,index[...,None])[...,0]
    known=(sample['pp_start']>=0)&(sample['controlled_actor']>=0)&output['active'].gather(2,index[...,None])[...,0]
    target=sample['pp_start'].clamp_min(0).to(pp_logits.dtype)
    pp_values=F.binary_cross_entropy_with_logits(pp_logits,target,reduction='none')
    pp=torch.where(known,pp_values,0).sum()/known.sum().clamp_min(1)
    value=weights['raw_actor']*raw+weights['controlled_actor']*de+weights['pp_start']*pp
    if not torch.isfinite(value):raise ValueError('Nonfinite source teacher loss')
    conflict=(sample['raw_actor']>=0)&(sample['controlled_actor']>=0)&(sample['raw_actor']!=sample['controlled_actor'])
    return value,{'raw_CE':raw,'DE_CE':de,'PP_start_BCE':pp,'raw_points':nraw,'DE_points':nde,
        'PP_start_points':(known&(target==1)).sum(),'PP_continuation_points':(known&(target==0)).sum(),
        'eligible_conflict_exposures':conflict.sum(),'original_conflict_exposures':sample['cross_stream_conflict'].sum()}


def forward_loss(model,sample,recipe,device,precision):
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=precision=='bf16'):
        output=model(sample['values'],core_start=recipe['context'],core_stop=recipe['context']+recipe['target'])
    return loss(output,sample,recipe['loss_weights'])


def profile(args):
    setup_started=time.monotonic()
    protocol,rows=load_rows(args.out);recipe=protocol['recipe'];torch.manual_seed(recipe['seed']);sampler=Sampler(rows,recipe['seed'])
    model=SourceControlTeacher(TeacherConfig(**protocol['architecture'])).to(args.device)
    assert model.required_context==recipe['context']
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay']);times=[];materialization=[];compute=[];parts=None;traces=[]
    setup_seconds=time.monotonic()-setup_started
    for _ in range(args.profile_iterations):
        if args.device=='mps':torch.mps.synchronize()
        started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        sample=batch(sampler,recipe,args.device)
        if args.device=='mps':torch.mps.synchronize()
        prepared=time.monotonic();materialization.append(prepared-started);traces.append(sample['trace'])
        value,parts=forward_loss(model,sample,recipe,args.device,args.precision);value.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()for p in model.parameters()):raise ValueError('Nonfinite source teacher profilegradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip']);optimizer.step()
        if args.device=='mps':torch.mps.synchronize()
        finished=time.monotonic();times.append(finished-started);compute.append(finished-prepared)
    result={'kind':'source-TRAIN16 timing/gradient smoke only; nocandidateweights saved','protocol_sha256':sha256(args.out/'protocol.json'),
        'device':args.device,'precision':args.precision,'seconds':times,'parameters':sum(p.numel()for p in model.parameters()),
        'setup_seconds':setup_seconds,'materialization_seconds':materialization,'compute_seconds':compute,
        'fresh_batch_each_iteration':True,'batch_traces':traces,
        'projected_fit_seconds_median':float(np.median(times[1:] or times)*recipe['steps']),
        'projected_fit_seconds_max':float(max(times)*recipe['steps']),
        'input_shape':list(sample['values']['player'].shape),'all_gradients_finite':True,
        'components':{k:float(v.detach())for k,v in parts.items()},'source_read_paths':sorted(READS),'Sportec_read':False,'metrics_computed':False}
    if args.device=='mps':result['MPS_bytes']={'current':torch.mps.current_allocated_memory(),'driver':torch.mps.driver_allocated_memory()}
    if args.profile_out:json_write(args.profile_out,result)
    print(json.dumps(result),flush=True)


def checkpoint_state(model,optimizer,scheduler,sampler,device,**metadata):
    if device=='mps':torch.mps.synchronize()
    return {**metadata,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
        'sampler':sampler.state(),'torch_rng':torch.get_rng_state(),
        'mps_rng':torch.mps.get_rng_state()if device=='mps'else None}


def restore_checkpoint(state,model,optimizer,scheduler,sampler,device):
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    sampler.restore(state['sampler']);torch.set_rng_state(state['torch_rng'])
    if device=='mps':torch.mps.set_rng_state(state['mps_rng'])


def fit(args):
    protocol,rows=load_rows(args.out);recipe=protocol['recipe']
    if args.authorization is None:raise PermissionError('Main fitting requires root go after profile/posterior review')
    authorization=meta(args.authorization)
    assert authorization['allow_source_teacher_fit'] is True and authorization['protocol_sha256']==sha256(args.out/'protocol.json')
    report=meta(ROOT/authorization['profile_path'],authorization['profile_sha256'])
    assert report['protocol_sha256']==sha256(args.out/'protocol.json') and report['all_gradients_finite']
    assert report['device']==recipe['device'] and report['precision']==recipe['precision']
    device=recipe['device'];torch.manual_seed(recipe['seed']);sampler=Sampler(rows,recipe['seed'])
    model=SourceControlTeacher(TeacherConfig(**protocol['architecture'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay'])
    def factor(step):
        if step<recipe['warmup_steps']:return (step+1)/recipe['warmup_steps']
        progress=(step-recipe['warmup_steps'])/max(1,recipe['steps']-recipe['warmup_steps'])
        minimum=recipe['minimum_lr']/recipe['lr'];return minimum+(1-minimum)*.5*(1+np.cos(np.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor);start=1;history=[];traces=[];exposures={};resume=args.out/'resume.pt'
    if args.resume:
        state=torch.load(resume,map_location='cpu',weights_only=False);assert state['protocol_sha256']==sha256(args.out/'protocol.json')
        assert state['authorization_sha256']==sha256(args.authorization)
        restore_checkpoint(state,model,optimizer,scheduler,sampler,device)
        start=state['step']+1;history=state['history'];traces=state['traces'];exposures=state['exposures']
    elif resume.exists() or (args.out/'freeze.json').exists():raise FileExistsError('Explicitresume required; neveroverwritefrozenfit')
    started=time.monotonic()
    for step in range(start,recipe['steps']+1):
        sample=batch(sampler,recipe,device);traces.append({'step':step,'rows':sample['trace']});model.train();optimizer.zero_grad(set_to_none=True)
        value,parts=forward_loss(model,sample,recipe,device,recipe['precision']);value.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'])
        if not torch.isfinite(norm):raise ValueError('Nonfinite teachergradient')
        optimizer.step();scheduler.step()
        for key,v in parts.items():
            if 'points'in key or 'exposures'in key:exposures[key]=exposures.get(key,0)+float(v.detach())
        if step%25==0:
            row={'step':step,'loss':float(value.detach()),**{k:float(v.detach())for k,v in parts.items()},'seconds':time.monotonic()-started};history.append(row);print('SOURCE TRAIN',json.dumps(row),flush=True)
        if step%100==0 or step==recipe['steps']:
            state=checkpoint_state(model,optimizer,scheduler,sampler,device,step=step,
                history=history,traces=traces,exposures=exposures,protocol_sha256=sha256(args.out/'protocol.json'),authorization_sha256=sha256(args.authorization))
            temporary=resume.with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(resume)
            json_write(args.out/'progress.json',{'step':step,'history':history[-4:],'exposures':exposures})
    checkpoint=args.out/'model.pt';torch.save({'architecture':protocol['architecture'],'state_dict':{k:v.detach().cpu()for k,v in model.state_dict().items()},'step':recipe['steps']},checkpoint)
    json_write(args.out/'freeze.json',{'kind':'source-only-control-teacher-final','model_sha256':sha256(checkpoint),
        'protocol_sha256':sha256(args.out/'protocol.json'),'authorization_sha256':sha256(args.authorization),'source_train_matches':list(TRAIN),
        'step':recipe['steps'],'exposures':exposures,'batch_trace_sha256':hashlib.sha256(json.dumps(traces,separators=(',',':')).encode()).hexdigest(),
        'source_read_paths':sorted(READS),'Sportec_read':False,'source_CAL_read':False,'checkpoint_selection':'Finalonly; no primaryorCAL selection.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','profile','fit'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--data-manifest',type=Path,default=ROOT/'private/improvement/source-control-teacher-20261029/manifest.json');parser.add_argument('--device',choices=('cpu','mps'),default='cpu')
    parser.add_argument('--precision',choices=('fp32','bf16'),default='fp32');parser.add_argument('--profile-iterations',type=int,default=3)
    parser.add_argument('--profile-out',type=Path);parser.add_argument('--authorization',type=Path);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.precision=='bf16'and args.device!='mps':raise ValueError('BF16 profiles require MPS')
    sys.addaudithook(guard);torch.set_num_threads(2);threadpool_limits(2);globals()[args.stage](args)
