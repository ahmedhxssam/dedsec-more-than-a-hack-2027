"""Versioned runtime/admission repair around immutable campaign producers.

The queue resolves interpreter symlinks. Explicit local package paths preserve
the verified environment even when that resolution bypasses venv discovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
PACKAGES=[ROOT/'private/improvement/runtime',ROOT/'private/improvement/vision-env/lib/python3.11/site-packages',ROOT/'.venv/lib/python3.11/site-packages']
sys.path[:0]=[str(ROOT),*[str(p) for p in PACKAGES]]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(ref):
    path=(ROOT/ref['path']).resolve(); path.relative_to(ROOT)
    if digest(path)!=ref['sha256']:
        raise ValueError('Changed admission artifact: '+ref['path'])
    return path


def entry_path(value):
    path=(ROOT/value).resolve()
    path.relative_to(ROOT/'scripts')
    if path.suffix!='.py' or not path.is_file() or path==Path(__file__).resolve():
        raise ValueError('A distinct repository Python entrypoint is required')
    return path


def preflight_full(study,permit):
    study=Path(study).resolve(); study.relative_to(ROOT/'runs/wqq-090-20261031/studies')
    protocol_path=study/'protocol.json'; protocol=read(protocol_path); nomination=read(permit)
    protocol_pin={'path':str(protocol_path.relative_to(ROOT)),'sha256':digest(protocol_path)}
    if (nomination.get('candidate_id')!=study.name or nomination.get('protocol')!=protocol_pin
            or nomination.get('selected_recipe')!=protocol['recipe']):
        raise PermissionError('Nomination must match this exact study and recipe before WQQ access')
    from scripts import wqq_campaign_evaluate as evaluation
    evaluation._nomination({'path':str(Path(permit).resolve().relative_to(ROOT)),'sha256':digest(permit)},study.name,protocol_pin,ROOT)
    audit=read(checked(nomination['oof_audit']))
    seal_path=checked(audit['prediction_seal'])
    seal,_=evaluation.load_seal(seal_path,stage='oof')
    if seal['run_id']!=study.name or seal['protocol']!=protocol_pin or audit['protocol']!=protocol_pin:
        raise PermissionError('OOF audit, source protocol and current constituent evidence disagree')
    return {'kind':'wqq-campaign-admission-v1','nomination':{'path':str(Path(permit).resolve().relative_to(ROOT)),'sha256':digest(permit)},
            'oof_seal':audit['prediction_seal'],'protocol':protocol_pin,
            'runtime':{'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':digest(__file__)},
            'reference_reads':[]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',required=True)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts=parser.parse_args(); args=opts.args
    if args[:1]==['--']: args=args[1:]
    entry=entry_path(opts.entry)
    if entry.name=='wqq_campaign_study.py' and args[:1]==['full']:
        study=args[args.index('--study')+1]; permit=args[args.index('--permit')+1]
        admission=preflight_full(study,permit)
        from scripts.wqq_campaign_model import atomic_json
        atomic_json(Path(study)/'full-admission.json',admission,immutable=True)
    # Scorer dependency is verified before expensive work starts.
    import pandas
    from second_zero.evaluate import official_metrics
    official_metrics()
    sys.argv=[str(entry),*args]
    runpy.run_path(str(entry),run_name='__main__')


if __name__=='__main__': main()
