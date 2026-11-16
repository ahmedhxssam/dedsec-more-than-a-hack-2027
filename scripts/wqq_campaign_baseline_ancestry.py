"""Read-only proof of the retained full baseline's functional fit ancestry.

Only metadata, source files, weights, and TRAIN feature-producer outputs are
hashed. No observation/reference array is loaded. Historical development use
is disclosed; it must not be confused with an untouched-test claim.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY')
CAL = 'J03WPY'
BASE = 'runs/full-baseline-gap-v1'
SPRINT = 'runs/two-hour-sprint'


class _Proof:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.evidence = {}
        self.nodes = {}

    def path(self, name):
        p = Path(name)
        p = (self.root / p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Ancestry path escapes workspace')
        s = str(p.relative_to(self.root))
        if any(x in s for x in ('J03WQQ', 'J03WR9', '.truth.', '.actor.', 'evaluation/gt/', 'runs/main/')):
            raise PermissionError('Ancestry proof cannot open reference/reserved assets')
        return p, s

    def pin(self, name, expected=None):
        p, s = self.path(name)
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        value = digest.hexdigest()
        if expected is not None and value != expected:
            raise ValueError('Changed historical dependency: ' + s)
        self.evidence[s] = {'path': s, 'sha256': value}
        return self.evidence[s]

    def read(self, name, expected=None):
        receipt = self.pin(name, expected)
        p, _ = self.path(name)
        doc = json.loads(p.read_text())
        self.pin(name, receipt['sha256'])
        return doc

    def sources(self, doc, folder=None, scalar_source=None):
        src = doc.get('source_sha256', doc.get('sources'))
        if isinstance(src, dict):
            for path, digest in src.items():
                self.pin(path, digest)
        elif isinstance(src, str):
            choices = [f'{folder}/source.py'] if folder else []
            choices += [scalar_source] if scalar_source else []
            for path in choices:
                p, _ = self.path(path)
                if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == src:
                    self.pin(path, src)
                    break
            else:
                raise ValueError('No matching historical source for ' + str(folder))
        else:
            raise ValueError('Historical source identity missing: ' + str(folder))

    def runtime_sources(self, entry):
        """Pin local imports without importing legacy scripts or their hooks."""
        queue = [entry]
        seen = set()
        while queue:
            path = queue.pop()
            if path in seen:
                continue
            seen.add(path)
            self.pin(path)
            source, _ = self.path(path)
            for node in ast.walk(ast.parse(source.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [x.name for x in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module] + [node.module + '.' + x.name for x in node.names]
                for name in names:
                    if name.split('.')[0] not in ('scripts', 'second_zero'):
                        continue
                    candidate = name.replace('.', '/') + '.py'
                    target, _ = self.path(candidate)
                    if target.is_file():
                        queue.append(candidate)

    @staticmethod
    def cohort(doc):
        value = doc.get('train_matches', doc.get('training_matches'))
        if value is None:
            value = doc.get('split', {}).get('train')
        if not isinstance(value, list) or not value or len(value) != len(set(value)):
            raise ValueError('Explicit historical fit cohort is missing')
        if not set(value) <= set(TRAIN):
            raise ValueError('Non-TRAIN match in historical fitting')
        return value

    @staticmethod
    def reference_roles(doc, fit):
        """Inspect receipt strings; never open their label-bearing payloads."""
        for field in ('reference_reads', 'truth_reads'):
            for row in doc.get(field, []):
                name = row.get('name', row.get('path', '')) if isinstance(row, dict) else row
                ids = set(re.findall(r'J03[A-Z0-9]{3}', str(name)))
                allowed = {CAL} if isinstance(row, dict) and row.get('phase') == 'calibrate' else set(fit)
                if not ids or not ids <= allowed:
                    raise ValueError('Historical label-read cohort mismatch')

    def model(self, name, item, *, fit=TRAIN, freeze='freeze.json',
              protocol=None, scalar_source=None, calibrated=True, legacy=False,
              preceding_reference_reads=None):
        folder = str(Path(item['path']).parent)
        proto_path = protocol or f'{folder}/protocol.json'
        frozen = self.read(f'{folder}/{freeze}')
        proto = self.read(proto_path, frozen.get('protocol_sha256'))
        digest = frozen.get('model_sha256', frozen.get('models_sha256'))
        if digest != item['sha256']:
            raise ValueError('Specification/model freeze mismatch: ' + name)
        artifact = self.pin(item['path'], digest)
        if legacy:
            split = self.read('private/improvement/external/split.lock.json', proto['split_sha256'])
            actual_fit = split['split']['train']
            # The old selected decoder used both development matches. Its
            # probabilities-only weights, not that decoder, enter this graph.
            if set(item) != {'path', 'sha256'}:
                raise ValueError('Legacy decoder configuration entered functional graph')
        else:
            actual_fit = self.cohort(proto)
        if set(actual_fit) != set(fit):
            raise ValueError('Historical model fit cohort mismatch: ' + name)
        self.sources(proto, folder, scalar_source)
        self.reference_roles(proto, fit)
        scoped_frozen = frozen
        if preceding_reference_reads:
            # The historical interpolation CLI trained both folds in one
            # process and retained its cumulative audit list. Bind the exact
            # prior fold prefix before checking this fit's new read suffix.
            reads = frozen.get('truth_reads', [])
            n = len(preceding_reference_reads)
            if reads[:n] != preceding_reference_reads:
                raise ValueError('Cumulative historical reference prefix differs')
            scoped_frozen = {**frozen, 'truth_reads': reads[n:]}
        self.reference_roles(scoped_frozen, fit)
        cal_roles = []
        cal = None
        if calibrated:
            cal_path = f'{folder}/calibration.json'
            p, _ = self.path(cal_path)
            if p.exists():
                cal = self.read(cal_path, frozen.get('calibration_sha256'))
                if 'match' in cal and cal['match'] != CAL:
                    raise ValueError('Historical calibration is not WPY')
            if proto.get('calibration_match', CAL) != CAL:
                raise ValueError('Historical calibration cohort mismatch')
            if 'calibration_match' not in proto and cal is None:
                if 'WPY' not in str(proto.get('calibration', '')):
                    raise ValueError('Missing WPY calibration evidence')
            cal_roles = [CAL]
        training_path = f'{folder}/training.json'
        p, _ = self.path(training_path)
        if p.exists():
            training = self.read(training_path)
            for key in training.get('training_keys', []):
                if str(key).split('-')[0] not in fit:
                    raise ValueError('Historical training-key cohort mismatch')
            for key in training.get('source_reads', []):
                ids = set(re.findall(r'J03[A-Z0-9]{3}', key))
                if ids and not ids <= set(fit):
                    raise ValueError('Historical training source-read mismatch')
        if 'temperature' in item:
            expected = frozen.get('temperature', (cal or {}).get('temperature'))
            if expected != item['temperature']:
                raise ValueError('Historical temperature differs from specification')
        self.nodes[name] = {'artifact': artifact, 'fit_matches': list(fit),
                            'calibration_matches': cal_roles,
                            'protocol': self.evidence[proto_path],
                            'freeze': self.evidence[f'{folder}/{freeze}'],
                            'dependencies': []}
        return proto, frozen, cal


def validate_full_baseline(root=ROOT):
    """Return evidence pins + verified roles, without importing model code.

    Deliberately revalidates on every call: cached Python objects cannot conceal
    a changed checkpoint or metadata file between calibration and inference.
    """
    p = _Proof(root)
    spec_path = f'{BASE}/full-reference-spec.json'
    spec = p.read(spec_path)
    replay = p.read(f'{BASE}/loader-parity.json')
    if (spec.get('kind') != 'complete-retained-raw-baseline-artifacts-v1'
            or replay.get('all_outputs_bit_exact') is not True
            or replay.get('inherited_predict_identity') is not True
            or replay.get('reference_matches') != []
            or replay.get('raw_matches') != [CAL]
            or replay.get('spec') != p.evidence[spec_path]):
        raise ValueError('Full baseline raw-loader parity is not bound to specification')
    for path, digest in replay['sources'].items():
        p.pin(path, digest)
    p.pin(spec['prediction_source']['path'], spec['prediction_source']['sha256'])
    p.runtime_sources('second_zero/full_baseline_pipeline.py')
    if len(spec['owners']) != 4:
        raise ValueError('Full baseline owner inventory changed')
    for i, item in enumerate(spec['owners']):
        p.model(f'owner{i}', item)
    p.model('imputer', spec['imputer'], calibrated=False,
            scalar_source='scripts/sprint_ball_imputer.py')
    p.model('interpolation_owner', spec['interpolation_owner'], freeze='complete.json')
    memory, memory_freeze, _ = p.model('memory', spec['memory'],
            scalar_source='scripts/sprint_memory.py')
    if any(memory[k] != spec['memory'][k] for k in ('channels', 'width', 'stride', 'context')):
        raise ValueError('Memory architecture differs from specification')
    carrier, carrier_freeze, _ = p.model('carrier', spec['carrier'],
            scalar_source='scripts/sprint_gap_specialist.py')
    if carrier['imputer_sha256'] != spec['imputer']['sha256']:
        raise ValueError('Carrier imputer ancestor mismatch')
    p.nodes['carrier']['dependencies'] = ['imputer']
    cmix_path = f'{SPRINT}/carrier-mixture-v1/freeze.json'
    cmix = p.read(cmix_path)
    p.sources(cmix, f'{SPRINT}/carrier-mixture-v1', 'scripts/sprint_carrier_mix.py')
    if (cmix['carrier_freeze_sha256'] != p.evidence[f'{SPRINT}/carrier-specialist-v1/freeze.json']['sha256']
            or cmix['WPY_weight'] != spec['carrier']['weight']):
        raise ValueError('Carrier mixture calibration differs from specification')
    p.model('legacy', spec['legacy'], legacy=True, calibrated=False,
            scalar_source='second_zero/contextual.py')
    temporal, tf, _ = p.model('temporal_status', spec['temporal_status'])
    owner_freeze = p.evidence['runs/ownership-temporal-v1/freeze.json']['sha256']
    if tf['owner_freeze_sha256'] != owner_freeze or temporal['owner_freeze_sha256'] != owner_freeze:
        raise ValueError('Temporal status owner provenance mismatch')
    _, _, ccal = p.model('context_status', spec['context_status'],
            scalar_source='scripts/sprint_status_context.py')
    if ccal != spec['context_status']['calibration']:
        raise ValueError('Context calibration differs from specification')
    rel_folder = f'{SPRINT}/status-mixture-v1'
    rel_protocol = p.read(f'{rel_folder}/protocol.json')
    rel_freeze = p.read(f'{rel_folder}/reliability/freeze.json')
    rel_cal = p.read(f'{rel_folder}/reliability/calibration.json', rel_freeze['calibration_sha256'])
    p.sources(rel_protocol, rel_folder, 'scripts/sprint_status_mixture.py')
    if (rel_protocol['calibration_match'] != CAL
            or rel_protocol['WQQ_labels_accessed'] is not False
            or rel_freeze['WQQ_labels_accessed'] is not False
            or rel_freeze['source_sha256'] != rel_protocol['source_sha256']
            or rel_cal['fit'] != spec['reliability']['fit']
            or rel_cal['selected'] != spec['reliability']['selection']
            or p.read(f'{rel_folder}/mixture_fit.json') != rel_cal['fit']):
        raise ValueError('Reliability calibration/provenance mismatch')
    p.nodes['reliability'] = {'fit_matches': [], 'calibration_matches': [CAL],
                            'dependencies': ['legacy', 'context_status', 'temporal_status']}
    boundary, bf, _ = p.model('boundary', spec['boundary'])
    if bf['config'] != spec['boundary']['config'] or boundary['fixed_owner_freeze_sha256'] != owner_freeze:
        raise ValueError('Boundary configuration/owner provenance mismatch')
    _, qf, _ = p.model('quality', spec['quality'], scalar_source='scripts/sprint_decoder_segments_geometry_v2.py')
    if qf['WPY_selected'] is not True or qf['public_final_labels_used'] is not False:
        raise ValueError('Quality selection provenance mismatch')
    if (qf['penalty'], qf['frame_weight']) != (spec['contract']['penalty'], spec['contract']['frame_weight']):
        raise ValueError('Quality decoder differs from specification')
    joint_folder = str(Path(spec['joint']['path']).parent)
    joint_weight = {k: spec['joint'][k] for k in ('path', 'sha256')}
    jp, jf, _ = p.model('joint', joint_weight, freeze='freeze-fit.json',
            protocol=str(Path(joint_folder).parent / 'protocol.json'), calibrated=False)
    scope = p.read(f'{joint_folder}/scope-freeze.json')
    p.read(f'{joint_folder}/scope-calibration.json', scope['calibration_sha256'])
    p.sources(scope, joint_folder, 'scripts/joint_controller_ball_scope.py')
    if (scope['model_sha256'] != spec['joint']['sha256'] or scope['reference_scope'] != 'WPY only'
            or any(scope[k] != spec['joint'][k] for k in ('temperature', 'weight', 'scope'))):
        raise ValueError('Joint composition calibration differs from specification')
    p.nodes['joint']['calibration_matches'] = [CAL]
    # Stacked training inputs have their own learned producers. Check actual
    # opposite-pair weights and calibrations, not just the top-level cohort.
    ensemble = p.read('runs/oof-uniform-ensemble-v1/protocol.json')
    if ensemble['WQQ_public_final_access'] is not False:
        raise ValueError('OOF owner feature ancestors accessed reserved labels')
    stacked = []
    preceding_interpolation_reads = []
    for fold, cohort in enumerate((TRAIN[2:], TRAIN[:2])):
        for arm, folder in ensemble['models'][str(fold)].items():
            frozen = p.read(f'{folder}/freeze.json')
            item = {'path': f'{folder}/model.pt', 'sha256': frozen['model_sha256']}
            node = f'feature_owner_{fold}_{arm}'
            p.model(node, item, fit=cohort)
            stacked.append(node)
        folder = memory['extra_models'][str(fold)]
        frozen = p.read(f'{folder}/complete.json')
        node = f'feature_interpolation_{fold}'
        p.model(node, {'path': f'{folder}/model.pt', 'sha256': frozen['model_sha256']},
                fit=cohort, freeze='complete.json',
                preceding_reference_reads=preceding_interpolation_reads)
        preceding_interpolation_reads = frozen['truth_reads']
        stacked.append(node)
        folder = f'{SPRINT}/ball-imputer-fold{fold}'
        frozen = p.read(f'{folder}/freeze.json')
        p.model(f'feature_imputer_{fold}', {'path': f'{folder}/models.pkl', 'sha256': frozen['models_sha256']},
                fit=cohort, calibrated=False, scalar_source='scripts/sprint_ball_imputer.py')
    if memory['extra_models']['full'] != str(Path(spec['interpolation_owner']['path']).parent):
        raise ValueError('Full interpolation ancestor mismatch')
    p.nodes['memory']['dependencies'] = stacked + [f'owner{i}' for i in range(4)] + ['interpolation_owner']
    crossfit = p.read('runs/overnight-boundary-v1/crossfit_manifest.json')
    if crossfit['whole_match_disjoint'] is not True or crossfit['public_or_development_or_final_fit'] is not False:
        raise ValueError('Boundary proposal fit ancestry mismatch')
    expected_keys = {f'{m}-{h}-{c}' for m in TRAIN for h in ('firstHalf', 'secondHalf') for c in ('clean', 'corrupted')}
    if set(crossfit['records']) != expected_keys:
        raise ValueError('Boundary feature cohort inventory mismatch')
    for key, row in crossfit['records'].items():
        held = key.split('-')[0]
        cohort = TRAIN[2:] if held in TRAIN[:2] else TRAIN[:2]
        if set(row['model_train_matches']) != set(cohort):
            raise ValueError('Held match in boundary feature producer')
        p.pin(f'runs/overnight-boundary-v1/crossfit/{key}.npz', row['prediction_sha256'])
    p.nodes['boundary']['dependencies'] = ['owner0', 'legacy', 'boundary_oof_proposals']
    p.nodes['boundary_oof_proposals'] = {'fit_matches': list(TRAIN), 'calibration_matches': [],
                                       'dependencies': [], 'per_target_fit_exclusion_verified': True}
    p.nodes['quality']['dependencies'] = stacked + ['feature_imputer_0', 'feature_imputer_1', 'boundary_oof_proposals']
    p.pin('scripts/wqq_campaign_baseline_ancestry.py')
    return {'kind': 'wqq-campaign-full-baseline-ancestry-v1', 'ancestry_verified': True,
            'fit_matches': list(TRAIN), 'calibration_matches': [CAL],
            'evidence': [p.evidence[k] for k in sorted(p.evidence)],
            'nodes': p.nodes, 'reference_arrays_opened': [],
            'legacy_decoder_selection_excluded': True,
            'historical_interpolation_read_receipt': 'Fold1 cumulative read prefix verified against fold0; only suffix belongs to fold1 fit.',
            'adaptivity': 'WQQ is reused development; this proves functional fit/calibration roles, not untouched-test independence.'}
