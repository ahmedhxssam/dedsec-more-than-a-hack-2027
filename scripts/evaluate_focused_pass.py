"""Official scoring plus explicit event accounting for the frozen bounded pass."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from second_zero.common import json_write, sha256
from second_zero.evaluate import evaluate_submission, official_metrics, PUBLIC_GT


def best_gain(x):
    value = best = 0.
    for t in x:
        value = max(0., value + float(t))
        best = max(value, best)
    return best


def diagnostic(pm, gt, pred, evidence):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gs, ps)
    matched = pm.matched_gt_flags(overlaps, len(gs), .5)
    short = []
    for i, (s, e, k) in enumerate(gs):
        if e - s > 125:
            continue
        candidates = [(v, ps[j]) for v, ii, j in overlaps if ii == i]
        best = candidates[0] if candidates else None
        span = best[1] if best else None
        local = [(a, b, int(c)) for a, b, c in pm.segments(filled, runs) if a < e and b > s]
        merge = sum(min(e2, span[1]) > max(s2, span[0]) and k2 == k for s2, e2, k2 in gs) if span else 0
        x = evidence['interaction_logit'][s:e] * (1 if k == pm.A else -1)
        short.append({'id': len(short) + 1, 'reference': [s, e, int(k)], 'duration_frames': e - s,
            'duration_seconds': (e - s) / 25, 'matched_iou50': bool(matched[i]), 'overlapping_predictions': local,
            'best_same_team_overlap': span, 'best_iou': best[0] if best else 0.,
            'onset_error_frames': span[0] - s if span else None,
            'offset_error_frames': span[1] - e if span else None,
            'same_team_reference_overlaps_for_prediction': int(merge),
            'favorable_contact_frames': int((x > 0).sum()), 'signed_contact_logit_sum': float(x.sum()),
            'maximum_favorable_subsequence_gain': best_gain(x),
            'description': ('missing same-team event / wrong ownership' if span is None else
                            'matched' if matched[i] else 'merged and/or displaced boundaries; inspect overlapping intervals')})
    gseg, pseg = pm.segments(gt, runs), pm.segments(filled, runs)
    stops = []
    buckets = dict(complete_miss=0, late_onset=0, early_restart=0, internal_holes=0)
    for s, e, k in gseg:
        if k != pm.D:
            continue
        spans = [(a, b) for a, b, c in pseg if c == pm.D and a < e and b > s]
        hit = np.flatnonzero(filled[s:e] == pm.D)
        if len(hit):
            lost = {'complete_miss': 0, 'late_onset': int(hit[0]), 'early_restart': int(e - s - hit[-1] - 1),
                    'internal_holes': int((filled[s + hit[0]:s + hit[-1] + 1] != pm.D).sum())}
        else:
            lost = {'complete_miss': e - s, 'late_onset': 0, 'early_restart': 0, 'internal_holes': 0}
        for key in buckets:
            buckets[key] += lost[key]
        stops.append({'id': len(stops) + 1, 'reference': [s, e], 'reference_onset_pair': [int(gt[s - 1]), pm.D],
            'reference_restart_pair': [pm.D, int(gt[e])], 'predicted_overlapping_D': spans,
            'onset_error_frames': spans[0][0] - s if spans else None,
            'offset_error_frames': spans[-1][1] - e if spans else None,
            'predicted_onset_pair': [int(filled[spans[0][0] - 1]), pm.D] if spans else None,
            'predicted_restart_pair': [pm.D, int(filled[spans[-1][1]])] if spans else None,
            'D_recovered_frames': int(len(hit)), 'missed_D_by_cause': lost,
            'winning_dead_emission_frames': int((evidence['p_dead'][s:e] > .5).sum()),
            'predicted_labels_inside_reference': [(a,b,int(c)) for a,b,c in pseg if a<e and b>s]})
    transitions = {}
    gtrans, ptrans = pm.transitions(gt, runs), pm.transitions(filled, runs)
    for name, predicate in [('onset', lambda v: v[1] in [pm.H, pm.A] and v[2] == pm.D),
                             ('restart', lambda v: v[1] == pm.D and v[2] in [pm.H, pm.A])]:
        transitions[name] = {}
        for tol in [6, 12, 25]:
            transitions[name][str(tol)] = pm.transition_scores([v for v in gtrans if predicate(v)],
                        [v for v in ptrans if predicate(v)], tol, typed='pair')
    gd, pd = np.zeros(len(gs), int), np.zeros(len(ps), int)
    for _, i, j in overlaps:
        gd[i] += 1; pd[j] += 1
    split = [{'reference': gs[i], 'same_team_overlapping_predictions': [ps[j] for _,ii,j in overlaps if ii==i]}
             for i in np.flatnonzero(gd > 1)]
    merge = [{'prediction': ps[j], 'same_team_overlapping_references': [gs[i] for _,i,jj in overlaps if jj==j]}
             for j in np.flatnonzero(pd > 1)]
    return {'short_events': short, 'short_matched': sum(x['matched_iou50'] for x in short),
        'stoppages': stops, 'missed_D_by_cause': buckets, 'transitions_strict_pair': transitions,
        'split_excess_overlap': int(np.maximum(gd - 1, 0).sum()), 'merge_excess_overlap': int(np.maximum(pd - 1, 0).sum()),
        'split_overlap_details': split, 'merge_overlap_details': merge,
        'overlap_warning': 'Positive same-team overlap degrees are segmentation diagnostics, not independently verified football errors.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    pm = official_metrics()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT)); mask = gt != pm.X
    protocol = json.loads((a.run / 'protocol.json').read_text())
    results = {'reference_kind': 'public development', 'reference_sha256': sha256(PUBLIC_GT),
        'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(), 'scored_frames': int(mask.sum()),
        'definitions': {'short_spell': 'Maximal reference H/A run <=125 frames (<=5.00 s at 25 fps), after clipping to valid public runs.',
            'short_match': 'Official greedy one-to-one same-team temporal IoU >=0.5, sorted by descending IoU.',
            'headline_spellF1': 'Official mean of F1 at IoU 0.3,0.4,0.5,0.6,0.7; distinct from short recovery.',
            'strict_boundary': 'Official one-to-one maximum-cardinality transition matching, typed=pair, +/-25 frames = +/-1.00 s. Also report +/-6 (0.24s), +/-12 (0.48s).',
            'intervals': 'Zero-based source-frame half-open [start,end); error = prediction minus reference; negative is early.',
            'eligible_events': '6 short possessions / 381 frames, out of 17 possessions; 4 D onsets and 4 restarts / 1469 D frames; all inside one 7500-frame valid run.'}, 'arms': {}}
    base = pm.label_codes(pm.read_label_csv(a.run / 'retained/predictions.csv'))
    for name in protocol['arms']:
        path = a.run / name / 'predictions.csv'
        report = evaluate_submission(path)
        pred = pm.label_codes(pm.read_label_csv(path))
        with np.load(a.run / name / 'evidence.npz') as evidence:
            aux = diagnostic(pm, gt, pred, evidence)
        m = report['metrics']
        item = {k:m[k] for k in ['spell_f1_mean','macro_f1','per_class','n_gt_spells','n_pred_spells']}
        item.update(aux)
        item['csv_sha256'] = sha256(path)
        change = pred != base
        item['changed_public_frames'] = int((change & mask).sum())
        item['public_errors_corrected'] = int((change & mask & (pred == gt) & (base != gt)).sum())
        item['public_correct_damaged'] = int((change & mask & (pred != gt) & (base == gt)).sum())
        item['changed_unlabelled_frames'] = int((change & ~mask).sum())
        results['arms'][name] = item
        json_write(a.out / (name + '_official.json'), report)
        print(json.dumps({'arm': name, **{k:item[k] for k in ['spell_f1_mean','macro_f1','short_matched','split_excess_overlap','merge_excess_overlap']},
                          'D':item['per_class']['D'], 'onsets':aux['transitions_strict_pair']['onset']['25']['n_matched'],
                          'restarts':aux['transitions_strict_pair']['restart']['25']['n_matched']}), flush=True)
    json_write(a.out / 'comparison.json', results)


if __name__ == '__main__':
    main()
