"""Independent, read-only audit of the 46-run evidence; writes only beside this script.

No training/evaluation helpers are imported. Statistics are rederived from saved
per-run sufficient statistics using Python statistics and scipy's t distribution.
No model inference or training is run. --hash-large also rehashes raw input PKLs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import scipy
from scipy.stats import t
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / '重新训练2026_10_5'
DATA = ROOT / '新训练结果' / 'data'
TASKS = ['temperature', 'humidity', 'component_of_wind', 'cloud_cover']
EXPECTED = {(task, kind, seed) for task in TASKS for kind in ('control', 'agf') for seed in range(2021, 2026)} | {
    (task, 'mean_fusion', seed) for task in ('temperature', 'humidity') for seed in range(2023, 2026)}
findings, checks, receipts, hash_cache = [], Counter(), {}, {}


def read(path):
    path = Path(path)
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    receipt(path)
    return result


def digest(path):
    path = Path(path).resolve()
    signature = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    if signature not in hash_cache:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        hash_cache[signature] = h.hexdigest()
    return hash_cache[signature]


def receipt(path):
    path = Path(path).resolve()
    receipts[str(path)] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def check(condition, label, detail=None, severity='error'):
    checks[label.split(':')[0]] += 1
    if not condition:
        findings.append({'severity': severity, 'check': label, 'detail': detail})


def near(actual, expected, label, abs_tol=2e-10, rel_tol=2e-9):
    a, b = float(actual), float(expected)
    check(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol),
          label, {'actual': a, 'recomputed': b, 'absolute_difference': abs(a-b)})


def table(filename, keys):
    path = DATA / filename
    receipt(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    index = {tuple(row[k] for k in keys): row for row in rows}
    check(len(index) == len(rows), 'csv_unique:' + filename)
    return rows, index


def adjust_bh(values):
    """Rank and apply the suffix minimum, independent of original helper."""
    ordered = sorted(enumerate(values), key=lambda row: row[1])
    result, running = [0.0] * len(values), 1.0
    for rank in range(len(values), 0, -1):
        original, p = ordered[rank-1]
        running = min(running, p * len(values) / rank)
        result[original] = running
    return result


def comparison(control, agf):
    cm, am = statistics.mean(control), statistics.mean(agf)
    cv, av = statistics.variance(control), statistics.variance(agf)
    vc, va = cv / len(control), av / len(agf)
    se = math.sqrt(vc + va)
    df = (vc + va)**2 / (vc**2/(len(control)-1) + va**2/(len(agf)-1))
    delta = am - cm
    half_width = float(t.isf(.025, df)) * se
    pooled = list(control) + list(agf)
    n, extreme, total = len(control), 0, 0
    for members in combinations(range(len(pooled)), n):
        left = set(members)
        difference = statistics.mean([pooled[i] for i in left]) - statistics.mean([pooled[i] for i in range(len(pooled)) if i not in left])
        extreme += abs(difference) >= abs(delta) - 1e-12
        total += 1
    return {'control_mean': cm, 'control_sd': math.sqrt(cv), 'agf_mean': am,
            'agf_sd': math.sqrt(av), 'difference': delta, 'ci95_low': delta-half_width,
            'ci95_high': delta+half_width, 'welch_df': df,
            'welch_p': float(2*t.sf(abs(delta/se), df)), 'exact_permutation_p': extreme/total,
            'relative_change_percent': 100*delta/cm, 'permutations': total}


def verify_metrics(metrics, shape, prefix):
    n, steps, nodes, channels = shape
    denominator = n * nodes * channels
    check(metrics['n_examples'] == n and metrics['elements_per_lead'] == denominator,
          'metric_grain:' + prefix)
    check(metrics['zero_inclusive'] is True, 'metric_protocol:' + prefix)
    absolute = metrics['absolute_error_sum_by_lead']
    squared = metrics['squared_error_sum_by_lead']
    wab = metrics['latitude_weighted_absolute_error_sum_by_lead']
    wsq = metrics['latitude_weighted_squared_error_sum_by_lead']
    check(all(len(x) == steps for x in (absolute, squared, wab, wsq)), 'metric_leads:' + prefix)
    for h in range(steps):
        near(metrics['exact_mae'][h], absolute[h]/denominator, 'exact_mae:' + prefix)
        near(metrics['exact_rmse'][h], math.sqrt(squared[h]/denominator), 'exact_rmse:' + prefix)
        near(metrics['cumulative_mae'][h], math.fsum(absolute[:h+1])/(denominator*(h+1)), 'cumulative_mae:' + prefix)
        near(metrics['cumulative_rmse'][h], math.sqrt(math.fsum(squared[:h+1])/(denominator*(h+1))), 'cumulative_rmse:' + prefix)
    for metric, expected in [('mae', math.fsum(absolute)/(denominator*steps)),
                             ('rmse', math.sqrt(math.fsum(squared)/(denominator*steps))),
                             ('weighted_mae', math.fsum(wab)/(denominator*steps)),
                             ('weighted_rmse', math.sqrt(math.fsum(wsq)/(denominator*steps)))]:
        near(metrics[metric], expected, 'aggregate_metric:' + prefix + '/' + metric)


def main(hash_large=False):
    queue, plan = read(ROOT/'queue_status.json'), read(ROOT/'training_plan.json')
    verification = read(ROOT/'新训练结果/verification_report.json')
    status = read(ROOT/'新训练结果/analysis_status.json')
    rows = [read(p) for p in sorted(DATA.glob('inference_*.json'))]
    index = {(r['dataset'], r['kind'], r['seed']): r for r in rows}
    check(len(rows) == len(index) == 46 and set(index) == EXPECTED, 'run_grid:inference')
    check(len(plan['runs']) == 46 and {(r['task'], r['variant'], r['seed']) for r in plan['runs']} == EXPECTED, 'run_grid:plan')
    check(queue['status'] == 'completed' and len(queue['runs']) == 46 and all(r['status'] == 'completed' for r in queue['runs']), 'run_grid:queue')
    check(verification['status'] == 'passed' and verification['verified_runs'] == 46 and not verification['errors'], 'verification_report')
    check(status['status'] == 'completed' and status['completed_inferences'] == 46, 'analysis_status')
    metas = {task: read(DATA/f'dataset_{task}.json') for task in TASKS}
    source_manifest = read(ROOT/'training_source_manifest.json')
    for name, expected in source_manifest['files'].items():
        check(digest(ROOT/'source_snapshot'/name) == expected, 'frozen_training_source:' + name)
    analysis_manifest = read(ROOT/'analysis_snapshot/analysis_source_manifest.json')
    for name, expected in analysis_manifest['files'].items():
        check(digest(ROOT/'analysis_snapshot'/name) == expected, 'frozen_analysis_source:' + name)
    max_eval_delta, run_details, checkpoint_files = 0.0, [], 0
    for (task, kind, seed), row in index.items():
        label = f'{task}/{kind}/{seed}'
        run = Path(row['experiment_dir'])
        check(run.resolve().is_relative_to((ROOT/'runs').resolve()), 'run_containment:' + label)
        summary, history, config, manifest = [read(run/name) for name in ('summary.json', 'training_history.json', 'model_param.json', 'run_manifest.json')]
        check([h['epoch'] for h in history] == list(range(1, 101)), 'epoch_history:' + label)
        check(all(math.isfinite(h[m]) for h in history for m in ('train_mae', 'val_mae')), 'finite_history:' + label)
        best = min(history, key=lambda h: h['val_mae'])
        check(summary['best_epoch'] == row['corrected_best_epoch'] == best['epoch'], 'validation_selection:' + label)
        near(summary['best_val_mae'], best['val_mae'], 'validation_minimum:' + label)
        check(summary['training_complete'] and summary['epochs_completed'] == summary['planned_epochs'] == 100 and
              summary['precision'] == 'fp32' and summary['mode'] == 'new_training' and summary['sample_count'] == 657,
              'run_completion:' + label)
        check(all(summary[x] == 'zero_inclusive' for x in ('metric_protocol', 'training_protocol', 'validation_protocol')), 'run_protocol:' + label)
        check((summary['task'], summary['variant'], summary['seed']) == (task, kind, seed), 'run_identity:' + label)
        train, model, data = config['train'], config['model'], config['data']
        check(train['epochs'] == 100 and train['patience'] == 0 and not train['use_amp'] and data['batch_size'] == 32,
              'configuration:' + label)
        check(model['use_asttn_encoder'] == (kind != 'control') and model['asttn_fusion_mode'] == ('mean' if kind == 'mean_fusion' else 'gated'), 'variant_configuration:' + label)
        cfgsha = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        check(cfgsha == summary['config_sha256'] == manifest['config_sha256'] and manifest['config'] == config, 'config_digest:' + label)
        check(manifest['numerics']['tf32'] is False, 'precision_manifest:' + label)
        for name, sha in manifest['source_hashes'].items():
            check(sha == source_manifest['files'][name.replace('\\', '/')], 'run_source_provenance:' + label + '/' + name)
        for field in ('checkpoint', 'config', 'summary', 'training_history', 'training_manifest'):
            source = row[field]
            check(digest(source['path']) == source['sha256'] and Path(source['path']).stat().st_size == source['bytes'], 'inference_source_hash:' + label + '/' + field)
        saved = sorted((run/'saved_model').glob('epo*.tar'), key=lambda p: int(p.stem[3:]))
        checkpoint_files += len(saved)
        check([int(p.stem[3:]) for p in saved] == list(range(1, 101)), 'checkpoint_inventory:' + label)
        for epoch in set((100, best['epoch'])):
            path = run/f'saved_model/epo{epoch}.tar'
            state = torch.load(path, map_location='cpu', weights_only=False)
            check(state['epoch'] == epoch and state['config_sha256'] == cfgsha and state['checkpoint_version'] == 2,
                  'checkpoint_identity:' + label + '/' + str(epoch))
            check(all(k in state for k in ('rng_state', 'optimizer_state_dict', 'scheduler_state_dict', 'history', 'initialization_audit')), 'checkpoint_resume_state:' + label)
            check([h['epoch'] for h in state['history']] == list(range(1, epoch+1)), 'checkpoint_history:' + label)
            # state_dict repeats shared convolution parameters under aliases and
            # includes BatchNorm buffers. Count storage-unique parameter tensors.
            unique_parameters = {(v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride())): v
                                 for name, v in state['model_state_dict'].items()
                                 if not name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
            near(sum(v.numel() for v in unique_parameters.values()), row['n_parameters'], 'parameter_count:' + label, abs_tol=0, rel_tol=0)
            near(sum(v['exp_avg'].numel() for v in state['optimizer_state_dict']['state'].values()), row['n_parameters'], 'optimizer_parameter_count:' + label, abs_tol=0, rel_tol=0)
            near(len(unique_parameters), state['initialization_audit']['parameter_tensor_count'], 'parameter_tensor_count:' + label, abs_tol=0, rel_tol=0)
            check(all(v.dtype == torch.float32 for v in state['model_state_dict'].values() if v.is_floating_point()), 'checkpoint_fp32:' + label)
            receipt(path)
            del state
        verify_metrics(row['metrics'], metas[task]['test_shape'], label)
        for metric in ('mae', 'rmse'):
            delta = row['metrics'][metric]-summary[metric]
            max_eval_delta = max(max_eval_delta, abs(delta))
            near(delta, row['new_training_metric_differences'][metric], 'reevaluation_delta:' + label + '/' + metric)
            near(row['metrics'][metric], summary[metric], 'summary_agreement:' + label + '/' + metric, abs_tol=2e-5)
        run_details.append({'task': task, 'kind': kind, 'seed': seed, 'best_epoch': best['epoch'], 'checkpoint_sha256': row['checkpoint']['sha256'], 'epochs': len(history)})
    results = {}
    for filename, metrics in [('table1_primary_statistics.csv', ('mae', 'rmse')),
                              ('latitude_weighted_sensitivity.csv', ('weighted_mae', 'weighted_rmse'))]:
        csvrows, csvindex = table(filename, ('dataset', 'metric'))
        recomputed = []
        check(len(csvrows) == 8, 'statistics_row_count:' + filename)
        for task in TASKS:
            for metric in metrics:
                control = [index[task, 'control', s]['metrics'][metric] for s in range(2021, 2026)]
                agf = [index[task, 'agf', s]['metrics'][metric] for s in range(2021, 2026)]
                values = {'dataset': task, 'metric': metric, **comparison(control, agf)}
                recomputed.append(values)
        for method in ('welch', 'exact_permutation'):
            for row, q in zip(recomputed, adjust_bh([r[method+'_p'] for r in recomputed])):
                row[method+'_bh_q'] = q
        for row in recomputed:
            saved = csvindex[row['dataset'], row['metric']]
            for key, expected in row.items():
                if key in saved and key not in ('dataset', 'metric'):
                    near(saved[key], expected, 'statistics:' + filename + '/' + row['dataset'] + '/' + row['metric'] + '/' + key)
        results[filename] = recomputed
    for filename, keys in [('primary_per_seed.csv', ('dataset', 'kind', 'seed')),
                           ('ablation_per_seed.csv', ('dataset', 'variant', 'seed'))]:
        csvrows, _ = table(filename, keys)
        check(len(csvrows) == (40 if filename.startswith('primary') else 18), 'per_seed_count:' + filename)
        for saved in csvrows:
            kind = saved.get('kind') or {'w/o adaptive graph':'control', 'Mean fusion':'mean_fusion', 'Full CLCRN-AGF':'agf'}[saved['variant']]
            row = index[saved['dataset'], kind, int(saved['seed'])]
            for metric in ('mae', 'rmse'):
                near(saved[metric], row['metrics'][metric], 'per_seed_metric:' + filename)
            check(saved['checkpoint'] == row['checkpoint']['path'] and saved['checkpoint_sha256'] == row['checkpoint']['sha256'], 'per_seed_checkpoint:' + filename)
            if 'epoch' in saved:
                near(saved['epoch'], row['corrected_best_epoch'], 'per_seed_epoch')
                near(saved['latitude_weighted_mae'], row['metrics']['weighted_mae'], 'per_seed_weighted_mae')
                near(saved['latitude_weighted_rmse'], row['metrics']['weighted_rmse'], 'per_seed_weighted_rmse')
    csvrows, _ = table('table3_ablation_statistics.csv', ('dataset', 'variant'))
    check(len(csvrows) == 6, 'ablation_row_count')
    for saved in csvrows:
        kind = {'w/o adaptive graph':'control', 'Mean fusion':'mean_fusion', 'Full CLCRN-AGF':'agf'}[saved['variant']]
        check(saved['seeds'] == '2023;2024;2025' and saved['n_seeds'] == '3', 'ablation_seeds')
        for metric in ('mae', 'rmse'):
            values = [index[saved['dataset'], kind, s]['metrics'][metric] for s in (2023, 2024, 2025)]
            near(saved[metric+'_mean'], statistics.mean(values), 'ablation_mean')
            near(saved[metric+'_sd'], statistics.stdev(values), 'ablation_sd')
    results['ablations'] = csvrows
    csvrows, _ = table('horizon_per_seed.csv', ('dataset', 'kind', 'seed', 'lead_hours'))
    check(len(csvrows) == 480, 'horizon_row_count')
    for saved in csvrows:
        row = index[saved['dataset'], saved['kind'], int(saved['seed'])]
        for metric in ('exact_mae', 'exact_rmse', 'cumulative_mae', 'cumulative_rmse'):
            near(saved[metric], row['metrics'][metric][int(saved['lead_hours'])-1], 'horizon_values')
    csvrows, _ = table('table2_current_context.csv', ('dataset', 'method'))
    check(len(csvrows) == 12, 'context_row_count')
    for saved in csvrows:
        task = saved['dataset']
        if saved['method'] == 'Persistence':
            met = read(DATA/f'persistence_{task}.json')
            verify_metrics(met, metas[task]['test_shape'], 'persistence/' + task)
        else:
            kind = {'Control':'control', 'CLCRN-AGF':'agf'}[saved['method']]
            met = {m: statistics.mean(index[task, kind, s]['metrics'][m] for s in range(2021,2026)) for m in ('mae', 'rmse')}
        for metric in ('mae', 'rmse'):
            near(saved[metric], met[metric], 'context_metric')
    robust_rows, robust_index = table('robustness_per_seed_mask.csv', ('dataset', 'kind', 'seed', 'ratio', 'maskseed'))
    summary_rows, summary_index = table('robustness_summary.csv', ('dataset', 'kind', 'missing_ratio'))
    check(len(robust_rows) == 520 and len(summary_rows) == 40, 'robustness_row_counts')
    robustness_results = []
    for task in TASKS:
        indices = read(DATA/f'robustness_indices_{task}.json')
        check(indices['zero_based_test_indices'] == np.rint(np.linspace(0,656,64)).astype(int).tolist() and indices['mask_seeds'] == [11,22,33], 'robustness_sampling:' + task)
        for kind in ('control', 'agf'):
            for ratio in (0.0, .1, .2, .3, .4):
                seed_values = []
                for seed in range(2021, 2026):
                    records = index[task, kind, seed]['robustness']
                    check(len(records) == 13 and {(r['ratio'], r['maskseed']) for r in records} == {(0.0,None)} | {(a,b) for a in (.1,.2,.3,.4) for b in (11,22,33)}, 'robustness_grid')
                    base = next(r['mae'] for r in records if r['ratio'] == 0)
                    selected = [r for r in records if r['ratio'] == ratio]
                    seed_values.append(statistics.mean((r['mae']-base)/base*100 for r in selected))
                    for r in selected:
                        saved = robust_index[task, kind, str(seed), str(ratio), '' if r['maskseed'] is None else str(r['maskseed'])]
                        for metric in ('mae','rmse','n_examples'):
                            near(saved[metric], r[metric], 'robustness_source_value')
                        near(saved['clean_mae_same_subset'], base, 'robustness_baseline')
                        near(saved['relative_increase_percent'], (r['mae']-base)/base*100, 'robustness_relative_value')
                        check(r['n_examples'] == 64, 'robustness_sample_size')
                saved = next(r for r in summary_rows if r['dataset'] == task and r['kind'] == kind and float(r['missing_ratio']) == ratio)
                near(saved['mean_relative_mae_increase_percent'], statistics.mean(seed_values), 'robustness_summary_mean')
                near(saved['sd_across_trained_seeds_percent'], statistics.stdev(seed_values), 'robustness_summary_sd')
                if ratio == .4:
                    robustness_results.append({'dataset': task, 'kind': kind, '40_percent_mask_relative_mae_increase': statistics.mean(seed_values)})
    gate_rows, gate_index = table('table4_gate_statistics.csv', ('dataset',))
    seed_rows, seed_index = table('gate_per_seed.csv', ('dataset','seed'))
    hist_rows, hist_index = table('gate_histogram.csv', ('dataset','seed','bin_left'))
    check(len(gate_rows) == 4 and len(seed_rows) == 20 and len(hist_rows) == 2000, 'gate_row_counts')
    for task in TASKS:
        gates = [index[task,'agf',seed]['gates'] for seed in range(2021,2026)]
        count = sum(g['n_scalar_gates'] for g in gates)
        mean = math.fsum(g['mean']*g['n_scalar_gates'] for g in gates)/count
        variance = math.fsum(g['n_scalar_gates']*(g['population_std']**2+(g['mean']-mean)**2) for g in gates)/count
        saved = gate_index[task,]
        for key,value in {'mean':mean,'pooled_population_std':math.sqrt(variance),'min':min(g['min'] for g in gates),'max':max(g['max'] for g in gates),'n_scalar_gates':count}.items():
            near(saved[key],value,'gate_pooled:'+task+'/'+key)
        for seed,g in zip(range(2021,2026),gates):
            check(g['n_scalar_gates'] == g['n_test_examples']*g['nodes']*g['input_steps']*g['gate_channels'] == sum(g['histogram_counts']), 'gate_count:'+task)
            check(0 <= g['min'] <= g['mean'] <= g['max'] <= 1, 'gate_range:'+task)
            for key,value in seed_index[task,str(seed)].items():
                if key not in ('dataset','seed'):near(value,g[key],'gate_per_seed_value')
            for i,n in enumerate(g['histogram_counts']):
                saved_bin = hist_index[task,str(seed),str(g['histogram_edges'][i])]
                near(saved_bin['count'],n,'gate_histogram_count',abs_tol=0,rel_tol=0)
                near(saved_bin['bin_right'],g['histogram_edges'][i+1],'gate_histogram_edge')
    zero_rows, _ = table('raw_target_zero_fractions.csv', ('dataset','split'))
    check(len(zero_rows) == 12, 'zero_fraction_rows')
    for saved in zero_rows:
        expected = metas[saved['dataset']]['raw_target_zeros'][saved['split']]
        for field in ('zero_count','total','fraction'):near(saved[field],expected[field],'zero_fraction_source')
        near(saved['fraction'],float(saved['zero_count'])/float(saved['total']),'zero_fraction_arithmetic')
    provenance = read(ROOT/'新训练结果/provenance_manifest.json')
    manifest_results, large_skipped, provenance_hashes_checked = [], [], 0
    for section in ('sources','generated'):
        for entry in provenance[section]:
            path = Path(entry['path'])
            if path.stat().st_size > 128*1024**2 and not hash_large:
                large_skipped.append({'path':str(path),'recorded_sha256':entry['sha256'],'bytes':path.stat().st_size})
                check(path.stat().st_size == entry['bytes'], 'large_input_size:'+str(path))
                continue
            actual = digest(path)
            receipt(path)
            provenance_hashes_checked += 1
            if actual != entry['sha256']:
                mutable_queue = path == ROOT/'queue_status.json'
                check(False,'provenance_hash:'+str(path),{'recorded':entry['sha256'],'current':actual,
                      'explanation':'Queue status is rewritten after the analysis snapshot when the completion hook finishes.' if mutable_queue else None},
                      severity='warning' if mutable_queue else 'error')
                manifest_results.append({'path':str(path),'recorded':entry['sha256'],'current':actual,'mutable_queue':mutable_queue})
    assets = read(DATA/'manuscript_assets.json')
    for key, filename in [('statistics','table1_primary_statistics.csv'),('weighted_sensitivity','latitude_weighted_sensitivity.csv')]:
        for actual, expected in zip(assets[key], results[filename]):
            for field, value in actual.items():
                if field in expected and isinstance(value,(float,int)):
                    near(value,expected[field],'manuscript_assets:'+key+'/'+field)
    raw_fingerprint = read(ROOT/'data_sha256.json')
    prior = {r['path']:r for r in raw_fingerprint['files']}
    for meta in metas.values():
        for source in meta['sources']:
            p = Path(source['path']); baseline = prior[str(p)]
            check(source['sha256'] == baseline['sha256'] and source['bytes'] == p.stat().st_size == baseline['bytes'] and baseline['mtime_ns'] == p.stat().st_mtime_ns,'raw_data_start_finish_fingerprint:'+str(p))
    limitations = [
        'Audit rederived all reported statistics and error-sum metrics; it did not rerun prediction or training.',
        'Raw per-window predictions/gates are not stored; their extraction is checked by recorded sufficient statistics and provenance, not an independent new inference.',
        'Raw PKL contents were not rehashed in this invocation; start/end recorded SHA256, current sizes and mtimes were compared.' if large_skipped else 'Raw input SHA256 was independently rechecked.',
        'Welch/BH primary results cover eight comparisons; exact randomization sensitivity assumes exchangeability and uses all 252 five-versus-five partitions.',
        'Five training seeds quantify initialization/training variation on one fixed test split, not uncertainty over climate years, sites or operational deployment.',
        'The ablation uses three seeds and reports descriptive mean/SD; it cannot alone establish a causal or significant mechanism.',
        'Missing-input diagnostics use 64 deterministic test windows and three masks; they are not whole-test accuracy or formal significance tests.',
        'Pooled gate spread is across gates and seeds; gate magnitude is not branch information contribution or a physical mechanism.',
        'Manuscript wording, page layout, bibliography and submission packaging are reviewed by other task owners.'
    ]
    report = {'audit_generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'readiness':'Needs revision' if any(f['severity']=='error' for f in findings) else 'Share with caveats',
              'scope':'Numerical consistency and provenance audit for manuscript production; no training/inference rerun.',
              'run_root':str(ROOT),'run_count':len(rows),'epoch_checkpoints_inventoried':checkpoint_files,
              'checks_by_family':dict(checks),'finding_count':len(findings),'findings':findings,
              'max_abs_summary_reevaluation_difference':max_eval_delta,
              'primary':results['table1_primary_statistics.csv'],'weighted':results['latitude_weighted_sensitivity.csv'],
              'ablations':results['ablations'],'gate_summary':gate_rows,'robustness_40_percent':robustness_results,
              'runs':run_details,'provenance_mismatches':manifest_results,'large_source_hashes_not_recomputed':large_skipped,
              'provenance_entries_rehashed':provenance_hashes_checked,
              'limitations':limitations,'software':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
              'audited_input_receipts':list(receipts.values()),'audit_script_sha256':digest(__file__)}
    (HERE/'independent_statistics_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=[report['readiness'],f'46-run independent audit: {len(rows)} runs; {checkpoint_files} epoch files inventoried.',
           f'Checks: {sum(checks.values())}; findings: {len(findings)}; maximum reevaluation difference: {max_eval_delta:.12g}.','']
    lines += [f"{r['dataset']} {r['metric']}: control {r['control_mean']:.8g} (SD {r['control_sd']:.5g}), AGF {r['agf_mean']:.8g} (SD {r['agf_sd']:.5g}); change {r['relative_change_percent']:+.4f}%; Welch/BH q={r['welch_bh_q']:.8g}; exact/BH q={r['exact_permutation_bh_q']:.8g}." for r in report['primary']]
    lines += ['', 'Findings:'] + [json.dumps(f,ensure_ascii=False) for f in findings]
    lines += ['', 'Scope and interpretation limits:'] + limitations
    (HERE/'independent_statistics_audit.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('readiness','run_count','epoch_checkpoints_inventoried','finding_count','findings','max_abs_summary_reevaluation_difference')},ensure_ascii=False,indent=2))
    return 1 if any(f['severity']=='error' for f in findings) else 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-large',action='store_true')
    args=parser.parse_args()
    raise SystemExit(main(args.hash_large))
