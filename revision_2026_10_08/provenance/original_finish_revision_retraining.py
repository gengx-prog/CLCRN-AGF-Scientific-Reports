"""Verify the 46 new runs, then create separate, traceable manuscript evidence.

No archived inference enters this workflow. Failed/incomplete runs prevent the
analysis stage. This script never edits the manuscript or claims it was updated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
REVISION = CODE.parent
TASKS = ['cloud_cover', 'temperature', 'humidity', 'component_of_wind']
EXPECTED = {(t, v, s) for t in TASKS for v in ['control', 'agf'] for s in range(2021, 2026)} | {
    (t, 'mean_fusion', s) for t in ['temperature', 'humidity'] for s in [2023, 2024, 2025]}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config_digest(config):
    value = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def contained(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent):
        raise ValueError(f'Run artifact outside authorized new run root: {path}')
    return path


def require(test, message):
    if not test:
        raise ValueError(message)


def verify_runs(run_root, output, *, allow_fixture=False):
    """Fail closed on any mismatch, and retain a readable verification report."""
    import torch
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    report = {'generated_at_utc': now(), 'status': 'failed', 'expected_runs': 46,
              'run_root': str(run_root), 'checks': [], 'errors': [],
              'manuscript_updated': False, 'fixture': bool(allow_fixture)}
    verified = []
    try:
        plan = read(run_root / 'training_plan.json')
        queue = read(run_root / 'queue_status.json')
        require(not plan.get('fixture') or allow_fixture, 'Synthetic fixtures cannot become manuscript evidence.')
        require(plan.get('protocol') == 'zero_inclusive', 'Plan must use zero-inclusive loss/validation.')
        require(plan.get('epochs') == 100, 'Plan must specify 100 epochs.')
        require(str(plan.get('precision', '')).lower() == 'fp32', 'Plan must specify FP32.')
        snapshot = run_root / 'source_snapshot'
        if not allow_fixture:
            require((snapshot / 'supervisor.py').is_file() and (snapshot / 'run_revision.py').is_file(),
                    'Frozen training source_snapshot is required for new-run evaluation.')
        require(queue.get('status') == 'completed', 'Training queue has not completed.')
        jobs = plan['runs']
        require(len(jobs) == 46, f'Expected 46 planned jobs; found {len(jobs)}.')
        keys = [(r['task'], r['variant'], int(r['seed'])) for r in jobs]
        require(set(keys) == EXPECTED and len(set(keys)) == 46, 'Run grid differs from the approved 40 + 6 design.')
        status = {r['run_id']: r for r in queue['runs']}
        require(len(status) == 46, 'Queue must contain 46 unique run IDs.')
        for job in jobs:
            key = (job['task'], job['variant'], int(job['seed']))
            try:
                require(status.get(job['run_id'], {}).get('status') == 'completed', 'Queue job is not completed.')
                exp = contained(job['output_dir'], run_root / 'runs')
                summary = read(exp / 'summary.json')
                config = read(exp / 'model_param.json')
                manifest = read(exp / 'run_manifest.json')
                history = read(exp / 'training_history.json')
                train = config['train']
                require(manifest.get('mode') == 'new_training', 'Manifest is not a new training run.')
                require(summary.get('training_performed') is True, 'Training was not performed.')
                require(summary.get('training_complete') is True, 'Training is incomplete.')
                require(summary.get('status') == 'completed', 'Run did not complete normally.')
                require(summary.get('epochs_completed') == 100 and summary.get('planned_epochs') == 100,
                        'Run must complete all 100 planned epochs.')
                require([r.get('epoch') for r in history] == list(range(1, 101)), 'History must contain epochs 1..100 exactly once.')
                require(train.get('epochs') == 100 and train.get('loss_protocol') == 'zero_inclusive' and
                        train.get('validation_protocol') == 'zero_inclusive', 'Training/configuration protocol mismatch.')
                if not allow_fixture:
                    require(train.get('patience') == 0 and train.get('seed') == key[2],
                            'Early stopping must be disabled and the training seed must match.')
                    model = config['model']
                    require(bool(model.get('use_asttn_encoder')) == (key[1] != 'control'), 'Variant adaptive graph switch mismatch.')
                    require(model.get('asttn_fusion_mode') == ('mean' if key[1] == 'mean_fusion' else 'gated'),
                            'Variant fusion mode mismatch.')
                require(not train.get('use_amp', False) and str(summary.get('precision', '')).lower() == 'fp32',
                        'Mixed precision is not part of this matched experiment.')
                require(summary.get('training_protocol') == 'zero_inclusive' and
                        summary.get('validation_protocol') == 'zero_inclusive' and
                        summary.get('metric_protocol') == 'zero_inclusive', 'Summary protocols are inconsistent.')
                require((summary.get('task'), summary.get('variant'), int(summary.get('seed', -1))) == key,
                        'Summary job identity does not match the plan.')
                require((manifest.get('task'), int(manifest.get('seed', -1))) == (key[0], key[2]),
                        'Manifest job identity does not match the plan.')
                require(summary.get('config_sha256') == config_digest(config), 'Configuration digest mismatch.')
                require(manifest.get('config') == config, 'Run manifest configuration mismatch.')
                if not allow_fixture:
                    require(bool(manifest.get('source_hashes')), 'Run has no frozen source hash record.')
                    for relative, expected_hash in manifest['source_hashes'].items():
                        source = contained(snapshot / relative, snapshot)
                        require(sha(source) == expected_hash, f'Frozen training source changed: {relative}')
                    require(manifest.get('precision') == 'fp32', 'Manifest precision mismatch.')
                    require(manifest.get('numerics', {}).get('tf32') is False, 'Manifest must disable TF32.')
                    require(manifest.get('variant') == key[1], 'Manifest variant mismatch.')
                vals = [float(r['val_mae']) for r in history]
                require(all(math.isfinite(v) for v in vals), 'History contains non-finite validation MAE.')
                best_epoch = int(summary['best_epoch'])
                require(best_epoch == 1 + min(range(100), key=vals.__getitem__), 'Best epoch is not the first validation minimum.')
                require(math.isclose(float(summary['best_val_mae']), min(vals), rel_tol=1e-7, abs_tol=1e-8),
                        'Best validation metric disagrees with history.')
                checkpoint = contained(exp / 'saved_model' / f'epo{best_epoch}.tar', exp)
                checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
                require(checkpoint_data.get('epoch') == best_epoch, 'Checkpoint epoch mismatch.')
                require(checkpoint_data.get('config') == config, 'Checkpoint config mismatch.')
                require(checkpoint_data.get('loss_protocol') == 'zero_inclusive', 'Checkpoint training protocol mismatch.')
                state = checkpoint_data['model_state_dict']
                require(bool(state), 'Checkpoint has no model weights.')
                require(all(v.dtype == torch.float32 for v in state.values() if torch.is_tensor(v) and v.is_floating_point()),
                        'Checkpoint contains non-FP32 floating tensors.')
                if not allow_fixture:
                    require(checkpoint_data.get('config_sha256') == config_digest(config), 'Checkpoint digest mismatch.')
                    require(checkpoint_data.get('processing_version') == summary.get('processing_version') ==
                            manifest.get('processing_version') == 'curated_float32_then_train_only_standardization_v1',
                            'Preprocessing version mismatch.')
                    final = torch.load(exp / 'saved_model/epo100.tar', map_location='cpu', weights_only=False)
                    require(final.get('epoch') == 100 and final.get('checkpoint_version') == 2 and
                            final.get('config_sha256') == config_digest(config), 'Missing/incompatible final epoch checkpoint.')
                    require(all(k in final for k in ['rng_state', 'optimizer_state_dict', 'scheduler_state_dict',
                                                   'scaler_state_dict', 'initialization_audit']),
                            'Final epoch checkpoint lacks resumable training state.')
                    del final
                del checkpoint_data, state
                row = {'job': job, 'output_dir': str(exp), 'summary': summary, 'config': config,
                       'checkpoint': record(checkpoint), 'config_source': record(exp / 'model_param.json'),
                       'summary_source': record(exp / 'summary.json'), 'history_source': record(exp / 'training_history.json'),
                       'manifest_source': record(exp / 'run_manifest.json')}
                verified.append(row)
                report['checks'].append({'run_id': job['run_id'], 'status': 'passed', 'best_epoch': best_epoch,
                                         'checkpoint_sha256': row['checkpoint']['sha256']})
            except Exception as e:
                report['errors'].append({'run_id': job['run_id'], 'error': str(e)})
        require(len(verified) == 46 and not report['errors'], 'One or more run checks failed.')
        report['status'] = 'passed'
        report['plan'] = record(run_root / 'training_plan.json')
        report['queue_status'] = record(run_root / 'queue_status.json')
    except Exception as e:
        report['errors'].append({'scope': 'grid', 'error': str(e)})
    report['verified_runs'] = len(verified)
    write(output / 'verification_report.json', report)
    if report['status'] != 'passed':
        raise RuntimeError(f"46-run verification failed; see {output / 'verification_report.json'}")
    return verified, report


class GateStats:
    def __init__(self):
        import numpy as np
        self.count, self.total, self.square, self.low, self.high = 0, 0., 0., 1., 0.
        self.hist = np.zeros(100, dtype=np.int64)

    def hook(self, module, args, output):
        import torch
        g = output.sigmoid().detach()
        require(bool(torch.isfinite(g).all()), 'Non-finite sigmoid gate.')
        self.count += g.numel()
        self.total += g.double().sum().item()
        self.square += g.double().square().sum().item()
        self.low, self.high = min(self.low, g.min().item()), max(self.high, g.max().item())
        self.hist += torch.histc(g, bins=100, min=0, max=1).cpu().numpy().astype('int64')

    def result(self, ntest, config):
        import numpy as np
        require(self.count > 0 and int(self.hist.sum()) == self.count, 'Incomplete gate histogram.')
        m = self.total / self.count
        return {'n_scalar_gates': self.count, 'mean': m,
                'population_std': math.sqrt(max(0, self.square / self.count - m * m)),
                'min': self.low, 'max': self.high, 'histogram_counts': self.hist.tolist(),
                'histogram_edges': np.linspace(0, 1, 101).tolist(), 'n_test_examples': ntest,
                'gate_channels': config['model']['asttn_hidden_dim'], 'input_steps': config['model']['seq_len'],
                'nodes': config['model']['node_num'],
                'grain': 'all full-test examples x input steps x nodes x gate channels; no channel averaging'}


def metric_arrays(pred, truth, weights):
    """Same inverse-transformed predictions; double-precision aggregation only."""
    import numpy as np
    import torch
    require(pred.shape == truth.shape and pred.ndim == 4, 'Expected matching [lead, example, node, channel] tensors.')
    h, ntest, nodes, channels = pred.shape
    total, square, wtotal, wsquare = (np.zeros(h, dtype=np.float64) for _ in range(4))
    weight = torch.as_tensor(weights, dtype=torch.float64).reshape(1, 1, -1, 1)
    for start in range(0, ntest, 16):
        err = (pred[:, start:start+16] - truth[:, start:start+16]).double()
        require(bool(torch.isfinite(err).all()), 'Non-finite prediction/target; cannot silently exclude entries.')
        total += err.abs().sum((1, 2, 3)).numpy()
        square += err.square().sum((1, 2, 3)).numpy()
        wtotal += (err.abs() * weight).sum((1, 2, 3)).numpy()
        wsquare += (err.square() * weight).sum((1, 2, 3)).numpy()
    n = ntest * nodes * channels
    return {'mae': float(total.sum() / (n * h)), 'rmse': float(np.sqrt(square.sum() / (n * h))),
            'weighted_mae': float(wtotal.sum() / (n * h)), 'weighted_rmse': float(np.sqrt(wsquare.sum() / (n * h))),
            'exact_mae': (total / n).tolist(), 'exact_rmse': np.sqrt(square / n).tolist(),
            'cumulative_mae': (total.cumsum() / n / np.arange(1, h + 1)).tolist(),
            'cumulative_rmse': np.sqrt(square.cumsum() / n / np.arange(1, h + 1)).tolist(),
            'absolute_error_sum_by_lead': total.tolist(), 'squared_error_sum_by_lead': square.tolist(),
            'latitude_weighted_absolute_error_sum_by_lead': wtotal.tolist(),
            'latitude_weighted_squared_error_sum_by_lead': wsquare.tolist(),
            'elements_per_lead': n, 'n_examples': ntest, 'zero_inclusive': True,
            'spatial_weighting': 'uniform for mae/rmse; normalized cos(latitude) for weighted_*'}


def sampled_predictions(supervisor, indices, ratio=0., maskseed=0, persistence=False):
    import numpy as np
    import torch
    dataset = supervisor.data['test_loader'].dataset
    preds, truths = [], []
    rng = np.random.default_rng(maskseed)
    with torch.no_grad():
        for start in range(0, len(indices), 32):
            batch = indices[start:start+32]
            xx, yy = supervisor.prepare(torch.from_numpy(dataset.x[batch].copy()), torch.from_numpy(dataset.y[batch].copy()))
            if ratio:
                mask = np.ones((xx.shape[1], xx.shape[2]), dtype=np.float32)
                for row in mask:
                    row[rng.choice(len(row), size=int(len(row) * ratio), replace=False)] = 0
                xx *= torch.as_tensor(mask, device=supervisor.device)[None, :, :, None]
            prediction = xx[-1:].repeat(yy.shape[0], 1, 1, 1) if persistence else supervisor.model(xx)
            preds.append(supervisor.inverse(prediction).cpu())
            truths.append(supervisor.inverse(yy).cpu())
    return torch.cat(preds, dim=1), torch.cat(truths, dim=1)


def dataset_metadata(supervisor, task):
    import numpy as np
    base = Path(supervisor.config['data']['dataset_dir'])
    metadata = {'task': task, 'sources': [], 'raw_target_zeros': {},
                'preprocessing': 'curated experiments.dataloader.Dataset: raw training x supplies scaler moments; '
                                 'cast split x/y to float32 BEFORE in-place standardization; Supervisor.inverse for physical units',
                'train_mean': [float(s.mean) for s in supervisor.scalers],
                'train_std': [float(s.std) for s in supervisor.scalers]}
    for split, filename in [('train', 'trn.pkl'), ('validation', 'val.pkl'), ('test', 'test.pkl')]:
        source = base / filename
        with source.open('rb') as f:
            raw = pickle.load(f)
        metadata['raw_target_zeros'][split] = {'zero_count': int((raw['y'] == 0).sum()),
                                             'total': int(raw['y'].size), 'fraction': float((raw['y'] == 0).mean())}
        metadata[split + '_raw_dtype'] = str(raw['x'].dtype)
        metadata[split + '_shape'] = list(raw['x'].shape)
        del raw
        gc.collect()
        metadata['sources'].append(record(source))
    position = Path(supervisor.config['data']['position_file'])
    with position.open('rb') as f:
        lonlat = pickle.load(f)['lonlat']
    weights = np.cos(np.deg2rad(lonlat[:, 1])).astype(np.float64)
    weights /= weights.mean()
    require(bool(np.isfinite(weights).all()) and bool((weights >= 0).all()), 'Invalid latitude weights.')
    metadata['sources'].append(record(position))
    metadata['ntest'] = len(supervisor.data['test_loader'].dataset)
    metadata['latitude_weight'] = 'cos(latitude) / mean(cos(latitude)); full finite-node support'
    return metadata, weights


def full_analysis(verified, report, run_root, output, device):
    import numpy as np
    import torch
    frozen_code = run_root / 'source_snapshot'
    sys.path.insert(0, str(frozen_code))
    from supervisor import Supervisor
    from run_revision import seed_all
    require(Path(sys.modules['supervisor'].__file__).resolve() == (frozen_code / 'supervisor.py').resolve(),
            'Supervisor must be imported from this run\'s frozen source snapshot.')
    seed_all(2021)
    torch.set_num_threads(4)
    data_dir, figures = output / 'data', output / 'figures'
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = [report['plan'], report['queue_status']]
    for task in TASKS:
        shared_data = None
        task_jobs = [v for v in verified if v['job']['task'] == task]
        for i, checked in enumerate(task_jobs):
            start = time.perf_counter()
            job, config = checked['job'], checked['config']
            stem = f"{job['variant']}_{task}_{job['seed']}"
            print('ANALYZE', stem, flush=True)
            supervisor = Supervisor(config, output / 'evaluation_logs' / stem, device=device,
                                    checkpoint_dir=Path(checked['output_dir']) / 'saved_model', data=shared_data)
            shared_data = supervisor.data
            try:
                if i == 0:
                    meta, weights = dataset_metadata(supervisor, task)
                    write(data_dir / f'dataset_{task}.json', meta)
                    sources.extend(meta['sources'])
                    ntest = len(supervisor.data['test_loader'].dataset)
                    require(ntest >= 64, 'Full analysis requires at least 64 test windows.')
                    subset = np.rint(np.linspace(0, ntest-1, 64)).astype(int)
                    write(data_dir / f'robustness_indices_{task}.json', {
                        'zero_based_test_indices': subset.tolist(),
                        'selection': '64 equally spaced deterministic indices across the chronological test set',
                        'mask_seeds': [11, 22, 33]})
                    pred, truth = sampled_predictions(supervisor, np.arange(ntest), persistence=True)
                    write(data_dir / f'persistence_{task}.json', metric_arrays(pred, truth, weights))
                    del pred, truth
                gates, gate_hook = GateStats(), None
                if job['variant'] == 'agf':
                    gate_hook = supervisor.model.asttn_encoder.gate_proj.register_forward_hook(gates.hook)
                try:
                    official, pred, truth = supervisor.evaluate('test', epoch=checked['summary']['best_epoch'], keep_predictions=True)
                finally:
                    if gate_hook is not None:
                        gate_hook.remove()
                metrics = metric_arrays(pred, truth, weights)
                del pred, truth
                difference = {m: metrics[m] - checked['summary'][m] for m in ['mae', 'rmse']}
                require(max(abs(d) for d in difference.values()) < 2e-5,
                        f'Reevaluation differs from new-run summary: {stem}: {difference}')
                require(max(abs(metrics[m] - official[m]) for m in ['mae', 'rmse']) < 2e-5,
                        'Streaming supplementary metrics disagree with Supervisor.evaluate.')
                require(ntest == official['sample_count'], 'Evaluator omitted test samples.')
                robust = []
                if job['variant'] in ['control', 'agf']:
                    for ratio, maskseed in [(0., None)] + [(r, s) for s in [11, 22, 33] for r in [.1, .2, .3, .4]]:
                        pred, truth = sampled_predictions(supervisor, subset, ratio, maskseed or 0)
                        result = metric_arrays(pred, truth, weights)
                        robust.append({'ratio': ratio, 'maskseed': maskseed, 'mae': result['mae'],
                                       'rmse': result['rmse'], 'n_examples': len(subset)})
                        del pred, truth
                payload = {'kind': job['variant'], 'dataset': task, 'seed': job['seed'],
                           'corrected_best_epoch': checked['summary']['best_epoch'],
                           'selection': 'new training: first full validation zero-inclusive MAE minimum among 100 epochs',
                           'experiment_dir': checked['output_dir'], 'checkpoint': checked['checkpoint'],
                           'config': checked['config_source'], 'summary': checked['summary_source'],
                           'training_history': checked['history_source'], 'training_manifest': checked['manifest_source'],
                           'new_training_metric_differences': difference, 'training_performed': True,
                           'training_precision': 'FP32', 'training_protocol': 'zero_inclusive',
                           'epochs_completed': 100, 'n_parameters': sum(p.numel() for p in supervisor.model.parameters()),
                           'metrics': metrics, 'gates': gates.result(ntest, config) if gate_hook is not None else None,
                           'robustness': robust, 'inference_seconds': time.perf_counter() - start}
                write(data_dir / f'inference_{stem}.json', payload)
                sources.extend(checked[k] for k in ['checkpoint', 'config_source', 'summary_source', 'history_source', 'manifest_source'])
                write(output / 'analysis_status.json', {'status': 'running', 'last_completed': stem,
                      'completed_inferences': len(list(data_dir.glob('inference_*.json'))), 'manuscript_updated': False,
                      'updated_at_utc': now()})
            finally:
                supervisor.close()
                del supervisor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        del shared_data
        gc.collect()
    builder = run_root / 'analysis_snapshot' / 'retrain_build_artifacts.py'
    require(builder.is_file(), 'Frozen retrain_build_artifacts.py is required.')
    spec = importlib.util.spec_from_file_location('retrain_build_artifacts', builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(run_root, data_dir, figures)
    for relative in ['supervisor.py', 'run_revision.py', 'experiments/dataloader.py', 'model/loss.py',
                     'model/clcnn/adaptive_attention.py', 'model/clcnn/recurrent/seq2seq_model.py']:
        sources.append(record(frozen_code / relative))
    sources.extend([record(Path(__file__)), record(builder)])
    write(output / 'analysis_code_manifest.json', {'scripts': [record(Path(__file__)), record(builder)],
          'training_source_root': str(frozen_code), 'generated_at_utc': now()})
    write(output / 'provenance_manifest.json', {
        'source_authority': '46 newly trained zero-inclusive FP32 runs; approved plan and verified new run records only',
        'training': '40 primary plus 6 mean-fusion runs; each completed 100 epochs; validation-selected checkpoint',
        'generated_at_utc': now(), 'sources': list({r['path']: r for r in sources}.values()),
        'generated': [record(p) for directory in [data_dir, figures] for p in directory.iterdir() if p.is_file()],
        'software': {'torch': torch.__version__, 'numpy': np.__version__, 'device': device},
        'manuscript_updated': False})
    write(output / 'analysis_status.json', {'status': 'completed', 'completed_inferences': 46,
          'new_training': True, 'manuscript_updated': False, 'manuscript_status': 'awaiting evidence-based revision',
          'assets': str(data_dir / 'manuscript_assets.json'), 'updated_at_utc': now()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=REVISION / '重新训练2026_10_5')
    parser.add_argument('--output', type=Path, help='Defaults to <run-root>/新训练结果.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--full-analysis', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    output = args.output.resolve() if args.output else run_root / '新训练结果'
    require(output.is_relative_to(run_root), 'Analysis output must remain within the new run root.')
    write(output / 'analysis_status.json', {'status': 'verifying', 'manuscript_updated': False, 'updated_at_utc': now()})
    try:
        verified, report = verify_runs(run_root, output)
        if args.full_analysis:
            full_analysis(verified, report, run_root, output, args.device)
        else:
            write(output / 'analysis_status.json', {'status': 'verified_only', 'verified_runs': 46,
                  'manuscript_updated': False, 'updated_at_utc': now()})
        print(json.dumps({'status': 'success', 'mode': 'full_analysis' if args.full_analysis else 'verify_only',
                          'output': str(output), 'manuscript_updated': False}, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        write(output / 'analysis_status.json', {'status': 'failed', 'error': str(error),
              'manuscript_updated': False, 'instruction': 'Do not update manuscript from incomplete/failed analysis.',
              'updated_at_utc': now()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
