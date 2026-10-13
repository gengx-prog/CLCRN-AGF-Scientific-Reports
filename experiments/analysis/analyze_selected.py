"""Recompute all 46 selected experiments, robustness, gate and horizon diagnostics.

Requires the hash-verified external processed weather data and substantial
memory/compute. The default CPU demo does not execute this analysis.
"""
from __future__ import annotations

import argparse
import copy
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

REVISION = Path(__file__).resolve().parents[1]
REPO = REVISION.parent
TASKS = ['cloud_cover', 'temperature', 'humidity', 'component_of_wind']

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


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def require(test, message):
    if not test:
        raise ValueError(message)


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


def analyze(data_root, output, device):
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO / 'reviewer_tools'))
    from reviewer_verify import check_manifest, check_run_records, check_data, runs, member
    preflight = {'package': check_manifest(REVISION),
                 'training_records': check_run_records(REVISION),
                 'data': check_data(REVISION, data_root, set(TASKS))}
    write(output / 'preflight.json', preflight)
    if device != 'cpu':
        require(device.startswith('cuda') and torch.cuda.is_available(), 'Requested CUDA device unavailable.')
    sys.path.insert(0, str(REVISION / 'code'))
    from supervisor import Supervisor
    from run_revision import seed_all
    require(Path(sys.modules['supervisor'].__file__).resolve() == (REVISION / 'code/supervisor.py').resolve(),
            'Supervisor must be imported from the supplied training source.')
    seed_all(2021)
    torch.set_num_threads(4)
    rows = runs(REVISION)
    data_dir, figures = output / 'data', output / 'figures'
    data_dir.mkdir()
    figures.mkdir()
    completed = []
    for task in TASKS:
        shared_data = None
        task_rows = [row for row in rows if row['task'] == task]
        # Retain the recorded ordering: control/AGF for each primary seed, then
        # the three mean-fusion seeds. This also retains random-stream ordering.
        task_rows.sort(key=lambda row: (row['variant'] == 'mean_fusion', row['seed'], row['variant'] != 'control'))
        for index, row in enumerate(task_rows):
            started = time.perf_counter()
            directory = member(REVISION, row['run_directory'])
            source_config = directory / 'model_param.json'
            config = copy.deepcopy(read(source_config))
            config['data'].update(dataset_dir=str(data_root / task),
                                  position_file=str(data_root / task / 'position_info.pkl'), num_workers=0)
            config['train'].update(log_dir=str(output / 'evaluation_logs'), experiment_name=row['run_id'])
            stem = f"{row['variant']}_{task}_{row['seed']}"
            print('ANALYZE', stem, flush=True)
            supervisor = Supervisor(config, output / 'evaluation_logs' / stem, device=device,
                                    checkpoint_dir=directory / 'saved_model', data=shared_data)
            shared_data = supervisor.data
            try:
                if index == 0:
                    metadata, weights = dataset_metadata(supervisor, task)
                    write(data_dir / f'dataset_{task}.json', metadata)
                    ntest = len(supervisor.data['test_loader'].dataset)
                    require(ntest == 657, 'Expected all 657 test windows.')
                    subset = np.rint(np.linspace(0, ntest-1, 64)).astype(int)
                    write(data_dir / f'robustness_indices_{task}.json', {
                        'zero_based_test_indices': subset.tolist(),
                        'selection': '64 equally spaced deterministic indices across the chronological test set',
                        'mask_seeds': [11, 22, 33]})
                    pred, truth = sampled_predictions(supervisor, np.arange(ntest), persistence=True)
                    write(data_dir / f'persistence_{task}.json', metric_arrays(pred, truth, weights))
                    del pred, truth
                gates, gate_hook = GateStats(), None
                if row['variant'] == 'agf':
                    gate_hook = supervisor.model.asttn_encoder.gate_proj.register_forward_hook(gates.hook)
                try:
                    official, pred, truth = supervisor.evaluate('test', epoch=row['selected_epoch'], keep_predictions=True)
                finally:
                    if gate_hook is not None:
                        gate_hook.remove()
                metrics = metric_arrays(pred, truth, weights)
                del pred, truth
                summary = read(directory / 'summary.json')
                difference = {metric: metrics[metric] - summary[metric] for metric in ['mae', 'rmse']}
                require(max(abs(value) for value in difference.values()) < 2e-5,
                        f'Reevaluation differs from recorded training summary: {stem}: {difference}')
                require(max(abs(metrics[metric] - official[metric]) for metric in ['mae', 'rmse']) < 2e-5,
                        'Supplementary metrics disagree with Supervisor.evaluate.')
                require(ntest == official['sample_count'], 'Evaluator omitted test samples.')
                robust = []
                if row['variant'] in ['control', 'agf']:
                    for ratio, maskseed in [(0., None)] + [(ratio, seed) for seed in [11, 22, 33] for ratio in [.1, .2, .3, .4]]:
                        pred, truth = sampled_predictions(supervisor, subset, ratio, maskseed or 0)
                        result = metric_arrays(pred, truth, weights)
                        robust.append({'ratio': ratio, 'maskseed': maskseed, 'mae': result['mae'],
                                       'rmse': result['rmse'], 'n_examples': len(subset)})
                        del pred, truth
                payload = {'kind': row['variant'], 'dataset': task, 'seed': row['seed'],
                           'corrected_best_epoch': row['selected_epoch'],
                           'selection': 'new training: first full validation zero-inclusive MAE minimum among 100 epochs',
                           'experiment_dir': str(directory),
                           'checkpoint': record(member(REVISION, row['checkpoint'])),
                           'config': record(source_config), 'summary': record(directory / 'summary.json'),
                           'training_history': record(directory / 'training_history.json'),
                           'training_manifest': record(directory / 'run_manifest.json'),
                           'new_training_metric_differences': difference,
                           'training_performed': True,
                           'training_performed_in_this_command': False,
                           'training_precision': 'FP32', 'training_protocol': 'zero_inclusive',
                           'epochs_completed': 100, 'n_parameters': sum(parameter.numel() for parameter in supervisor.model.parameters()),
                           'metrics': metrics, 'gates': gates.result(ntest, config) if gate_hook is not None else None,
                           'robustness': robust, 'inference_seconds': time.perf_counter() - started}
                write(data_dir / f'inference_{stem}.json', payload)
                completed.append(row['run_id'])
                write(output / 'analysis_status.json', {'status': 'running', 'completed_runs': completed,
                      'training_performed_in_this_command': False, 'updated_at_utc': now()})
            finally:
                supervisor.close()
                del supervisor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        del shared_data
        gc.collect()
    builder = REVISION / 'analysis/retrain_build_artifacts.py'
    spec = importlib.util.spec_from_file_location('retrain_build_artifacts', builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(output, data_dir, figures)
    write(output / 'analysis_status.json', {'status': 'passed', 'completed_runs': completed,
          'training_performed_in_this_command': False, 'updated_at_utc': now(),
          'source': record(Path(__file__)), 'builder': record(builder)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Parent directory of the four processed task datasets.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory.')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    for name in ('experiments', 'reviewer_tools', 'validation', '.git', '.github'):
        require(not output.is_relative_to((REPO / name).resolve()), 'Output must be outside supplied sources and evidence.')
    require(not output.exists(), 'Use a fresh output directory.')
    output.mkdir(parents=True)
    write(output / 'analysis_status.json', {'status': 'running', 'started_at_utc': now(),
          'training_performed_in_this_command': False})
    try:
        analyze(args.data_root.resolve(), output, args.device)
    except Exception as error:
        write(output / 'analysis_status.json', {'status': 'failed', 'error': str(error), 'updated_at_utc': now(),
              'training_performed_in_this_command': False})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
