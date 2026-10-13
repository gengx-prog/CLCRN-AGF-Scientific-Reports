"""Curated CLCRN training with atomic epoch checkpoints and exact resume state."""
from pathlib import Path
import hashlib
import json
import os
import random
import time
import numpy as np
import torch
from experiments import dataloader
from lib.utils import get_logger
from model.clcnn import CLCRNModel
from model.loss import masked_mae_loss

CHECKPOINT_VERSION = 2
PROCESSING_VERSION = 'curated_float32_then_train_only_standardization_v1'


def config_digest(config):
    value = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA RNG checkpoint requires CUDA-capable hardware.')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


class Supervisor:
    def __init__(self, config, output_dir, device='cpu', checkpoint_dir=None, data=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir/'saved_model'
        self.device = torch.device(device)
        self.logger = get_logger(self.output_dir)
        self.data = data if data is not None else dataloader.load_dataset(**config['data'])
        self.scalers = self.data['scaler']
        self.model_kwargs = dict(config['model'])
        if self.model_kwargs.get('model_name', 'CLCRN') != 'CLCRN':
            raise ValueError('This curated package supports CLCRN and its AGF/mean-fusion variants.')
        self.model_kwargs.pop('asttn_heads', None)
        ki = self.data['kernel_info']
        self.model = CLCRNModel(
            loc_info=torch.as_tensor(ki['MLP_inputs'], dtype=torch.float32, device=self.device),
            sparse_idx=torch.as_tensor(ki['sparse_idx'], dtype=torch.long, device=self.device),
            geodesic=torch.as_tensor(ki['geodesic'], dtype=torch.float32, device=self.device),
            angle_ratio=torch.as_tensor(ki['angle_ratio'], dtype=torch.float32, device=self.device),
            logger=self.logger, **self.model_kwargs,
        ).to(self.device)
        self.protocol = config['train'].get('loss_protocol', 'zero_inclusive')
        self.validation_protocol = config['train'].get('validation_protocol', self.protocol)
        for protocol in [self.protocol, self.validation_protocol]:
            if protocol not in ('zero_inclusive', 'legacy_zero_exclusive'):
                raise ValueError(f'Unknown loss protocol: {protocol}')
        self.initialization_audit = None

    def prepare(self, x, y):
        return (x.permute(1, 0, 2, 3).float().to(self.device),
                y.permute(1, 0, 2, 3).float().to(self.device))

    def inverse(self, values):
        return torch.stack([self.scalers[i].inverse_transform(values[..., i])
                            for i in range(values.shape[-1])], dim=-1)

    @staticmethod
    def null_value(protocol):
        return 0.0 if protocol == 'legacy_zero_exclusive' else None

    def materialize(self):
        """Fixed sample eval warmup before Adam; no shuffle or teacher forcing.

        Current CLCRN parameters are fully eager. Eval preserves BatchNorm state;
        RNG is restored when no lazy parameters were created or replaced.
        """
        before = {name: id(p) for name, p in self.model.named_parameters()}
        state, was_training = rng_state(), self.model.training
        self.model.eval()
        x, y = self.data['train_loader'].dataset[0]
        x, y = self.prepare(torch.as_tensor(x).unsqueeze(0), torch.as_tensor(y).unsqueeze(0))
        with torch.no_grad():
            if not torch.isfinite(self.model(x)).all():
                raise FloatingPointError('Model initialization produced nonfinite predictions.')
        after = {name: id(p) for name, p in self.model.named_parameters()}
        added = sorted(set(after)-set(before))
        replaced = sorted(name for name in before if after.get(name) != before[name])
        if not added and not replaced:
            restore_rng(state)
        self.model.train(was_training)
        self.initialization_audit = {
            'warmup': 'fixed_training_sample_0_eval_no_grad_before_optimizer',
            'lazy_parameter_names': added, 'replaced_parameter_names': replaced,
            'rng_preserved': not added and not replaced,
            'trainable_parameter_count': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'parameter_tensor_count': len(after), 'optimizer_coverage': 'not_yet_checked'}
        return self.initialization_audit

    def assert_optimizer_coverage(self, optimizer):
        expected = {id(p): name for name, p in self.model.named_parameters() if p.requires_grad}
        optimized = [id(p) for group in optimizer.param_groups for p in group['params']]
        missing = [name for ident, name in expected.items() if ident not in optimized]
        if missing or len(optimized) != len(set(optimized)) or set(optimized) != set(expected):
            raise RuntimeError(f'Optimizer parameter coverage mismatch: missing={missing}')

    def save(self, epoch, optimizer=None, scheduler=None, scaler=None, **training_state):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir/f'epo{epoch}.tar'
        payload = {'model_state_dict': self.model.state_dict(), 'epoch': epoch,
                   'config': self.config, 'config_sha256': config_digest(self.config),
                   'loss_protocol': self.protocol, 'validation_protocol': self.validation_protocol,
                   'processing_version': PROCESSING_VERSION}
        if optimizer is not None:
            payload.update(checkpoint_version=CHECKPOINT_VERSION,
                           optimizer_state_dict=optimizer.state_dict(),
                           scheduler_state_dict=scheduler.state_dict(),
                           scaler_state_dict=scaler.state_dict(), rng_state=rng_state(),
                           initialization_audit=self.initialization_audit,
                           device_type=self.device.type, **training_state)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if optimizer is not None:
            atomic_json(self.output_dir/'last_checkpoint.json',
                        {'epoch': epoch, 'path': str(path.resolve()),
                         'config_sha256': payload['config_sha256'],
                         'best_epoch': training_state['best_epoch'],
                         'batches_seen': training_state['batches_seen']})
        return path

    def load(self, epoch):
        checkpoint = torch.load(self.checkpoint_dir/f'epo{epoch}.tar', map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.logger.info('Loaded checkpoint epoch %d', epoch)

    def evaluate(self, split='test', epoch=None, keep_predictions=False, full_metrics=True):
        if epoch is not None:
            self.load(epoch)
        self.model.eval()
        protocol = self.validation_protocol if split == 'val' else 'zero_inclusive'
        null = self.null_value(protocol)
        sums_abs = sums_sq = counts = None
        sample_count = 0
        preds, truths = [], []
        with torch.no_grad():
            for x, y in self.data[f'{split}_loader']:
                x, y = self.prepare(x, y)
                pred, truth = self.inverse(self.model(x)), self.inverse(y)
                valid = torch.isfinite(truth)
                if null is not None:
                    valid &= truth != null
                if torch.any(valid & ~torch.isfinite(pred)):
                    raise FloatingPointError('Nonfinite evaluation prediction at a valid target.')
                difference = torch.where(valid, pred, 0.0)-torch.where(valid, truth, 0.0)
                dimensions = tuple(range(1, difference.ndim))
                batch_abs = difference.abs().sum(dim=dimensions, dtype=torch.float64)
                batch_count = valid.sum(dim=dimensions)
                sums_abs = batch_abs if sums_abs is None else sums_abs+batch_abs
                counts = batch_count if counts is None else counts+batch_count
                if full_metrics:
                    batch_sq = difference.square().sum(dim=dimensions, dtype=torch.float64)
                    sums_sq = batch_sq if sums_sq is None else sums_sq+batch_sq
                sample_count += int(pred.shape[1])
                if keep_predictions:
                    preds.append(pred.cpu())
                    truths.append(truth.cpu())
        if counts is None or torch.any(counts == 0):
            raise ValueError('No valid targets for one or more evaluation horizons.')
        result = {'mae': float(sums_abs.sum()/counts.sum()), 'metric_protocol': protocol,
                  'spatial_weighting': 'uniform_gridpoint', 'sample_count': sample_count}
        if full_metrics:
            a, s, c = sums_abs.cpu().numpy(), sums_sq.cpu().numpy(), counts.cpu().numpy()
            result.update(rmse=float(np.sqrt(s.sum()/c.sum())), step_metrics={}, exact_step_metrics={})
            for step in range(1, len(c)+1):
                result['step_metrics'][f'mae_{step}'] = float(a[:step].sum()/c[:step].sum())
                result['step_metrics'][f'rmse_{step}'] = float(np.sqrt(s[:step].sum()/c[:step].sum()))
                result['exact_step_metrics'][f'mae_{step}'] = float(a[step-1]/c[step-1])
                result['exact_step_metrics'][f'rmse_{step}'] = float(np.sqrt(s[step-1]/c[step-1]))
        if keep_predictions:
            return result, torch.cat(preds, dim=1), torch.cat(truths, dim=1)
        return result

    def train(self, resume=False, stop_after_epoch=None):
        kw = self.config['train']
        self.materialize()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=kw['base_lr'], eps=kw.get('epsilon', 1e-3))
        self.assert_optimizer_coverage(optimizer)
        self.initialization_audit['optimizer_coverage'] = 'all_trainable_parameters_exactly_once'
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=kw['steps'], gamma=kw['lr_decay_ratio'])
        use_amp = bool(kw.get('use_amp', False)) and self.device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        best, best_epoch, wait, batches_seen, start_epoch = float('inf'), 0, 0, 0, 1
        history = []
        if resume:
            paths = sorted(self.checkpoint_dir.glob('epo*.tar'), key=lambda p: int(p.stem[3:]))
            if not paths:
                raise FileNotFoundError('Resume requested but no completed epoch checkpoint exists.')
            state = torch.load(paths[-1], map_location='cpu', weights_only=False)
            if state.get('checkpoint_version') != CHECKPOINT_VERSION:
                raise ValueError('Checkpoint lacks complete optimizer/RNG state and cannot resume.')
            if state.get('config_sha256') != config_digest(self.config) or state['config'] != self.config:
                raise ValueError('Resume configuration, seed or protocol differs from the checkpoint.')
            if state.get('processing_version') != PROCESSING_VERSION or state.get('device_type') != self.device.type:
                raise ValueError('Resume processing version or device type differs from checkpoint.')
            self.model.load_state_dict(state['model_state_dict'], strict=True)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            scaler.load_state_dict(state['scaler_state_dict'])
            best, best_epoch, wait = state['best'], state['best_epoch'], state['wait']
            batches_seen, history = state['batches_seen'], state['history']
            start_epoch = state['epoch']+1
            self.initialization_audit = state['initialization_audit']
            self.assert_optimizer_coverage(optimizer)
            restore_rng(state['rng_state'])
            atomic_json(self.output_dir/'training_history.json', history)
            self.logger.info('Resuming after epoch %d with %d batches seen', start_epoch-1, batches_seen)
        atomic_json(self.output_dir/'initialization_audit.json', self.initialization_audit)
        end_epoch = min(kw['epochs'], stop_after_epoch) if stop_after_epoch is not None else kw['epochs']
        patience = kw.get('patience', 50)
        for epoch in range(start_epoch, end_epoch+1):
            if patience > 0 and wait >= patience:
                break
            start = time.perf_counter()
            self.model.train()
            losses = []
            for batch_index, (x, y) in enumerate(self.data['train_loader']):
                optimizer.zero_grad(set_to_none=True)
                x, y = self.prepare(x, y)
                with torch.amp.autocast(self.device.type, enabled=use_amp):
                    pred = self.model(x, y, batches_seen=batches_seen)
                    loss = masked_mae_loss(self.inverse(pred), self.inverse(y), null_val=self.null_value(self.protocol))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, batch {batch_index}.')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if batch_index == 0:
                    self.assert_optimizer_coverage(optimizer)
                    absent = [name for name, p in self.model.named_parameters() if p.requires_grad and p.grad is None]
                    if absent:
                        raise RuntimeError(f'Trainable parameters without gradients: {absent}')
                    self.initialization_audit['gradient_coverage'] = 'all_trainable_parameters_have_gradients'
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), kw.get('max_grad_norm', 5), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                batches_seen += 1
                losses.append(float(loss.detach()))
            if not losses:
                raise ValueError('Training loader contains no batches.')
            scheduler.step()
            val = self.evaluate('val', full_metrics=False)
            row = {'epoch': epoch, 'train_mae': float(np.mean(losses)), 'val_mae': val['mae'],
                   'seconds': time.perf_counter()-start, 'learning_rate': scheduler.get_last_lr()[0],
                   'batches_seen': batches_seen}
            history.append(row)
            if val['mae'] < best:
                best, best_epoch, wait = val['mae'], epoch, 0
            else:
                wait += 1
            self.save(epoch, optimizer, scheduler, scaler, best=best, best_epoch=best_epoch,
                      wait=wait, batches_seen=batches_seen, history=history)
            atomic_json(self.output_dir/'training_history.json', history)
            atomic_json(self.output_dir/'initialization_audit.json', self.initialization_audit)
            self.logger.info('%s', row)
        if not best_epoch:
            raise RuntimeError('No valid best checkpoint was produced.')
        complete = len(history) >= kw['epochs']
        early_stopped = patience > 0 and wait >= patience and not complete
        result = self.evaluate('test', epoch=best_epoch) if complete or early_stopped else {}
        result.update(best_epoch=best_epoch, best_val_mae=best, training_protocol=self.protocol,
                      protocol=self.protocol, validation_protocol=self.validation_protocol,
                      epochs_completed=len(history), planned_epochs=kw['epochs'], training_complete=complete,
                      status='completed' if complete else ('early_stopped' if early_stopped else 'interrupted'),
                      precision='amp' if use_amp else 'fp32', batches_seen=batches_seen,
                      config_sha256=config_digest(self.config), processing_version=PROCESSING_VERSION,
                      checkpoint=str((self.checkpoint_dir/f'epo{best_epoch}.tar').resolve()),
                      last_checkpoint=str((self.checkpoint_dir/f'epo{len(history)}.tar').resolve()))
        return result

    def close(self):
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
