"""Portable train / archived-checkpoint evaluation / synthetic smoke entry point."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import numpy as np
import torch
from supervisor import Supervisor, atomic_json, config_digest, PROCESSING_VERSION

ROOT = Path(__file__).resolve().parent
TASKS = ['temperature','humidity','component_of_wind','cloud_cover']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path, value):
    atomic_json(path, value)


def source_hashes():
    paths = [ROOT/'supervisor.py', ROOT/'run_revision.py']
    for folder in ['model', 'experiments', 'lib']:
        paths.extend((ROOT/folder).rglob('*.py'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true',help='Small synthetic CPU train/save/reload/evaluation; no benchmark data needed.')
    p.add_argument('--data-root',type=Path,help='Directory containing the four task directories with trn/val/test/position_info.pkl.')
    p.add_argument('--task',choices=TASKS,default='temperature')
    p.add_argument('--variant',choices=['agf','control','mean_fusion'],default='agf')
    p.add_argument('--seed',type=int,default=2021)
    p.add_argument('--config',type=Path,default=ROOT/'configs/primary.json')
    p.add_argument('--protocol',choices=['zero_inclusive','legacy_zero_exclusive'],default='zero_inclusive')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,help='Early-stopping patience; 0 disables early stopping for the full 100-epoch budget.')
    p.add_argument('--resume',action='store_true',help='Resume the same output/config/seed from its latest complete atomic epoch checkpoint.')
    p.add_argument('--stop-after-epoch',type=int,help='Stop cleanly at this epoch, retaining the configured full budget for a later --resume.')
    p.add_argument('--amp',action='store_true',help='Explicit opt-in to CUDA AMP. All revision benchmark runs default to FP32.')
    p.add_argument('--batch-size',type=int,help='Optional training batch size override.')
    p.add_argument('--eval-batch-size',type=int,help='Optional validation/test batch size override.')
    p.add_argument('--device',default='cpu',help='cpu or cuda:0, etc.; CPU is used by --smoke.')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,required=True,help='New output directory, or the original run directory with --resume.')
    p.add_argument('--evaluate-archive',type=Path,help='Historical experiment directory with model_param.json and saved_model/*.tar.')
    p.add_argument('--epoch',type=int,help='Evaluate a fixed historical checkpoint; omit to reselect among all saved epochs using corrected validation MAE.')
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume and (args.smoke or args.evaluate_archive):
        raise ValueError('--resume is only supported for new training, not smoke or archived evaluation.')
    if args.resume and not (args.output/'run_manifest.json').is_file():
        raise FileNotFoundError('--resume requires an existing run_manifest.json in the original output directory.')
    if not args.resume and ((args.output/'run_manifest.json').exists() or (args.output/'smoke_results.json').exists()):
        raise FileExistsError('Choose a fresh output directory; an existing run will not be reused or overwritten.')
    if args.smoke:
        from smoke import run_smoke
        run_smoke(args.output)
        return
    if args.data_root is None:
        raise ValueError('--data-root is required except for --smoke.')
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive.')
    if args.patience is not None and args.patience < 0:
        raise ValueError('--patience must be nonnegative; 0 disables early stopping.')
    if args.stop_after_epoch is not None and not 0 < args.stop_after_epoch <= args.epochs:
        raise ValueError('--stop-after-epoch must be between 1 and --epochs.')
    seed_all(args.seed)
    data_dir = (args.data_root/args.task).resolve()
    for name in ['trn.pkl','val.pkl','test.pkl','position_info.pkl']:
        if not (data_dir/name).is_file():
            raise FileNotFoundError(data_dir/name)
    source_config = args.evaluate_archive/'model_param.json' if args.evaluate_archive else args.config
    config = json.loads(source_config.read_text(encoding='utf-8'))
    config = copy.deepcopy(config)
    config['data'].update(dataset_dir=str(data_dir),position_file=str(data_dir/'position_info.pkl'),num_workers=0)
    if args.batch_size is not None:
        if args.batch_size <= 0: raise ValueError('--batch-size must be positive.')
        config['data']['batch_size'] = args.batch_size
    if args.eval_batch_size is not None:
        if args.eval_batch_size <= 0: raise ValueError('--eval-batch-size must be positive.')
        config['data']['test_batch_size'] = args.eval_batch_size
        config['data']['val_batch_size'] = args.eval_batch_size
    config['model'].pop('asttn_heads',None)
    checkpoints = None
    if args.evaluate_archive:
        archive = args.evaluate_archive.resolve()
        if archive == args.output or archive in args.output.parents:
            raise ValueError('Archived evaluation output must be outside the historical experiment directory.')
        checkpoints = archive/'saved_model'
        config['train']['validation_protocol'] = 'zero_inclusive'
        config['train'].setdefault('loss_protocol', 'legacy_zero_exclusive')
        epochs = sorted(int(p.stem[3:]) for p in checkpoints.glob('epo*.tar'))
        if not epochs:
            raise FileNotFoundError(f'No saved checkpoints: {checkpoints}')
        if args.epoch is not None and args.epoch not in epochs:
            raise ValueError(f'Checkpoint epoch {args.epoch} is absent; available epochs: {epochs}')
    else:
        channels = 2 if args.task == 'component_of_wind' else 1
        config['model'].update(input_dim=channels,output_dim=channels,
                               use_asttn_encoder=args.variant!='control',
                               asttn_fusion_mode='mean' if args.variant=='mean_fusion' else 'gated')
        config['train'].update(epochs=args.epochs,loss_protocol=args.protocol,validation_protocol=args.protocol,
                               epoch=0,seed=args.seed,use_amp=args.amp)
        if args.patience is not None:
            config['train']['patience'] = args.patience
    config['train'].update(log_dir=str(args.output.parent),experiment_name=args.output.name)
    manifest = {'release':'2026-10-05 local revision; not published', 'seed':args.seed,
                'mode':'archived_checkpoint_evaluation' if args.evaluate_archive else 'new_training',
                'task':args.task,'variant':args.variant,'config':config,'python':platform.python_version(),
                'torch':torch.__version__,'numpy':np.__version__,'device':args.device,
                'threads':args.threads,'precision':'amp' if args.amp and args.device.startswith('cuda') else 'fp32',
                'processing_version':PROCESSING_VERSION,'config_sha256':config_digest(config),
                'source_hashes':source_hashes(),
                'numerics':{'tf32':False,'cudnn_deterministic':True,'cudnn_benchmark':False},
                'source_config':str(source_config.resolve()),
                'source_config_sha256':hashlib.sha256(source_config.read_bytes()).hexdigest(),
                'source_model_protocol':'historical model; training provenance is not changed by reevaluation' if args.evaluate_archive else args.protocol,
                'command':sys.argv}
    if args.resume:
        original = json.loads((args.output/'run_manifest.json').read_text(encoding='utf-8'))
        keys = ['mode','seed','task','variant','config','device','threads','precision','python','torch','numpy',
                'processing_version','source_hashes','numerics','source_config_sha256']
        differences = [key for key in keys if original.get(key) != manifest.get(key)]
        if differences:
            raise ValueError(f'Resume rejected: original manifest differs in {differences}. Use the original configuration and software.')
        events_path = args.output/'resume_events.json'
        events = json.loads(events_path.read_text(encoding='utf-8')) if events_path.exists() else []
        events.append({'command':sys.argv,'config_sha256':manifest['config_sha256']})
        write_json(events_path, events)
    else:
        write_json(args.output/'run_manifest.json',manifest)
        write_json(args.output/'model_param.json',config)
    supervisor = Supervisor(config,args.output,args.device,checkpoint_dir=checkpoints)
    try:
        if args.evaluate_archive:
            if args.epoch is None:
                val_rows = [{'epoch':e,**supervisor.evaluate('val',epoch=e)} for e in epochs]
                best = min(val_rows,key=lambda row:row['mae'])
                write_json(args.output/'validation_candidates.json',val_rows)
                selected = best['epoch']
            else:
                selected = args.epoch
                best = supervisor.evaluate('val',epoch=selected)
            result = supervisor.evaluate('test',epoch=selected)
            result.update(best_epoch=selected,best_val_mae=best['mae'],saved_epochs=epochs,
                          selection='corrected_validation_among_saved_checkpoints' if args.epoch is None else 'fixed_checkpoint',
                          training_performed=False)
        else:
            result = supervisor.train(resume=args.resume,stop_after_epoch=args.stop_after_epoch)
            result['training_performed'] = True
        result.update(seed=args.seed,task=args.task,variant=args.variant,mode=manifest['mode'])
        write_json(args.output/'summary.json',result)
        print(json.dumps(result,indent=2))
    finally:
        supervisor.close()


if __name__ == '__main__':
    main()
