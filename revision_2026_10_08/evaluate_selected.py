"""Evaluate a supplied new-training checkpoint on external WeatherBench data."""
import argparse, copy, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import torch
from run_revision import seed_all
from supervisor import Supervisor

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists(): raise FileExistsError('Use a fresh evaluation output directory.')
    if output.is_relative_to(ROOT/'selected_runs'): raise ValueError('Do not write inside supplied run records.')
    rows=json.loads((ROOT/'selected_checkpoints.json').read_text(encoding='utf-8'))['runs']
    row=next(r for r in rows if r['run_id']==args.run_id)
    checkpoint=ROOT/row['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=row['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    source=ROOT/row['run_directory']
    cfg=copy.deepcopy(json.loads((source/'model_param.json').read_text(encoding='utf-8')))
    data=(args.data_root/row['task']).resolve()
    for name in ['trn.pkl','val.pkl','test.pkl','position_info.pkl']:
        if not (data/name).is_file(): raise FileNotFoundError(data/name)
    cfg['data'].update(dataset_dir=str(data),position_file=str(data/'position_info.pkl'),num_workers=0)
    cfg['train'].update(log_dir=str(output.parent),experiment_name=output.name)
    torch.set_num_threads(4)
    seed_all(row['seed'])
    supervisor=Supervisor(cfg,output,args.device,checkpoint_dir=source/'saved_model')
    try:
        result=supervisor.evaluate('test',epoch=row['selected_epoch'])
        result.update(run_id=row['run_id'],selected_epoch=row['selected_epoch'],checkpoint_sha256=row['checkpoint_sha256'],
                      source_training_protocol='zero_inclusive',source_training_epochs=100,training_performed_in_this_command=False)
        result['differences_from_supplied_analysis']={m:result[m]-row['expected_test_'+m] for m in ['mae','rmse']}
        result['within_2e_minus_5']=all(abs(v)<2e-5 for v in result['differences_from_supplied_analysis'].values())
        (output/'reevaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        if not result['within_2e_minus_5']: raise RuntimeError('Review hardware/software/data discrepancy; do not silently overwrite evidence.')
    finally: supervisor.close()
if __name__=='__main__': main()
