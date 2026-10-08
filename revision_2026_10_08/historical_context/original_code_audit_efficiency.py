"""Matched archived control/AGF inference timing; run after GPU audit completes."""
from pathlib import Path
import json
import sys
import time
import gc
import platform
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
REV=Path(__file__).resolve().parents[1]/'\u53ef\u590d\u73b0\u4ee3\u7801'
sys.path.insert(0,str(REV))
from model.clcnn import CLCRNModel

torch.set_num_threads(4)
if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no GPU benchmark performed.')
device=torch.device('cuda:0')
fixture=Path(__file__).resolve().parents[1]/'\u8865\u5145\u6750\u6599/data_audit/efficiency_temperature_inputs.npz'
batch=np.load(fixture)
inputs=torch.as_tensor(batch['x'],dtype=torch.float32,device=device).permute(1,0,2,3)
shared={key:torch.as_tensor(batch[key],dtype=torch.long if key=='sparse_idx' else torch.float32,device=device)
        for key in ['loc_info','sparse_idx','geodesic','angle_ratio']}
rows=[]
for kind,dirname,suffix in [('control','weatherbench_publication_control_multiseed','wo_adaptive_graph'),
                            ('agf','weatherbench_publication_multiseed','full_model')]:
    exp=ROOT/'experiments'/dirname/'seed_2021/temperature'/f'CLCRN_temperature_{suffix}'
    config=json.loads((exp/'model_param.json').read_text())
    selected=json.loads((exp/'summary_corrected_valbest.json').read_text())
    model=CLCRNModel(**shared,**config['model']).to(device)
    checkpoint=exp/'saved_model'/f"epo{selected['corrected_best_epoch']}.tar"
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
    model.load_state_dict(state,strict=True)
    model.eval()
    with torch.no_grad():
        for _ in range(10): output=model(inputs)
        del output
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline=torch.cuda.memory_allocated(device)
        wall_times=[];cuda_times=[]
        for _ in range(30):
            torch.cuda.synchronize(device)
            start_event=torch.cuda.Event(enable_timing=True)
            end_event=torch.cuda.Event(enable_timing=True)
            start=time.perf_counter()
            start_event.record()
            output=model(inputs)
            end_event.record()
            torch.cuda.synchronize(device)
            wall_times.append((time.perf_counter()-start)*1000)
            cuda_times.append(start_event.elapsed_time(end_event))
            del output
        peak=torch.cuda.max_memory_allocated(device)
    row={'model':kind,'seed':2021,'task':'temperature','epoch':selected['corrected_best_epoch'],
         'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/'),
         'parameter_count':sum(p.numel() for p in model.parameters()),
         'wall_ms_mean':float(np.mean(wall_times)),'wall_ms_sd':float(np.std(wall_times,ddof=1)),
         'wall_ms_median':float(np.median(wall_times)),'wall_ms_samples':wall_times,
         'cuda_ms_mean':float(np.mean(cuda_times)),'cuda_ms_samples':cuda_times,
         'base_allocated_mib':baseline/2**20,'peak_allocated_mib':peak/2**20,
         'incremental_peak_mib':(peak-baseline)/2**20}
    rows.append(row)
    print(kind,row['wall_ms_mean'],row['peak_allocated_mib'],flush=True)
    del model,state
    gc.collect();torch.cuda.empty_cache()
result={'artifact_kind':'matched inference microbenchmark; first 32 real temperature test windows, actual archived weights',
        'date':'2026-10-05','python':platform.python_version(),'torch':torch.__version__,
        'device':torch.cuda.get_device_name(0),'precision':'FP32; eval; no_grad; autocast disabled',
        'input_shape':[12,32,2048,1],'output_shape':[12,32,2048,1],
        'input_distribution':'first 32 temperature test windows, standardized using training x statistics; pre-created on GPU; no disk I/O or host-to-device copy in timed region',
        'input_fixture':str(fixture.relative_to(ROOT)).replace('\\','/'),
        'warmup_iterations':10,'timed_iterations':30,'cpu_threads':4,
        'memory_definition':'torch.cuda.max_memory_allocated after warmup/reset, including resident model/input/geometry tensors; caching allocator reserved memory is not reported',
        'scope':'same real test batch; both current primary seed-2021 archived checkpoints; not training cost, not full test-set throughput',
        'results':rows}
out=REV/'validation/efficiency';out.mkdir(parents=True,exist_ok=True)
(out/'matched_inference.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
