"""Meaningful regression tests plus synthetic end-to-end training on CPU."""
import copy
import json
import pickle
from pathlib import Path
import platform
import numpy as np
import torch
from model.clcnn.adaptive_attention import AdaptiveGraphSpatialAttention
from model.loss import masked_mae_loss, masked_mse_loss
from supervisor import Supervisor


def metric_checks():
    # Zero contributes error; a negative physical wind component also contributes.
    target = torch.tensor([0.,-2.,2.,float('nan'),float('inf')])
    pred = torch.tensor([3.,-1.,4.,float('nan'),float('nan')],requires_grad=True)
    assert torch.isclose(masked_mae_loss(pred,target),torch.tensor(2.))
    assert torch.isclose(masked_mse_loss(pred,target),torch.tensor(14./3.))
    assert torch.isclose(masked_mae_loss(pred,target,null_val=0),torch.tensor(1.5))
    masked_mae_loss(pred,target).backward()
    assert torch.isfinite(pred.grad).all()
    try:
        masked_mae_loss(torch.tensor([float('nan')]),torch.tensor([0.]))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite prediction was silently accepted.')
    return 'zero/negative inclusion, explicit legacy zero exclusion, missing targets, finite gradients, invalid-prediction rejection passed'


def graph_checks():
    torch.manual_seed(123)
    layer = AdaptiveGraphSpatialAttention(6,4,3,adaptive_node_dim=2,topk=3)
    x = torch.randn(2,3,6,4,requires_grad=True)
    index, weight = layer._adaptive_topk()
    assert torch.equal(index[:,0],torch.arange(6))
    assert torch.allclose(weight.sum(-1),torch.ones(6))
    dense = torch.zeros(6,6).scatter(1,index,weight)
    mixed = layer.out_proj(torch.einsum('ij,btjf->btif',dense,x))
    residual = layer.residual_proj(x)
    gate = torch.sigmoid(layer.gate_proj(torch.cat([residual,mixed],dim=-1)))
    expected = gate*mixed+(1-gate)*residual
    actual = layer(x)
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.node_emb_src.grad is not None and torch.isfinite(layer.node_emb_src.grad).all()
    for k in [1,6]:
        other=AdaptiveGraphSpatialAttention(6,4,3,topk=k)
        assert other(x.detach()).shape == (2,3,6,3)
    return 'self-inclusive normalized top-k, independent dense formula, gate operand order, backward gradients, k=1 and k=N passed'


def write_fixture(path):
    path.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1234)
    for split,count in [('trn',6),('val',3),('test',3)]:
        x=rng.normal(size=(count,3,6,2)).astype('float32')
        y=np.repeat(x[:,-1:,:,:],3,axis=1)+rng.normal(scale=.1,size=(count,3,6,2)).astype('float32')
        y[0,:,0,0]=0
        with (path/f'{split}.pkl').open('wb') as f:pickle.dump({'x':x,'y':y},f,protocol=4)
    coords=np.array([[-150,-45],[-90,0],[-30,45],[30,-45],[90,0],[150,45]],dtype='float32')
    with (path/'position_info.pkl').open('wb') as f:pickle.dump({'lonlat':coords},f,protocol=4)


def run_smoke(output):
    output=Path(output)
    results={'artifact_kind':'synthetic software verification, NOT WeatherBench evidence',
             'device':'cpu','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
             'metrics':metric_checks(),'graph':graph_checks(),'runs':[]}
    fixture=output/'synthetic_data'
    write_fixture(fixture)
    base=json.loads((Path(__file__).parent/'configs/primary.json').read_text())
    base['data'].update(dataset_dir=str(fixture),position_file=str(fixture/'position_info.pkl'),batch_size=2,test_batch_size=3,k_neighbors=3)
    base['model'].update(node_num=6,seq_len=3,horizon=3,input_dim=2,output_dim=2,rnn_units=4,layer_num=1,
                         embed_dim=3,asttn_hidden_dim=3,asttn_node_dim=2,asttn_topk=3,lck_structure=[4,3])
    base['train'].update(epochs=1,steps=[1],base_lr=.001,loss_protocol='zero_inclusive')
    for variant in ['agf','control','mean_fusion','legacy_agf']:
        torch.manual_seed(2021);np.random.seed(2021)
        config=copy.deepcopy(base)
        config['model'].update(use_asttn_encoder=variant!='control',asttn_fusion_mode='mean' if variant=='mean_fusion' else 'gated')
        if variant=='legacy_agf':config['train']['loss_protocol']='legacy_zero_exclusive'
        sv=Supervisor(config,output/variant,device='cpu')
        (output/variant/'model_param.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        try:
            assert sv.data['val_loader'].batch_size == 3
            # Corrected explicit geometry argument wiring must agree in all components.
            assert torch.equal(sv.model.angle_ratio,sv.model.decoder_model.angle_ratio)
            assert torch.equal(sv.model.geodesic,sv.model.decoder_model.geodesic)
            before={k:v.detach().clone() for k,v in sv.model.named_parameters()}
            summary=sv.train()
            assert summary['epochs_completed']==1
            assert summary['metric_protocol']=='zero_inclusive'
            assert summary['validation_protocol']==config['train']['loss_protocol']
            assert any(not torch.equal(before[k],v) for k,v in sv.model.named_parameters())
            a,pred,truth=sv.evaluate('test',keep_predictions=True)
            sv.load(summary['best_epoch'])
            b,pred2,truth2=sv.evaluate('test',keep_predictions=True)
            assert torch.equal(pred,pred2) and torch.equal(truth,truth2)
            assert a==b
            assert np.isclose(a['mae'],np.mean(np.abs(pred.numpy()-truth.numpy())),rtol=1e-6)
            results['runs'].append({'variant':variant,'status':'passed','summary':summary,
                                    'checks':'optimizer changed parameters; geometry; configured eval batch size; strict checkpoint round trip; independent NumPy MAE'})
        finally:sv.close()
    results['status']='passed'
    (output/'smoke_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
