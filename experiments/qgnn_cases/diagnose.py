"""Frozen QGNN/GNN case diagnosis on V_select; no optimization or checkpoint writes."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.qgnn_readout.run import Predictor as QPredictor, configuration, shared, training
from experiments.qgat_candidate.run_formal import Predictor as GPredictor

REPORT = ROOT/'reports/qgnn_cases'
RUNS = dict(qgnn=ROOT/'results/qgnn_readout/seed2026/original',
            gnn=ROOT/'results/qgat_extend100/seed2026/gnn')


def target_metrics(prediction, truth, eligible):
    valid = truth['label_valid'] & eligible[:, None]
    distance = torch.linalg.vector_norm(torch.where(valid[..., None], prediction-truth['future_position'], 0), dim=-1)
    counts = valid.sum(1)
    return distance.sum(1)/counts.clamp_min(1), distance[:, -1], counts > 0, valid[:, -1]


@torch.no_grad()
def evaluate(model, valid):
    predictions, ade, fde, ade_mask, fde_mask, scene_ade, scene_fde = [], [], [], [], [], [], []
    for start in range(0, valid.n, 16):
        inputs, truth = valid.batch(np.arange(start, min(start+16, valid.n)), 20, 'cuda:0')
        out = model(**inputs)
        metric = training.scene_metrics(**out, **truth)
        a, f, am, fm = target_metrics(out['prediction'], truth, out['origin_eligible'])
        for rows, tensor in zip((predictions, ade, fde, ade_mask, fde_mask, scene_ade, scene_fde),
                                (out['prediction'], a, f, am, fm, metric['scene_ade'], metric['scene_fde'])):
            rows.append(tensor.cpu().numpy())
    return dict(zip(('prediction', 'target_ADE', 'target_FDE', 'ADE_mask', 'FDE_mask', 'scene_ADE', 'scene_FDE'),
                    (np.concatenate(x) for x in (predictions, ade, fde, ade_mask, fde_mask, scene_ade, scene_fde))))


class LayerTrace:
    """Observe actual tensors, then verify the reconstructed pre-normalization sum."""
    def __init__(self, model):
        self.data = [{} for _ in model.graph.graph_layers]
        self.handles = []
        for layer, row in zip(model.graph.graph_layers, self.data):
            def before(module, args, row=row):
                row.update(nodes=args[0].detach().clone(), edges=args[1].detach().clone(), adj=args[2].detach().clone())
            self.handles.append(layer.register_forward_pre_hook(before))
            for key, module in (('score', layer.score), ('gate_logits', layer.gate), ('value', layer.value),
                                ('edge_value', layer.edge_value), ('readouts', layer.core)):
                self.handles.append(module.register_forward_hook(
                    lambda module, args, output, key=key, row=row: row.__setitem__(key, output.detach().clone())))
            def dropout_input(module, args, row=row):
                if args[0].ndim == 4:
                    row['attention'] = args[0].detach().clone()
            self.handles.append(layer.dropout.register_forward_pre_hook(dropout_input))
            self.handles.append(layer.output_norm.register_forward_pre_hook(
                lambda module, args, row=row: row.__setitem__('residual_input', args[0].detach().clone())))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def reconstruct(self, model, exists):
        active = exists.reshape(-1, exists.shape[-1]).any(-1).nonzero().flatten()
        reconstructed = []
        for layer, row in zip(model.graph.graph_layers, self.data):
            nodes, adj = row['nodes'], row['adj']
            b, n, _ = nodes.shape
            selected = adj.reshape(-1).nonzero().flatten()
            logits = nodes.new_zeros(b*n*n, layer.heads).index_copy(0, selected, row['gate_logits']).view(b,n,n,layer.heads)
            gates = 2*torch.sigmoid(logits)
            base = row['value'].view(b,n,layer.heads,layer.head_dim)[:, None] + row['edge_value'].view(b,n,n,layer.heads,layer.head_dim)
            contribution = row['attention'][..., None]*gates[..., None]*base
            aggregate = contribution.sum(2).reshape(b,n,-1)
            residual = nodes+torch.sigmoid(layer.residual_logit)*aggregate
            torch.testing.assert_close(residual, row['residual_input'], rtol=1e-6, atol=1e-6)
            relations = layer.relation_features(nodes, row['edges'])
            angles = layer.encode_angles(relations)
            outputs = dict(attention=row['attention'], gate=gates, base_norm=base.flatten(-2).norm(dim=-1),
                           contribution_norm=contribution.flatten(-2).norm(dim=-1), adjacency=adj,
                           relations=relations, angles=angles)
            full = {}
            for key, value in outputs.items():
                padded = value.new_zeros(exists.shape[1], *value.shape[1:])
                full[key] = padded.index_copy(0, active, value).cpu().numpy()
            full['residual_strength'] = float(torch.sigmoid(layer.residual_logit).detach())
            reconstructed.append(full)
        return reconstructed


def shared_gate_handles(model, errors):
    handles = []
    for layer in model.graph.graph_layers:
        cache = {}
        def before(module, args, cache=cache):
            adj = args[2]
            cache['rows'] = adj.reshape(-1).nonzero().flatten()//adj.shape[-1]
            cache['count'] = adj.shape[0]*adj.shape[1]
        def replace(module, args, output, cache=cache):
            rows, count = cache['rows'], cache['count']
            gates = 2*torch.sigmoid(output)
            totals = gates.new_zeros(count, gates.shape[-1]).index_add_(0, rows, gates)
            sizes = torch.bincount(rows, minlength=count).clamp_min(1)[:,None]
            mean = totals/sizes
            assigned = mean[rows].clamp(1e-6, 2-1e-6)
            logits = torch.log(assigned/(2-assigned))
            after = gates.new_zeros(count, gates.shape[-1]).index_add_(0, rows, 2*torch.sigmoid(logits))/sizes
            error = float((after-mean).abs().max().detach())
            errors.append(error)
            assert error < 2e-6
            return logits
        handles.append(layer.register_forward_pre_hook(before))
        handles.append(layer.gate.register_forward_hook(replace))
    return handles


def summary_metrics(result):
    return {key:float(result['scene_'+key].mean()) for key in ('ADE','FDE')}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origins', nargs='+', type=int, required=True)
    args = parser.parse_args()
    assert len(args.origins) == len(set(args.origins)) and all(0 <= i < 400 for i in args.origins)
    assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                              capture_output=True,text=True,check=True).stdout.strip(), 'GPU busy'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, valid = configuration(), shared.dataset('V_select')
    assert valid.n == 400
    metadata = json.loads((ROOT/'data/f01d/metadata/V_select.json').read_text())['samples']
    models, checkpoints, baseline, archive_errors = {}, {}, {}, {}
    for name in ('qgnn','gnn'):
        model = (QPredictor('original', cfg) if name=='qgnn' else GPredictor('gnn', 2026, cfg)).to('cuda:0').eval()
        saved = torch.load(RUNS[name]/'best.pt', map_location='cpu', weights_only=False)
        assert all(saved['config']['validation_hashes'][k] == v for k,v in valid.input_hashes.items())
        training.load_weights(model, saved['model'])
        result = evaluate(model, valid)
        epoch = saved['progress']['epoch']
        reference = json.loads((RUNS[name]/f'validation_epoch_{epoch:02d}.json').read_text())
        rows = [row for row in reference['per_scene'] if row['snr_db']==20]
        assert [row['origin'] for row in rows] == list(range(valid.n))
        error = max(abs(float(result['scene_'+k][i])-row[k]) for i,row in enumerate(rows) for k in ('ADE','FDE'))
        assert error < 1e-5, (name, error)
        models[name], checkpoints[name], baseline[name], archive_errors[name] = model, saved, result, error
        print('BASELINE', name, epoch, summary_metrics(result), 'max_error', error, flush=True)
    for key in ('ADE_mask','FDE_mask'):
        assert np.array_equal(baseline['qgnn'][key], baseline['gnn'][key])
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPORT/'predictions.npz', **{f'{name}_{key}':value for name,result in baseline.items() for key,value in result.items()})
    target_rows = []
    for i in range(valid.n):
        for slot in range(8):
            if not baseline['qgnn']['ADE_mask'][i,slot]:
                continue
            row = dict(origin=i, slot=slot, episode_id=metadata[i]['episode_id'], source_block=metadata[i]['source_block'])
            for name in ('qgnn','gnn'):
                row[name+'_ADE'] = float(baseline[name]['target_ADE'][i,slot])
                row[name+'_FDE'] = float(baseline[name]['target_FDE'][i,slot]) if baseline[name]['FDE_mask'][i,slot] else None
            target_rows.append(row)
    with (REPORT/'target_errors.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=list(target_rows[0]));writer.writeheader();writer.writerows(target_rows)

    qmodel, qstate = models['qgnn'], checkpoints['qgnn']['model']
    controls, mean_errors = {}, []
    for mode in ('uniform_attention','unit_gate','neighbor_mean_gate'):
        training.load_weights(qmodel, qstate)
        handles = []
        for layer in qmodel.graph.graph_layers:
            if mode=='uniform_attention':layer.score.weight.zero_()
            if mode=='unit_gate':layer.gate.weight.zero_()
        if mode=='neighbor_mean_gate':handles = shared_gate_handles(qmodel, mean_errors)
        try:result = evaluate(qmodel, valid)
        finally:
            for handle in handles:handle.remove()
        scores = summary_metrics(result)
        controls[mode] = dict(metrics=scores, change_percent={k:100*(scores[k]/summary_metrics(baseline['qgnn'])[k]-1) for k in scores})
        print('CONTROL',mode,controls[mode],flush=True)
    training.load_weights(qmodel, qstate)
    case_rows, arrays = [], {}
    for origin in args.origins:
        inputs, truth = valid.batch([origin],20,'cuda:0')
        delta = baseline['qgnn']['target_ADE'][origin]-baseline['gnn']['target_ADE'][origin]
        worse = baseline['qgnn']['scene_ADE'][origin] > baseline['gnn']['scene_ADE'][origin]
        choices = np.flatnonzero(baseline['qgnn']['ADE_mask'][origin])
        target = int(choices[np.argmax(delta[choices]) if worse else np.argmin(delta[choices])])
        trace = LayerTrace(qmodel)
        try:
            output = qmodel(**inputs)
            layers = trace.reconstruct(qmodel, inputs['track_exists'])
        finally:trace.close()
        assert np.max(np.abs(output['prediction'][0].cpu().numpy()-baseline['qgnn']['prediction'][origin])) < 1e-4
        prefix = f'case_{origin}_'
        arrays.update({prefix+k:v[0].cpu().numpy() for k,v in {**inputs,**truth}.items()})
        for name in ('qgnn','gnn'):arrays[prefix+name+'_prediction'] = baseline[name]['prediction'][origin]
        neighbor_rows = []
        for j in range(8):
            if j == target or not layers[0]['adjacency'][:,target,j].any():continue
            handles = []
            def remove_edge(module, values, sender=j):
                nodes, edges, adj = values
                changed = adj.clone();changed[:,target,sender]=False
                return nodes,edges,changed
            handles = [layer.register_forward_pre_hook(remove_edge) for layer in qmodel.graph.graph_layers]
            try:removed = qmodel(**inputs)
            finally:
                for handle in handles:handle.remove()
            a,f,am,fm = target_metrics(removed['prediction'],truth,removed['origin_eligible'])
            neighbor_rows.append(dict(sender=j, source_key=metadata[origin]['source_keys'][j],
                target_ADE_without_edge=float(a[0,target]),
                target_ADE_change=float(a[0,target])-float(baseline['qgnn']['target_ADE'][origin,target]),
                target_FDE_change=float(f[0,target])-float(baseline['qgnn']['target_FDE'][origin,target]) if fm[0,target] else None))
        layer_stats = []
        for number, layer in enumerate(layers):
            for key,value in layer.items():
                if isinstance(value,np.ndarray):arrays[prefix+f'layer{number}_'+key]=value
            neighbors = []
            for sender in range(8):
                mask = layer['adjacency'][:,target,sender]
                if not mask.any():continue
                neighbors.append(dict(sender=sender, frames=int(mask.sum()),
                    attention_mean=float(layer['attention'][mask,target,sender].mean()),
                    gate_mean=float(layer['gate'][mask,target,sender].mean()),
                    base_norm_mean=float(layer['base_norm'][mask,target,sender].mean()),
                    contribution_norm_mean=float(layer['contribution_norm'][mask,target,sender].mean())))
            layer_stats.append(dict(layer=number,residual_strength=layer['residual_strength'],neighbors=neighbors))
        row = dict(origin=origin, selection='Q_worse' if worse else 'Q_better', target_slot=target,
                   target_source_key=metadata[origin]['source_keys'][target],metadata=metadata[origin],
                   scores={name:{k:float(baseline[name]['scene_'+k][origin]) for k in ('ADE','FDE')} for name in ('qgnn','gnn')},
                   target_scores={name:{k:float(baseline[name]['target_'+k][origin,target]) for k in ('ADE','FDE')} for name in ('qgnn','gnn')},
                   layer_statistics=layer_stats, direct_edge_interventions=neighbor_rows)
        case_rows.append(row);print('CASE',origin,'target',target,'interventions',len(neighbor_rows),flush=True)
    np.savez_compressed(REPORT/'case_traces.npz',**arrays)
    report = dict(status='complete',seed=2026,split='V_select',snr_db=20,origins=args.origins,
                  checkpoint_epochs={name:s['progress']['epoch'] for name,s in checkpoints.items()},
                  archived_scene_metrics_max_error=archive_errors,baseline={n:summary_metrics(v) for n,v in baseline.items()},
                  control_experiments=controls,neighbor_mean_gate_preservation_max_error=max(mean_errors),
                  cases=case_rows,optimizer_steps=0,checkpoint_files_modified=False,
                  limitations=['Historical GNN batching differs; diagnostic reference, not matched model attribution.',
                               'Post-hoc selected cases are explanatory examples, not independent validation.',
                               'Frozen-model interventions change internal distributions; attention/norm are not causal importance.',
                               'Edge removal targets a direct sender-to-receiver edge in both layers over history; indirect paths may remain.'])
    training.write_json(REPORT/'case_diagnostics.json',report)
    print('COMPLETE',flush=True)


if __name__ == '__main__':main()
