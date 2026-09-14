"""Independent F02-F interface checks; training data only, no optimizer steps."""
import json
from pathlib import Path
import torch
from frontend.symbol_dataset import SharedPredictionInputs
from .classical import ClassicalGNN
from .temporal import NumericTokenAdapter, TrajectoryPredictor, masked_trajectory_loss

ROOT = Path(__file__).resolve().parents[1]


def run(device='cuda:1'):
    torch.manual_seed(20260912)
    result = {'device': device, 'formal_training': False, 'checks': {}}
    checks = result['checks']

    def record(name, passed, **evidence):
        checks[name] = dict(passed=bool(passed), **evidence)
        print(json.dumps({'check': name, **checks[name]}), flush=True)
        if not passed:
            raise AssertionError(name)

    # Compare the batched GNN against a literal per-node/per-edge evaluation.
    g = ClassicalGNN().double().to(device)
    x = (torch.randn(2, 3, 4, device=device, dtype=torch.float64) * 10).requires_grad_()
    z = x / 12
    mask = torch.tensor([[1, 1, 1], [1, 0, 1]], device=device, dtype=torch.bool)
    detected = mask.clone()
    out = g(x, z, mask, detected)
    rows = []
    for b in range(2):
        h = [g.input(torch.cat((z[b, i] if mask[b, i] else z[b, i]*0,
                               detected[b, i:i+1].double(), mask[b, i:i+1].double()))) * mask[b, i]
             for i in range(3)]
        for message, update, norm in zip(g.messages, g.updates, g.norms):
            new = []
            for i in range(3):
                u = h[i] * 0
                for j in range(3):
                    if i != j and mask[b, i] and mask[b, j]:
                        delta = x[b, j] - x[b, i]
                        edge = delta / delta.new_tensor([45., 45., 15., 15.])
                        kappa = torch.exp(-delta[:2].square().sum() / (2 * 45**2))
                        u = u + kappa * message(torch.cat((h[i], h[j], edge))) / 7
                new.append(norm(h[i] + update(torch.cat((h[i], u)))) * mask[b, i])
            h = new
        rows.append(torch.stack(h))
    reference = torch.stack(rows)
    error = (out-reference).abs().max().item()
    weights = torch.randn_like(out)
    ga = torch.autograd.grad((out*weights).sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad((reference*weights).sum(), x)[0]
    grad_error = (ga-gb).abs().max().item()
    record('GNN_literal_edge_formula', error <= 1e-10 and grad_error <= 1e-8,
           output_max_abs=error, input_gradient_max_abs=grad_error,
           message_input_width=g.messages[0][0].in_features, edge_features=4)
    perm = torch.tensor([2, 0, 1], device=device)
    xp = x.detach()[:, perm].requires_grad_()
    reordered = g(xp, xp/12, mask[:, perm], detected[:, perm])
    perm_error = (reordered-out.detach()[:, perm]).abs().max().item()
    gp = torch.autograd.grad((reordered*weights[:, perm]).sum(), xp)[0]
    perm_grad_error = (gp-ga[:, perm]).abs().max().item()
    padded_x = torch.cat((x.detach(), torch.randn(2, 5, 4, device=device, dtype=torch.float64)*1000), 1)
    padded_m = torch.cat((mask, torch.zeros(2, 5, device=device, dtype=torch.bool)), 1)
    padded = g(padded_x, padded_x/12, padded_m, padded_m)
    record('GNN_permutation_padding', perm_error <= 1e-10 and perm_grad_error <= 1e-8
           and (padded[:, :3]-out.detach()).abs().max().item() <= 1e-10 and padded[:, 3:].abs().max().item() == 0,
           permutation_abs=perm_error, permutation_gradient_abs=perm_grad_error)
    del out, reference, rows, g

    # Q21 -> Q24 token nesting, independent of a particular trained circuit.
    old = NumericTokenAdapter(21).double().to(device)
    new = NumericTokenAdapter(24).double().to(device)
    with torch.no_grad():
        new.state_projection.load_state_dict(old.state_projection.state_dict())
        new.marker_projection.load_state_dict(old.marker_projection.state_dict())
        new.graph_projection.weight[:, :21].copy_(old.graph_projection.weight)
    s = torch.randn(1, 20, 8, 4, device=device, dtype=torch.float64)
    m = torch.ones(1, 20, 8, device=device, dtype=torch.bool)
    q = torch.randn(1, 20, 8, 24, device=device, dtype=torch.float64, requires_grad=True)
    a, b = old(s, m, m, q[..., :21]), new(s, m, m, q)
    nesting = (a-b).abs().max().item()
    grad = torch.autograd.grad(b.square().sum(), q)[0]
    record('Q21_Q24_token_nesting', nesting <= 1e-10 and grad[..., 21:].abs().max().item() == 0,
           output_max_abs=nesting, token_shape=list(b.shape),
           original_columns_nonzero=bool(new.graph_projection.weight[:, :21].abs().sum() > 0))

    # A known censored-label case distinguishes training point means from evaluation target means.
    pred = torch.zeros(3, 20, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
    truth = torch.zeros_like(pred)
    truth[0, :, 0, 0] = 1
    truth[0, 0, 1, 0] = 9
    truth[1, :2, 0, 0] = 2
    valid = torch.zeros(3, 20, 2, device=device, dtype=torch.bool)
    valid[0, :, 0] = True
    valid[0, 0, 1] = True
    valid[1, :2, 0] = True
    eligible = torch.ones(3, 2, device=device, dtype=torch.bool)
    losses = masked_trajectory_loss(pred, truth, valid, eligible)
    expected = (29/21 + .5 + 2) / 3
    loss_error = abs(losses['loss'].item()-expected)
    losses['loss'].backward()
    empty = masked_trajectory_loss(pred, truth, torch.zeros_like(valid), eligible)
    record('censoring_and_loss_denominators', loss_error <= 1e-12
           and losses['fde_count'].tolist() == [1, 0, 0] and empty['skip_optimizer']
           and torch.isfinite(pred.grad).all(), max_abs=loss_error,
           training_scene0_ADE=29/21, evaluation_scene0_target_mean_ADE=5.,
           no_last_visible_substitution=True, zero_supervision_optimizer_skip=empty['skip_optimizer'])

    loader = SharedPredictionInputs(ROOT/'data/f01d/inputs/train_snr_20.npz')
    # At least three currently eligible vehicles make the routing test non-vacuous.
    import numpy as np
    active_counts = loader.arrays['track_exists'].sum(-1)
    available = (loader.arrays['track_exists'][:, -1] & (loader.arrays['track_exists'].sum(1) >= 3)).sum(1) >= 3
    index = int(np.argmin(np.where(available, active_counts.max(1), 100)))
    sample = loader[index]
    batch = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in sample.items()}
    physical = batch['state_hat']
    standardized = batch['standardized_state']
    exists, detected = batch['track_exists'], batch['detected']
    # Supervision remains outside every model/loader forward call.
    with np.load(ROOT/'data/f01d/labels/train.npz', allow_pickle=False) as labels:
        future = torch.as_tensor(labels['future_position'][index:index+1], device=device)
        label_valid = torch.as_tensor(labels['label_valid'][index:index+1], device=device)
    result['real_cache'] = dict(split='train', snr_db=20, sample_index=index,
                                max_active_nodes=int(active_counts[index].max()), input_hash=loader.input_hash)
    from .quantum import QuantumGraph
    q_model = QuantumGraph(loader.normalization['quantum_scale']).to(device)
    q_features = q_model.forward_history(physical.double(), exists)
    q_features.retain_grad()
    graph_models = [(24, q_model, q_features),
                    (128, ClassicalGNN().to(device), None)]
    for dim, graph, features in graph_models:
        if features is None:
            features = graph(physical, standardized, exists, detected)
            features.retain_grad()
        predictor = TrajectoryPredictor(dim).to(device).eval()
        trainable_llm = [(name, p) for name, p in predictor.llm.named_parameters() if p.requires_grad]
        record(f'actual_{dim}_LoRA_and_base_freeze', len(trainable_llm) == 24
               and all('lora_A' in name or 'lora_B' in name for name, _ in trainable_llm)
               and sum(p.numel() for _, p in trainable_llm) == 294912,
               trainable_LLM_tensors=len(trainable_llm),
               trainable_LLM_parameters=sum(p.numel() for _, p in trainable_llm), rank=8, alpha=16)
        predictor.requires_grad_(False)
        captured = {}
        def llm_input(module, args, kwargs):
            captured['tokens'] = kwargs['inputs_embeds'].detach()
            captured['mask'] = kwargs['attention_mask'].detach()
            captured['position_ids'] = kwargs['position_ids'].detach()
        position_calls = []
        hooks = [predictor.llm.register_forward_pre_hook(llm_input, with_kwargs=True),
                 predictor.llm.wpe.register_forward_hook(lambda module, args, output: position_calls.append(args[0].detach()))]
        output = predictor(physical, standardized, exists, detected, features)
        prediction = output['prediction']
        active = output['origin_eligible'].reshape(-1)
        tokens = predictor.adapter(standardized, exists, detected, features)
        position_ok = len(position_calls) == 1 and torch.equal(captured['position_ids'], torch.arange(20, device=device)[None])
        token_error = (captured['tokens']-tokens[active]).abs().max().item()
        record(f'actual_{dim}_token_and_positions', tokens.shape == (8, 20, 768)
               and prediction.shape == (1, 20, 8, 2) and token_error == 0 and position_ok
               and torch.equal(output['origin_eligible'], batch['origin_eligible']),
               tokens_shape=list(tokens.shape), prediction_shape=list(prediction.shape),
               active_sequences=int(active.sum()), position_embedding_calls=len(position_calls), token_error=token_error)
        for hook in hooks:
            hook.remove()
        # Freeze the full time predictor: gradients must still reach graph features and graph parameters.
        real_loss = masked_trajectory_loss(prediction, future, label_valid, output['origin_eligible'])
        record(f'actual_{dim}_supervision_sidecar', torch.isfinite(real_loss['loss']) and not real_loss['skip_optimizer'],
               valid_future_points=int(real_loss['ade_count'].sum()),
               valid_final_step_targets=int(real_loss['fde_count'].sum()),
               labels_path='data/f01d/labels/train.npz', labels_passed_to_model=False)
        real_loss['loss'].backward()
        feature_grad = features.grad.abs().max().item()
        graph_grads = [p.grad for p in graph.parameters() if p.grad is not None]
        graph_grad = max((v.abs().max().item() for v in graph_grads), default=0.)
        base_grad_free = all(p.grad is None for p in predictor.parameters())
        record(f'actual_{dim}_frozen_LLM_input_backprop', feature_grad > 0 and graph_grad > 0 and base_grad_free
               and all(torch.isfinite(v).all() for v in graph_grads),
               graph_feature_gradient_max=feature_grad, graph_parameter_gradient_max=graph_grad,
               frozen_parameter_gradients_absent=base_grad_free)
        with torch.no_grad():
            # Isolate temporal routing by permuting fixed graph features and the matching local history.
            perm8 = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3], device=device)
            permuted = predictor(physical[:, :, perm8], standardized[:, :, perm8], exists[:, :, perm8],
                                 detected[:, :, perm8], features.detach()[:, :, perm8])['prediction']
            err = (permuted-prediction.detach()[:, :, perm8]).abs().max().item()
            changed = features.detach().clone()
            selected = int(active.nonzero()[0])
            changed[:, :, selected] += .1
            altered = predictor(physical, standardized, exists, detected, changed)['prediction']
            others = torch.arange(8, device=device) != selected
            independence = (altered[:, :, others]-prediction.detach()[:, :, others]).abs().max().item()
            record(f'actual_{dim}_vehicle_routing', err <= 2e-5 and independence <= 2e-5,
                   permutation_abs=err, other_vehicle_change_abs=independence,
                   precision='float32 GPT-2; quantum math independently float64/complex128')
            predictor.head[-1].weight.zero_()
            predictor.head[-1].bias.zero_()
            cv = predictor(physical, standardized, exists, detected, features.detach())['prediction']
            horizon = torch.arange(1, 21, device=device)[None, :, None, None]*.1
            target_cv = physical[:, -1:, :, :2] + horizon*physical[:, -1:, :, 2:]
            target_cv = torch.where(output['origin_eligible'][:, None, :, None], target_cv, 0)
            cv_error = (cv-target_cv).abs().max().item()
            all_missing = torch.zeros_like(exists)
            missing = predictor(torch.zeros_like(physical), torch.zeros_like(standardized), all_missing,
                                all_missing, torch.zeros_like(features))
            record(f'actual_{dim}_estimated_origin_and_missing', cv_error <= 1e-6
                   and not missing['origin_eligible'].any() and missing['prediction'].abs().max().item() == 0,
                   constant_velocity_error=cv_error, all_missing_no_prediction=True)
        del predictor, prediction, output, tokens
        torch.cuda.empty_cache()

    result['passed'] = all(v['passed'] for v in checks.values())
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    report = run(args.device)
    path = ROOT/'reports/f02/interface.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
