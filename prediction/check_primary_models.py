"""Real training-input and label-sidecar check for the A04 public model factory."""
import json
from pathlib import Path
import numpy as np
import torch
from frontend.symbol_dataset import SharedPredictionInputs
from .model import build_model
from .classical import GNNGraph
from .temporal import masked_trajectory_loss

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.manual_seed(20260912)
    device = 'cuda:1'
    loader = SharedPredictionInputs(ROOT/'data/f01d/inputs/train_snr_20.npz')
    index = 97
    batch = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in loader[index].items()}
    args = [batch[k] for k in ('state_hat', 'standardized_state', 'track_exists', 'detected')]
    with np.load(ROOT/'data/f01d/labels/train.npz', allow_pickle=False) as f:
        truth = torch.as_tensor(f['future_position'][index:index+1], device=device)
        valid = torch.as_tensor(f['label_valid'][index:index+1], device=device)
    checks = {}
    for name in ('QGNN', 'GNN'):
        model = build_model(name, loader.normalization, device).eval()
        # cuDNN inference-mode GRU does not retain backward workspaces; this is local validation only.
        with torch.backends.cudnn.flags(enabled=False):
            output = model(*args)
            losses = masked_trajectory_loss(output['prediction'], truth, valid, output['origin_eligible'])
            losses['loss'].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        passed = output['prediction'].shape == (1, 20, 8, 2) and torch.isfinite(output['prediction']).all()
        passed = bool(passed and torch.equal(output['origin_eligible'], batch['origin_eligible'])
                      and grads and all(torch.isfinite(g).all() for g in grads)
                      and any(g.abs().max() > 0 for g in grads))
        if name == 'QGNN':
            passed &= model.graph.__class__.__module__ == 'prediction.quantum'
            passed &= model.graph.theta.grad is not None and model.graph.theta.grad.abs().max().item() > 0
            implementation = 'PennyLane QuantumGraph plus numeric GPT-2 temporal model'
        else:
            passed &= isinstance(model.graph, GNNGraph) and hasattr(model.temporal, 'llm')
            passed &= not any(isinstance(module, torch.nn.GRU) for module in model.modules())
            implementation = 'Original Plain graph-attention core and common numeric GPT-2 temporal model'
        llm_config = model.temporal.llm.config
        lora_count = sum(p.numel() for p in model.temporal.llm.parameters() if p.requires_grad)
        passed &= (llm_config.n_layer, llm_config.n_head, llm_config.n_embd) == (12, 12, 768) and lora_count == 294912
        checks[name] = dict(passed=passed, display_name=model.display_name, implementation=implementation,
                            prediction_shape=list(output['prediction'].shape),
                            parameter_count=sum(p.numel() for p in model.parameters()),
                            common_LLM=[llm_config.n_layer, llm_config.n_head, llm_config.n_embd],
                            LLM_trainable_parameters=lora_count,
                            gradient_max=max(g.abs().max().item() for g in grads))
        print(json.dumps(checks[name]), flush=True)
        del model, output, losses, grads
        torch.cuda.empty_cache()
    report = dict(passed=all(c['passed'] for c in checks.values()), checks=checks,
                  split='train', sample_index=index, snr_db=20, device=device, optimizer_steps=0)
    (ROOT/'reports/f02/primary_models.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
