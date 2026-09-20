# Raj PennyLane P2 Self-stage training protocol (2026-09-20)

## Purpose

Train only the own-history LLM path in

`Y_hat = Y_CV + delta_self + gate * delta_interaction`.

During this stage Raj is frozen and is not executed. The interaction interface
is also frozen. Consequently, no other-vehicle relation can enter the Self LLM.

## Frozen data and optimization

- Dataset: SinD Route-B sensing cache, train split for updates and validation
  split for checkpoint observation.
- Formal SNR: 0 dB.
- Test split: never constructed.
- Seed: 2026.
- Train/validation samples: 15,802 / 1,880.
- Epochs: 20; batch size: 32.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `2e-4`.
- GPT-2 LoRA learning rate: one quarter of the main learning rate.
- Scheduler: cosine decay to 8% of the initial learning rate.
- Gradient clipping: 3.0.
- Training loss:
  `mean(ADE_scene + 0.5 * FDE_scene) + 0.035 * token_cross_entropy`.

ADE and FDE remain separately reported. Their coordinate component is also
used as the checkpoint-selection score so optimization and selection do not
pull in different directions.

## Checkpoint rule

The run retains one `best.pt`, selected by the lowest validation
`ADE + 0.5 * FDE`. This is a balanced checkpoint rule over the two required
trajectory metrics. ADE and FDE are still stored separately in every epoch and
must be reported separately; the combined value is only an internal ordering
rule, not a third paper metric.

## Temporal baselines

The Self-stage comparison contains exactly four trained models:

- `llm`: four-layer GPT-2 with LoRA and the Self residual head;
- `lstm`: two-layer LSTM encoder and two-layer future-query decoder;
- `transformer`: four-layer classical Transformer encoder with 20 learned
  future queries;
- `tcn`: four causal residual TCN blocks with dilations 1, 2, 4, and 8.

The three classical baselines use only each vehicle's own displacement/velocity
history. All four models share the same SinD split, sensing input, CV residual,
16 m bounded correction, timestamps, seed, exposure, optimizer family, batch
size, validation set, and single-checkpoint selection rule. CV is additionally
reported as the common zero-initialized epoch-0 physical reference. If a trained
model never improves the combined validation criterion, `best.pt` remains this
epoch-0 checkpoint. No GRU baseline is included.

## Commands

One-epoch smoke, not performance evidence:

```bash
CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python \
  scripts/train_raj_residual_self.py \
  --model llm \
  --epochs 1 --batch-size 4 --train-limit 8 --val-limit 8 \
  --run-dir reports/qgnn/raj_pennylane_p2/self_smoke_seed2026
```

Formal run, to be started only after reviewing the smoke result:

```bash
CUDA_VISIBLE_DEVICES=0 /home/dell/YrM/envs/ICCT/bin/python -u \
  scripts/train_raj_residual_self.py \
  --model llm \
  --run-dir results/qgnn/raj_pennylane_p2/self_0db_seed2026 \
  --max-seconds 7200
```

Resume the same frozen run with the same arguments plus `--resume`. Increasing
`--max-seconds` does not alter the experiment contract.

For the formal four-model comparison, run `llm` and `lstm` on GPU 0 and
`transformer` and `tcn` on GPU 1 as four concurrent processes. Their measured
batch-32 peak reserved memory is about 3.89, 0.31, 0.74, and 0.19 GiB,
respectively, so the paired processes remain far below either 24 GiB limit.

## Gate after completion

A completed process is not automatically a successful Self model. Review:

- ADE and FDE separately across all epochs;
- the selected epoch's ADE and FDE together, rather than the combined score alone;
- training-loss and validation-metric divergence;
- whether either validation metric is still improving at epoch 20;
- checkpoint replay before Interaction training.

No 4096 screening, full Raj training, multi-seed run, multi-SNR run, or test
evaluation is authorized by this protocol.
