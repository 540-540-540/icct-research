# Round4 Raj Paper-Math 4096 Paired Diagnostic Findings — 2026-09-19

## 1. Preregistered decision

The single-j paper-math Raj adaptation **fails** the preregistered primary transfer gate against the existing single-j paper-native JohnsonGIN adaptation.

Protocol was frozen at commit `0d9f164` before either paired result was available.

## 2. Result

| Model | ADE | FDE | J = ADE + 0.5 FDE | Best epoch |
|---|---:|---:|---:|---:|
| Raj paper-math QGNN, j=3 | 0.810536438 | 1.707538617 | 1.664305747 | 1 |
| Raj JohnsonGIN, j=3 | 0.550160027 | 1.155770682 | 1.128045368 | 20 |

Quantum relative gain over JohnsonGIN:
- ADE: -47.327%
- FDE: -47.740%
- J: -47.539%

Primary gate (both ADE and FDE lower): **FAIL**.
Strong gate (primary + J gain >=1%): **FAIL**.

This is not a small statistical margin. The paper-math adaptation is substantially worse under this SinD transfer protocol.

## 3. Integrity

Both arms:
- SinD 0 dB
- seed2026
- deterministic 4096 train windows
- all 1880 validation windows
- 20 epochs
- batch32
- LR 3e-4
- same GPT-2 / tokenizer / trajectory+token objective
- same correction cap 16
- test closed

Hashes are identical across arms:
- train-index SHA256: `dd7b0bd5ff75e728dcca1dd61abc913595f70953ae6c3265271ae9806d53bd68`
- downstream trainable initialization SHA256: `0d35048bfb3385d55aaab30cca077f76a7bc52701da744e786d67462a276b57f`

Both completed 20/20 epochs without OOM, deadlock, non-finite loss or early termination.

Graph parameters:
- paper-math Quantum: 56,763
- JohnsonGIN: 81,248

## 4. Learning-curve evidence

Quantum:
- epoch1: train 1.592729 / J 1.664306
- epoch2: train 1.550440 / J 1.671087
- epoch5: train 1.383838 / J 1.736023
- epoch10: train 1.162342 / J 1.731212
- epoch15: train 1.046104 / J 1.783571
- epoch20: train 0.999255 / J 1.775574

JohnsonGIN:
- epoch1: train 1.556452 / J 1.559657
- epoch2: train 1.413379 / J 1.501549
- epoch5: train 1.145123 / J 1.383588
- epoch10: train 0.843968 / J 1.220443
- epoch15: train 0.701725 / J 1.151710
- epoch20: train 0.642918 / J 1.128045

The Quantum training objective decreases steadily, but validation performance is best at epoch1 and then degrades. Therefore the main failure mode is **not insufficient optimization time**. It is consistent with a generalization / representation-interface mismatch.

## 5. What this result does NOT prove

It does **not** prove that Raj's quantum substrate is 47% worse than JohnsonGIN.

The comparison is not substrate-only:
- the new paper-math Quantum uses graph-conditioned weighted node-register Givens mixing;
- existing `raj_johnson` uses fixed binary Johnson aggregation;
- existing Johnson readout includes an ICCT `local(own)` residual branch;
- the paper-math Quantum deliberately omitted that local graph residual to stay closer to the paper;
- SinD j>=2 requires an ICCT-specific subset-to-agent incidence readout absent from the paper's graph-level j>=2 tasks;
- the public Raj repository does not provide the paper's complete final joint-mixer + conditional-1RDM task implementation, so the missing blocks necessarily required a documented interpretation/adaptation.

Accordingly, this experiment tests one frozen paper-math SinD adaptation, not the universal value of the Raj quantum construction.

## 6. Relation to earlier Raj adaptations

The observed ordering is informative:

- simpler ICCT Raj j=3 subset QGNN (historical 12-epoch diagnostic): approximately ADE 0.6096 / FDE 1.2533;
- prior `raj_paper_quantum` intermediate (compound evolution + conditional 1-RDM, no final joint M), 4096/12e: ADE 0.7655 / FDE 1.5727 / J 1.5518;
- current paper-math prototype (restored controlled loader + final joint M), 4096/20e: ADE 0.8105 / FDE 1.7075 / J 1.6643.

More paper-math machinery has **not** monotonically improved the SinD transfer. This strongly argues against blindly adding fidelity blocks and calling that progress.

## 7. Immediate research consequence

Freeze this negative result. Do not tune V/A/W/M/readout from it.

The next preregistered diagnostic must separate at least two confounds:
1. weighted graph conditioning versus fixed Johnson aggregation;
2. local-history residual / task-interface support versus pure subset quantum representation.

A weighted single-j classical comparator without the Johnson `local(own)` advantage is the next causal reference. The existing `raj_paper_johnson` family is a suitable first comparator because it uses weighted Johnson dynamics and no local residual.

Only after that diagnostic should we decide whether:
- the failure is mainly due to the guessed full-paper quantum blocks;
- the failure is primarily an ICCT per-agent readout/interface issue;
- or the Raj paper's useful advantage simply does not map to this continuous trajectory-prediction problem.

