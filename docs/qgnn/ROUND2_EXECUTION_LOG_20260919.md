# Round 2 execution log — 2026-09-19

Base: 2882cb4, branch qgnn, /home/dell/YrM/ICCT. Remote Desktop Commander via authorized Windows Git-bundled SSH. Windows system OpenSSH exited 255 without output; Git SSH works.

Read in full: main task, SinD Q0 diagnostic + JSON, SinD migration handoff, current Q0 graph/token/train/eval/dataset implementations, four historical QGNN/Q0 documents, old prediction/quantum.py. Existing untracked reports preserved.

Decisions before experiments:
- SinD train/validation only; test never loaded by this round. Existing sensing-test summary is historical, not a new test access.
- Preserve controlled-sensing boundary: no detector/association failures currently modelled. Do not claim real detection.
- Historical oracle uses future errors and separate ADE/FDE choices; it is not an achievable policy or a quantum-advantage proof.
- Build a graph-conditioned joint ZZ/ZZZ quantum evolution with noncommuting local rotations, shared parameters and continuous inputs. No classical message-passing before the circuit.
- Cap a simulated register at 8 qubits, not the number of scene vehicles. Full graph at N<=8; deterministic history-only ego patches for N>8, same for classical.
- Compare to adaptive pair+triplet classical core with the same input, temporal encoder, GPT-2 and token interface.
- Use exact batched Torch statevector, independently check against PennyLane; not claim hardware speedup.
- Need measured runtime, gradients, permutation, masking, multi-SNR tokens and train/val pilot before final freeze.

Engineering: independent PennyLane state/gradient, N=1/2/4/8/12/16/20 permutation, padding and end-to-end checks PASS. N8 B32 quantum step 0.09541s / peak 3.613GiB. Core 17,296 trainable vs adaptive classical 320,816. Matched shared-GPT2 initialization SHA256 verified equal for all three models.

Pilot v1 launched: seed2026, 0dB, deterministic 4096 train subset, all1880 validation, 12epochs, B32, same LR/loss and initialization. Two separate GPU processes; checkpoint every100steps and each epoch, heartbeat every25steps, atomic summaries. Additional legacy pair+triplet control will use the same new tokenizer/GPT2. These are limited-exposure pilots, not final test results.
