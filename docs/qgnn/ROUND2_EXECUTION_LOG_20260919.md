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

## V2 refinement declared before validation
V1 commit269544d. Q ADE/FDE=0.601817/1.255976; adaptive classical=0.583580/1.205876; legacy higher-order=0.610172/1.138172. Same4096/1880/12epoch exposure. No quantum victory.
Train-only64scene diagnostic: local RMS0.998 vs quantum readout RMS0.0629; pair std0.0058-0.0178, triple std0.0010-0.0043, singleton std0.167-0.521. Entangler removal worsens ADE only0.26%, ZZZ removal0.037%. Marginal purity mean0.99218: weak entanglement.
V2 keeps joint ZZ/ZZZ structure: four independently conditioned8qubit channels, weighted phase-energy normalization max(1,sqrt(sum w^2 / active_N)), asinh readout scaling .02(pair)/.005(triple). No extra observations. Classical control strengthened to four-head pair and four-head rooted-triplet attention; width64/three layers unchanged. Shared GPT2, tokens, seed, subset and exposure unchanged. A second refinement requires a new mechanism diagnosis.

## Continuation audit and final permitted refinement (V3)
Recovered server state at 53844e5; no active previous training process. V2 finished, Q0.608360/1.265798 versus classical0.581917/1.198834: still no advantage. Independently reran V2 mechanism audit: deleting entanglers degrades ADE to0.658697; deleting ZZZ to0.640063. Mean marginal purity0.92444. This supports functional dependence, NOT a quantum win or a retrained causal claim.
V3 is the second and last structural refinement: keep 4x8-qubit, 3-round graph-conditioned ZZ/ZZZ evolution; retain relation identities in cumulant-weighted continuous message readout instead of only per-node mean/RMS. All learned cross-agent routing coefficients still originate in the joint quantum circuit; no classical GNN precedes it. Same own-history encoder, bounded patches, GPT2, tokenizer, loss and data exposure. Keep the strong 4-head/3-layer adaptive classical comparator; do not weaken it. First compare on the identical4096train/1880val/12epoch/seed2026 protocol. No further architecture search this round, and test remains closed.
Shared decoder4m saturation is separately documented by train-only five-SNR audit; do not silently attribute a future shared-interface fix to the quantum core.

## Shared five-SNR output interface freeze
After the isolated V3 pilot, freeze the common correction as 16*tanh(z/4) instead of4*tanh(z), applied identically to all interaction cores. The derivative atzero remains4 and initial prediction remainsCV. Justification is the prior train-only five-SNR audit:4m causes an unavoidable -10dB FDE floor0.15336m;16m covers all observed train residual coordinates. This is a common-interface robustness repair, not a third quantum architecture refinement or claimed quantum improvement. Historical checkpoints explicitly reconstructcap4. Final controlled pilot: full15802train, all1880val,0dB,seed2026,20epochs, B32, same optimizer/scheduler/order/LLM; no test.

## Final acceptance
Primary RC-HQGNN-v3 frozen. Full0dB/seed2026/15802train/1880val/20epoch: Q0.51094460/1.12605076 (1146.55s), matched classical0.49146503/1.07928883 (891.65s). Q is worse3.96%ADE/4.33%FDE globally; predefined closing>=15 subset140 windows improves4.29%/6.35%. No quantum advantage claim. Two architecture refinements exhausted.
Installed padding/root-duplication boundary fix8305cf4 after experiments; N8 prediction reload identical. Independent PennyLane observables and state/gradient checks PASS. Interrupted/resumed64-sample smoke equals uninterrupted mutable parameters exactly; ADE/FDE equal. All jobs completed. Test closed. Canonical freeze and compact results identify remaining multi-seed/five-SNR/finite-shot validation.
