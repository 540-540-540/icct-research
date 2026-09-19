# Round4 Raj Paper-Math Prototype Findings — 2026-09-19

## 1. Decision

A first isolated **Raj paper-math faithful SinD adaptation** has been implemented and passed mathematical correctness gates.

This is not promoted to the accepted QGNN main model. The accepted main remains `raj_weighted_multij_quantum`.

The prototype exists to answer a narrower scientific question:

> Does restoring the missing two-register / joint-mixer / conditional-1RDM structure change the behaviour of the Raj route on SinD?

No ADE/FDE superiority claim is made at this stage.

## 2. Source audit completed

Official repository audited:
- `SnehalRaj/mp-qgnns`
- commit `851537589d61bcce96130b055e5724291e7ea318`

The official WL-climb pipeline was reproduced directly on the server.

Prism vs K3, quick / 2 seeds:
- QGNN: j=1 0.350, j=2 0.350, j=3 1.000
- JohnsonGIN: j=1 0.350, j=2 0.350, j=3 1.000

CFI(K3), quick / 2 seeds:
- QGNN: j=1 0.350, j=2 0.350, j=3 1.000
- JohnsonGIN: j=1 0.350, j=2 0.350, j=3 1.000

This confirms that the paper's hard WL-climb signal is a higher-order-j effect shared by the same-j JohnsonGIN comparator; it is not evidence that the quantum substrate must beat JohnsonGIN on downstream prediction error.
## 3. Fidelity boundary

The paper full mathematical pipeline contains:
- hierarchical loader V(x)
- graph-conditioned adjacency A(G)
- embedding evolution W(theta)
- data re-upload
- joint-register mixer M(theta_M)
- conditional embedding 1-RDM readout

The public CFI/QM9 implementation does not expose all of those blocks as one executable model. The public TSP implementation includes graph-conditioned adjacency and compound evolution, but still does not expose the final paper joint-register mixer + conditional 1-RDM pipeline.

The current accepted ICCT model also omits the full joint-register mixer.

The first prototype therefore implements a paper-math adaptation rather than claiming an official-code reproduction.

## 4. New isolated modules

Added:
- `prediction/qgnn_paper_native/raj_joint.py`
- `prediction/qgnn_paper_native/raj_paper_math.py`

New model kind:
- `raj_paper_math_quantum`

Core structure:
- single j=3 branch
- D=6, k=3
- 3 rounds
- globally normalised factored quantum state
- subset-controlled orthogonal feature re-upload
- graph-conditioned node-register Givens message passing
- compound/Cayley embedding evolution
- one final total-weight j+k joint-register mixer
- conditional projection
- full complex embedding 1-RDM
- incidence-weighted subset-to-agent readout
- output contract [B,N,64]

No multi-j fusion and no local-history residual are used inside this graph core.
## 5. Joint-register primitive verification

Result:
`reports/qgnn/round4_raj_joint_mixer_unit_20260919.json`

Interpretation frozen for this prototype:
- one final M after L rounds
- all three Supplementary-Note-1 blocks:
  - node-node
  - embedding-embedding
  - node-embedding cross block

Small deterministic test:
- identity error at zero angles: 0
- conditional projection probability error: 0
- conditional state error: 0
- norm error: 0
- node-permutation equivariance error: 0
- non-zero cross-register leakage mass: 0.0286668
- finite non-zero gradients for node / embedding / cross parameters

Production dimensions:
- N=8
- j=3
- D=6
- k=3
- factored state: 56 x 20 = 1120 amplitudes
- total-weight joint shell: C(14,6) = 3003 amplitudes

An initial orientation implementation failed the permutation test because it oriented Givens gates by numeric node labels. That bug was fixed by graph-derived canonical orientation before the PASS result was frozen.
## 6. Full paper-math core correctness

Result:
`reports/qgnn/round4_raj_paper_math_unit_20260919.json`

Passed:
- loader global norm error: 0
- conditional embedding norm error: 1.11e-16
- feature target preparation error: 1.39e-16
- re-upload norm error: 1.78e-15
- adjacency norm error: 1.11e-16
- adjacency permutation equivariance error: 0
- 1-RDM trace-k error: 1.78e-15
- full-core node permutation equivariance error: 1.33e-15
- output shape: [1,8,64]
- masked-agent output: exactly zero
- finite, non-zero gradient flow through V / A / W / M / readout

These checks use no test split.

## 7. Real SinD tiny smoke

Result:
`reports/qgnn/round4_raj_paper_math_sind_smoke_20260919.json`

One real 0 dB train sample with 8 active vehicles:
- full GPT-2 forward/backward completed
- prediction shape: [1,20,8,2]
- graph shape: [1,8,64]
- finite predictions
- forward: ~0.623 s
- backward: ~0.149 s
- peak allocated GPU memory: ~0.398 GiB
- graph trainable parameters: 56,763

The random-initialisation ADE/FDE from this smoke is not a model-performance result and must not be compared with trained baselines.
## 8. Scientific consequences

Three distinctions are now frozen:

1. **Paper mathematical architecture != public task code != ICCT current main.**
2. **Raj j-WL expressivity != quantum-vs-JohnsonGIN predictive superiority.**
3. **SinD j>=2 requires a subset-to-agent readout absent from the paper's graph-level j>=2 tasks.**

A further baseline issue was identified:
- accepted Quantum `raj_weighted_multij_quantum` uses graph-conditioned weighted Johnson mixing;
- accepted `raj_multij_johnson` uses fixed binary Johnson aggregation.

Thus the current main paired comparison is valid under the frozen paper-native policy, but it is not a strict substrate-only causal comparison.

## 9. Next step

Before any 4096-window performance comparison:
- run a small fixed-set overfit/sanity diagnostic for the new prototype;
- verify loss can decrease and GPU/runtime remain tractable;
- then write a separate paired 4096 preregistration.

No full-train, five-SNR, or test-set experiment is authorised yet.
