# Round4 Raj Paper-Math SinD Prototype Preregistration — 2026-09-19

## Purpose

Build and validate an isolated single-j Raj-style two-register prototype that restores the major mathematical blocks missing from the accepted ICCT main model, before any further full-train confirmation.

This prototype is for **mechanism/fidelity diagnosis**, not for replacing the accepted main model and not for producing a favourable result by architecture search.

## Naming boundary

The prototype must be described as:

**Raj paper-math faithful SinD adaptation**

It must **not** be described as:
- an official-code reproduction;
- a verbatim full-paper reproduction;
- proof of quantum advantage.

Reasons are frozen in `ROUND4_RAJ_PAPER_FIDELITY_AUDIT_20260919.md`.

## Frozen architecture target

Single branch only:
- node-register particle number: j=3
- embedding register: D=6, k=3
- trainable rounds L=3
- no j=2 branch
- no multi-j fusion
- no classical/quantum router
- no RC-HQGNN controller
ICCT infrastructure retained:
- SinD history-only sensing input
- 20 history frames
- N<=8 bounded graph interface
- same final 64D graph-feature contract
- same fixed GPT-2 / Tokenizer / trajectory objective when training is later enabled
- test split closed

## Frozen block interpretation

The prototype follows the main-text composition:

`readout o M(theta_M) o [ W(theta) o A(G) o V(x) ]^L`

with one final joint mixer after L rounds.

### V(x): controlled feature loader

Because the public repository does not ship the paper's full hierarchical loader, the reduced-basis SinD adaptation will implement its structural role explicitly:
- one symmetric continuous feature vector per j-subset;
- a shared feature map produces non-negative amplitudes over the C(D,k) embedding basis for every valid j-subset;
- the initial C(N,j) x C(D,k) amplitude table is normalised **globally**, so it represents one valid two-register quantum state rather than independently normalised subset rows;
- each subset's conditional embedding target is obtained by row normalisation of that same amplitude table;
- later re-upload is controlled by the node-subset basis: for each subset separately, a differentiable orthogonal Householder map sends a fixed embedding reference basis state to that subset's conditional target and acts only on the embedding axis; therefore re-upload never performs cross-subset message passing;
- each controlled Householder map acts unitarily inside the fixed-k embedding subspace and can be decomposed into Givens rotations;
- no additive MLP state injection.

This is a reduced-basis functional adaptation of the hierarchical controlled-Givens loader, not a claim of gate-for-gate identity to reference [42].
### A(G): equivariant graph-conditioned node mixing

- use all active node pairs;
- graph weights are history-only continuous physical interaction weights from the existing frozen `physical_graph` interface;
- one node-qubit Givens per active pair;
- gate order is canonical and graph-derived;
- Givens endpoint orientation must also be graph-derived, never numeric-label-derived;
- no future information.

### W(theta): embedding evolution

- D=6, k=3 compound/Cayley evolution;
- one trainable W per round;
- fixed-particle-number preserving.

### M(theta_M): joint-register mixer

Use the already isolated `JointRegisterMixer`:
- total shell H_(N+D)^(j+k);
- node-pair block;
- embedding-pair block;
- cross-register block;
- canonical rank parameter assignment;
- one final M after L rounds.

The alternative Supplementary-Note-11 reading ("M per iteration", cross-only resource count) remains an explicitly separate future ambiguity ablation and is not allowed in this first prototype.
### Readout

- project the joint state onto each j-subset / k-embedding sector;
- normalise the conditional embedding state;
- compute full complex embedding-register 1-RDM;
- map each subset's 1-RDM through a shared readout;
- convert subset features to per-agent features by incidence pooling.

The final incidence pooling is an ICCT task adaptation: Raj j>=2 experiments use graph-level subset pooling, whereas trajectory prediction requires one representation per vehicle.

No additional local-history residual is added inside this graph core in the first prototype; GPT-2 already receives the motion history separately.

## Correctness gates before any training

All must pass:
1. loader target preparation error <= 1e-10 on a small deterministic case;
2. loader/re-upload norm preservation <= 1e-10;
3. adjacency norm preservation <= 1e-10;
4. adjacency node-permutation equivariance <= 1e-10 on unique canonical keys;
5. joint mixer identity test <= 1e-10 at zero angles;
6. joint mixer norm preservation <= 1e-10;
7. joint mixer node-permutation equivariance <= 1e-10;
8. non-zero cross-register mixing moves non-zero mass outside the factored (j,k) sector;
9. 1-RDM trace equals k on accepted conditional states within numerical tolerance;
10. gradients are finite/non-zero through V/A/W/M/readout;
11. graph output shape is [B,N,64], finite, and zero on masked agents;
12. no test dataset is constructed.
## Experimental sequence after correctness

No full train is authorised by this preregistration.

After all correctness gates pass:
1. tiny forward/backward smoke on SinD;
2. short overfit/sanity diagnostic;
3. write a separate 4096-window experimental preregistration before any paired 20-epoch result is read.

First 4096 paired comparator, if authorised later:
- Quantum: the frozen single-j=3 paper-math adaptation;
- Classical: single-j=3 paper-native JohnsonGIN adaptation (`raj_johnson`);
- same downstream, same seed, same data, same training exposure.

A weighted classical diagnostic will be added separately to isolate graph-conditioning from quantum-substrate effects. It does not replace the frozen paper-native baseline policy.

## Stop rule

If the prototype cannot satisfy the mathematical correctness/equivariance gates, do not train it.

If it passes correctness but fails a preregistered 4096 paired diagnostic, freeze the result before modifying V/A/W/M/readout.
