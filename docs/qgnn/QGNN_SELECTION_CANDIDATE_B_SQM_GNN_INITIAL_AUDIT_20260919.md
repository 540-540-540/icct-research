# QGNN Selection Candidate B — SQM-GNN Initial Route Audit — 2026-09-19

## Status

The project is formally back in QGNN architecture selection.

Candidate B is the **SQM-GNN-style quantum message-passing route**.

This document is an initial research audit. It does not freeze an ICCT implementation yet.

## Primary paper

Le Tung Giang, Nguyen Xuan Tung, Trinh Van Chien, Lajos Hanzo, Won-Joo Hwang,
"Scalable Quantum Message Passing Graph Neural Networks for Next-Generation Wireless Communications: Architectures, Use Cases, and Future Directions",
arXiv:2601.18198 / IEEE Network 2026.

Primary source:
https://arxiv.org/abs/2601.18198

## Name and positioning

SQM-GNN = **Scalable Quantum Message Passing Graph Neural Network**.

The key quantum claim is not a VQC replacement of a classical MLP.

The paper explicitly places the message-generation and aggregation/update mechanism inside a parameterized quantum circuit.

## Core paper mechanism

For a center node i:

1. Build a k-neighbor star subgraph.
2. Encode node features into a Node Quantum Register.
3. Encode edge attributes into an Edge Quantum Register.
4. For every sampled neighbor-edge pair, apply shared U_MSG to the neighbor-node and edge registers.
5. The resulting neighbor message states are sequentially coupled to the center-node state through U_UPD.
6. Measure the updated center-node state to obtain its refined embedding.
7. Repeat across QGCL layers.
8. Use a classical task-specific readout after the quantum graph layers.

The center node is intentionally left untouched during U_MSG and is only updated in U_UPD.

The same PQC design/parameters are shared across subgraphs.

## Encoding

The 2026 SQM-GNN paper uses rotation-based angle encoding.

Each scalar feature is mapped to a one-qubit rotation R(x), R in {RX, RY, RZ}.

The authors explicitly move away from the amplitude encoding used in their earlier work.

## Star-subgraph scaling

For one scalar node feature and one scalar edge attribute per slot, the architecture implies:

- center node: 1 qubit
- k neighbor nodes: k qubits
- k center-neighbor edges: k qubits

Therefore:

Q(k) = 2k + 1.

This matches Table II:
- k = 6
- Q = 13 qubits.

For current ICCT model-facing N<=8, using all seven neighbors would imply 15 qubits under this scalar-per-slot interpretation.

This is an inference from the paper architecture and Table II, not an explicit formula printed by the authors.

## Paper complexity claim

The paper measures complexity using cost-function evaluations (CFEs).

For fixed-size k-neighbor stars, it claims:
- one full QGCL forward requires O(N) CFEs;
- classical message passing is described as O(d_bar N);
- training complexity is written as T * B * N * N_shot.

Important caveat:
this is a CFE-level accounting argument, not a demonstrated wall-clock quantum speedup.

It abstracts away:
- state-preparation cost,
- gate count/depth inside U_MSG/U_UPD,
- simulator complexity,
- parameter-shift multiplicative cost,
- and, in the printed training expression, explicit QGCL depth L.

ICCT must therefore benchmark real GPU statevector cost rather than inherit the paper's O(N) claim as an engineering guarantee.

## D2D experiment in the 2026 paper

Training scenario:
- K=20 D2D pairs
- 500m x 500m area
- p_bar=1 W
- 10,000 training realizations
- 10,000 testing realizations
- five random seeds
- PennyLane + PyTorch
- CPU workstation reported.

After 100 epochs:
- SQM-GNN testing sum-rate is approximately 2.6 bps/Hz
- classical GNN is approximately 2.3 bps/Hz
- the paper reports SQM-GNN slightly exceeding the WMMSE stationary point.

Table II:
- SQM-GNN(k=6): 13 qubits
- SQM-GNN parameters: 5,925
- classical GNN parameters: 67,073.

This is not a parameter-matched comparison.

## Generalization table

Both models are trained at K=20, p_bar=1 and tested without retraining.

Reported percentage of WMMSE:

| K | p_bar | GNN | SQM-GNN |
|---|---:|---:|---:|
| 10 | 1 | 102.21% | 107.02% |
| 20 | 1 | 95.16% | 105.12% |
| 40 | 1 | 93.99% | 100.59% |
| 80 | 1 | 93.38% | 97.87% |
| 10 | 2 | 102.00% | 106.88% |
| 20 | 2 | 95.99% | 103.28% |
| 40 | 2 | 93.74% | 99.63% |
| 80 | 2 | 92.01% | 98.26% |

The cross-size result is relevant to scalability/generalization, but it does not by itself isolate a quantum-substrate advantage because the architectures and parameter counts differ.

## Classical baseline

The main classical GNN reference is:

Yifei Shen, Yuanming Shi, Jun Zhang, Khaled B. Letaief,
"Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and Theoretical Analysis",
IEEE JSAC 2021 / arXiv:2007.07632.

This is a legitimate wireless MPGNN baseline, not a toy MLP.

However, for ICCT final selection it is not sufficient by itself.

We still need:
- a functionally matched SQM-style classical message/update control;
- the existing strong project classical baseline;
- and, because SinD shows higher-order value, a higher-order classical reference.

## Official-code audit

No official public repository for the 2026 SQM-GNN paper was found in:
- targeted GitHub repository searches for the title/acronym;
- author-name GitHub repository searches;
- the arXiv article links.

Current status:
**no verified official implementation found**.

Therefore any ICCT implementation must be labelled an adaptation/reconstruction unless the authors' code is later found.

## Critical reproducibility gap

The 2026 paper describes U_MSG and U_UPD semantically but does not provide a sufficiently explicit gate-level ansatz in the text.

It states:
- U_MSG acts jointly on each neighbor-node state and its edge register;
- U_UPD sequentially amalgamates message qubits into the center-node state;
- shared parameters preserve permutation equivariance.

But the paper does not provide enough detail to directly reconstruct:
- the exact sequence of parameterized gates;
- the observable set;
- the exact feature-to-qubit assignment beyond scalar angle encoding;
- or the condition under which sequential U_UPD operations are strictly permutation-equivariant.

This is the central technical uncertainty for an ICCT reproduction.

## Important theoretical question for ICCT

Sequential U_UPD operations share the center qubit.

Generic two-qubit unitaries on (center, neighbor_j) and (center, neighbor_k) do not necessarily commute.

Therefore parameter sharing alone is not obviously sufficient to guarantee order independence.

Before calling our adaptation permutation-equivariant, we must either:
- recover the authors' exact commuting/symmetric update construction;
- or design and prove/test an order-invariant replacement.

This must be treated as a first-class correctness requirement.

## Research lineage

### 2025 ICAIIC precursor

"Quantum Graph Neural Network for Resource Management in Wireless Communication"

The earlier model:
- used amplitude encoding;
- represented full-graph adjacency with N(N-1)/2 qubits;
- used controlled-U graph operations;
- therefore scaled poorly with N.

This motivates the later star-subgraph approach.

### 2025 ICMU / arXiv:2511.15246

"D2D Power Allocation via Quantum Graph Neural Network"

Same core author line.

It introduces:
- stochastic k-neighbor star decomposition;
- shared QGCL processing;
- a 4-node star example encoded with 7 qubits;
- 300 training channel realizations;
- reported approximately 6% sum-rate gain over its GCN baseline.

It is closer to the 2026 SQM-GNN concept, but still does not provide a complete U_MSG/U_UPD gate-level specification.

### 2026 SQM-GNN

The 2026 paper formalizes the route as:
- separate node and edge quantum registers;
- rotation encoding;
- U_MSG + U_UPD;
- fixed-size star subgraphs;
- k=6 / 13-qubit demonstration;
- stronger 10k/10k, five-seed D2D evaluation.

## Fit to SinD / ICCT

Strong fit:
1. node-level output naturally matches per-agent trajectory interaction features;
2. edge attributes are first-class quantum inputs;
3. continuous wireless/physical graphs are much closer to SinD than graph-isomorphism benchmarks;
4. graph message generation and aggregation both belong to the quantum core;
5. shared local PQCs are compatible with N<=8 and bounded simulation.

Major mismatch:
1. original D2D node and edge features are essentially scalar channel gains;
2. ICCT node histories and physical edge features are high-dimensional;
3. original architecture is pairwise-star based;
4. SinD evidence shows explicit triplet/higher-order interaction has additional value;
5. random neighbor sampling can discard the most safety-critical vehicles;
6. no official code and incomplete circuit specification.

## Key opportunity

For current N<=8, ICCT does not need to copy the paper's random neighbor dropping.

A full star with k=N-1<=7 is possible.

Under the scalar-per-node/scalar-per-edge paper interpretation:
- k=7 -> 15 qubits.

This avoids one of the paper's own acknowledged failure modes: information loss from subgraph sampling.

The engineering feasibility still depends on feature encoding, circuit depth and batching.

## Higher-order opportunity

Even though the original SQM-GNN is pairwise-star based, U_UPD repeatedly couples multiple neighbor message states into the same center state.

If the update mechanism is genuinely joint and order-invariant, this can create multi-neighbor dependencies without explicitly enumerating triplets.

This is a potentially attractive bridge to the SinD higher-order evidence.

It must be tested rather than assumed.

## Candidate-B design questions before implementation

1. Can the exact U_MSG/U_UPD ansatz be recovered from related author material or source files?
2. How is permutation equivariance guaranteed under sequential U_UPD?
3. What node/edge dimensionality can be encoded without classical preprocessing becoming the real graph model?
4. Should the first ICCT prototype use all N-1 neighbors instead of stochastic sampling?
5. Can multi-neighbor quantum aggregation capture the observed triplet benefit?
6. What is the minimum strong matched classical control?
7. What is real statevector wall-clock cost on the 2x4090 server?
8. Is the 5,925-vs-67,073 parameter gap an inductive-bias advantage or simply unmatched capacity/design?

## Initial route decision

Candidate B remains **open and worth prototyping**, but should not yet be treated as directly reproducible from the paper.

Recommended next step:
- do a dedicated circuit-reconstruction study of U_MSG/U_UPD and permutation symmetry;
- then build only a mathematical/engineering smoke prototype;
- only after correctness and cost gates pass, preregister a 4096-window SinD comparison.

No full-train, five-SNR or test-set experiment is authorized at this stage.

