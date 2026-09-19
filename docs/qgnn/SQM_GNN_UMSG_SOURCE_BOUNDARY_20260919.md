# SQM-GNN Candidate B — U_MSG Source Boundary and ICCT Design Entry — 2026-09-19

## Purpose

Freeze what the SQM-GNN sources actually specify about the quantum message unitary U_MSG before designing an ICCT implementation.

Primary sources:
- Le Tung Giang et al., arXiv:2601.18198 (2026)
- Tung Giang Le et al., arXiv:2511.15246 (2025 precursor)

This document separates source-supported architecture from ICCT inference/design.

## 1. What the 2026 paper explicitly states

For each center-node star subgraph:
- node features are encoded into a Node Quantum Register;
- edge attributes are encoded into an Edge Quantum Register;
- rotation-based angle encoding is used;
- each scalar feature is mapped to a single-qubit rotation R(x), R in {RX, RY, RZ};
- U_MSG is applied jointly to each neighbor-node state and its corresponding edge-register state;
- U_MSG generates a quantum message represented on the neighbor-node qubit;
- the center-node state is deliberately untouched during U_MSG;
- the same U_MSG design/parameters are shared over neighbor-edge pairs;
- U_UPD later couples message-bearing neighbor qubits to the center node;
- after the update phase, the center node is measured to produce the next node embedding.

This establishes the semantic role:

m_{j->i}^Q = U_MSG(h_j, e_ij)

with the center state excluded from message generation.

## 2. What the source does NOT specify

The public 2026 paper does not provide a complete gate-level ansatz for U_MSG.

It does not uniquely specify:
- the exact RX/RY/RZ choice for every feature;
- the exact trainable one-qubit rotations;
- the exact entangling gate family;
- gate ordering;
- number of U_MSG layers/repetitions;
- exact parameter count attributable to U_MSG;
- exact observable/measurement of the message qubit;
- how a multi-dimensional node/edge feature vector is compressed when there is one qubit per scalar slot.

No verified official source repository was found.

Therefore ICCT must not call any concrete U_MSG gate sequence an "official reproduction" unless new source material is later found.

## 3. 2025 precursor

The 2025 D2D precursor already uses:
- N overlapping k-neighbor star subgraphs;
- one shared PQC per star;
- quantum encoding;
- a shared unitary processing center node, leaf node, and connecting edge;
- quantum-compatible aggregation;
- a 4-node example represented by a 7-qubit PQC.

But it also does not provide a fully reproducible gate-level U_MSG/U_UPD decomposition.

The 2026 paper is therefore best read as a clearer architectural formalization of the same research line, not a code-complete implementation paper.

## 4. Consequence for ICCT

We should preserve the source-supported invariants:

1. Graph edge attributes are first-class quantum inputs.
2. Message generation is quantum-native.
3. The center state is untouched during U_MSG.
4. U_MSG is shared across neighbor-edge pairs.
5. U_UPD performs the center aggregation/update in the quantum domain.
6. Node-level output is measured only after the update phase.

Everything beyond these constraints is an ICCT design choice and must be named as such.

## 5. First ICCT U_MSG design question

For SinD, one neighbor relation contains much more than one scalar:
- neighbor motion/history representation;
- relative position and velocity;
- distance;
- closing rate;
- TCPA/DCPA;
- risk/history-change features.

Therefore a literal "one scalar = one qubit" encoding is unsuitable without either:
- many more qubits;
- aggressive feature compression;
- or data re-upload over a small fixed register.

The first architecture decision is thus not the entangling gate.

It is:

**How should a high-dimensional SinD neighbor-edge relation be encoded into a small U_MSG quantum register without a classical network already performing the graph reasoning?**

## 6. Candidate encoding families for later evaluation

Not yet selected:

A. one-qubit node + one-qubit edge scalar summary
- closest to paper spirit
- lowest qubits
- severe information compression risk

B. small multi-qubit node register + small multi-qubit edge register
- preserves more continuous information
- more faithful to SinD complexity
- moderate qubit cost

C. fixed small registers with data re-upload
- keeps qubit count bounded
- richer feature access
- increases depth and optimization risk

No option is selected by this document.

## 7. Stop condition before U_UPD

Do not design U_UPD until the U_MSG input representation and message state are explicit.

For any U_MSG candidate, require:
- graph edge features directly alter quantum evolution;
- center node remains untouched;
- parameter sharing over neighbor identities;
- no classical GNN/message MLP before the quantum core;
- finite gradients;
- a matched classical message-function control can be defined;
- real GPU cost is measurable on the ICCT server.


