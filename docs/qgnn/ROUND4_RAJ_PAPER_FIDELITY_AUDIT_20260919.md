# Round4 Raj Paper Fidelity Audit — 2026-09-19

## 1. Scope

本审计回答两个问题：

1. 当前 ICCT Raj 路线是否已经完整复现 Raj et al. 2026 的 QGNN？
2. “使用 paper-native JohnsonGIN baseline”是否意味着 ICCT 应当复现出论文中的量子优势？

Canonical ICCT server: `/home/dell/YrM/ICCT`, branch `qgnn`.

Raj official repository audited at:
`SnehalRaj/mp-qgnns@851537589d61bcce96130b055e5724291e7ea318`.

Paper:
Snehal Raj et al., *Scalable Message-Passing Quantum Graph Neural Networks in the Weisfeiler–Leman Hierarchy*, arXiv:2606.26873v1.

## 2. Highest-level conclusion

**No: the current ICCT main model is not a verbatim/full reproduction of the paper architecture.**

The paper's mathematical pipeline is:

`V(x) -> A(G) -> W(theta) -> data re-upload, repeated L times -> M(theta_M) -> conditional embedding-register 1-RDM gamma -> task head`.

The current ICCT accepted main `raj_weighted_multij_quantum` preserves several Raj ideas, but:
- uses two parallel j branches (j=2+j=3), whereas the paper studies a chosen fixed j as the WL level;
- uses ICCT continuous physical subset features and MLP amplitude construction;
- uses graph-conditioned weighted Johnson mixing;
- uses compound/RBS-style embedding evolution and data re-upload;
- **does not implement the paper's final joint-register mixer M(theta_M)**;
- **does not use the full post-joint-mixer conditional 1-RDM pipeline**;
- adds ICCT-specific per-agent local fusion and multi-j classical fusion.

Therefore all current results must remain labelled **Raj-inspired ICCT adaptation**.

## 3. Paper architecture versus public repository

A second distinction is equally important: the current public repository does **not** expose one reusable model that matches every block described in the paper.

### 3.1 Paper full mathematical architecture

Main text Sec. I.B, Methods M1, Supplementary Note 1 and Note 11 specify:
- N-qubit node register at Hamming weight j;
- D-qubit embedding register at Hamming weight k;
- hierarchical controlled-Givens loader V(x);
- graph-conditioned Givens adjacency A(G), canonically ordered;
- trainable embedding evolution W(theta), with compound/RBS construction;
- layerwise re-upload;
- joint mixer M(theta_M);
- conditional embedding-register 1-RDM readout after projecting node register;
- joint mixer has node-pair, embedding-pair, and cross-register Givens blocks in Supplementary Note 1;
- the joint state lies in total-weight j+k space after M.

### 3.2 Official CFI / QM9 public implementations

Files:
- `src/mp_qgnns/models/cfi.py`
- `src/mp_qgnns/models/qm9.py`

The public reduced-basis implementations use:
- j-subset lift;
- isomorphism-type features;
- `exp(i alpha A_J)` Johnson mixing;
- MLP re-upload and per-row normalisation;
- real/imag jumping-knowledge pooling;
- task head.

They do **not** call `CompoundPyramidLayer`, a joint-register mixer, or a conditional 1-RDM readout.

### 3.3 Official TSP public implementation

File:
`src/mp_qgnns/models/tsp.py`

It includes:
- coordinate MLP embedding;
- canonically ordered graph-conditioned node-register Givens mixing;
- `CompoundPyramidLayer(D,k)`;
- repeated re-upload;
- classical symmetric edge head.

It still does **not** expose the full paper joint-register `M(theta_M)` + conditional 1-RDM pipeline.

A repository-wide search at the audited commit found no Python implementation named/described as the paper's final joint-register mixer.

## 4. Fidelity matrix

| Component | Paper full | Official CFI/QM9 | Official TSP | ICCT current main |
|---|---|---|---|---|
| j-subset/node register | yes | yes | j=1 node register | yes, j=2+j=3 |
| fixed embedding register (D,k) | yes | implicit reduced basis | yes | yes, D=6,k=3 |
| hierarchical controlled-Givens loader | yes | no; MLP subset encoder | no; coordinate MLP | no; ICCT MLP subset encoder |
| graph-conditioned adjacency A(G) | yes | fixed Johnson A_J | yes | yes, weighted Johnson |
| canonical gate ordering | yes | not needed for exp(i alpha A_J) | yes | matrix Hamiltonian, not paper gate-order implementation |
| compound/RBS W(theta) | yes | no | yes | yes |
| data re-upload | yes | yes, additive MLP | yes | yes, additive MLP |
| final joint-register M(theta_M) | **yes** | **no** | **no** | **no** |
| post-M conditional 1-RDM | **yes** | **no** | **no** | **no in accepted main** |
| single chosen j as WL level | yes | yes | j=1 | **no; multi-j fusion** |
| per-agent local residual/fusion | task dependent | no | no | yes |
| GPT-2 downstream | no | no | no | yes |

Note: `raj_paper.py::RajPaperQGNNCore` and `raj_fidelity_a.py` are closer intermediate research implementations: they include compound dynamics and 1-RDM-style readout. Both explicitly omit the full final cross-register joint mixer. They are not the accepted main model.

## 5. Official minimal reproduction completed

Environment:
- official repo commit: `851537589d61bcce96130b055e5724291e7ea318`
- ICCT Python: Python 3.11.15, torch 2.13.0+cu130, PennyLane 0.45.1
- official scripts used directly with `PYTHONPATH=src`

### Prism vs K3, quick, two seeds

Official QGNN:
- j=1: test acc 0.350
- j=2: test acc 0.350
- j=3: test acc 1.000
- j=3 invariance spread: 1.1e-12

Official JohnsonGIN:
- j=1: 0.350
- j=2: 0.350
- j=3: 1.000
- j=3 invariance spread: 7.2e-13

### CFI(K3), quick, two seeds

Official QGNN:
- j=1: 0.350
- j=2: 0.350
- j=3: 1.000
- j=3 invariance spread: 2.2e-03

Official JohnsonGIN:
- j=1: 0.350
- j=2: 0.350
- j=3: 1.000
- j=3 invariance spread: 2.1e-07

Conclusion:
the official WL-climb behaviour is reproducible in our server environment.

Critically, **JohnsonGIN reproduces the same set-j-WL climb**. The hard expressivity result is therefore a higher-order-j versus 1-WL result, not evidence that the quantum substrate must beat a same-j JohnsonGIN on downstream error.

## 6. Baseline matching audit inside ICCT

Current accepted quantum kind:
`raj_weighted_multij_quantum`
-> `RajWeightedMultiJQGNNCore`.

Current primary baseline kind:
`raj_multij_johnson`
-> `RajMultiJJohnsonCore`.

They share:
- j=2+j=3 subset identities;
- the same ICCT `SubsetFeatureBuilder`;
- same per-agent `MultiJFusion`;
- same GPT-2 / Tokenizer / loss / data protocol.

But they are **not a strict substrate-only matched pair**:
- Quantum uses `WeightedJohnsonMix`: physical edge features directly determine the subset-transition Hamiltonian.
- `RajMultiJJohnsonCore` uses a fixed binary Johnson adjacency for aggregation.

A weighted classical implementation exists historically as `RajWeightedJohnsonGINCore` in `raj_fidelity_a.py`, but it is single-j and is not the current primary multi-j baseline.

This does not invalidate the frozen baseline policy, but it matters for mechanism attribution:
winning `raj_multij_johnson` cannot by itself isolate a quantum-substrate effect, because graph conditioning differs.

## 7. Why the full-train near-tie does not contradict the Raj paper

The Raj paper's strongest supported claims are:
- quantum message passing;
- exact permutation equivariance;
- set-j-WL expressivity controlled by j;
- trainability/scalability in the structured subspace;
- parameter-efficient higher-order representation.

The paper does not establish a general theorem that QGNN must outperform a same-j JohnsonGIN in predictive error.

The paper itself reports a classical message-passing model with lower QM9 error, albeit at much larger parameter count, and a matched classical GCN with lower TSP edge-prediction BCE.

Therefore the ICCT observation
`4096 windows: Q > Johnson by ~2-3%; full 15802: near tie`
does not constitute a failed reproduction of a paper claim.

What remains genuinely untested is:
**whether the full paper two-register pipeline, especially M(theta_M) + conditional 1-RDM, provides useful inductive bias on SinD.**

## 8. Frozen next research sequence

Do not modify the current accepted main model yet.

1. Build an isolated **paper-math fidelity prototype** at single j=3.
2. Implement and unit-test the missing joint-register mixer in total Hamming-weight j+k space.
3. Implement conditional projection back to a j-subset and full embedding 1-RDM readout.
4. Keep the implementation explicitly separate from `raj_weighted_multij_quantum`.
5. Add permutation-equivariance / norm-preservation / zero-angle / projection tests before any trajectory training.
6. Add a **weighted multi-j Johnson diagnostic** later to isolate graph-conditioning versus quantum-substrate effects; this is an additional diagnostic, not a rewrite of the frozen baseline policy.
7. Only after the fidelity prototype passes tests, run a small 4096-window paired diagnostic before any new full-train experiment.

No five-SNR or final-test use is authorized by this audit.


## 9. Paper-internal implementation ambiguity that must be frozen before coding

The paper itself contains an ambiguity about the exact placement/content of `M(theta_M)`:

- Main-text Fig. 1 and the composition formula write a **single final** mixer:
  `readout o M o (W o A o V)^L`.
- Methods M1 also describes trainable iterations followed by the joint-register mixing layer.
- Supplementary Note 11 resource accounting instead states one `M` **per iteration**, repeated `L` times.
- Supplementary Note 1 defines `M` as three blocks: node-pair + embedding-pair + cross-register, with parameter count `C(N,2)+C(D,2)+N*D`.
- Supplementary Note 11's resource table counts the joint mixer mainly as the `N*D` cross-register block.

Therefore an ICCT implementation cannot honestly be called a verbatim full-paper reproduction without choosing an interpretation. The prototype must expose this choice in config and tests rather than silently guessing.

Recommended first interpretation for an isolated prototype:
- follow the main-text composition: one final `M` after `L` rounds;
- implement all three Supplementary Note 1 blocks;
- keep a second `cross_only_per_layer` ablation only as a later paper-ambiguity check;
- do not promote either interpretation to the accepted main model before reproduction diagnostics.

## 10. Task-interface mismatch: Raj j>=2 is graph-level, ICCT requires per-agent output

This is not a minor implementation detail.

In the paper's j>=2 CFI/QM9 uses, the final subset-conditioned features are pooled over j-subsets for a graph-level target. By contrast, ICCT must output one interaction representation per vehicle so that the fixed trajectory stack can produce per-agent futures.

Therefore any j=2/j=3 SinD adaptation needs an additional map from subset rows back to agents (for example incidence pooling). That map is not part of the paper's j>=2 task interface and can change which higher-order information survives.

Consequently, even a prototype with the missing joint mixer restored is still a **paper-math faithful task adaptation**, not a zero-change reproduction of the paper experiment.

## 11. Why j-WL advantage may be weak on SinD

The paper's strongest expressivity demonstration deliberately uses graph pairs that lower-order WL cannot distinguish. SinD is qualitatively different:
- node/edge inputs contain continuous position, velocity, distance, closing speed, TCPA/DCPA and related physical features;
- these continuous attributes often break graph symmetries before higher-order WL refinement becomes necessary;
- the target is continuous future motion, not graph identity or a graph-level molecular scalar.

Therefore the relevant unanswered question is not merely whether j=3 is more expressive in principle. It is whether the SinD prediction problem actually contains a meaningful population of examples for which lower-order interaction representations collapse distinct future-relevant situations and the higher-order subset representation resolves them.

This must be diagnosed empirically before interpreting any QGNN-vs-Johnson margin as a WL advantage.

## 12. Public-code provenance check

The official repository has no tags and only one public branch (`main`). Full history was fetched. The initial public commit is `34acb1a`; current audited commit is `8515375`.

A history-wide source search found no Python implementation of the paper's final joint-register mixer / conditional 1-RDM pipeline in the initial public tree either. The current public repository should therefore not be treated as a complete executable transcription of every block in the paper's mathematical diagram.
