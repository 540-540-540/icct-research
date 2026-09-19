# Round 3 execution log — 2026-09-19

Base server commit dc5daca (ahead of remote by2); no reset/pull. Required task, new diagnostic MD+summary+full1880 rows, R2 freeze/log, all qgnn_final sources, train/evaluation scripts and SinD/Q0 handoffs read. Existing untracked data/reports preserved. Test closed.

## A. Pre-registered candidate decision
A: history scene+relation-conditioned ZZ/ZZZ multipliers, channel/round specialization; no intermediate feedback. Targets coarse/late adaptation, does not exploit K3 dynamically. Backup only.
B (PRIMARY): A plus bounded per-relation K2/K3 feedback into the next quantum round. Targets late gate, loss of high-order mode adaptation and channel homogeneity; all scenes always run C4/D3 joint quantum evolution. No classical graph encoder before quantum, no classical/quantum selector, no oracle supervision.
C: global scalar schedule alone: rejected as too coarse given edge-identity bottleneck and weak external-router CV.
No expansion of qubits/channels/depth. At most2 evidence-led structural revisions after initial B; stop after limit.

Primary design: same R2 own-history encoder and relation readout. A small permutation-invariant physical summary controller modulates only ZZ/ZZZ, never sends scene information straight to output or local rotations. Positive bounded multipliers in[.2,1.8], learned channel/round priors; previous-round signed/squared bounded cumulants modulate individual relations. No entropy/oracle/extra label loss. Functional classical has the exact same controller, receives its previous pair/triplet score feedback, and preserves existing 3-layer4-head higher-order attention capacity.

Hardware boundary: simulator expectation feedback is deterministic differentiable and does not collapse state; hardware requires ensemble expectation estimates and fresh preparation/replay of prefixes. NOT coherent feedback and NOT free mid-circuit measurement. Sources checked: Skolik2023 doi10.1038/s41534-023-00710-y; PennyLane0.45.1 qp.measure docs; Koh etal2024 arXiv2406.07611. No literature guarantees ADE advantage.

Reference calibration launched before new model: R2 quantum and classical, 4096/1880/12epoch/cap16, same seed2026. Previous small R2 pilots usedcap4; not mix them into cap16 claims.

Full-train gate registered before adaptive validation: J at least1% better than cap16 frozen reference; relative Q/C J gap at least30% smaller than frozen gap (or positive); ordinary final-history closing<10 J improves; final-history>=15 J not worse than frozen by more than1%, with Q/C high-dynamic gap no worse. Also report recent5-frame/wedge/triangle/frozenK3 strata without cherry-picking. No multiSNR or multiSeed while overall is negative.

## B. Engineering v1
Controller2112 parameters identically added on both sides. Quantum69469 core vs classical325250. Independent PennyLane prefix replay state/feedback-gradient agrees at1e-15. PermutationN1/2/4/8/12/16/20, padding4-to20 and no-entangler dependency PASS. B32 Q0.11615s/3.641GiB, C0.08437s/3.666GiB. Full20 training-only projected1148s/834s plus loader/validation/checkpoint. Empty-triple warning guarded before pilot (N>=3 unchanged). Reference cap16 small Q0.602135/1.246336/J1.225303; C0.585355/1.208482/J1.189596.
A REPL compound-line error skipped the planned commit but did not alter model sources; pilot launched from dirty dc5daca with complete code hashes, exact source committed immediately afterward.

## C. V1 result and first structural revision declared before next validation
V1 small Q0.609190/1.256607/J1.237493 vs matched C0.588472/1.214436/J1.195691. Worse frozen cap16Q J by0.995%; full gate FAIL. Final-history high140 windows still Q positive2.36%ADE/3.88%FDE. No full training launched.
Post-hoc no-feedback J1.238243 vsnormal1.237493 (only0.061% dependency); controller removal J1.255438. First64 train scenes lambda scene std mostly0.001-0.036, no saturation. Thus feedback is weakly used, not evidence of causal benefits.
Revision1 (v2 phase_feedback): keep controller2112, C4/D3/Q8, readout/LLM/loss unchanged. Change ONLY quantum modulation from multiplicative lambda*base to base +1.5*(lambda-1)*risk/norm. Derivative of old path is proportional to base angle; at small base angle feedback cannot activate a relation or change its sign through a positive multiplier. The additive bounded phase path removes this representational/gradient restriction. Same classical controller/own relation feedback and attention capacity retained, re-run paired exposure. This is one evidence-led structural revision, not proof the bottleneck caused v1 failure. Maximum2 revisions remains in force.

Revision1 engineering PASS: B32Q0.11153s/3.641GiB, unchanged2112controller; all symmetry/mask/gradient and independent PennyLane reference rechecked. Caveat: first64 chronological train windows are not representative of full train variability; low lambda scene std is not extrapolated. Weak full-validation feedback dependency and multiplicative coupling derivative restriction remain reasons for targeted v2.
