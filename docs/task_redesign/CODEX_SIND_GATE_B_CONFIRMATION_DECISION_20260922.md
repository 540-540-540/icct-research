# SinD Gate B independent confirmation decision (2028/2029/2030)

## Decision

**Strict stability gate: FAIL.**

The frozen stochastic multi-order Raj candidate has a useful mean advantage over Strong Graph and Matched Johnson, but does not win ADE and FDE on every independent seed. It also loses the three-seed mean to Large Johnson. Test remains sealed.

## IC4 validation results

| Model | ADE mean ± sample SD | FDE mean ± sample SD | Paired outcome versus Raj |
|---|---:|---:|---|
| Raj stochastic multi-order | 1.02363 ± 0.01996 | 2.64767 ± 0.03823 | — |
| Strong Graph N<=8 | 1.04262 ± 0.01181 | 2.68803 ± 0.01729 | Raj wins both metrics on 2028/2030; loses both on 2029 |
| Matched Johnson N<=8 | 1.02884 ± 0.02785 | 2.65581 ± 0.09488 | Raj wins both metrics on 2028/2030; loses both on 2029 |
| Large Johnson N<=8 | 1.01343 ± 0.02137 | 2.64215 ± 0.05774 | Raj loses both metrics on 2028/2029; wins both on 2030 |

Raj improves the three-seed mean over Strong Graph by 1.82% ADE and 1.50% FDE, and over Matched Johnson by 0.51% ADE and 0.31% FDE. These aggregate gains are insufficient for the requested stable quantum-mechanism claim.

## Mechanism diagnosis

The failed Raj seed 2029 has pooled j2/j3 linear CKA 0.637, versus 0.310 for seed 2028 and 0.528 for seed 2030. This supports a concrete hypothesis: stochastic branch dropping reduced fragile co-adaptation, but did not consistently prevent the two quantum orders from learning redundant interaction representations.

The next bounded development experiment therefore adapts redundancy reduction from Barlow Twins/VICReg to the QGNN setting: penalize cross-correlation between attention-pooled j2 and j3 representations while retaining branch drop and the supervised trajectory objective. It does not align the orders or add a classical prediction path.

Seeds 2028/2029/2030 are now exhausted confirmation evidence. Any revised method must be developed on 2026/2027 and, if frozen, confirmed on new seeds rather than reusing these as independent confirmation.
