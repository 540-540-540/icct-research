# Automatum QGNN Task-Space Audit — 0 dB validation pilot

Reference: frozen strong Local residual; comparator: frozen-base Pairwise residual. Validation only; no test use.

## Global

- Strong Local: ADE 0.456623, FDE 0.757666
- Pairwise: ADE 0.499086, FDE 0.858638
- Pairwise gain vs Local: ADE -9.30%, FDE -13.33%
- Pairwise beats Local on both metrics in 27.0% of validation scenes.
- Per-scene Local/Pairwise oracle gain: ADE 3.20%, FDE 5.09%.

## Predefined interaction strata

| Stratum | Scenes | Local error share ADE/FDE | Pairwise gain ADE/FDE | Required internal gain for 5% global ADE/FDE |
|---|---:|---:|---:|---:|
| N>=4 | 482 | 53.1% / 54.1% | -7.0% / -8.0% | 9.4% / 9.2% |
| N>=5 | 322 | 36.3% / 37.6% | -5.7% / -6.0% | 13.8% / 13.3% |
| N>=6 | 133 | 15.2% / 15.3% | -3.8% / -3.0% | 33.0% / 32.6% |
| close20>=2 | 419 | 46.4% / 47.5% | -6.3% / -6.9% | 10.8% / 10.5% |
| close20>=3 | 273 | 31.1% / 32.6% | -4.9% / -3.8% | 16.1% / 15.3% |
| close30>=3 | 453 | 50.3% / 51.4% | -7.5% / -8.4% | 9.9% / 9.7% |
| closing30>=1 | 563 | 62.1% / 63.1% | -9.2% / -12.4% | 8.1% / 7.9% |
| closing30>=2 | 307 | 34.6% / 35.3% | -6.7% / -8.1% | 14.4% / 14.2% |
| CPA<5m | 394 | 42.7% / 43.1% | -5.3% / -6.0% | 11.7% / 11.6% |
| CPA<10m | 511 | 55.4% / 55.4% | -6.5% / -7.3% | 9.0% / 9.0% |
| complex_combo | 394 | 43.8% / 45.0% | -6.2% / -6.6% | 11.4% / 11.1% |
| closing_combo | 289 | 32.8% / 33.5% | -6.4% / -7.4% | 15.2% / 14.9% |

## Interpretation

- A stratum with error share alpha can mathematically support a 5% global improvement only if its internal error can be reduced by at least 0.05/alpha, assuming all other scenes are unchanged.
- The current Pairwise residual does **not** establish marginal interaction value over the Strong Local control; it is worse globally and in most predefined high-interaction strata.
- However, high-interaction strata carry a large fraction of Strong-Local error, so Automatum still has mathematical room for an interaction-specialized mechanism if a better interaction model can exploit it.
- The Local/Pairwise per-scene oracle quantifies heterogeneity: some scenes do benefit from Pairwise, but current gating/interaction learning does not identify and exploit them reliably.
