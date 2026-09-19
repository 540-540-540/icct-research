# Round4 Raj Full-Train 0 dB Paired Confirmation Findings

Date: 2026-09-19

## 1. Decision

The preregistered full-train confirmation **does not pass the primary gate**.

Protocol was frozen before launch in `ROUND4_RAJ_FULLTRAIN_PREREG_20260919.md` and committed as `c654ae6`.

Both arms completed:
- SinD, 0 dB, seed 2026
- full 15,802 train windows
- all 1,880 validation windows
- 20 epochs, batch 32, LR 3e-4
- depth 3, correction cap 16
- same GPT-2 / Tokenizer / loss / optimizer / scheduler
- same deterministic train indices
- same downstream initialization hash
- test split closed

## 2. Final best-validation result

| Model | ADE | FDE | J = ADE + 0.5 FDE | Best epoch |
|---|---:|---:|---:|---:|
| Weighted multi-j Quantum | 0.481258665 | 1.051172996 | 1.006845163 | 9 |
| multi-j JohnsonGIN | 0.482314337 | 1.049728396 | 1.007178535 | 9 |

Quantum relative gain over JohnsonGIN:
- ADE: **+0.2189%**
- FDE: **-0.1376%**
- J: **+0.0331%**

Preregistered gates:
- Primary: Quantum ADE < Johnson ADE **and** Quantum FDE < Johnson FDE -> **FAIL**
- Strong: primary passes and J gain >= 1% -> **FAIL**

This is effectively a near-tie under the full-train seed2026 regime, not a confirmed quantum advantage.

## 3. Integrity checks

- Quantum status: COMPLETED, 20/20 epochs, 9,880 steps
- Johnson status: COMPLETED, 20/20 epochs, 9,880 steps
- train samples: 15,802 for both
- validation samples: 1,880 for both
- train-index SHA256 identical: `18eeffab9a311866f3cb67c352e9882d86e7068fb5527d82cd2d15d100ef73ca`
- shared downstream initialization SHA256 identical: `0d35048bfb3385d55aaab30cca077f76a7bc52701da744e786d67462a276b57f`
- launch Git HEAD: `c654ae68c5f4dfc76cf78e6151ba3a2b183648b7`
- test set used: false

## 4. Comparison with the 4096-window evidence

Earlier 4096-train results:

Seed2026 Quantum gain:
- ADE +1.805%
- FDE +2.390%
- J +2.108%

Seed2027 Quantum gain:
- ADE +3.548%
- FDE +3.501%
- J +3.523%

Two-seed 4096-window mean gain:
- ADE +2.696%
- FDE +2.960%
- J +2.833%

The full 15,802-window seed2026 run shrinks this margin to approximately zero and loses the FDE component of the preregistered gate.

Therefore the earlier positive signal **did not scale cleanly to the stronger full-training regime**.

## 5. What the learning curves show

Confirmed observations:
1. Both models reached their best validation J at epoch 9.
2. After epoch 9, training loss continued to decrease for both models while validation metrics did not establish a new best.
3. At epoch 9, train losses were already very close:
   - Quantum: 0.510013
   - JohnsonGIN: 0.513056
4. At epoch 20, Quantum still had slightly lower training loss:
   - Quantum: 0.338061
   - JohnsonGIN: 0.342933
5. Quantum therefore did not lose because it simply failed to optimize or needed more than 20 epochs.
6. The final comparison is metric-split: Quantum is slightly better on average displacement, JohnsonGIN slightly better on final displacement.

## 6. Interpretation

### Supported by this experiment

- The full-data regime materially reduces the previously observed Raj-style Quantum advantage over the matched JohnsonGIN baseline.
- The remaining difference is too small and internally split across ADE/FDE to support a stable quantum-superiority claim.
- The failure is not explained by unequal data exposure, unequal initialization, early termination, OOM, or test leakage.
- Extending the same 20-epoch run further is not justified by the observed curves: both validation optima occurred at epoch 9 and later epochs overfit / fail to improve validation.

### Plausible but not yet proven

- The Raj-style subset quantum inductive bias may help more in the data-limited 4096-window regime, while the stronger-capacity JohnsonGIN (179,201 graph params vs 126,535 Quantum) catches up when all 15,802 training windows are available.
- The 4096-window 2-3% gain may contain a genuine low-data advantage plus seed/subset variance, rather than a full-data asymptotic advantage.
- JohnsonGIN's tiny FDE edge may indicate that additional classical capacity helps terminal-state accuracy once sufficient data are available.

These are hypotheses, not established mechanisms. No architecture change or baseline weakening is licensed by this result.

## 7. Decision for the main line

Per preregistration:
- do not tune the Raj architecture from this result;
- do not weaken JohnsonGIN;
- do not cherry-pick ADE or J and call the gate passed;
- do not start five-SNR or final test as if full-train confirmation succeeded.

The Raj route remains scientifically useful as a documented positive low-data signal, but **full-train seed2026 does not confirm a stable paper-native quantum advantage**.

Before switching methods, the next research decision should be based on this stronger negative/neutral confirmation. If continuing the quantum main-line search, the preregistered fallback candidate is SQM-GNN-style `U_MSG + U_UPD`, rather than another RC-HQGNN / router / controller variant.

Machine-readable result:
`reports/qgnn/ROUND4_RAJ_FULLTRAIN_RESULTS_20260919.json`
