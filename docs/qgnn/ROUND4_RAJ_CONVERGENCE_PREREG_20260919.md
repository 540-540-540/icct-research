# Round4 Raj weighted multi-j 20-epoch convergence preregistration

Purpose: determine whether the remaining ~1-2% gap to paper-native multi-j JohnsonGIN is primarily incomplete convergence under the 12-epoch pilot.

Fixed before launch:
- SinD, 0 dB, seed2026
- 4096 train indices, all 1880 validation windows
- batch32, correction cap16
- same Motion-Token GPT-2 / Tokenizer / loss / optimizer family
- 20 epochs for BOTH models from fresh same-seed initialization
- no architecture, LR, loss, width, subset, token, or data change
- test split closed
- Quantum = weighted multi-j j=2+j=3 subset QGNN
- Classical = paper-native multi-j JohnsonGIN

Interpretation gate:
- If Quantum best validation ADE AND FDE are both lower than JohnsonGIN, Raj route obtains a positive paper-native paired signal and may proceed to stronger confirmation.
- If either remains worse, do not add another Raj architecture variant in this round; freeze the negative/partial result and move primary exploration to SQM-GNN.

This is a convergence check, not another architecture search.
