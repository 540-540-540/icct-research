# Round4 Raj weighted multi-j seed2027 confirmation preregistration

Purpose: independently check whether the seed2026 paper-native JohnsonGIN win replicates before any full-15802 or multi-SNR run.

Frozen before launch:
- architecture unchanged from seed2026 accepted candidate
- Quantum: weighted multi-j j=2+j=3 subset QGNN
- Classical: matched multi-j JohnsonGIN
- SinD 0 dB
- seed 2027 for both arms
- train-limit 4096 using the seed2027 deterministic train subset
- all 1880 validation windows
- 20 epochs, batch32, same base LR/optimizer/cosine schedule
- same GPT-2, Tokenizer, loss, correction cap16
- test split closed
- no hyperparameter or architecture tuning from seed2027 outcome

Confirmation interpretation:
- Primary: Quantum ADE and FDE both lower than matched JohnsonGIN.
- Strong replication: both gains positive and J gain >=1%.
- If either ADE or FDE is not lower, do not call the Raj advantage replicated; inspect only after freezing the result.
- No full 15802 / five-SNR run starts before this pair completes.

Seed2026 accepted reference:
Quantum 0.516457940 / 1.102781152 / J 1.067848516
Johnson 0.525952385 / 1.129781810 / J 1.090843291
gains: +1.805% ADE / +2.390% FDE / +2.108% J.
