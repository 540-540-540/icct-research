# Gate A strict-causal Lankershim data contract

SinD is not used for the formal Gate A benchmark because the available history fields are produced by an RTS-smoothed pipeline. The fallback source is the repository's raw Lankershim trajectory table, fixed by SHA-256 in the configuration.

For every prediction origin `t0`, model history consists only of raw positions timestamped at or before `t0`, past-only OLS velocities (current plus at most four prior contiguous observations), and 0 dB same-frame 3-BS controlled sensing outputs. Target membership, context membership, interaction edges, neighbor ordering, the ego coordinate frame, and every diagnostic stratum are functions of these history-side values only.

Positions after `t0` are read only when constructing `future_xy` and `future_mask`. They cannot create or remove a sample, node, edge, or stratum. A vehicle may remain in the benchmark even when some or all future labels are unavailable. `Full2`, `IC2`, `Full4`, and `IC4` are views of the same samples and the same frozen `k`; only the horizon and `k>=1` evaluation mask differ.

The temporal split is the user-selected `80% train / 10% validation / 10% sealed test`, with six-second gaps. Conservative cross-record `source_group` identities are purged if they touch more than one split. No test cache is materialized, and the loader rejects a test split.

The five controls share one node TCN, CV residual anchor, decoder, loss, optimizer budget, and checkpoint rule. `Self` uses target history; `Own` adds target-only capacity; `Pool` sees a permutation-invariant neighbor summary; `Graph` uses edge-aware messages on target plus seven neighbors; `All Graph` uses every history-valid neighbor within 50 m.
