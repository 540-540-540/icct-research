"""SENS-AUDIT-01 SNR probe: read-only, paired multi-SNR rerun of the frozen frontend.

Runs the production symbol-level GPU kernels on a small train/V_select sample at
-20..20 dB with the production seed recipe (identical noise realization across
SNR). Writes only under reports/sensing_audit/sens_audit_01/.

Usage: python -u code/06_sensing_diagnostics/probe_snr_response.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/sensing_audit/sens_audit_01"
SNRS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
PROBE_EPISODES = {"train": 2, "V_select": 2}
PROBE_FRAMES = [30, 90, 150, 190]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row.get(f, "")) for f in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rd_map(B, w, min_range, max_range, max_speed):
    import torch

    nr, nv = w.K, w.N
    spectrum = torch.fft.fft(torch.fft.ifft(B, n=nr, dim=-2), n=nv, dim=-1)
    power = spectrum.abs().square()
    ranges = torch.arange(nr, device=B.device, dtype=B.real.dtype) * w.unambiguous_range / nr
    velocities = torch.fft.fftfreq(nv, d=w.T, device=B.device, dtype=B.real.dtype) * w.c / (2 * w.fc)
    valid = ((ranges >= min_range) & (ranges <= max_range))[:, None] & (velocities.abs() <= max_speed)[None, :]
    masked = power.masked_fill(~valid, -1.0)
    peak = float(masked.max())
    floor = float(power[valid].median())
    index = int(masked.flatten().argmax())
    peak_range = float(ranges[index // nv])
    peak_velocity = float(velocities[index % nv])
    return peak, floor, peak_range, peak_velocity


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import torch
    from frontend.echo_source import SourceEpisodes
    from frontend.symbol_level_gpu import PaperWaveform, divide_symbols, estimate_batch, synthesize_batch
    from frontend.generate_symbol_gpu import measurement_seed

    config = json.loads((ROOT / "configs/symbol_frontend.json").read_text())
    assert config["revision"] == "A03-symbol-v4-cuda-stable-solver" and config["snr_db"] == [5, 10, 15, 20]
    w = PaperWaveform(**config["waveform"])
    stations = torch.tensor(config["stations"], device="cuda:0", dtype=torch.float64)
    xy_bounds = torch.tensor(config["xy_bounds"], device="cuda:0", dtype=torch.float64)
    v_bounds = torch.tensor(config["v_bounds"], device="cuda:0", dtype=torch.float64)
    source = SourceEpisodes()

    corners = torch.tensor([[x, y] for x in config["xy_bounds"][0] for y in config["xy_bounds"][1]],
                           device="cuda:0", dtype=torch.float64)
    max_ranges = torch.linalg.vector_norm(corners[:, None] - stations[None], dim=-1).amax(0)
    nearest = torch.clamp(stations, min=xy_bounds[:, 0], max=xy_bounds[:, 1])
    min_ranges = torch.linalg.vector_norm(nearest - stations, dim=-1)

    selection = []
    for split, count in PROBE_EPISODES.items():
        indices = [i for i, ep in enumerate(source.episodes) if ep["split"] == split][:count]
        for index in indices:
            selection.append((split, index))

    rows = []
    failure_rows = []
    for split, index in selection:
        episode = source.episodes[index]
        start = int(episode["start_ms"])
        for frame in PROBE_FRAMES:
            deadline = start + (frame + 1) * 100
            states, slots, _ = source.at_time(episode, deadline * 1_000_000)
            if not len(states):
                continue
            states_t = torch.tensor(np.asarray(states, np.float64), device="cuda:0", dtype=torch.float64)
            seeds = [measurement_seed(index, frame + 1, int(slot)) for slot in slots]
            deltas = stations[None] - states_t[:, None, :2]
            radii = torch.linalg.vector_norm(deltas, dim=-1)
            radial = (deltas * states_t[:, None, 2:]).sum(-1) / radii
            for snr in SNRS:
                with torch.inference_mode():
                    noisy = synthesize_batch(states_t[:, :2], states_t[:, 2:], config["stations"],
                                             waveform=config["waveform"], snr_db=snr, seeds=seeds,
                                             device="cuda:0", noise=True)
                    clean = synthesize_batch(states_t[:, :2], states_t[:, 2:], config["stations"],
                                             waveform=config["waveform"], snr_db=snr, seeds=seeds,
                                             device="cuda:0", noise=False)
                    noise = noisy["Y"] - clean["Y"]
                    b = divide_symbols(**noisy)
                    result = estimate_batch(b, config["stations"], config["xy_bounds"], config["v_bounds"],
                                            waveform=config["waveform"], search=config["search"])
                    estimates = result["state_hat"]
                    for i, (slot, seed) in enumerate(zip(slots, seeds)):
                        estimate = estimates[i].cpu().numpy().astype(np.float64)
                        truth = np.asarray(states[i], np.float64)
                        diag = result["diagnostics"][i]
                        signal_power = float(clean["Y"][i].abs().square().mean())
                        noise_power = float(noise[i].abs().square().mean())
                        row = {
                            "split": split, "episode_index": index, "frame_index": frame,
                            "slot": int(slot), "seed": int(seed), "snr_db": snr,
                            "truth_x": truth[0], "truth_y": truth[1], "truth_vx": truth[2], "truth_vy": truth[3],
                            "est_x": estimate[0], "est_y": estimate[1], "est_vx": estimate[2], "est_vy": estimate[3],
                            "pos_err_m": float(np.linalg.norm(estimate[:2] - truth[:2])),
                            "vel_err_mps": float(np.linalg.norm(estimate[2:] - truth[2:])),
                            "pos_err_x": float(estimate[0] - truth[0]), "pos_err_y": float(estimate[1] - truth[1]),
                            "vel_err_x": float(estimate[2] - truth[2]), "vel_err_y": float(estimate[3] - truth[3]),
                            "search_failure": bool(diag["search_failure"]),
                            "optimizer_success": bool(diag["localization_optimizer_success"]),
                            "pos_coarse_edge": bool(diag["position_search"]["coarse_local_edge"]),
                            "pos_fine_edge": bool(diag["position_search"]["fine_local_edge"]),
                            "vel_coarse_edge": bool(diag["velocity_search"]["coarse_local_edge"]),
                            "vel_fine_edge": bool(diag["velocity_search"]["fine_local_edge"]),
                            "signal_rms": math.sqrt(signal_power), "noise_rms": math.sqrt(noise_power),
                            "empirical_snr_db": 10 * math.log10(signal_power / noise_power) if noise_power else float("inf"),
                        }
                        for station in range(3):
                            row[f"coarse_range_s{station}"] = float(result["coarse"]["range_radial_velocity"][i, station, 0])
                            row[f"coarse_radial_v_s{station}"] = float(result["coarse"]["range_radial_velocity"][i, station, 1])
                        for station in range(3):
                            peak, floor, peak_range, peak_velocity = rd_map(
                                b[i, station], w, float(min_ranges[station]), float(max_ranges[station]),
                                float(torch.linalg.vector_norm(v_bounds.abs().amax(-1))))
                            row[f"rd_peak_floor_db_s{station}"] = 10 * math.log10(peak / floor) if floor else float("inf")
                            row[f"rd_peak_range_s{station}"] = peak_range
                            row[f"rd_peak_velocity_s{station}"] = peak_velocity
                            row[f"true_range_s{station}"] = float(radii[i, station])
                            row[f"true_radial_v_s{station}"] = float(radial[i, station])
                            row[f"rd_range_err_s{station}"] = abs(peak_range - float(radii[i, station]))
                            row[f"coarse_range_err_s{station}"] = abs(row[f"coarse_range_s{station}"] - float(radii[i, station]))
                        rows.append(row)
                frame_failed = any(r["search_failure"] for r in rows[-len(slots):])
                if frame_failed:
                    failure_rows.append({"split": split, "episode_index": index, "frame_index": frame, "snr_db": snr})

    fields = list(rows[0].keys())
    write_csv(OUT / "tables/probe_snr_response.csv", rows, fields)

    aggregate = []
    for snr in SNRS:
        group = [r for r in rows if r["snr_db"] == snr]
        base = [r for r in rows if r["snr_db"] == 20]
        lookup = {(r["episode_index"], r["frame_index"], r["slot"]): r for r in base}
        identical = sum(1 for r in group
                        if r["est_x"] == lookup[(r["episode_index"], r["frame_index"], r["slot"])]["est_x"]
                        and r["est_y"] == lookup[(r["episode_index"], r["frame_index"], r["slot"])]["est_y"]
                        and r["est_vx"] == lookup[(r["episode_index"], r["frame_index"], r["slot"])]["est_vx"]
                        and r["est_vy"] == lookup[(r["episode_index"], r["frame_index"], r["slot"])]["est_vy"])
        pos = np.array([r["pos_err_m"] for r in group])
        vel = np.array([r["vel_err_mps"] for r in group])
        rd = np.array([r[f"rd_peak_floor_db_s{s}"] for r in group for s in range(3)])
        aggregate.append({
            "snr_db": snr, "targets": len(group),
            "pos_err_median_m": float(np.median(pos)), "pos_err_p95_m": float(np.quantile(pos, 0.95)),
            "pos_err_max_m": float(pos.max()),
            "vel_err_median_mps": float(np.median(vel)), "vel_err_p95_mps": float(np.quantile(vel, 0.95)),
            "vel_err_max_mps": float(vel.max()),
            "search_failures": sum(1 for r in group if r["search_failure"]),
            "identical_to_20db": identical,
            "identical_fraction": identical / len(group) if group else 1.0,
            "empirical_snr_median_db": float(np.median([r["empirical_snr_db"] for r in group])),
            "rd_peak_floor_median_db": float(np.median(rd)),
            "rd_range_err_median_m": float(np.median([r[f"rd_range_err_s{s}"] for r in group for s in range(3)])),
        })
    write_csv(OUT / "tables/probe_snr_aggregate.csv", aggregate, list(aggregate[0].keys()))
    write_json(OUT / "probe_summary.json", {
        "probe_snrs_db": SNRS, "official_snrs_db": config["snr_db"],
        "episodes": selection, "frames": PROBE_FRAMES,
        "seed_recipe": "production measurement_seed(episode, frame+1, slot); identical noise across SNR",
        "aggregate": aggregate, "search_failure_rows": failure_rows,
        "coherent_processing_gain_db": 10 * math.log10(w.K * w.N),
        "k_times_n": w.K * w.N,
    })

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
        snr_axis = [a["snr_db"] for a in aggregate]
        axes[0].plot(snr_axis, [a["pos_err_median_m"] for a in aggregate], "o-", label="median")
        axes[0].plot(snr_axis, [a["pos_err_p95_m"] for a in aggregate], "s--", label="p95")
        axes[0].set(xlabel="SNR (dB)", ylabel="position error (m)", title="Paired probe: estimation error")
        axes[1].plot(snr_axis, [a["vel_err_median_mps"] for a in aggregate], "o-", label="median")
        axes[1].plot(snr_axis, [a["vel_err_p95_mps"] for a in aggregate], "s--", label="p95")
        axes[1].set(xlabel="SNR (dB)", ylabel="velocity error (m/s)", title="Paired probe: estimation error")
        for ax in axes:
            ax.axvspan(5, 20, color="#cccccc", alpha=0.35, zorder=0)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        path = OUT / "figures/probe_error_vs_snr.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)

        gain = 10 * math.log10(w.K * w.N)
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot(snr_axis, [a["rd_peak_floor_median_db"] for a in aggregate], "o-", label="range-Doppler peak/floor (median)")
        ax.plot(snr_axis, [a["empirical_snr_median_db"] for a in aggregate], "s--", label="per-entry received SNR (median)")
        ax.plot(snr_axis, [s + gain for s in snr_axis], ":", label=f"per-entry SNR + 10log10(KN)={gain:.1f} dB")
        ax.set(xlabel="SNR (dB)", ylabel="dB", title="Observation-level SNR and RD map peak-to-floor")
        ax.axvspan(5, 20, color="#cccccc", alpha=0.35, zorder=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = OUT / "figures/probe_rd_peak_vs_snr.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as error:
        write_json(OUT / "probe_figure_error.json", {"error": repr(error)})

    print(json.dumps({"aggregate": aggregate, "search_failure_rows": failure_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()