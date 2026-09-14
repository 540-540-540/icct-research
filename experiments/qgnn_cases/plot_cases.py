"""Plot saved diagnostic cases only; no model, dataset, or training access."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/qgnn_cases'
OUTPUT = REPORT / 'figures'
Q_COLOR, G_COLOR = '#0F4D92', '#D55E00'
IMPROVE, DEGRADE = '#33866B', '#B64342'


def square_limits(ax, points, pad=0.13):
    points = np.concatenate(points)
    assert np.isfinite(points).all()
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    span = max(float((high - low).max()), 4.0) * (1 + 2 * pad)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('Relative x (m)')
    ax.set_ylabel('Relative y (m)')
    ax.grid(color='#E7EAED', linewidth=0.65, zorder=0)
    return span


def plot_trajectory(ax, case, traces):
    prefix, slot = f"case_{case['origin']}_", case['target_slot']
    state = traces[prefix + 'state_hat']
    exists = traces[prefix + 'track_exists'][:, slot]
    valid = traces[prefix + 'label_valid'][:, slot]
    assert exists[-1] and valid.any()
    origin = state[-1, slot, :2]
    history = state[exists, slot, :2] - origin
    truth = traces[prefix + 'future_position'][valid, slot] - origin
    quantum = traces[prefix + 'qgnn_prediction'][valid, slot] - origin
    classical = traces[prefix + 'gnn_prediction'][valid, slot] - origin
    for name, predicted in (('qgnn', quantum), ('gnn', classical)):
        errors = np.linalg.norm(predicted - truth, axis=1)
        np.testing.assert_allclose(errors.mean(), case['target_scores'][name]['ADE'], atol=1e-5)
        np.testing.assert_allclose(errors[-1], case['target_scores'][name]['FDE'], atol=1e-5)
    ax.plot(*history.T, color='#80858A', linestyle=':', linewidth=2.0, zorder=2)
    # Add the present point for visual continuity; metrics above use future only.
    paths = [np.vstack([np.zeros(2), p]) for p in (truth, quantum, classical)]
    ax.plot(*paths[0].T, color='#20252B', linewidth=2.2, marker='o', markersize=3.0,
            markevery=4, zorder=3)
    ax.plot(*paths[1].T, color=Q_COLOR, linewidth=1.8, marker='^', markersize=4.0,
            markerfacecolor='white', markevery=4, zorder=4)
    ax.plot(*paths[2].T, color=G_COLOR, linestyle='--', linewidth=1.9, marker='s',
            markersize=3.5, markerfacecolor='white', markevery=4, zorder=5)
    for path, color, marker in zip(paths, ('#20252B', Q_COLOR, G_COLOR), ('o', '^', 's')):
        ax.scatter(*path[-1], s=38, c=color, marker=marker, edgecolor='white', linewidth=0.65, zorder=6)
    ax.scatter(0, 0, marker='*', s=100, color='#20252B', edgecolor='white', linewidth=.8, zorder=7)
    square_limits(ax, [history, *paths])
    q, g = case['target_scores']['qgnn'], case['target_scores']['gnn']
    ax.set_title(f"Case {case['origin']} | target slot {slot}\n"
                 f"ADE Q {q['ADE']:.2f} / G {g['ADE']:.2f} m; FDE Q {q['FDE']:.2f} / G {g['FDE']:.2f} m",
                 loc='left', fontsize=10.3, pad=11)


def plot_neighbors(ax, case, traces, max_contribution):
    prefix, target = f"case_{case['origin']}_", case['target_slot']
    state = traces[prefix + 'state_hat'][-1]
    exists = traces[prefix + 'track_exists'][-1]
    senders = [row['sender'] for row in case['direct_edge_interventions']]
    contributions = {row['sender']: row['contribution_norm_mean']
                     for row in case['layer_statistics'][1]['neighbors']}
    relative = state[:, :2] - state[target, :2]
    shown = [sender for sender in senders if exists[sender]]
    span = square_limits(ax, [relative[[target, *shown]]], pad=.20)
    for sender in shown:
        xy = relative[sender]
        width = .6 + 4.0 * contributions[sender] / max_contribution
        ax.plot([0, xy[0]], [0, xy[1]], color='#A6B4C3', linewidth=width, alpha=.8, zorder=1)
        ax.scatter(*xy, s=140, color='#647F98', edgecolor='white', linewidth=.8, zorder=3)
        ax.text(*xy, str(sender), ha='center', va='center', fontsize=8.5,
                weight='bold', color='white', zorder=6)
    ax.scatter(0, 0, marker='*', s=145, color=Q_COLOR, edgecolor='white', linewidth=.8, zorder=5)
    ax.annotate(f'Target {target}', (0, 0), xytext=(-8, 8), textcoords='offset points',
                ha='right', fontsize=9, weight='bold', color=Q_COLOR, zorder=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
    velocity = state[target, 2:]
    speed = np.linalg.norm(velocity)
    if speed > 1e-6:
        endpoint = velocity / speed * span * .14
        ax.annotate('', endpoint, xytext=(0, 0), zorder=7,
                    arrowprops=dict(arrowstyle='-|>', color='#20252B', lw=1.6, mutation_scale=13))
        ax.annotate('heading', endpoint, xytext=(5, 1), textcoords='offset points',
                    color='#20252B', fontsize=8.3, zorder=8)
    unavailable = [str(sender) for sender in senders if not exists[sender]]
    if unavailable:
        ax.text(.03, .03, 'Not present now: ' + ', '.join(unavailable), transform=ax.transAxes, fontsize=8)
    ax.set_title('Direct neighbors seen in history\nPositions now; width = layer-2 contribution proxy',
                 loc='left', fontsize=10.3, pad=11)


def plot_interventions(ax, case, x_limit):
    rows = case['direct_edge_interventions']
    senders = [row['sender'] for row in rows]
    changes = np.array([row['target_ADE_change'] for row in rows])
    base = case['target_scores']['qgnn']['ADE']
    np.testing.assert_allclose(changes, [row['target_ADE_without_edge'] - base for row in rows], atol=1e-6)
    positions = np.arange(len(rows))
    ax.barh(positions, changes, height=.58, color=[IMPROVE if x < 0 else DEGRADE for x in changes],
            edgecolor='white', linewidth=.6, zorder=3)
    ax.axvline(0, color='#3C4249', linewidth=1.0, zorder=2)
    ax.set_yticks(positions, [f'Slot {s}' for s in senders])
    ax.invert_yaxis()
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(len(rows) - .35, -.65)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis='x', color='#E7EAED', linewidth=.65, zorder=0)
    for y, value in zip(positions, changes):
        ax.text(value + np.sign(value) * x_limit * .025, y, f'{value:+.3f}',
                va='center', ha='left' if value >= 0 else 'right', fontsize=9,
                color=IMPROVE if value < 0 else DEGRADE)
    ax.set_xlabel('Target ADE change after edge removal (m)')
    ax.set_title('Remove one neighbor-to-target edge\nFixed QGNN; no retraining', loc='left', fontsize=10.3, pad=11)


def make_figure(report, traces, selection, stem):
    cases = [case for case in report['cases'] if case['selection'] == selection]
    assert len(cases) == 3
    maximum = max(row['contribution_norm_mean'] for case in report['cases']
                  for row in case['layer_statistics'][1]['neighbors'] if row['sender'] != case['target_slot'])
    x_limit = max(abs(row['target_ADE_change']) for case in cases for row in case['direct_edge_interventions']) * 1.30
    fig, axes = plt.subplots(3, 3, figsize=(16.8, 14.0), gridspec_kw={'width_ratios': [1.10, 1.05, 1.10]})
    fig.subplots_adjust(left=.062, right=.98, bottom=.085, top=.835, wspace=.31, hspace=.42)
    direction = 'underperforms' if selection == 'Q_worse' else 'outperforms'
    fig.suptitle(f'Where inherited QGNN {direction} the strong GNN',
                 x=.062, y=.978, ha='left', fontsize=20, weight='bold', color='#20252B')
    fig.text(.062, .947, f"Selected cases | {report['split']}, SNR {report['snr_db']} dB | "
             f"QGNN checkpoint {report['checkpoint_epochs']['qgnn']}; GNN checkpoint {report['checkpoint_epochs']['gnn']}",
             ha='left', fontsize=10.5, color='#5A6169')
    handles = [Line2D([0], [0], color='#80858A', lw=2, ls=':', label='Past estimate'),
               Line2D([0], [0], color='#20252B', lw=2.2, marker='o', markersize=4, label='Future truth'),
               Line2D([0], [0], color=Q_COLOR, lw=1.8, marker='^', markersize=5, label='QGNN'),
               Line2D([0], [0], color=G_COLOR, lw=1.9, ls='--', marker='s', markersize=4, label='GNN'),
               Line2D([0], [0], color='#20252B', marker='*', markersize=10, ls='none', label='Present target')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.056, .928), ncol=5,
               fontsize=10, handlelength=2.7, columnspacing=2.1)
    fig.text(.062, .866, 'TARGET TRAJECTORY', fontsize=10.5, weight='bold', color='#354A5D')
    fig.text(.393, .866, 'NEIGHBOR CONTEXT', fontsize=10.5, weight='bold', color='#354A5D')
    fig.text(.719, .866, 'FIXED-MODEL INTERVENTION', fontsize=10.5, weight='bold', color='#354A5D')
    for row, case in enumerate(cases):
        plot_trajectory(axes[row, 0], case, traces)
        plot_neighbors(axes[row, 1], case, traces, maximum)
        plot_interventions(axes[row, 2], case, x_limit)
    fig.text(.062, .044, 'Coordinates are translated to the present target; spatial axes have equal meter scales. '
             'Neighbor line widths encode history-mean layer-2 contribution norms (proxy only).', fontsize=9.0, color='#59616A')
    fig.text(.062, .026, 'Green / negative: removal improves target ADE. Red / positive: removal worsens it. '
             'Fixed-model edge removal is a sensitivity diagnostic, not evidence of real-world causal importance.',
             fontsize=9.0, color='#59616A')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Check that text lies within the canvas; images are also visually inspected.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_tightbbox(renderer)
    width, height = fig.get_size_inches()
    assert bounds.x0 >= -.02 and bounds.y0 >= -.02 and bounds.x1 <= width + .02 and bounds.y1 <= height + .02, bounds
    for extension in ('png', 'pdf'):
        path = OUTPUT / f'{stem}.{extension}'
        fig.savefig(path, dpi=200, facecolor='white', metadata={'Title': f'QGNN selected {selection} cases'})
        print(path)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'axes.edgecolor': '#8C9299',
                         'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 200})
    report = json.loads((REPORT / 'case_diagnostics.json').read_text(encoding='utf-8'))
    assert report['status'] == 'complete'
    with np.load(REPORT / 'case_traces.npz', allow_pickle=False) as traces:
        make_figure(report, traces, 'Q_worse', 'worse_cases')
        make_figure(report, traces, 'Q_better', 'better_cases')


if __name__ == '__main__':
    main()
