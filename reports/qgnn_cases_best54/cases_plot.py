"""Render absolute-ADE extreme cases from adjacent, frozen inference artifacts."""
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import FormatStrFormatter

OUT = Path(__file__).resolve().parent
meta = json.loads((OUT / 'cases_metadata.json').read_text(encoding='utf-8'))
data = np.load(OUT / 'cases_data.npz', allow_pickle=False)
assert meta['checkpoint_epochs'] == {'qgnn': 54, 'gnn': 97}
assert meta['candidate_count'] == 1338 and meta['full_population_candidate_count'] == 2808
assert meta['threshold_m'] == 5 and meta['worst_preserved_from_full_population']
COLORS = {'truth': '#20252B', 'qgnn': '#0F4D92', 'gnn': '#D55E00', 'history': '#80858A'}
FONT = FontProperties(family=['Times New Roman', 'SimSun'])
plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'], 'font.size': 11,
    'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .75, 'axes.edgecolor': '#6D7277', 'legend.frameon': False,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'savefig.dpi': 300})
STYLES = {
    'truth': dict(ls='-', marker='o', markevery=[1, 7, 13], lw=1.35),
    'qgnn': dict(ls=(0, (4, 1.3, 1, 1.3)), marker='^', markevery=[3, 9, 15], lw=1.5),
    'gnn': dict(ls=(0, (5, 2.8)), marker='s', markevery=[5, 11, 17], lw=1.5),
}

def draw_paths(ax, paths, history=None, zoom=False):
    if history is not None:
        ax.plot(*history.T, color=COLORS['history'], ls=':', lw=1.5,
                marker='D', ms=4.3, mfc='none', mew=1, markevery=[0, 2], zorder=7)
    for name in ['truth', 'gnn', 'qgnn']:
        style = dict(STYLES[name])
        ax.plot(*paths[name].T, color=COLORS[name], ms=4.6 if zoom else 4.1,
                mfc='none', mew=.95, zorder=4, **style)
        if not zoom:
            ax.plot(*paths[name][-1], color=COLORS[name], marker=style['marker'],
                    ms=5, mfc='none', mew=1.05, zorder=6)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(color='#E7EAED', lw=.5, zorder=0)
    ax.tick_params(labelsize=8.3 if zoom else 10, length=2.5, pad=2)

fig = plt.figure(figsize=(400 / 72, 400 / 72), dpi=300, facecolor='white')
axes = [fig.add_axes([.122, .305, .335, .600]), fig.add_axes([.625, .305, .335, .600])]
for ax, case, title in zip(axes, meta['cases'], ['QGNN最佳案例', 'QGNN最差案例']):
    name = case['name']
    present = data[name + '_present']
    history = (data[name + '_history'] - present)[-4:]
    paths = {model: np.vstack([np.zeros(2), data[name + '_' + model] - present])
             for model in ['truth', 'qgnn', 'gnn']}
    metrics = case['metrics']
    draw_paths(ax, paths, history)
    if name == 'best':
        yspan = 34
        xspan = yspan * .335 / .600
        ax.set_xlim(2.8 - xspan, 2.8)
        ax.set_ylim(-4.5, 29.5)
        ax.set_xticks([-15, -10, -5, 0])
        ax.set_yticks([0, 10, 20])
    else:
        yspan = 23.2
        xspan = yspan * .335 / .600
        ax.set_xlim(2.5 - xspan, 2.5)
        ax.set_ylim(-4.2, 19)
        ax.set_xticks([-10, -5, 0])
        ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel(r'$x\;(\mathrm{m})$', fontsize=12, labelpad=2)
    ax.set_ylabel(r'$y\;(\mathrm{m})$', fontsize=12, labelpad=0)
    left, bottom, width, height = ax.get_position().bounds
    center = left + width / 2
    fig.text(center, .965, title, ha='center', va='top', fontsize=13.5, fontproperties=FONT)

    if name in ('best', 'worst'):
        inset = ax.inset_axes([.045, .45, .48, .48 * width / height])
        inset.set_zorder(10)
        draw_paths(inset, paths, zoom=True)
        bounds = (-.9, .3, 22.4, 23.6) if name == 'best' else (-1.05, .45, 6, 7.5)
        xmin, xmax, ymin, ymax = bounds
        inset.set_xlim(xmin, xmax)
        inset.set_ylim(ymin, ymax)
        inset.set_xticks([-.5, 0] if name == 'best' else [-1, 0])
        inset.set_yticks([22.5, 23, 23.5] if name == 'best' else [6, 7])
        if name == 'best':
            for model in ['truth', 'qgnn']:
                inset.plot(*paths[model][-1], color=COLORS[model], marker=STYLES[model]['marker'],
                           ms=5, mfc='none', mew=1.05, zorder=6)
        inset.yaxis.tick_right()
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.8)
            spine.set_edgecolor('#818A93')
        inset.set_title('局部放大', fontsize=10, pad=4, fontproperties=FONT)
        inset.set_facecolor('white')
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False,
                              ec='#8795A3', ls=(0, (2, 2)), lw=.7, zorder=7))
        fig.add_artist(ConnectionPatch(xyA=(1, 1) if name == 'best' else (1, .5), coordsA='axes fraction', axesA=inset,
            xyB=(xmin, (ymin+ymax)/2), coordsB='data', axesB=ax,
            lw=.7, color='#8795A3', ls=(0, (2, 2)), zorder=8))

    fig.text(center, .204, 'ADE / FDE (m)', ha='center', fontsize=10.8, fontproperties=FONT)
    fig.text(center, .163, f"QGNN  {metrics['qgnn']['ADE']:.4f} / {metrics['qgnn']['FDE']:.4f}",
             color=COLORS['qgnn'], ha='center', fontsize=10.8, fontproperties=FONT)
    fig.text(center, .124, f"GNN     {metrics['gnn']['ADE']:.4f} / {metrics['gnn']['FDE']:.4f}",
             color=COLORS['gnn'], ha='center', fontsize=10.8, fontproperties=FONT)
    all_points = np.concatenate([history, *paths.values()])
    assert (all_points.min(0) >= [ax.get_xlim()[0], ax.get_ylim()[0]]).all()
    assert (all_points.max(0) <= [ax.get_xlim()[1], ax.get_ylim()[1]]).all()

handles = [Line2D([], [], color=COLORS['history'], ls=':', lw=1.5, marker='D',
                  ms=4, mfc='none', label='历史末4帧')]
for model, label in [('truth', '真值'), ('qgnn', 'QGNN'), ('gnn', 'GNN')]:
    style = dict(STYLES[model])
    style.pop('markevery')
    handles.append(Line2D([], [], color=COLORS[model], ms=4.2, mfc='none', label=label, **style))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .060), ncol=4,
    handlelength=1.85, handletextpad=.35, columnspacing=.9,
    prop=FontProperties(family=['Times New Roman', 'SimSun'], size=10.2), borderaxespad=0)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    fig.canvas.draw()
    for ax in [*axes, *[child for a in axes for child in a.child_axes]]:
        pts = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_allclose(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[0]), rtol=1e-8)
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= w and box.y1 <= h, box
    fig.savefig(OUT / 'cases.png', dpi=300, facecolor='white')
    assert not [str(w.message) for w in caught if 'Glyph' in str(w.message)]
print(OUT / 'cases.png')
