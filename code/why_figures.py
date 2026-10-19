"""v3 stage 2e figures (post-hoc), drawn from the CSV / npz outputs of why_errors.py (--corrected, --s6) and why_tests.py (--2d, --2e).

Usage: python why_figures.py RESULTS_DIR OUT_DIR
  RESULTS_DIR  results/v3 (episode_transitions_2d.csv, rates_by_condition.csv, s5_profiles.csv, detection_recheck.csv, past_end_example_data.npz)
  OUT_DIR      figures are written here as PNG
Nothing here recomputes a statistic: every number drawn is read from those files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

SRC = ['paris4_heldout', 'PHerc0139', 'PHerc1451', 'PHerc0332']
NAME = {'paris4_heldout': 'Paris4', 'PHerc0139': 'PHerc0139', 'PHerc1451': 'PHerc1451', 'PHerc0332': 'PHerc0332', 'unseen pooled': 'Unseen pooled'}
# reference palette (categorical slots 1-3, light surface) and recessive ink
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9a96'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'figure.dpi': 100,
                     'savefig.dpi': 160, 'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans'})


def what_happened(res, out):
    t = pd.read_csv(res/'episode_transitions_2d.csv')
    h = t[t['orig_cls'] == 'hard_switch']
    order = [('hard_switch', 'still hard switch', BLUE), ('drift', 'now drift', ORANGE), ('ended', 'now ended', AQUA), ('removed', 'removed (onset after the cut or run cut short)', GREY)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bottom = np.zeros(len(SRC))
    for cls, lab, col in order:
        v = np.array([(h[(h['src'] == s)]['cls'] == cls).sum() for s in SRC], float)
        ax.bar(range(len(SRC)), v, bottom=bottom, color=col, edgecolor='white', linewidth=1.5, label=lab, width=0.62)
        for i, (b, x) in enumerate(zip(bottom, v)):
            if x >= 6:
                ax.text(i, b+x/2, f'{int(x)}', ha='center', va='center', color='white' if cls != 'removed' else INK, fontsize=9)
        bottom += v
    for i, s in enumerate(SRC):
        ax.text(i, bottom[i]+4, f'{int(bottom[i])}', ha='center', va='bottom', color=INK, fontsize=9)
    ax.set_xticks(range(len(SRC)))
    ax.set_xticklabels([NAME[s] for s in SRC])
    ax.set_ylabel('original hard switches (count)')
    ax.set_xlabel('scroll')
    ax.set_title(f'Fate of the {int(len(h))} original hard switches after scoring only where a human label exists', fontsize=10, loc='left')
    ax.set_ylim(0, bottom.max()*1.32)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.grid(axis='x', visible=False)
    fig.savefig(out/'what_happened.png')
    plt.close(fig)


def rates_by_condition(res, out):
    r = pd.read_csv(res/'rates_by_condition.csv')
    panels = [('Q2 bend', 'bend (lowest to highest third)', ['lowest', 'middle', 'highest']),
              ('S2 kink, hard switches', 'kink (lowest to highest third)', ['lowest', 'middle', 'highest']),
              ('Q3 crossing', 'distance to a crossing fiber', ['nearest', 'middle', 'farthest']),
              ('Q4 parallel', 'distance to a parallel fiber', ['nearest', 'middle', 'farthest']),
              ('S4 seed distance, hard switches', 'distance from the seed (vx)', ['< 100', '100-400', '> 400']),
              ('S3 junction, hard switches', 'distance to a junction (vx)', ['<= 16', '> 16'])]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey=True)
    for ax, (key, xl, labs) in zip(axes.ravel(), panels):
        g = r[r['panel'] == key].reset_index(drop=True)
        x = np.arange(len(g))
        ax.bar(x, g['rate_per_cm'], color=BLUE, width=0.55, edgecolor='white')
        ax.errorbar(x, g['rate_per_cm'], yerr=[g['rate_per_cm']-g['lo'], g['hi']-g['rate_per_cm']], fmt='none', ecolor=INK2, elinewidth=1.2, capsize=3)
        for xi, ev, rt, hi in zip(x, g['events'], g['rate_per_cm'], g['hi']):
            ax.text(xi, hi, f'n={int(ev)}', ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_xlabel(xl)
        ax.grid(axis='x', visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel('hard switches per cm at risk')
    axes[0, 0].set_ylim(0, max(0.001, float(r['hi'].max()))*1.18)
    fig.suptitle('Corrected hard switches per cm by condition, all scrolls pooled (bars: events/exposure; whiskers: 95% fiber-bootstrap CI)', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out/'rates_by_condition.png')
    plt.close(fig)


def profiles(res, out):
    p = pd.read_csv(res/'s5_profiles.csv')
    panels = [('bend', 'bend (degrees)'), ('kink', 'kink (degrees)'), ('cross_d', 'distance to nearest crossing fiber (vx)'), ('par_d', 'distance to nearest parallel fiber (vx)')]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True)
    for ax, (k, yl) in zip(axes.ravel(), panels):
        g = p[p['covariate'] == k]
        ax.fill_between(g['offset_vx'], g['lo'], g['hi'], color=BLUE, alpha=0.22, linewidth=0)
        ax.plot(g['offset_vx'], g['mean'], color=BLUE, linewidth=2, label='hard switches (mean, 95% band)')
        b = g.iloc[0]
        ax.axhspan(b['baseline_lo'], b['baseline_hi'], color=GREY, alpha=0.25, linewidth=0)
        ax.axhline(b['baseline'], color=INK2, linestyle='--', linewidth=1.4, label='exposure baseline (scroll-mix weighted)')
        ax.axvline(0, color=INK2, linewidth=0.8, alpha=0.6)
        ax.set_ylabel(yl)
    for ax in axes[1]:
        ax.set_xlabel('offset along the GT, direction of travel (vx)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc='lower center', ncol=2)
    fig.suptitle('Covariate profiles around the corrected hard switches (S5, pre-registered, run on corrected episodes)', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out/'profiles.png')
    plt.close(fig)


def detector_recall(res, out):
    d = pd.read_csv(res/'detection_recheck.csv')
    groups = [*SRC, 'unseen pooled']
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    w = 0.19
    specs = [('original', 'caught_fb', 'recall_fb', BLUE, '///', 'fb check, original'), ('corrected', 'caught_fb', 'recall_fb', BLUE, None, 'fb check, corrected'),
             ('original', 'caught_conf', 'recall_conf', ORANGE, '///', 'confidence stop, original'), ('corrected', 'caught_conf', 'recall_conf', ORANGE, None, 'confidence stop, corrected')]
    for j, (ver, cnt, rec, col, hatch, lab) in enumerate(specs):
        xs, ys, labs = [], [], []
        for i, g in enumerate(groups):
            row = d[(d['version'] == ver) & (d['scroll'] == g)].iloc[0]
            xs.append(i+(j-1.5)*w); ys.append(100*row[rec]); labs.append(f'{int(row[cnt])}/{int(row["n"])}')
        ax.bar(xs, ys, width=w*0.92, color=col if hatch is None else 'white', edgecolor=col, hatch=hatch, linewidth=1.2, label=lab)
        for x, y, t in zip(xs, ys, labs):
            ax.text(x, y+1, t, ha='center', va='bottom', rotation=90, fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([NAME[g] for g in groups])
    ax.set_ylabel('recall of hard switches (%)')
    ax.set_xlabel('scroll (bar labels: caught / hard switches checked)')
    ax.set_ylim(0, 112)
    ax.set_title('Detector recall on hard switches: original vs after scoring only where a human label exists', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper left')
    ax.grid(axis='x', visible=False)
    fig.savefig(out/'detector_recall.png')
    plt.close(fig)


def past_end_example(res, out):
    z = np.load(res/'past_end_example_data.npz')
    P, L, ds, G = z['path'], z['lengths'], z['stored_ds'], z['gt_points']
    cut, onset = float(z['cut_len']), int(z['onset_idx'])
    sign = float(z['sign'])
    toward = G[-1] if sign > 0 else G[0]
    cut_pt = np.array([np.interp(cut, L, P[:, k]) for k in range(3)])
    allp = np.vstack([G, P])
    c = allp.mean(0)
    _, _, vt = np.linalg.svd(allp-c, full_matrices=False)
    proj = lambda X: (np.atleast_2d(X)-c) @ vt[:2].T
    Gp, Pp = proj(G), proj(P)
    e, cp, op, sp = proj(toward)[0], proj(cut_pt)[0], proj(P[onset])[0], proj(P[0])[0]
    on = ds[1:] <= 3
    seg = np.stack([Pp[:-1], Pp[1:]], 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax, zoom in ((a1, False), (a2, True)):
        ax.plot(Gp[:, 0], Gp[:, 1], color=GREY if zoom else INK, linewidth=5.5 if zoom else 2.2, solid_capstyle='round', zorder=1, alpha=0.55 if zoom else 1)
        ax.add_collection(LineCollection(seg[on], colors=BLUE, linewidths=2.2 if zoom else 1.6, zorder=3))
        ax.add_collection(LineCollection(seg[~on], colors=ORANGE, linewidths=1.8 if zoom else 1.2, zorder=2))
        ax.plot(*e, marker='*', color=INK, markersize=18 if zoom else 15, markeredgecolor='white', linestyle='none', zorder=6)
        ax.plot(*cp, marker='D', color='white', markeredgecolor=INK, markersize=9, markeredgewidth=1.8, linestyle='none', zorder=7)
        ax.plot(*op, marker='v', color=AQUA, markeredgecolor=INK, markersize=11, markeredgewidth=1.2, linestyle='none', zorder=7)
        ax.plot(*sp, marker='o', color='white', markeredgecolor=INK, markersize=7, linestyle='none', zorder=7)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('first principal axis (vx)')
    a1.set_ylabel('second principal axis (vx)')
    a1.autoscale_view()
    r = max(np.linalg.norm(sp-e), np.linalg.norm(op-e), np.linalg.norm(cp-e), 30.0)*1.7
    a2.set_xlim(e[0]-r, e[0]+r); a2.set_ylim(e[1]-r, e[1]+r)
    a2.set_title(f'zoom on the GT end (box {2*r:.0f} vx wide)', fontsize=9, loc='left')
    a1.add_patch(plt.Rectangle((e[0]-r, e[1]-r), 2*r, 2*r, fill=False, edgecolor=INK2, linewidth=1, linestyle=':'))
    handles = [Line2D([], [], color=GREY, linewidth=4, label='human GT fiber'), Line2D([], [], color=BLUE, linewidth=2, label='path within 3 vx of GT (stored distance)'),
               Line2D([], [], color=ORANGE, linewidth=2, label='path farther than 3 vx from GT'),
               Line2D([], [], marker='*', color=INK, markersize=13, linestyle='none', label='GT end the trace travels toward'),
               Line2D([], [], marker='D', color='white', markeredgecolor=INK, markeredgewidth=1.8, markersize=8, linestyle='none', label=f'cut: path crosses the GT end plane (path length {cut:.0f} vx)'),
               Line2D([], [], marker='v', color=AQUA, markeredgecolor=INK, markersize=10, linestyle='none', label=f'original hard-switch onset (path length {float(L[onset]):.0f} vx, after the cut)'),
               Line2D([], [], marker='o', color='white', markeredgecolor=INK, markersize=7, linestyle='none', label='path start')]
    a1.legend(handles=handles, frameon=True, framealpha=0.9, edgecolor='none', fontsize=8, loc='upper left')
    a1.set_title(f'{NAME[str(z["src"])]} trace {int(z["trace"])}: hard switch removed, {float(z["after_cut_len"]):.0f} vx of path after the cut', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(out/'past_end_example.png')
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for f in (what_happened, rates_by_condition, profiles, detector_recall, past_end_example):
        f(res, out)
        print('wrote', f.__name__, flush=True)
