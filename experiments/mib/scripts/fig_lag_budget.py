"""Fig — LAG (Local Approximation Gap) noise budget, 4 models x 2 tasks.
Reads experiments/mib/method_comparison/lag/{model}_{task}.json (from lag_opgap.py).
(a) EAP LAG per op, IOI          (b) EAP LAG per op, MCQA
(c) softmax IG convergence       (d) task effect: MCQA vs IOI EAP LAG per (op, model)
"""
import json, os, sys, argparse
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
PROJ = '/home/dacslab/djk/lasse-circuit/circuit_discovery'
LAG_DIR = f'{PROJ}/experiments/mib/method_comparison/lag'
mpl.rcParams.update({
    'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6,
    'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'text.color': '#0b0b0b', 'axes.labelcolor': '#0b0b0b', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})
INK = '#0b0b0b'; SEC = '#52514e'; GRID = '#e6e5e2'
MODELS = ['gpt2', 'qwen2.5', 'gemma2', 'llama3']
LABEL = {'gpt2': 'GPT-2 small', 'qwen2.5': 'Qwen2.5-0.5B', 'gemma2': 'Gemma-2-2B', 'llama3': 'Llama-3.1-8B'}
COLOR = {'gpt2': '#2a78d6', 'qwen2.5': '#e0862a', 'gemma2': '#1baf7a', 'llama3': '#8a4fd6'}
OPS = [('softmax', 'softmax', 'IG (Z×)'), ('QK', 'Q@K', 'midpoint'), ('QKc', 'Q@K (row-centered)', 'midpoint'), ('act', 'MLP act', 'secant'),
       ('gateup', 'gate·up', 'midpoint'), ('AV', 'A@V', 'midpoint'), ('norm', 'norm', 'freeze'),
       ('softcap', 'softcap', 'secant')]
ZS = [1, 2, 5, 10, 20]
ap = argparse.ArgumentParser(); ap.add_argument('--tasks', nargs='+', default=['ioi', 'mcqa']); ap.add_argument('--out', default='fig_lag_budget')
args = ap.parse_args(); TASKS = args.tasks
TNAME = {'ioi': 'IOI', 'mcqa': 'MCQA', 'arc_easy': 'ARC-easy', 'arc_challenge': 'ARC-challenge', 'arithmetic_addition': 'arithmetic (+)', 'arithmetic_subtraction': 'arithmetic (−)'}

R = {}
for m in MODELS:
    for t in TASKS:
        p = f'{LAG_DIR}/{m}_{t}.json'
        if os.path.exists(p): R[(m, t)] = json.load(open(p))
def eap(m, t, op):
    r = R.get((m, t)); return np.nan if r is None or op not in r['ops'] else r['ops'][op]['eap']

NB = len(TASKS); NC = 2 if NB <= 2 else 3
fig, axes = plt.subplots(2, NC, figsize=(5.25 * NC, 8.2))
budget_axes = list(axes[0]) + list(axes[1])
budget_axes, rest = budget_axes[:NB], budget_axes[NB:]
axC, axD = rest[0], (rest[1] if len(rest) > 1 else None)
for ax in rest[2:]: ax.axis('off')

def budget_panel(ax, task, title):
    n_ops = len(OPS); y = np.arange(n_ops)[::-1]; h = 0.19
    xmax = 0
    for j, m in enumerate(MODELS):
        vals = [eap(m, task, op) for op, _, _ in OPS]
        off = (1.5 - j) * (h + 0.015)
        ax.barh(y + off, vals, height=h, color=COLOR[m], label=LABEL[m], zorder=3)
        for yi, v in zip(y, vals):
            if not np.isnan(v):
                ax.text(v + 0.02, yi + off, f'{v:.2f}', va='center', ha='left', fontsize=5.8, color=INK)
                xmax = max(xmax, v)
    ax.axvline(1.0, color=SEC, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([nm for _, nm, _ in OPS], fontsize=8.3)
    for yi, (_, _, c) in zip(y, OPS):
        ax.text(1.72, yi, c, va='center', ha='left', fontsize=7, color=SEC, style='italic')
    ax.text(1.72, n_ops - 0.6, 'corrector', va='center', ha='left', fontsize=7, color=SEC, weight='bold')
    ax.set_xlim(0, 2.05); ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_xlabel('EAP LAG  (relative first-order error)', fontsize=8.3)
    ax.set_title(title, fontsize=9, loc='left', color=INK, pad=8)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
for i, (ax, t) in enumerate(zip(budget_axes, TASKS)): budget_panel(ax, t, f'({chr(97+i)})  noise budget — {TNAME[t]}')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COLOR[m], label=LABEL[m]) for m in MODELS], frameon=False, fontsize=8,
           loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.005), handlelength=1.1, handleheight=1.1)

# softmax IG convergence
LS = ['-', (0, (3, 2)), (0, (1, 1.2)), (0, (4, 1.5, 1, 1.5))]
for m in MODELS:
    for t, ls in zip(TASKS, LS):
        r = R.get((m, t))
        if r is None: continue
        ig = [r['ops']['softmax']['ig'][str(z)] for z in ZS]
        axC.plot(ZS, ig, ls=ls, marker='o', color=COLOR[m], lw=1.6, ms=4.5, mfc=COLOR[m], mec='white', mew=0.8,
                 label=f'{LABEL[m]} / {TNAME[t]}', zorder=4)
axC.set_xscale('log'); axC.set_yscale('log'); axC.set_xticks(ZS); axC.set_xticklabels(ZS)
axC.set_xlim(0.9, 22)
axC.set_xlabel('integration steps  Z', fontsize=8.3); axC.set_ylabel('softmax LAG', fontsize=8.3)
axC.set_title(f'({chr(97+NB)})  softmax: Z-step IG residual (line style = task, see legend)', fontsize=8.5, loc='left', color=INK, pad=8)
axC.grid(color=GRID, lw=0.6, which='both', zorder=0); axC.set_axisbelow(True)
for s in ['top', 'right']: axC.spines[s].set_visible(False)
axC.tick_params(length=0, which='both')
axC.legend(frameon=False, fontsize=6.5, loc='lower left', handlelength=1.6, ncol=2)

# task effect scatter (last task vs first task)
T0, T1 = TASKS[0], TASKS[-1]
MARK = {'softmax': 'o', 'QK': 's', 'QKc': 'p', 'act': '^', 'gateup': 'D', 'AV': 'v', 'norm': 'P', 'softcap': 'x'}
lo, hi = 1e9, 0
for m in MODELS:
    for op, nm, _ in OPS:
        a, b = eap(m, T0, op), eap(m, T1, op)
        if axD is None or np.isnan(a) or np.isnan(b): continue
        axD.scatter(a, b, marker=MARK[op], s=34, color=COLOR[m], edgecolor='white', lw=0.6, zorder=4)
        lo = min(lo, a, b); hi = max(hi, a, b)
lo = lo / 1.6 if lo < 1e9 else 1e-3; hi = hi * 1.6 if hi > 0 else 2
if axD is not None:
  axD.plot([lo, hi], [lo, hi], color=SEC, lw=0.8, ls=(0, (4, 3)), zorder=2)
  axD.set_xscale('log'); axD.set_yscale('log'); axD.set_xlim(lo, hi); axD.set_ylim(lo, hi)
  axD.set_xlabel(f'EAP LAG — {TNAME[T0]}', fontsize=8.3); axD.set_ylabel(f'EAP LAG — {TNAME[T1]}', fontsize=8.3)
  axD.set_title(f'({chr(98+NB)})  task effect: same op, {TNAME[T1]} vs {TNAME[T0]} (marker = op, color = model)', fontsize=8.5, loc='left', color=INK, pad=8)
  axD.grid(color=GRID, lw=0.6, which='both', zorder=0); axD.set_axisbelow(True)
for s in ['top', 'right']: axD.spines[s].set_visible(False)
axD.tick_params(length=0, which='both')
from matplotlib.lines import Line2D
axD.legend(handles=[Line2D([], [], marker=MARK[op], ls='', color=SEC, ms=5, label=nm) for op, nm, _ in OPS],
           frameon=False, fontsize=6.5, loc='lower right', ncol=2, handlelength=1.0)

plt.tight_layout(w_pad=2.0, h_pad=2.2, rect=(0, 0, 1, 0.975))
for ext in ['pdf', 'png']:
    fig.savefig(f'{PROJ}/experiments/mib/method_comparison/{args.out}.{ext}', dpi=200, bbox_inches='tight')
print(f'saved {args.out}.pdf / .png   cells:', sorted(R))
