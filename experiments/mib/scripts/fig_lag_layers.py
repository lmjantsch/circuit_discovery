"""Fig — LAG per-layer profiles (EAP LAG vs normalized depth), 4 models x 2 tasks.
Rows: IOI, MCQA.  Columns: softmax, Q@K (row-centered), MLP act, gate·up, norm.
Reads experiments/mib/method_comparison/lag/{model}_{task}.json.
"""
import json, os, argparse
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
PROJ = '/home/dacslab/djk/lasse-circuit/circuit_discovery'
LAG_DIR = f'{PROJ}/experiments/mib/method_comparison/lag'
mpl.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.6, 'axes.edgecolor': '#52514e',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b', 'axes.labelcolor': '#0b0b0b',
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
INK = '#0b0b0b'; SEC = '#52514e'; GRID = '#e6e5e2'
MODELS = ['gpt2', 'qwen2.5', 'gemma2', 'llama3']
LABEL = {'gpt2': 'GPT-2 small', 'qwen2.5': 'Qwen2.5-0.5B', 'gemma2': 'Gemma-2-2B', 'llama3': 'Llama-3.1-8B'}
COLOR = {'gpt2': '#2a78d6', 'qwen2.5': '#e0862a', 'gemma2': '#1baf7a', 'llama3': '#8a4fd6'}
ap = argparse.ArgumentParser(); ap.add_argument('--tasks', nargs='+', default=['ioi', 'mcqa']); ap.add_argument('--out', default='fig_lag_layers')
args = ap.parse_args(); TASKS = args.tasks
TNAME = {'ioi': 'IOI', 'mcqa': 'MCQA', 'arc_easy': 'ARC-easy', 'arc_challenge': 'ARC-chal.', 'arithmetic_addition': 'arith (+)', 'arithmetic_subtraction': 'arith (−)'}
COLS = [('softmax', 'softmax'), ('QKc', 'Q@K (row-centered)'), ('act', 'MLP act'), ('gateup', 'gate·up'), ('norm', 'norm')]
R = {(m, t): json.load(open(f'{LAG_DIR}/{m}_{t}.json')) for m in MODELS for t in TASKS if os.path.exists(f'{LAG_DIR}/{m}_{t}.json')}
fig, axes = plt.subplots(len(TASKS), len(COLS), figsize=(13, 2.8 * len(TASKS)), sharex=True, squeeze=False)
for ri, task in enumerate(TASKS):
    for ci, (op, nm) in enumerate(COLS):
        ax = axes[ri, ci]
        for m in MODELS:
            r = R.get((m, task))
            if r is None or op not in r['ops']: continue
            NL = r['arch']['NL']; pl = r['ops'][op]['per_layer']['eap']
            v = np.array(pl if op == 'norm' else pl[:NL], dtype=float); x = np.arange(len(v)) / max(len(v) - 1, 1)
            ax.plot(x, v, '-', color=COLOR[m], lw=1.3, alpha=0.9, label=LABEL[m], zorder=3)
            ax.scatter(x, v, s=7, color=COLOR[m], zorder=4)
        ax.axhline(1.0, color=SEC, lw=0.7, ls=(0, (4, 3)), zorder=2)
        ax.set_yscale('log'); ax.set_ylim(0.005, 5)
        ax.grid(color=GRID, lw=0.5, which='major', zorder=0); ax.set_axisbelow(True)
        for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
        ax.tick_params(length=0, which='both')
        if ri == 0: ax.set_title(nm, fontsize=9, color=INK, pad=6)
        if ci == 0: ax.set_ylabel(f'{TNAME[task]}\nEAP LAG (per layer)', fontsize=8.5)
        if ri == len(TASKS) - 1: ax.set_xlabel('depth  (layer / n_layers)', fontsize=8.3)
axes[0, 0].legend(frameon=False, fontsize=7, loc='lower left', handlelength=1.4)
plt.tight_layout(w_pad=1.2, h_pad=1.4)
for ext in ['pdf', 'png']:
    fig.savefig(f'{PROJ}/experiments/mib/method_comparison/{args.out}.{ext}', dpi=200, bbox_inches='tight')
print(f'saved {args.out}.pdf / .png   cells:', sorted(R))
