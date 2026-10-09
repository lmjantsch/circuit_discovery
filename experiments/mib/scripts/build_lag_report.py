"""Build the LAG (Local Approximation Gap) tables from experiments/mib/method_comparison/lag/*.json
and splice them into experiments/mib/method_comparison/lag_report.md between
<!-- LAG-TABLES-START --> / <!-- LAG-TABLES-END --> (the narrative outside the markers is kept).
"""
import json, glob, os, math
PROJ = '/home/dacslab/djk/lasse-circuit/circuit_discovery'
LAG_DIR = f'{PROJ}/experiments/mib/method_comparison/lag'
REPORT = f'{PROJ}/experiments/mib/method_comparison/lag_report.md'
MODELS = ['gpt2', 'qwen2.5', 'gemma2', 'llama3']
GROUPS = {'MAIN': ['ioi', 'mcqa'], 'ARC': ['arc_easy', 'arc_challenge', 'arithmetic_addition', 'arithmetic_subtraction']}
MIB = {('gpt2','ioi'),('qwen2.5','ioi'),('gemma2','ioi'),('llama3','ioi'),('qwen2.5','mcqa'),('gemma2','mcqa'),('llama3','mcqa'),
       ('gemma2','arc_easy'),('llama3','arc_easy'),('llama3','arc_challenge'),('llama3','arithmetic_addition'),('llama3','arithmetic_subtraction')}
def tag(m, t): return '' if (m, t) in MIB else '†'
SHORT = {'ioi': 'ioi', 'mcqa': 'mcqa', 'arc_easy': 'arcE', 'arc_challenge': 'arcC', 'arithmetic_addition': 'arith+', 'arithmetic_subtraction': 'arith−'}
OPS = [('softmax', 'softmax', 'IG (Z×)'), ('QK', 'Q@K', 'midpoint'), ('QKc', 'Q@K row-centered', 'midpoint'), ('act', 'MLP act', 'secant'),
       ('gateup', 'gate·up', 'midpoint'), ('AV', 'A@V', 'midpoint'), ('norm', 'norm', 'freeze'),
       ('softcap', 'softcap', 'secant')]
ARCH_ACT = {'gpt2': 'gelu_new', 'qwen2.5': 'SiLU', 'gemma2': 'gelu_tanh', 'llama3': 'SiLU'}
ARCH_NORM = {'gpt2': 'LN', 'qwen2.5': 'RMS(w)', 'gemma2': 'RMS(1+w)', 'llama3': 'RMS(w)'}

R = {}
for m in MODELS:
    for t in sum(GROUPS.values(), []):
        p = f'{LAG_DIR}/{m}_{t}.json'
        if os.path.exists(p): R[(m, t)] = json.load(open(p))
print('cells found:', sorted(R))

def f4(x): return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.4f}'
def fe(x): return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.1e}' if x < 1e-3 else f'{x:.4f}')
def cell_hdr(): return ' | '.join(f'{m}/{SHORT[t]}' for m, t in CELLS)
def get(m, t, op, key):
    r = R.get((m, t));
    if r is None or op not in r['ops']: return None
    return r['ops'][op][key]

def build_tables(TASKS):
    CELLS = [(m, t) for t in TASKS for m in MODELS]
    def cell_hdr(): return ' | '.join(f'{m}/{SHORT[t]}{tag(m, t)}' for m, t in CELLS)
    L = []
    L.append('† = MIB circuit track 에 없는 참고 cell (CPR 대응값 없음). 나머지는 MIB cell.\n')
    L.append('### T1. EAP LAG — noise budget (first-order relative error per op)\n')
    L.append('행 = op (큰 잡음원 순), 열 = model/task. 값 = `‖J_clean·Δ − Δz_true‖ / ‖Δz_true‖` (전 example·layer·head·pos·dim RMS). `—` = 그 아키텍처에 없는 op.\n')
    L.append(f'| op | corrector | {cell_hdr()} |')
    L.append('|---|---|' + '---|' * len(CELLS))
    for op, nm, corr in OPS:
        L.append(f'| **{nm}** | {corr} | ' + ' | '.join(f4(get(m, t, op, 'eap')) for m, t in CELLS) + ' |')
    L.append('')
    L.append('### T1b. EAP LAG — per-layer median (robust to norm-dominant layers)\n')
    L.append('T1 은 전 layer 를 norm 으로 합산(pooled)하므로 `‖Δz_true‖` 가 큰 소수 layer 가 지배할 수 있다. 여기서는 layer 별 LAG 의 중앙값 (norm 은 final norm 포함).\n')
    L.append(f'| op | {cell_hdr()} |')
    L.append('|---|' + '---|' * len(CELLS))
    import statistics
    def layer_median(m, t, op):
        r = R.get((m, t))
        if r is None or op not in r['ops']: return None
        NL = r['arch']['NL']; pl = r['ops'][op]['per_layer']['eap']
        v = [x for x in (pl if op == 'norm' else pl[:NL]) if x == x]
        return statistics.median(v) if v else None
    for op, nm, corr in OPS:
        L.append(f'| **{nm}** | ' + ' | '.join(f4(layer_median(m, t, op)) for m, t in CELLS) + ' |')
    L.append('')
    L.append('### T2. module LAG — residual after the corrector (softmax: IG Z=5)\n')
    L.append(f'| op | corrector | {cell_hdr()} |')
    L.append('|---|---|' + '---|' * len(CELLS))
    for op, nm, corr in OPS:
        vals = []
        for m, t in CELLS:
            if op == 'softmax':
                r = R.get((m, t)); vals.append(fe(r['ops']['softmax']['ig']['5']) if r else '—')
            else: vals.append(fe(get(m, t, op, 'mod')))
        L.append(f'| **{nm}** | {corr} | ' + ' | '.join(vals) + ' |')
    L.append('')
    L.append('### T3. perturbation size at the op — `‖Δz_true‖ / ‖z_clean‖`\n')
    L.append('clean→corrupt 전이가 각 op 출력을 상대적으로 얼마나 움직이는가. 2차항은 섭동의 제곱이라 상대 LAG 는 이 값과 함께 커진다.\n')
    L.append(f'| op | {cell_hdr()} |')
    L.append('|---|' + '---|' * len(CELLS))
    for op, nm, corr in OPS:
        L.append(f'| **{nm}** | ' + ' | '.join(f4(get(m, t, op, 'pert')) for m, t in CELLS) + ' |')
    L.append('')
    L.append('### T4. softmax IG convergence (LAG vs Z)\n')
    L.append('| cell | EAP tangent | Z=1 | Z=2 | Z=5 | Z=10 | Z=20 |')
    L.append('|---|---|---|---|---|---|---|')
    for m, t in CELLS:
        r = R.get((m, t))
        if r is None: continue
        ig = r['ops']['softmax']['ig']
        L.append(f'| {m}/{SHORT[t]}{tag(m, t)} | {f4(r["ops"]["softmax"]["eap"])} | ' + ' | '.join(f4(ig[str(z)]) for z in [1, 2, 5, 10, 20]) + ' |')
    L.append('')
    L.append('### T5. norm — EAP (J_full) vs FrLN (J_frozen)\n')
    L.append('| cell | norm type | #norms | EAP LAG | FrLN LAG | FrLN/EAP | verdict |')
    L.append('|---|---|---|---|---|---|---|')
    for m, t in CELLS:
        r = R.get((m, t))
        if r is None: continue
        e = r['ops']['norm']['eap']; f = r['ops']['norm']['mod']; ratio = f / e
        verdict = '개선' if ratio < 0.9 else ('악화' if ratio > 1.1 else '중립')
        L.append(f'| {m}/{SHORT[t]}{tag(m, t)} | {ARCH_NORM[m]} | {r["arch"]["n_norms"]} | {f4(e)} | {f4(f)} | {ratio:.2f}× | {verdict} |')
    L.append('')
    L.append(f'### T6. task effect — {SHORT[TASKS[-1]]} / {SHORT[TASKS[0]]} ratio of EAP LAG and of perturbation size\n')
    L.append('| op | ' + ' | '.join(f'{m} LAG' for m in MODELS) + ' | ' + ' | '.join(f'{m} pert' for m in MODELS) + ' |')
    L.append('|---|' + '---|' * (2 * len(MODELS)))
    for op, nm, corr in OPS:
        cols = []
        for key in ['eap', 'pert']:
            for m in MODELS:
                a = get(m, TASKS[0], op, key); b = get(m, TASKS[-1], op, key)
                cols.append('—' if a is None or b is None else f'{b/a:.2f}×')
        L.append(f'| **{nm}** | ' + ' | '.join(cols) + ' |')
    L.append('')
    L.append('### T7. run metadata\n')
    L.append('| cell | model id | dtype | layers | heads (Q/KV) | head_dim | act | norm | aligned ex. | time | ‖Δq‖/‖q‖ | ‖Δk‖/‖k‖ |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for m, t in CELLS:
        r = R.get((m, t))
        if r is None: continue
        a = r['arch']
        L.append(f'| {m}/{SHORT[t]}{tag(m, t)} | `{r["model_id"]}` | {r["dtype"].replace("torch.", "")} | {a["NL"]} | {a["nH"]}/{a["nKV"]} | {a["HD"]} | {ARCH_ACT[m]} | {ARCH_NORM[m]} | {r["n_aligned"]}/{r["num_requested"]} | {r["elapsed_s"]:.0f}s | {f4(r.get("qk_rel_dq"))} | {f4(r.get("qk_rel_dk"))} |')
    L.append('')
    return '\n'.join(L)


def splice(txt, START, END, tables):
    if START in txt and END in txt:
        pre = txt[:txt.index(START) + len(START)]; post = txt[txt.index(END):]
        return pre + '\n' + tables + '\n' + post
    return txt + '\n' + START + '\n' + tables + '\n' + END + '\n'
def build_summary():
    ALL = sum(GROUPS.values(), [])
    L = ['### T8. cross-task summary — pooled EAP LAG, min–max over the 4 models (softcap: gemma2 only)\n',
         '| op | ' + ' | '.join(SHORT[t] for t in ALL) + ' |', '|---|' + '---|' * len(ALL)]
    for op, nm, corr in OPS:
        cells = []
        for t in ALL:
            v = [get(m, t, op, 'eap') for m in MODELS]; v = [x for x in v if x is not None]
            cells.append('—' if not v else (f'{v[0]:.2f}' if len(v) == 1 else f'{min(v):.2f}–{max(v):.2f}'))
        L.append(f'| **{nm}** | ' + ' | '.join(cells) + ' |')
    L.append('')
    L.append('perturbation size `pert`, min–max over models:\n')
    L.append('| op | ' + ' | '.join(SHORT[t] for t in ALL) + ' |'); L.append('|---|' + '---|' * len(ALL))
    for op, nm, corr in OPS:
        cells = []
        for t in ALL:
            v = [get(m, t, op, 'pert') for m in MODELS]; v = [x for x in v if x is not None]
            cells.append('—' if not v else (f'{v[0]:.2f}' if len(v) == 1 else f'{min(v):.2f}–{max(v):.2f}'))
        L.append(f'| **{nm}** | ' + ' | '.join(cells) + ' |')
    L.append('')
    L.append('FrLN/EAP norm ratio per model (개선 < 1):\n')
    L.append('| model | ' + ' | '.join(SHORT[t] for t in ALL) + ' |'); L.append('|---|' + '---|' * len(ALL))
    for m in MODELS:
        cells = []
        for t in ALL:
            e, f = get(m, t, 'norm', 'eap'), get(m, t, 'norm', 'mod')
            cells.append('—' if e is None else f'{f/e:.2f}×')
        L.append(f'| {m} ({ARCH_NORM[m]}) | ' + ' | '.join(cells) + ' |')
    L.append('')
    L.append('softmax IG Z=2 residual, min–max over models:\n')
    L.append('| ' + ' | '.join(SHORT[t] for t in ALL) + ' |'); L.append('|' + '---|' * len(ALL))
    cells = []
    for t in ALL:
        v = [R[(m, t)]['ops']['softmax']['ig']['2'] for m in MODELS if (m, t) in R]
        cells.append('—' if not v else f'{min(v):.3f}–{max(v):.3f}')
    L.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(L)

txt = open(REPORT).read() if os.path.exists(REPORT) else '# LAG report\n'
txt = splice(txt, '<!-- LAG-SUMMARY-START -->', '<!-- LAG-SUMMARY-END -->', build_summary())
for g, TASKS in GROUPS.items():
    gtag = '' if g == 'MAIN' else f'-{g}'
    if not any((m, t) in R for m in MODELS for t in TASKS): continue
    txt = splice(txt, f'<!-- LAG-TABLES{gtag}-START -->', f'<!-- LAG-TABLES{gtag}-END -->', build_tables(TASKS))
open(REPORT, 'w').write(txt)
print('wrote', REPORT)
