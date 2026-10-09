"""LAG (Local Approximation Gap) — unified op-level Taylor-gap measurement.

Generalizes the 7 per-model op-gap scripts (oplevel_*_{gpt2,gemma2}_ioi.py) to
{gpt2, qwen2.5, gemma2, llama3} x {ioi, mcqa}.  Same definition:

    LAG(Ê) = sqrt( Σ ‖Ê − Δz_true‖²_F / Σ ‖Δz_true‖²_F )

for one isolated nonlinear op f, with Δz_true = f(corrupt in) − f(clean in) taken from
two real forwards (full clean->corrupt transition), Ê = EAP first-order (clean-point
Jacobian) or the module corrector (midpoint / secant / frozen / IG).  Sums run over all
aligned examples, layers, heads, positions, hidden dims.

Ops per model
  Q@K      bilinear   midpoint   (post-rotary, post-GQA-repeat for qwen/gemma/llama)
  Q@K-c    bilinear   midpoint   same, row-centered over causal keys (softmax-invariant part)
  softmax  multi-D    IG(Z)      input = scaled (softcapped for gemma) attn logits
  softcap  elemwise   secant     gemma2 only
  A@V      bilinear   midpoint
  act      elemwise   secant     gelu_new (gpt2) / gelu_tanh (gemma2) / silu (qwen, llama)
  gate*up  bilinear   midpoint   gated MLP only (qwen/gemma/llama)
  norm     multi-D    freeze     LayerNorm (gpt2) / RMSNorm (others; gemma uses (1+w))

Extra column: pert = sqrt(Σ‖Δz_true‖² / Σ‖z_clean‖²)  (relative perturbation size at the op).

Usage:
  CUDA_VISIBLE_DEVICES=3 python experiments/mib/scripts/lag_opgap.py --model qwen2.5 --task ioi
Writes experiments/mib/method_comparison/lag/{model}_{task}.json and prints a table.
"""
import os, sys, time, math, json, argparse
PROJ = '/home/dacslab/djk/lasse-circuit/circuit_discovery'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from experiments.mib.data_utils import MIBDataset

MODEL_IDS = {
    'gpt2': 'openai-community/gpt2',
    'qwen2.5': 'Qwen/Qwen2.5-0.5B',
    'gemma2': 'google/gemma-2-2b',
    'llama3': 'meta-llama/Llama-3.1-8B',
}
# fp32 for small models (matches previous gpt2 runs); bf16 for the 2B/8B models (matches gemma2 runs)
DTYPES = {'gpt2': torch.float32, 'qwen2.5': torch.float32, 'gemma2': torch.bfloat16, 'llama3': torch.bfloat16}
ZS = [1, 2, 5, 10, 20]

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=list(MODEL_IDS))
ap.add_argument('--task', required=True, choices=['ioi', 'mcqa', 'arc_easy', 'arc_challenge', 'arithmetic_addition', 'arithmetic_subtraction'])
ap.add_argument('--num', type=int, default=100)
ap.add_argument('--split', default='train')
ap.add_argument('--out-dir', default='experiments/mib/method_comparison/lag')
args = ap.parse_args()
MODEL, TASK, NUM = args.model, args.task, args.num
MID = MODEL_IDS[MODEL]
print(f'LAG  model={MODEL} ({MID})  task={TASK}  split={args.split}  num={NUM}')
print(f'device: {torch.cuda.get_device_name(0)}  free={torch.cuda.mem_get_info()[0]/2**30:.1f} GiB')

tok = AutoTokenizer.from_pretrained(MID); tok.padding_side = 'right'
if not tok.pad_token: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=DTYPES[MODEL],
                                             attn_implementation='eager', device_map='cuda').eval()
cfg = model.config
IS_GPT2 = MODEL == 'gpt2'
IS_GEMMA = MODEL == 'gemma2'

# ---------------------------------------------------------------- architecture facts
if IS_GPT2:
    NL = cfg.n_layer; nH = cfg.n_head; nKV = nH; HD = cfg.n_embd // nH; NG = 1
    SCALE = HD ** -0.5; CAP = None; EPS = cfg.layer_norm_epsilon
    def act(x): return F.gelu(x, approximate='tanh')            # gelu_new
    layers = model.transformer.h
    norm_mods = [(f'n{L}_{nm}', L, getattr(layers[L], nm)) for L in range(NL) for nm in ['ln_1', 'ln_2']]
    norm_mods.append(('nf', NL, model.transformer.ln_f))
else:
    NL = cfg.num_hidden_layers; nH = cfg.num_attention_heads; nKV = cfg.num_key_value_heads
    HD = getattr(cfg, 'head_dim', None) or cfg.hidden_size // nH; NG = nH // nKV
    SCALE = (cfg.query_pre_attn_scalar ** -0.5) if IS_GEMMA else HD ** -0.5
    CAP = cfg.attn_logit_softcapping if IS_GEMMA else None
    EPS = cfg.rms_norm_eps
    if cfg.hidden_act in ('silu', 'swish'):
        def act(x): return F.silu(x)
    elif 'gelu' in cfg.hidden_act:                                # gelu_pytorch_tanh
        def act(x): return F.gelu(x, approximate='tanh')
    else:
        raise ValueError(cfg.hidden_act)
    if MODEL == 'qwen2.5':
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    elif IS_GEMMA:
        from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb, repeat_kv
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    layers = model.model.layers
    norm_names = ['input_layernorm', 'post_attention_layernorm', 'pre_feedforward_layernorm', 'post_feedforward_layernorm']
    norm_mods = [(f'n{L}_{nm}', L, getattr(layers[L], nm)) for L in range(NL) for nm in norm_names
                 if getattr(layers[L], nm, None) is not None]
    norm_mods.append(('nf', NL, model.model.norm))
print(f'NL={NL} nH={nH} nKV={nKV} HD={HD} groups={NG} scale={SCALE:.5f} softcap={CAP} act={"gelu_new" if IS_GPT2 else cfg.hidden_act} norms={len(norm_mods)} dtype={DTYPES[MODEL]}')

def act_prime(x):
    xx = x.detach().clone().float().requires_grad_(True)
    with torch.enable_grad(): act(xx).sum().backward()
    return xx.grad

def softcap(x): return CAP * torch.tanh(x / CAP)
def softcap_d(u): return 1.0 - torch.tanh(u / CAP) ** 2

def norm_fwd(x, mod):
    if IS_GPT2:
        mu = x.mean(-1, keepdim=True); xc = x - mu; var = xc.pow(2).mean(-1, keepdim=True)
        return mod.weight.float() * xc / (var + EPS).sqrt() + mod.bias.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    w = mod.weight.float()
    return x * r * ((1.0 + w) if IS_GEMMA else w)

def norm_jac(x, D, mod):
    """returns (J_full·D, J_frozen·D) at clean point x for perturbation D"""
    if IS_GPT2:
        g = mod.weight.float()
        mu = x.mean(-1, keepdim=True); xc = x - mu; var = xc.pow(2).mean(-1, keepdim=True); std = (var + EPS).sqrt()
        Dc = D - D.mean(-1, keepdim=True); mean_xcDc = (xc * Dc).mean(-1, keepdim=True)
        return g * (Dc / std - xc * mean_xcDc / std.pow(3)), g * (Dc / std)
    W = (1.0 + mod.weight.float()) if IS_GEMMA else mod.weight.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS); mean_xD = (x * D).mean(-1, keepdim=True)
    return W * (D * r - x * r.pow(3) * mean_xD), W * (D * r)

# ---------------------------------------------------------------- hooks
cap = {}
def mk(key):
    def h(m, i, o): cap[key] = o.detach().float()
    return h
def mkpre(key):
    def h(m, a): cap[key] = a[0].detach().float()
    return h
def mkrot(m, i, o): cap['cos'], cap['sin'] = o[0].detach().float(), o[1].detach().float()
for L in range(NL):
    if IS_GPT2:
        layers[L].attn.c_attn.register_forward_hook(mk(f'qkv{L}'))
        layers[L].mlp.c_fc.register_forward_hook(mk(f'g{L}'))
    else:
        a = layers[L].self_attn
        a.q_proj.register_forward_hook(mk(f'q{L}')); a.k_proj.register_forward_hook(mk(f'k{L}')); a.v_proj.register_forward_hook(mk(f'v{L}'))
        layers[L].mlp.gate_proj.register_forward_hook(mk(f'g{L}'))
        layers[L].mlp.up_proj.register_forward_hook(mk(f'u{L}'))
for key, _, mod in norm_mods: mod.register_forward_pre_hook(mkpre(key))
if not IS_GPT2: model.model.rotary_emb.register_forward_hook(mkrot)

def heads(t, h):  # (B,S,h*HD) -> (B,h,S,HD)
    B, S, _ = t.shape; return t.view(B, S, h, HD).transpose(1, 2)

def qkv(d, L):
    """post-rotary, post-repeat q,k,v: (B,nH,S,HD) each"""
    if IS_GPT2:
        q, k, v = d[f'qkv{L}'].split(cfg.n_embd, dim=-1)
        return heads(q, nH), heads(k, nH), heads(v, nH)
    q = heads(d[f'q{L}'], nH); k = heads(d[f'k{L}'], nKV); v = heads(d[f'v{L}'], nKV)
    q, k = apply_rotary_pos_emb(q, k, d['cos'], d['sin'])
    return q, repeat_kv(k, NG), repeat_kv(v, NG)

def jvp_softmax(s, ds, m):
    A = torch.softmax(s + m, dim=-1); return A * (ds - (A * ds).sum(-1, keepdim=True))

# ---------------------------------------------------------------- accumulators (per layer; index NL = final norm)
OPS = ['QK', 'QKc', 'softmax', 'softcap', 'AV', 'act', 'gateup', 'norm']
qk_stats = {k: 0.0 for k in ['dq', 'q', 'dk', 'k']}
acc = {op: {k: [0.0] * (NL + 1) for k in ['eap', 'mod', 'true', 'clean']} for op in OPS}
accZ = {z: [0.0] * NL for z in ZS}
def add(op, L, est_eap, est_mod, true, clean):
    a = acc[op]
    a['eap'][L] += (est_eap - true).pow(2).sum().item()
    if est_mod is not None: a['mod'][L] += (est_mod - true).pow(2).sum().item()
    a['true'][L] += true.pow(2).sum().item(); a['clean'][L] += clean.pow(2).sum().item()

ds = MIBDataset(TASK, tok, MODEL, split=args.split, num_examples=NUM)
n = 0; skipped = 0; t0 = time.time()
with torch.no_grad():
    for i in range(len(ds)):
        prompt, base, *_ = ds[i]
        ic = tok(prompt, return_tensors='pt').to('cuda'); ib = tok(base, return_tensors='pt').to('cuda')
        if ic['input_ids'].shape[1] != ib['input_ids'].shape[1]: skipped += 1; continue
        cap.clear(); model(**ic); cl = {k: v.clone() for k, v in cap.items()}
        cap.clear(); model(**ib); co = {k: v.clone() for k, v in cap.items()}
        for L in range(NL):
            q, k, v = qkv(cl, L); qs, ks, vs = qkv(co, L)
            S = q.shape[2]; m = torch.triu(torch.full((S, S), float('-inf'), device='cuda'), 1)
            # --- Q@K (bilinear; scale-invariant ratio, use raw matmul) ---
            QK = q @ k.transpose(-1, -2); QKs = qs @ ks.transpose(-1, -2); dq = qs - q; dk = ks - k
            qk_fe = dq @ k.transpose(-1, -2) + q @ dk.transpose(-1, -2)
            qk_mid = dq @ ((k + ks) / 2).transpose(-1, -2) + ((q + qs) / 2) @ dk.transpose(-1, -2)
            add('QK', L, qk_fe, qk_mid, QKs - QK, QK)
            # --- Q@K row-centered over causal keys (the part softmax actually sees; removes per-query constants
            #     such as q·b_kᵀ from a large k_proj bias). Centering is linear, so midpoint stays exact. ---
            valid = torch.tril(torch.ones(S, S, device='cuda')); cnt = valid.sum(-1, keepdim=True)
            def rc(t): return (t - (t * valid).sum(-1, keepdim=True) / cnt) * valid
            add('QKc', L, rc(qk_fe), rc(qk_mid), rc(QKs - QK), rc(QK))
            qk_stats['dq'] += dq.pow(2).sum().item(); qk_stats['q'] += q.pow(2).sum().item()
            qk_stats['dk'] += dk.pow(2).sum().item(); qk_stats['k'] += k.pow(2).sum().item()
            # --- softcap (gemma only; element-wise) ---
            u = QK * SCALE; us = QKs * SCALE
            if CAP is not None:
                du = us - u; sc_true = softcap(us) - softcap(u); sc_first = softcap_d(u) * du
                add('softcap', L, sc_first, torch.where(du.abs() > 1e-6, sc_true, sc_first), sc_true, softcap(u))
                s, ss = softcap(u), softcap(us)
            else:
                s, ss = u, us
            # --- softmax (multi-D; tangent + IG) ---
            d = ss - s; A = torch.softmax(s + m, dim=-1); As = torch.softmax(ss + m, dim=-1)
            add('softmax', L, jvp_softmax(s, d, m), None, As - A, A)
            for z in ZS:
                ig = sum(jvp_softmax(s + ((j + 0.5) / z) * d, d, m) for j in range(z)) / z
                accZ[z][L] += (ig - (As - A)).pow(2).sum().item()
            # --- A@V (bilinear) ---
            dA = As - A; dv = vs - v
            add('AV', L, dA @ v + A @ dv, dA @ ((v + vs) / 2) + ((A + As) / 2) @ dv, As @ vs - A @ v, A @ v)
            # --- MLP activation (element-wise) ---
            g = cl[f'g{L}']; gs = co[f'g{L}']; dg = gs - g
            a_true = act(gs) - act(g); a_first = act_prime(g) * dg
            add('act', L, a_first, torch.where(dg.abs() > 1e-6, a_true, a_first), a_true, act(g))
            # --- gated MLP bilinear: x = act(gate), y = up ---
            if not IS_GPT2:
                x, xs = act(g), act(gs); y, ys = cl[f'u{L}'], co[f'u{L}']; dx = xs - x; dy = ys - y
                add('gateup', L, dx * y + x * dy, dx * (y + ys) / 2 + (x + xs) / 2 * dy, xs * ys - x * y, x * y)
        # --- norms (all, incl. final) ---
        for key, L, mod in norm_mods:
            x = cl[key]; xs = co[key]; D = xs - x
            Jfull, Jfroz = norm_jac(x, D, mod)
            add('norm', L, Jfull, Jfroz, norm_fwd(xs, mod) - norm_fwd(x, mod), norm_fwd(x, mod))
        n += 1
        if n % 25 == 0: print(f'  {n} examples  ({time.time()-t0:.0f}s)', flush=True)
elapsed = time.time() - t0
print(f'aligned examples: {n}/{len(ds)} (skipped {skipped})  ({elapsed:.0f}s)')

# ---------------------------------------------------------------- summarize
def ratio(num, den): return math.sqrt(num / den) if den > 0 else float('nan')
rows = []
MODNAME = {'QK': 'Bilinear(mid)', 'QKc': 'Bilinear(mid)', 'AV': 'Bilinear(mid)', 'gateup': 'Bilinear(mid)', 'act': 'SM(secant)',
           'softcap': 'secant', 'norm': 'FrLN(freeze)', 'softmax': 'IG(Z)'}
summary = {}
for op in OPS:
    a = acc[op]; T = sum(a['true'])
    if T == 0: continue
    eap = ratio(sum(a['eap']), T); mod = ratio(sum(a['mod']), T) if op != 'softmax' else None
    pert = ratio(T, sum(a['clean']))
    summary[op] = {'eap': eap, 'mod': mod, 'pert': pert,
                   'per_layer': {'eap': [ratio(a['eap'][L], a['true'][L]) for L in range(NL + 1)],
                                 'mod': [ratio(a['mod'][L], a['true'][L]) for L in range(NL + 1)] if op != 'softmax' else None,
                                 'pert': [ratio(a['true'][L], a['clean'][L]) for L in range(NL + 1)],
                                 'true_sq': a['true'], 'eap_sq': a['eap'], 'mod_sq': a['mod'], 'clean_sq': a['clean']}}
    rows.append((op, MODNAME[op], eap, mod, pert))
Tsm = sum(acc['softmax']['true'])
summary['softmax']['ig'] = {z: ratio(sum(accZ[z]), Tsm) for z in ZS}
summary['softmax']['ig_per_layer'] = {z: [ratio(accZ[z][L], acc['softmax']['true'][L]) for L in range(NL)] for z in ZS}

print(f'\n{"op":<9} {"corrector":<15} {"EAP LAG":>9} {"module LAG":>11} {"pert":>7}')
for op, mn, eap, mod, pert in rows:
    ms = f'{mod:.3e}' if mod is not None else '(IG below)'
    print(f'{op:<9} {mn:<15} {eap:>9.4f} {ms:>11} {pert:>7.4f}')
print('softmax IG:  ' + '  '.join(f'Z={z}: {summary["softmax"]["ig"][z]:.4f}' for z in ZS))
rel_dq = ratio(qk_stats['dq'], qk_stats['q']); rel_dk = ratio(qk_stats['dk'], qk_stats['k'])
print(f'attn inputs: ‖Δq‖/‖q‖ = {rel_dq:.4f}   ‖Δk‖/‖k‖ = {rel_dk:.4f}')

os.makedirs(args.out_dir, exist_ok=True)
out = {'model': MODEL, 'model_id': MID, 'task': TASK, 'split': args.split, 'num_requested': NUM,
       'n_aligned': n, 'skipped': skipped, 'dtype': str(DTYPES[MODEL]), 'elapsed_s': elapsed,
       'arch': {'NL': NL, 'nH': nH, 'nKV': nKV, 'HD': HD, 'scale': SCALE, 'softcap': CAP, 'eps': EPS,
                'act': 'gelu_new' if IS_GPT2 else cfg.hidden_act, 'norm': 'LayerNorm' if IS_GPT2 else 'RMSNorm',
                'n_norms': len(norm_mods)},
       'ZS': ZS, 'qk_rel_dq': rel_dq, 'qk_rel_dk': rel_dk, 'ops': summary}
path = os.path.join(args.out_dir, f'{MODEL}_{TASK}.json')
with open(path, 'w') as f: json.dump(out, f, indent=1)
print(f'saved {path}')
