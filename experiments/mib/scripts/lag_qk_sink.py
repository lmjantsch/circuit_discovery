"""Diagnostic: is a small Q@K LAG an attention-sink artifact?
Q@K LAG = ‖Δq·Δkᵀ‖ / ‖Δ(q kᵀ)‖.  If key position 0 (BOS / sink) carries a massive, prompt-invariant key,
column 0 of Δ(q kᵀ) = Δq·k₀ᵀ is large and purely linear (Δk₀ = 0), which dilutes the ratio.
Reports Q@K EAP LAG and softmax EAP LAG (a) all columns, (b) excluding key column 0, (c) column 0 only,
plus the share of ‖Δz_true‖² and ‖z_clean‖² sitting in column 0.
Usage: CUDA_VISIBLE_DEVICES=3 python experiments/mib/scripts/lag_qk_sink.py --model qwen2.5 --task ioi
"""
import os, sys, time, math, json, argparse
PROJ = '/home/dacslab/djk/lasse-circuit/circuit_discovery'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from experiments.mib.data_utils import MIBDataset
MODEL_IDS = {'gpt2': 'openai-community/gpt2', 'qwen2.5': 'Qwen/Qwen2.5-0.5B', 'gemma2': 'google/gemma-2-2b', 'llama3': 'meta-llama/Llama-3.1-8B'}
DTYPES = {'gpt2': torch.float32, 'qwen2.5': torch.float32, 'gemma2': torch.bfloat16, 'llama3': torch.bfloat16}
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--task', required=True); ap.add_argument('--num', type=int, default=100)
args = ap.parse_args(); MODEL, TASK = args.model, args.task; MID = MODEL_IDS[MODEL]
tok = AutoTokenizer.from_pretrained(MID)
if not tok.pad_token: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MID, dtype=DTYPES[MODEL], attn_implementation='eager', device_map='cuda').eval()
cfg = model.config; IS_GPT2 = MODEL == 'gpt2'; IS_GEMMA = MODEL == 'gemma2'
if IS_GPT2:
    NL, nH = cfg.n_layer, cfg.n_head; nKV = nH; HD = cfg.n_embd // nH; NG = 1; SCALE = HD ** -0.5; CAP = None; layers = model.transformer.h
else:
    NL, nH, nKV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    HD = getattr(cfg, 'head_dim', None) or cfg.hidden_size // nH; NG = nH // nKV
    SCALE = (cfg.query_pre_attn_scalar ** -0.5) if IS_GEMMA else HD ** -0.5; CAP = cfg.attn_logit_softcapping if IS_GEMMA else None
    mod = {'qwen2.5': 'qwen2', 'gemma2': 'gemma2', 'llama3': 'llama'}[MODEL]
    M = __import__(f'transformers.models.{mod}.modeling_{mod}', fromlist=['apply_rotary_pos_emb', 'repeat_kv'])
    apply_rotary_pos_emb, repeat_kv = M.apply_rotary_pos_emb, M.repeat_kv; layers = model.model.layers
cap = {}
def mk(key):
    def h(m, i, o): cap[key] = o.detach().float()
    return h
def mkrot(m, i, o): cap['cos'], cap['sin'] = o[0].detach().float(), o[1].detach().float()
for L in range(NL):
    if IS_GPT2: layers[L].attn.c_attn.register_forward_hook(mk(f'qkv{L}'))
    else: layers[L].self_attn.q_proj.register_forward_hook(mk(f'q{L}')); layers[L].self_attn.k_proj.register_forward_hook(mk(f'k{L}'))
if not IS_GPT2: model.model.rotary_emb.register_forward_hook(mkrot)
def heads(t, h): B, S, _ = t.shape; return t.view(B, S, h, HD).transpose(1, 2)
def qk(d, L):
    if IS_GPT2: q, k, _ = d[f'qkv{L}'].split(cfg.n_embd, dim=-1); return heads(q, nH), heads(k, nH)
    q = heads(d[f'q{L}'], nH); k = heads(d[f'k{L}'], nKV); q, k = apply_rotary_pos_emb(q, k, d['cos'], d['sin']); return q, repeat_kv(k, NG)
def softcap(x): return CAP * torch.tanh(x / CAP) if CAP else x
def jvp(s, ds, m): A = torch.softmax(s + m, dim=-1); return A * (ds - (A * ds).sum(-1, keepdim=True))
ds = MIBDataset(TASK, tok, MODEL, split='train', num_examples=args.num)
K = ['all', 'col0', 'rest']; acc = {op: {k: {'eap': 0., 'true': 0., 'clean': 0.} for k in K} for op in ['QK', 'softmax']}
knorm0 = 0.; knormR = 0.; n = 0; t0 = time.time(); qn = {'dq': 0., 'q': 0., 'dk': 0., 'k': 0., 'cross': 0., 'lin_q': 0., 'lin_k': 0.}
with torch.no_grad():
    for i in range(len(ds)):
        p, b, *_ = ds[i]; ic = tok(p, return_tensors='pt').to('cuda'); ib = tok(b, return_tensors='pt').to('cuda')
        if ic['input_ids'].shape[1] != ib['input_ids'].shape[1]: continue
        cap.clear(); model(**ic); cl = {k: v.clone() for k, v in cap.items()}
        cap.clear(); model(**ib); co = {k: v.clone() for k, v in cap.items()}
        for L in range(NL):
            q, k = qk(cl, L); qs, ks = qk(co, L); S = q.shape[2]; m = torch.triu(torch.full((S, S), float('-inf'), device='cuda'), 1)
            knorm0 += k[:, :, 0].pow(2).sum().item(); knormR += k[:, :, 1:].pow(2).sum().item() / (S - 1)
            QK = q @ k.transpose(-1, -2); QKs = qs @ ks.transpose(-1, -2)
            qn['dq'] += (qs - q).pow(2).sum().item(); qn['q'] += q.pow(2).sum().item(); qn['dk'] += (ks - k).pow(2).sum().item(); qn['k'] += k.pow(2).sum().item()
            qn['cross'] += ((qs - q) @ (ks - k).transpose(-1, -2)).pow(2).sum().item(); qn['lin_q'] += ((qs - q) @ k.transpose(-1, -2)).pow(2).sum().item(); qn['lin_k'] += (q @ (ks - k).transpose(-1, -2)).pow(2).sum().item()
            fe = (qs - q) @ k.transpose(-1, -2) + q @ (ks - k).transpose(-1, -2); tr = QKs - QK
            s, ss = softcap(QK * SCALE), softcap(QKs * SCALE)
            A = torch.softmax(s + m, -1); As = torch.softmax(ss + m, -1); sm_fe = jvp(s, ss - s, m); sm_tr = As - A
            for op, e, t, c in [('QK', fe - tr, tr, QK), ('softmax', sm_fe - sm_tr, sm_tr, A)]:
                for key, sl in [('all', slice(None)), ('col0', slice(0, 1)), ('rest', slice(1, None))]:
                    acc[op][key]['eap'] += e[..., sl].pow(2).sum().item(); acc[op][key]['true'] += t[..., sl].pow(2).sum().item(); acc[op][key]['clean'] += c[..., sl].pow(2).sum().item()
        n += 1
print(f'{MODEL}/{TASK}  aligned {n}  ({time.time()-t0:.0f}s)   mean‖k_0‖² / mean‖k_j≠0‖² = {knorm0/knormR:.1f}×')
print(f'relative ‖Δq‖/‖q‖ = {math.sqrt(qn["dq"]/qn["q"]):.4f}   ‖Δk‖/‖k‖ = {math.sqrt(qn["dk"]/qn["k"]):.4f}   term norms  Δq·kᵀ : q·Δkᵀ : Δq·Δkᵀ = 1 : {math.sqrt(qn["lin_k"]/qn["lin_q"]):.3f} : {math.sqrt(qn["cross"]/qn["lin_q"]):.3f}')
print(f'{"op":<8} {"cols":<6} {"EAP LAG":>8} {"share true²":>12} {"share clean²":>13}')
for op in ['QK', 'softmax']:
    for key in K:
        a = acc[op][key]; T = acc[op]['all']
        print(f'{op:<8} {key:<6} {math.sqrt(a["eap"]/a["true"]):>8.4f} {a["true"]/T["true"]:>12.3f} {a["clean"]/T["clean"]:>13.3f}')
os.makedirs('experiments/mib/method_comparison/lag', exist_ok=True)
json.dump({'model': MODEL, 'task': TASK, 'n': n, 'k0_over_rest_sq': knorm0 / knormR, 'rel_dq': math.sqrt(qn['dq']/qn['q']), 'rel_dk': math.sqrt(qn['dk']/qn['k']),
           'term_ratio_linq_link_cross': [1.0, math.sqrt(qn['lin_k']/qn['lin_q']), math.sqrt(qn['cross']/qn['lin_q'])],
           'ops': {op: {key: {'eap_lag': math.sqrt(acc[op][key]['eap'] / acc[op][key]['true']), 'share_true_sq': acc[op][key]['true'] / acc[op]['all']['true'],
                              'share_clean_sq': acc[op][key]['clean'] / acc[op]['all']['clean']} for key in K} for op in ['QK', 'softmax']}},
          open(f'experiments/mib/method_comparison/lag/sink_{MODEL}_{TASK}.json', 'w'), indent=1)
