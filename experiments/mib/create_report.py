"""
Generate experiment_results.md by comparing DPA CPR scores with MIB paper SOTA.
Reads evaluation pkl files and writes a formatted markdown report.
"""
import os
import pickle
import sys

RESULT_DIR = sys.argv[1] if len(sys.argv) > 1 else "experiments/mib/MIB-circuit-track/results"
OUTPUT_MD = sys.argv[2] if len(sys.argv) > 2 else "experiments/mib/experiment_results.md"

# SOTA from MIB paper Table 14 (CPR, PUBLIC test set).
# We use public test set because that's the only split we can replicate.
# Private test set (Table 16) is hidden — accessible only via HF leaderboard.
#
# Verified against arXiv:2504.13151v2:
#   - Table 14 caption cross-references Table 17 "public test sets"
#   - Tables 15/16 are explicitly labeled "private test set"
#   - MIB README: --split test = public test set
#
# For apples-to-apples comparison, our DPA must also evaluate on --split test.
SOTA = {
    "ioi_qwen2.5":                 {"method": "EAP-IG-inputs", "cpr": 1.63},  # T14 (bold)
    "mcqa_qwen2.5":                {"method": "EAP-IG-inputs", "cpr": 1.16},  # T14 (underlined)
    "ioi_llama3":                  {"method": "EAP-IG-inputs", "cpr": 2.08},  # T14 (bold)
    "mcqa_llama3":                 {"method": "NAP-IG",        "cpr": 1.87},  # T14 (bold)
    "arithmetic-addition_llama3":  {"method": "EAP-IG-act",    "cpr": 0.98},  # T17 (bold)
    "arithmetic-subtraction_llama3":{"method": "EAP-IG-inputs","cpr": 1.03},  # T17 (bold)
    "arc-easy_llama3":             {"method": "EAP-IG-inputs", "cpr": 1.04},  # T14 (bold)
    "arc-challenge_llama3":        {"method": "EAP-IG-inputs", "cpr": 0.98},  # T14 (bold)
}

# CMD from Table 2 (public test set, main body)
SOTA_CMD = {
    "ioi_qwen2.5":                 {"method": "EAP-IG-act",    "cmd": 0.02},
    "mcqa_qwen2.5":                {"method": "EAP-IG-act",    "cmd": 0.02},
    "ioi_llama3":                  {"method": "EAP-IG-act",    "cmd": 0.03},
    "mcqa_llama3":                 {"method": "EAP (CF)",       "cmd": 0.04},
    "arithmetic-addition_llama3":  {"method": "EAP-IG-inputs", "cmd": 0.00},
    "arithmetic-subtraction_llama3":{"method": "EAP-IG-inputs","cmd": 0.00},
    "arc-easy_llama3":             {"method": "EAP (CF)",       "cmd": 0.11},
    "arc-challenge_llama3":        {"method": "EAP (CF)",       "cmd": 0.18},
}

# Also store CMD SOTA for reference
SOTA_CMD = {
    "ioi_qwen2.5":                {"method": "EAP-IG-act", "cmd": 0.02},
    "mcqa_qwen2.5":               {"method": "EAP-IG-act", "cmd": 0.02},
    "ioi_llama3":                  {"method": "EAP-IG-act", "cmd": 0.03},
    "mcqa_llama3":                 {"method": "EAP (CF)",   "cmd": 0.04},
    "arithmetic-addition_llama3":  {"method": "EAP-IG-inputs","cmd": 0.00},
    "arithmetic-subtraction_llama3":{"method": "EAP-IG-inputs","cmd": 0.00},
    "arc-easy_llama3":             {"method": "EAP (CF)",   "cmd": 0.11},
    "arc-challenge_llama3":        {"method": "EAP (CF)",   "cmd": 0.18},
}

METHOD_DIR = "dpa_patching_edge"

def load_results():
    results = {}
    method_path = os.path.join(RESULT_DIR, METHOD_DIR)
    if not os.path.exists(method_path):
        return results
    for fname in os.listdir(method_path):
        if not fname.endswith(".pkl"):
            continue
        parts = os.path.splitext(fname)[0].split("_")
        # format: task_model_split_abs-{True/False}.pkl
        # e.g. ioi_qwen2.5_validation_abs-True.pkl
        task = parts[0]
        model = parts[1]
        key = f"{task}_{model}"
        fpath = os.path.join(method_path, fname)
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        results[key] = d
    return results


def generate_report(results):
    lines = []
    lines.append("# DPA Edge Extension — MIB Benchmark Results")
    lines.append("")
    lines.append("## CPR Comparison (higher is better)")
    lines.append("")
    lines.append("| Task / Model | DPA (Ours) | SOTA Method | SOTA CPR | Winner |")
    lines.append("|---|---|---|---|---|")

    wins = 0
    losses = 0
    ties = 0

    for key in SOTA:
        sota_info = SOTA[key]
        if key in results:
            dpa_cpr = results[key].get("area_under", "N/A")
            if isinstance(dpa_cpr, float):
                if dpa_cpr > sota_info["cpr"]:
                    winner = "DPA"
                    wins += 1
                elif dpa_cpr < sota_info["cpr"]:
                    winner = sota_info["method"]
                    losses += 1
                else:
                    winner = "Tie"
                    ties += 1
                lines.append(f"| {key} | {dpa_cpr:.4f} | {sota_info['method']} | {sota_info['cpr']:.2f} | **{winner}** |")
            else:
                lines.append(f"| {key} | {dpa_cpr} | {sota_info['method']} | {sota_info['cpr']:.2f} | - |")
        else:
            lines.append(f"| {key} | (not run) | {sota_info['method']} | {sota_info['cpr']:.2f} | - |")

    lines.append("")
    lines.append(f"**Score: DPA wins {wins}, loses {losses}, ties {ties}**")

    # CMD comparison
    lines.append("")
    lines.append("## CMD Comparison (lower is better)")
    lines.append("")
    lines.append("| Task / Model | DPA (Ours) | SOTA Method | SOTA CMD | Winner |")
    lines.append("|---|---|---|---|---|")

    cmd_wins = 0
    cmd_losses = 0

    for key in SOTA_CMD:
        sota_info = SOTA_CMD[key]
        if key in results:
            dpa_cmd = results[key].get("area_from_1", "N/A")
            if isinstance(dpa_cmd, float):
                if dpa_cmd < sota_info["cmd"]:
                    winner = "DPA"
                    cmd_wins += 1
                else:
                    winner = sota_info["method"]
                    cmd_losses += 1
                lines.append(f"| {key} | {dpa_cmd:.4f} | {sota_info['method']} | {sota_info['cmd']:.2f} | **{winner}** |")
            else:
                lines.append(f"| {key} | {dpa_cmd} | {sota_info['method']} | {sota_info['cmd']:.2f} | - |")
        else:
            lines.append(f"| {key} | (not run) | {sota_info['method']} | {sota_info['cmd']:.2f} | - |")

    lines.append("")
    lines.append(f"**Score: DPA wins {cmd_wins}, loses {cmd_losses}**")

    # Faithfulness curves
    lines.append("")
    lines.append("## Faithfulness Curves (per-percentage)")
    lines.append("")
    pcts = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    for key, d in sorted(results.items()):
        if "faithfulnesses" in d:
            lines.append(f"### {key}")
            lines.append(f"| % edges | faithfulness |")
            lines.append(f"|---|---|")
            for p, f in zip(pcts, d["faithfulnesses"]):
                lines.append(f"| {p*100:.1f}% | {f:.4f} |")
            lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")

    total = wins + losses + ties
    if total == 0:
        lines.append("No results available yet.")
    elif wins > losses:
        lines.append(f"DPA Edge Extension beats SOTA on {wins}/{total} task/model combinations.")
        lines.append("")
        lines.append("### Strengths")
        lines.append("- DPA's typed edge decomposition (v/q/k + up/gate) provides finer-grained attribution")
        lines.append("- Single backward pass is computationally efficient compared to EAP-IG's multi-step IG")
    else:
        lines.append(f"DPA Edge Extension does NOT beat SOTA overall ({wins} wins vs {losses} losses).")
        lines.append("")
        lines.append("### Why DPA may underperform")
        lines.append("")
        lines.append("1. **Linear approximation vs Integrated Gradients**: DPA uses a single backward pass")
        lines.append("   (first-order Taylor approximation), while EAP-IG averages gradients across 5-30")
        lines.append("   interpolation steps, capturing nonlinear effects more faithfully.")
        lines.append("")
        lines.append("2. **Scaling factors**: DPA uses fixed scaling {v:0.5, q:0.25, k:0.25, up:0.5, gate:0.5}.")
        lines.append("   These may not be optimal for MIB's evaluation (patching ablation at varying circuit sizes).")
        lines.append("")
        lines.append("3. **Attribution target mismatch**: DPA attributes w.r.t. a single target token logit,")
        lines.append("   while MIB evaluates with logit_diff (correct - incorrect). This mismatch could cause")
        lines.append("   DPA to rank edges differently from what the evaluation metric rewards.")
        lines.append("")
        lines.append("4. **No counterfactual signal in backward pass**: EAP uses activation differences")
        lines.append("   (corrupt - clean) multiplied by gradients. DPA's backward pass only uses clean")
        lines.append("   activations, then separately computes source differences. The interaction between")
        lines.append("   the two may lose important cross-terms.")
        lines.append("")
        lines.append("### Suggested improvements")
        lines.append("")
        lines.append("1. **DPA-IG variant**: Add integrated gradients to DPA's backward pass (interpolate")
        lines.append("   between clean and corrupt embeddings, average the typed edge scores)")
        lines.append("2. **Logit diff target**: Use (correct - incorrect) logit difference as the attribution")
        lines.append("   target instead of single token logit")
        lines.append("3. **Scaling factor search**: Grid search over v/q/k/up/gate scaling factors")
        lines.append("4. **Remove scaling entirely**: Set all scaling to 1.0 and let raw scores determine ranking")
        lines.append("5. **Absolute value strategy**: Try using absolute values of edge scores for ranking")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {OUTPUT_MD}")


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} result files")
    generate_report(results)