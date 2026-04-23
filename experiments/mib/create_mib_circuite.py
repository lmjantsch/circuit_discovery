"""
Lightweight MIB Graph JSON builder.

Produces the same JSON format as eap.graph.Graph.to_json() without
requiring transformer_lens. This allows the DPA attribution step
to run independently of the TransformerLens-based evaluation step.
"""

import json
from typing import Dict


def build_graph_json(
    scores, cfg: Dict, output_path: str
):
    """
    Build and save a MIB-compatible Graph JSON from a scores tensor.

    Args:
        scores: (n_forward, n_backward) tensor of edge importance scores
        cfg: dict with keys: n_layers, n_heads, d_model, parallel_attn_mlp
        output_path: path to save the JSON file
    """
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    parallel = cfg.get('parallel_attn_mlp', False)

    n_forward = 1 + n_layers * (n_heads + 1)
    n_backward = n_layers * (3 * n_heads + 1) + 1

    assert scores.shape == (n_forward, n_backward), \
        f"Score shape {scores.shape} != expected ({n_forward}, {n_backward})"

    # Build node list
    nodes = {}
    # Input node (forward only, always in graph)
    nodes['input'] = {'in_graph': True}

    # Per-layer nodes
    for l in range(n_layers):
        for h in range(n_heads):
            nodes[f'a{l}.h{h}'] = {'in_graph': True}
        nodes[f'm{l}'] = {'in_graph': True}

    # Logits node (always in graph)
    nodes['logits'] = {'in_graph': True}

    # Build edge list with scores
    edges = {}
    residual_stream = ['input']

    for l in range(n_layers):
        attn_names = [f'a{l}.h{h}' for h in range(n_heads)]
        mlp_name = f'm{l}'

        if parallel:
            # Parallel: attn and mlp see the same residual stream
            for src in residual_stream:
                for attn_name in attn_names:
                    for qkv in 'qkv':
                        edge_name = f'{src}->{attn_name}<{qkv}>'
                        fwd_idx = _forward_index(src, n_heads)
                        bwd_idx = _backward_index(attn_name, qkv, n_heads, n_layers)
                        edges[edge_name] = {
                            'score': float(scores[fwd_idx, bwd_idx].item()),
                            'in_graph': True,
                        }
                edge_name = f'{src}->{mlp_name}'
                fwd_idx = _forward_index(src, n_heads)
                bwd_idx = _backward_index(mlp_name, None, n_heads, n_layers)
                edges[edge_name] = {
                    'score': float(scores[fwd_idx, bwd_idx].item()),
                    'in_graph': True,
                }
            residual_stream += attn_names + [mlp_name]
        else:
            # Sequential: attn first, then mlp sees attn outputs too
            for src in residual_stream:
                for attn_name in attn_names:
                    for qkv in 'qkv':
                        edge_name = f'{src}->{attn_name}<{qkv}>'
                        fwd_idx = _forward_index(src, n_heads)
                        bwd_idx = _backward_index(attn_name, qkv, n_heads, n_layers)
                        edges[edge_name] = {
                            'score': float(scores[fwd_idx, bwd_idx].item()),
                            'in_graph': True,
                        }
            residual_stream += attn_names

            for src in residual_stream:
                edge_name = f'{src}->{mlp_name}'
                fwd_idx = _forward_index(src, n_heads)
                bwd_idx = _backward_index(mlp_name, None, n_heads, n_layers)
                edges[edge_name] = {
                    'score': float(scores[fwd_idx, bwd_idx].item()),
                    'in_graph': True,
                }
            residual_stream.append(mlp_name)

    # Logits edges
    for src in residual_stream:
        edge_name = f'{src}->logits'
        fwd_idx = _forward_index(src, n_heads)
        bwd_idx = n_backward - 1  # logits is always last
        edges[edge_name] = {
            'score': float(scores[fwd_idx, bwd_idx].item()),
            'in_graph': True,
        }

    d = {
        'cfg': cfg,
        'nodes': nodes,
        'edges': edges,
    }

    with open(output_path, 'w') as f:
        json.dump(d, f)


def _forward_index(node_name: str, n_heads: int) -> int:
    if node_name == 'input':
        return 0
    elif node_name[0] == 'a':
        layer, head = node_name.split('.')
        layer = int(layer[1:])
        head = int(head[1:])
        return 1 + layer * (n_heads + 1) + head
    elif node_name[0] == 'm':
        layer = int(node_name[1:])
        return 1 + layer * (n_heads + 1) + n_heads
    else:
        raise ValueError(f"Invalid node: {node_name}")


def _backward_index(
    node_name: str, qkv: str, n_heads: int, n_layers: int
) -> int:
    if node_name == 'logits':
        return n_layers * (3 * n_heads + 1)
    elif node_name[0] == 'a':
        layer, head = node_name.split('.')
        layer = int(layer[1:])
        head = int(head[1:])
        qkv_offset = 'qkv'.index(qkv) * n_heads
        return layer * (3 * n_heads + 1) + qkv_offset + head
    elif node_name[0] == 'm':
        layer = int(node_name[1:])
        return layer * (3 * n_heads + 1) + 3 * n_heads
    else:
        raise ValueError(f"Invalid node: {node_name}")