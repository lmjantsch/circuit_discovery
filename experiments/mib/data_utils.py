from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

import torch

TASKS_TO_HF_NAMES = {
    'ioi': 'ioi',
    'mcqa': 'copycolors_mcqa',
    'arithmetic_addition': 'arithmetic_addition',
    'arithmetic_subtraction': 'arithmetic_subtraction',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
}

class MIBDataset(Dataset):
    """Minimal MIB dataset loader."""

    def __init__(self, task, tokenizer, model_name, split='train', num_examples=100):
        self.task = task
        self.tokenizer = tokenizer
        self.model_name = model_name

        hf_url = f"mib-bench/{TASKS_TO_HF_NAMES[task]}"
        if task == 'mcqa':
            self.dataset = load_dataset(hf_url, '4_answer_choices', split=split)
            self.counterfactual_type = "symbol_counterfactual"
        elif task.startswith('arc'):
            self.dataset = load_dataset(hf_url, split=split)
            self.counterfactual_type = "symbol_counterfactual"
        elif task.startswith('arithmetic'):
            self.dataset = load_dataset(hf_url, split=split)
            self.operator = "-" if "subtraction" in task else "+"
        else:
            self.dataset = load_dataset(hf_url, split=split)

        self.dataset = self._filter()
        if num_examples and num_examples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_examples))

    def _filter(self):
        tok = self.tokenizer
        if self.task == 'ioi':
            return self.dataset.filter(
                lambda x: (
                    len(tok(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) ==
                    len(tok(f" {x['metadata']['subject']}", add_special_tokens=False).input_ids) and
                    len(tok(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) ==
                    len(tok(f" {x['metadata']['random_c']}", add_special_tokens=False).input_ids)
                )
            )
        elif self.task == 'mcqa' or self.task.startswith('arc'):
            ct = self.counterfactual_type
            return self.dataset.filter(
                lambda x: (
                    len(tok(x["choices"]["label"][x["answerKey"]], add_special_tokens=False).input_ids) ==
                    len(tok(str(x[ct]["choices"]["label"][x[ct]["answerKey"]]), add_special_tokens=False).input_ids)
                )
            )
        elif self.task.startswith('arithmetic'):
            op = self.operator
            return self.dataset.filter(
                lambda x: (
                    len(tok(str(x["label"]), add_special_tokens=False).input_ids) == 1 and
                    x["random_counterfactual"] is not None and
                    x["random_counterfactual"]["prompt"] is not None and
                    x["operator"] == op and
                    len(tok(str(x["random_counterfactual"]["label"]), add_special_tokens=False).input_ids) == 1
                )
            )
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        tok = self.tokenizer

        if self.task == 'ioi':
            correct_idx = tok(f" {row['metadata']['indirect_object']}", add_special_tokens=False).input_ids[0]
            incorrect_idx = tok(f" {row['metadata']['subject']}", add_special_tokens=False).input_ids[0]
            cf = row.get("s2_io_flip_counterfactual", row.get("counterfactual", {}))
            return row["prompt"], cf.get("prompt", row["prompt"]), [correct_idx, incorrect_idx]

        elif self.task == 'mcqa' or self.task.startswith('arc'):
            ct = self.counterfactual_type
            correct_idx = tok(row["choices"]["label"][row["answerKey"]], add_special_tokens=False).input_ids[0]
            cf = row[ct]
            incorrect_idx = tok(str(cf["choices"]["label"][cf["answerKey"]]), add_special_tokens=False).input_ids[0]
            return row["prompt"], cf["prompt"], [correct_idx, incorrect_idx]

        elif self.task.startswith('arithmetic'):
            correct_idx = tok(str(row["label"]), add_special_tokens=False).input_ids[0]
            cf = row["random_counterfactual"]
            incorrect_idx = tok(str(cf["label"]), add_special_tokens=False).input_ids[0]
            return row["prompt"], cf["prompt"], [correct_idx, incorrect_idx]
        
    def dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size,
            collate_fn=self._collate_fn, shuffle=False,
        )

    def _collate_fn(self, xs):
        clean, corrupted, labels = zip(*xs)
        if self.model_name == 'gpt2':
            clean = ['<|endoftext|>' + prompt for prompt in clean]
            corrupted = ['<|endoftext|>' + prompt for prompt in corrupted]
        clean_labels, corrupt_labels = zip(*labels)
        return list(clean), list(corrupted), clean_labels, corrupt_labels


  
# embedding (src): input

# attention src: a{layer_id}.h{head_id} e.g. a0.h21
# attention tgt: a{layer_id}.h{head_id}<{head_type}> e.g. a0.h21<k>

# mlp src / tgt: m{layer_id} e.g. m2

# lm_head (tgt): logit

# edge: {src}->{tgt} : {'score': {score}, 'in_graph': False}

# Json Schema
# {
#   "cfg": {
#     "n_layers": 32,
#     "n_heads": 32,
#     "parallel_attn_mlp": False,
#     "d_model": 4096
#   },
#   "nodes": {},
#   "edges": {},
# }

def get_src_id(name: str, n_heads):
    if name == 'input':
        return 0
    if name[0] == 'm':
        layer_id = int(name[1:])
        return 1 + layer_id * (n_heads + 1) + n_heads
    if name[0] == 'a':
        attn_mod, head = name.split('.')
        layer_id = int(attn_mod[1:])
        head_id = int(head[1:])
        return 1 + layer_id * (n_heads + 1) + head_id
    raise NotImplementedError(f"Name '{name}' does not fit the patterns.")

def get_tgt_id(name: str, n_heads):
    if name == 'logits':
        return -1
    if name[0] == 'm':
        layer_id = int(name[1:])
        return layer_id * (3 * n_heads + 1) + 3 * n_heads
    if name[0] == 'a':
        attn_mod, head = name.split('.')
        layer_id = int(attn_mod[1:])
        head_id = int(head[1:-3])
        head_type = head[-2]
        if head_type == 'q':
            return layer_id * (3 * n_heads + 1) + head_id
        if head_type == 'k':
            return layer_id * (3 * n_heads + 1) + n_heads + head_id
        if head_type == 'v':
            return layer_id * (3 * n_heads + 1) + 2 * n_heads + head_id
    raise NotImplementedError(f"Name '{name}' does not fit the patterns.")

def create_mib_circuit(scores: torch.Tensor, n_layers, n_heads, d_model, circuit_level: str = 'edge'):
    circuit = {  
        "cfg": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "parallel_attn_mlp": False,
            "d_model": d_model
        },
        'nodes': {},
        'edges': {}
    }


    def _inner_loop(src_name: str, src_layer_id: int = 0, src_head_id: int = None):
        src_id = get_src_id(src_name, n_heads)
        for tgt_layer_id in range(src_layer_id, n_layers):

            if tgt_layer_id > src_layer_id or src_name == 'input':
                for tgt_head_type in ['q', 'k', 'v']:
                    for tgt_head_id in range(n_heads):
                        tgt_name = f"a{tgt_layer_id}.h{tgt_head_id}<{tgt_head_type}>"
                        tgt_id = get_tgt_id(tgt_name, n_heads)
                        circuit['edges'][f"{src_name}->{tgt_name}"] = {'score': scores[src_id, tgt_id].item(), 'in_graph': False}

            if not (src_name[0] == 'm' and tgt_layer_id == src_layer_id):
                tgt_name = f"m{tgt_layer_id}"
                tgt_id = get_tgt_id(tgt_name, n_heads)
                circuit['edges'][f"{src_name}->{tgt_name}"] = {'score': scores[src_id, tgt_id].item(), 'in_graph': False}
        
        circuit['edges'][f"{src_name}->logits"] = {'score': scores[src_id, -1].item(), 'in_graph': False}

    
    # outer loop
    _inner_loop("input") 
    circuit["nodes"]["input"] = {'in_graph': False}

    for layer_id in range(n_layers):
        for head_id in range(n_heads):
            src_name = f"a{layer_id}.h{head_id}"
            _inner_loop(src_name, src_layer_id=layer_id, src_head_id=head_id)
            circuit["nodes"][src_name] = {'in_graph': False}
        src_name = f"m{layer_id}"
        _inner_loop(src_name, src_layer_id=layer_id)
        circuit["nodes"][src_name] = {'in_graph': False}


    return circuit