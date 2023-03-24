import copy
import torch
import re
from nn_pruning.inference_model_patcher import InferenceModelPatcher
from nn_pruning.modules.nonorm import NoNormPatcher
from e2eAIOK.DeNas.pruner.model_speedup.model_structure import struct_from_config, struct_from_name, struct_from_model


def optimize_model(model, mode="dense", prune_heads=True, clone=False):
    """
        convert sparse model to dense representation by removing zero parameters
        mode in ["dense", "heads", "block_sparse"]
    """

    if clone == True:
        model = copy.deepcopy(model)

    if hasattr(model, 'config_class'):
        model_structure = struct_from_config(model.config_class)
    elif hasattr(model, 'name'):
        model_structure = struct_from_name(model.name)
    else:
        raise RuntimeError("Model optimize not supported yet, please define model structure")

    # Further prune
    params = {}
    for name, parameter in model.named_parameters():
        params[name] = parameter
        if name.endswith("weight"):
            if model_structure.is_ffn(name):
                pos = model_structure.get_position_ffn(name)
                if pos == 0:
                    output_mask = params[name].abs().sum(1) == 0
                    n0 = name
                else:
                    input_mask = params[name].abs().sum(0) == 0
                    print(f"in sum: {output_mask.sum()}, out: {input_mask.sum()}")
                    # with torch.no_grad():
                    #     params[n0][input_mask] = 0
                    #     params[name][:, output_mask] = 0

    mp = InferenceModelPatcher(prune_heads=prune_heads, mode=mode)
    pattern_prefix = model_structure.PATTERN_PREFIX
    for i, pattern in enumerate(model_structure.FFN_LAYERS):
        pattern_name = (pattern_prefix + model_structure.LAYER_PATTERNS[pattern]).replace(".", "\\.")
        if i == 0:
            mp.add_pattern(
                pattern_name,
                {"input_keep_dimension": True, "output_keep_dimension": False, "prune_input":False, "prune_output":True},
            )
        else:
            mp.add_pattern(
                pattern_name,
                {"output_keep_dimension": True, "input_keep_dimension": False, "prune_input":True, "prune_output":False},
            )

    mp.patch_model(model)

    if hasattr(model, 'config') and hasattr(model.config, "layer_norm_type") and model.config.layer_norm_type == "no_norm":
        nnc = NoNormPatcher()
        if nnc.needs_patch(model):
            nnc.patch(model)

    return model

def pattern_match(module_name, patterns):
    for pattern in patterns:
        if re.match(pattern, module_name):
            return True
    return False

def get_sparsity(w, dim, prune=True):
    if prune:
        l = (w != 0).sum(dim)
        nnz = (l != 0).sum()
        idx = l.nonzero(as_tuple=False).squeeze(-1)
        if idx.shape[0] == 0:
            idx = torch.tensor([0], device=idx.device)
            nnz = 1
        return 1.0 - (float(nnz) / l.numel()), idx
    else:
        return 0.0, None

def optimize_model_directly(model):
    if hasattr(model, 'config_class'):
        model_structure = struct_from_config(model.config_class)
    elif hasattr(model, 'name'):
        model_structure = struct_from_name(model.name)
    else:
        raise RuntimeError("Model optimize not supported yet, please define model structure")

    # Further prune
    params = {}
    for name, parameter in model.named_parameters():
        params[name] = parameter
        if name.endswith("weight"):
            if model_structure.is_ffn(name):
                pos = model_structure.get_position_ffn(name)
                if pos == 0:
                    output_mask = params[name].abs().sum(1) == 0
                    n0 = name
                else:
                    input_mask = params[name].abs().sum(0) == 0
                    with torch.no_grad():
                        params[n0][input_mask] = 0
                        params[name][:, output_mask] = 0
                    print(f"in sum: {output_mask.sum()}, out: {input_mask.sum()}")
    
    pattern_prefix = model_structure.PATTERN_PREFIX
    patterns = [(pattern_prefix + model_structure.LAYER_PATTERNS[pattern]).replace(".", "\\.") for pattern in model_structure.FFN_LAYERS]
    for k, v in model.named_modules():
        if pattern_match(k, patterns):
            c_sparsity, c = get_sparsity(v.weight, 0)
            r_sparsity, r = get_sparsity(v.weight, 1)
            print(f"{k}, sparsity = {c_sparsity:0.2f}, {r_sparsity:0.2f}")
            if r is not None:
                v.weight.data = v.weight[r, :]
                v.bias.data = v.bias[r]
            if c is not None:
                v.weight.data = v.weight[:, c]

    return model