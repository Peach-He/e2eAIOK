import torch
import torch.nn.utils.prune as prune
import functools
import re
from e2eAIOK.DeNas.pruner.model_speedup.model_structure import struct_from_config, struct_from_name


class PytorchPruner():
    def __init__(self, algo_name, layer_list=[], exclude_list=[], type="structure_aware", **kargs):
        self.layer_list = layer_list
        self.exclude_list = exclude_list
        self.kargs = kargs
        self.algo_name = algo_name
        self.type = type
        self.algo = self.get_prune_algo(algo_name)

    def get_prune_algo(self, algo):
        """
            get pytorch pruning algorithm based on algo name
        """
        if algo.lower() == "l1unstructured":
            return prune.l1_unstructured
        elif algo.lower() == "randomunstructured":
            return prune.random_unstructured
        elif algo.lower() == "lnstructured":
            return functools.partial(prune.ln_structured, n=1)
        elif algo.lower() == "randomstructured":
            return functools.partial(prune.random_structured)
        elif algo.lower() == "globalrandomunstructured":
            return functools.partial(prune.global_unstructured, pruning_method=prune.RandomUnstructured)
        elif algo.lower() == "globall1unstructured":
            return functools.partial(prune.global_unstructured, pruning_method=prune.L1Unstructured)
        else:
            raise RuntimeError(f"Pruning algorithm {algo} is not supported yet")
    
    def pattern_match(self, patterns, module_name):
        for pattern in patterns:
            if re.match(pattern, module_name):
                return True
        return False
    
    def type_match(self, module):
        return isinstance(module, torch.nn.Linear)
    
    def prune(self, model, sparsity):
        """
            prune model with the given target sparsity
        """
        mask = {}
        if self.type == "structure_aware":
            if hasattr(model, 'config_class'):
                model_structure = struct_from_config(model.config_class)
            elif hasattr(model, 'name'):
                model_structure = struct_from_name(model.name)
            else:
                raise RuntimeError(f"Model structure unknown")
            pattern_prefix = model_structure.PATTERN_PREFIX
            pattern_name = [(pattern_prefix + model_structure.LAYER_PATTERNS[pattern]).replace(".", "\\.") for pattern in model_structure.FFN_LAYERS]

        if self.algo_name.startswith("global"):
            params_to_prune = tuple([(layer, 'weight') for name, layer in model.named_modules() if hasattr(layer, 'weight') and name not in self.exclude_list])
            self.algo(params_to_prune, amount=sparsity)
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and name not in self.exclude_list:
                    mask[name] = module.named_buffers()
                    prune.remove(module, 'weight')
        else:
            for name, module in model.named_modules():
                if self.type == "structure_aware": 
                    if self.pattern_match(pattern_name, name):
                        if self.algo_name.endswith("structured"):
                            pos = model_structure.get_position_ffn(name)
                            self.algo(module, name='weight', amount=sparsity, dim=pos)
                        elif self.algo_name.endswith("unstructured"):
                            self.algo(module, name='weight', amount=sparsity)
                        mask[name] = module.named_buffers()
                        prune.remove(module, 'weight')
                else:
                    if self.type_match(module):
                        if self.algo_name.endswith("structured"):
                            self.algo(module, name='weight', amount=sparsity, dim=1)
                        elif self.algo_name.endswith("unstructured"):
                            self.algo(module, name='weight', amount=sparsity)
                        mask[name] = module.named_buffers()
                        prune.remove(module, 'weight')

        return model, mask