import torch
import os
from e2eAIOK.common.trainer.model_builder import ModelBuilder
from e2eAIOK.DeNas.asr.lib.convolution import ConvolutionFrontEnd
from e2eAIOK.DeNas.module.asr.linear import Linear
from e2eAIOK.DeNas.asr.data.processing.features import InputNormalization
from e2eAIOK.DeNas.module.asr.utils import gen_transformer
from e2eAIOK.DeNas.utils import decode_arch_tuple

class ModelBuilderASR(ModelBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_model(self):
        if self.cfg.best_model_structure != None:
            with open(self.cfg.best_model_structure, 'r') as f:
                arch = f.readlines()[-1]
            num_encoder_layers, mlp_ratio, encoder_heads, d_model = decode_arch_tuple(arch)
            self.cfg["num_encoder_layers"] = num_encoder_layers
            self.cfg["mlp_ratio"] = mlp_ratio
            self.cfg["encoder_heads"] = encoder_heads
            self.cfg["d_model"] = d_model
        return self.get_model()
    
    def get_model(self):
        modules = {}
        cnn = ConvolutionFrontEnd(
            input_shape = self.cfg["input_shape"],
            num_blocks = self.cfg["num_blocks"],
            num_layers_per_block = self.cfg["num_layers_per_block"],
            out_channels = self.cfg["out_channels"],
            kernel_sizes = self.cfg["kernel_sizes"],
            strides = self.cfg["strides"],
            residuals = self.cfg["residuals"]
        )
        transformer = gen_transformer(
            input_size=self.cfg["input_size"],
            output_neurons=self.cfg["output_neurons"], 
            d_model=self.cfg["d_model"], 
            encoder_heads=self.cfg["encoder_heads"], 
            nhead=self.cfg["nhead"], 
            num_encoder_layers=self.cfg["num_encoder_layers"], 
            num_decoder_layers=self.cfg["num_decoder_layers"], 
            mlp_ratio=self.cfg["mlp_ratio"], 
            d_ffn=self.cfg["d_ffn"], 
            transformer_dropout=self.cfg["transformer_dropout"]
        )
        ctc_lin = Linear(input_size=self.cfg["d_model"], n_neurons=self.cfg["output_neurons"])
        seq_lin = Linear(input_size=self.cfg["d_model"], n_neurons=self.cfg["output_neurons"])
        normalize = InputNormalization(norm_type="global", update_until_epoch=4)
        modules["CNN"] = cnn
        modules["Transformer"] = transformer
        modules["seq_lin"] = seq_lin
        modules["ctc_lin"] = ctc_lin
        modules["normalize"] = normalize
        model = torch.nn.ModuleDict(modules)
        return model

    def load_pretrained_model(self):
        self._pre_process()
        if not os.path.exists(self.cfg.ckpt):
            raise RuntimeError(f"Can not find pre-trained model {self.cfg.ckpt}!")
        self.logger.info(f"loading pretrained model at {self.cfg.ckpt}")

        super_model = self.get_model()
        super_model_list = torch.nn.ModuleList([super_model["CNN"], super_model["Transformer"], super_model["seq_lin"], super_model["ctc_lin"]])
        pretrained_dict = torch.load(self.cfg["ckpt"], map_location=torch.device('cpu'))
        super_model_list_dict = super_model_list.state_dict()
        super_model_list_keys = list(super_model_list_dict.keys())
        pretrained_keys = pretrained_dict.keys()
        for i, key in enumerate(pretrained_keys):
            super_model_list_dict[super_model_list_keys[i]].copy_(pretrained_dict[key])

        sub_model = self._init_model()
        sub_model_list = torch.nn.ModuleList([sub_model["CNN"], sub_model["Transformer"], sub_model["seq_lin"], sub_model["ctc_lin"]])
        sub_model_list_dict = sub_model_list.state_dict()
        for k in sub_model_list_dict:
            super_state = super_model_list_dict[k]
            sub_state = super_state
            for dim, size in enumerate(sub_model_list_dict[k].size()):
                sub_state = sub_state.index_select(dim, torch.tensor(range(size)))
            sub_model_list_dict[k].copy_(sub_state)

        del super_model, pretrained_dict
        self.model = sub_model
        self._post_process()
        return sub_model