import torch

from asr.supernet_asr import TransformerASRSuper


def gen_transformer(
    input_size=1280, output_neurons=5000, d_model=512, encoder_heads=[4]*12,
    decoder_heads=[4]*6, num_encoder_layers=12, num_decoder_layers=6, encoder_mlp_ratio=[4.0]*12, 
    decoder_mlp_ratio=[4.0]*6, transformer_dropout=0.1
):
    model = TransformerASRSuper(
        input_size = input_size,
        tgt_vocab = output_neurons,
        d_model = d_model,
        encoder_heads = encoder_heads,
        decoder_heads = decoder_heads,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_decoder_layers,
        encoder_mlp_ratio = encoder_mlp_ratio,
        decoder_mlp_ratio = decoder_mlp_ratio,
        dropout = transformer_dropout,
        activation = torch.nn.GELU,
        normalize_before = True
    )
    return model