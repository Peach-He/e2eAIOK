import ast

def decode_arch_tuple(arch_tuple):
    arch_tuple = ast.literal_eval(arch_tuple)
    encoder_depth = int(arch_tuple[0])
    encoder_mlp_ratio = [float(x) for x in (arch_tuple[1:encoder_depth+1])]
    encoder_num_heads = [int(x) for x in (arch_tuple[encoder_depth + 1: 2 * encoder_depth + 1])]

    decoder_depth = int(arch_tuple[2 * encoder_depth + 1])
    decoder_mlp_ratio = [float(x) for x in (arch_tuple[2*encoder_depth+2: 2*encoder_depth+2+decoder_depth])]
    decoder_num_heads = [int(x) for x in (arch_tuple[2*encoder_depth+2+decoder_depth: 2*encoder_depth+2+2*decoder_depth])]
    model_dim = int(arch_tuple[-1])
    return encoder_depth, encoder_mlp_ratio, encoder_num_heads, decoder_depth, decoder_mlp_ratio, decoder_num_heads, model_dim