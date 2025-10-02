# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------

from .mlla import MLLA
from .sema import SEMA


def build_model(config):
    model_dict = {
        'mlla': MLLA,
        'sema': SEMA,}
    model_type = config.MODEL.TYPE
    if model_type not in model_dict:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    
    if model_type == 'mlla':
        model = MLLA(img_size=config.DATA.IMG_SIZE,
                     patch_size=config.MODEL.CONFIGS.PATCH_SIZE,
                     in_chans=config.MODEL.CONFIGS.IN_CHANS,
                     num_classes=config.MODEL.NUM_CLASSES,
                     embed_dim=config.MODEL.CONFIGS.EMBED_DIM,
                     depths=config.MODEL.CONFIGS.DEPTHS,
                     num_heads=config.MODEL.CONFIGS.NUM_HEADS,
                     mlp_ratio=config.MODEL.CONFIGS.MLP_RATIO,
                     qkv_bias=config.MODEL.CONFIGS.QKV_BIAS,
                     drop_rate=config.MODEL.DROP_RATE,
                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                     ape=config.MODEL.CONFIGS.APE,
                     use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    else:
        model = model_dict[model_type](img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.CONFIGS.PATCH_SIZE,
                        in_chans=config.MODEL.CONFIGS.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.CONFIGS.EMBED_DIM,
                        depths=config.MODEL.CONFIGS.DEPTHS,
                        num_heads=config.MODEL.CONFIGS.NUM_HEADS,
                        window_size=config.MODEL.CONFIGS.WINDOW_SIZES,
                        mlp_ratio=config.MODEL.CONFIGS.MLP_RATIO,
                        qkv_bias=config.MODEL.CONFIGS.QKV_BIAS,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.CONFIGS.APE,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    return model
