"""
❤Descripttion: build neural network
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-01 08:54:00
❤LastEditTime: 2024-03-22 10:39:26
❤Github: https://github.com/MilknoCandy
"""


def build_network(model_name, num_classes=1):
############################################   shformer
    if model_name == "shformer_add_base":
        from ..shformer.shformer_add_base import shformer_add_base
        model = shformer_add_base(num_classes=num_classes)
        return model  

    if model_name == "shformer_add_dy":      # dynamic merge tokens
        from ..shformer.shformer_add_dy import shformer_add_dy
        model = shformer_add_dy(num_classes=num_classes)
        return model  

    if model_name == "shformer_add":
        from ..shformer.shformer_add import shformer_add
        model = shformer_add(num_classes=num_classes)
        return model  

    if model_name == "shformer_add_wo_sci":
        from ..shformer.shformer_add_wo_sci import shformer_add_wo_sci
        model = shformer_add_wo_sci(num_classes=num_classes)
        return model  
    
    if model_name == "shformer_best":
        from ..shformer.shformer_best import shformer_best
        model = shformer_best(num_classes=num_classes)
        return model
    
############################################   shformer transformers' depth ablation
    if model_name == "shformer_add_2":
        from ..shformer.depth_compare.shfromer_add_2 import shformer_add_2
        model = shformer_add_2(num_classes=num_classes)
        return model
    
    if model_name == "shformer_add_4":
        from ..shformer.depth_compare.shfromer_add_4 import shformer_add_4
        model = shformer_add_4(num_classes=num_classes)
        return model
    
    if model_name == "shformer_add_6":
        from ..shformer.depth_compare.shfromer_add_6 import shformer_add_6
        model = shformer_add_6(num_classes=num_classes)
        return model

    if model_name == "shformer_add_8":
        from ..shformer.depth_compare.shfromer_add_8 import shformer_add_8
        model = shformer_add_8(num_classes=num_classes)
        return model

    if model_name == "shformer_add_10":
        from ..shformer.depth_compare.shfromer_add_10 import shformer_add_10
        model = shformer_add_10(num_classes=num_classes)
        return model
    
############################################   shformer decoder ablation
    if model_name == "shformer_add_MLA":
        from ..shformer.decoder_compare.shfromer_add_MLA import shformer_add_MLA
        model=shformer_add_MLA(num_classes=num_classes)
        return model
    
    if model_name == "shformer_add_SeD":
        from ..shformer.decoder_compare.shfromer_add_SeD import shformer_add_SeD
        model = shformer_add_SeD(num_classes=num_classes)
        return model

    if model_name == "shformer_add_PLD":
        from ..shformer.decoder_compare.shfromer_add_PLD import shformer_add_PLD
        model = shformer_add_PLD(num_classes=num_classes)
        return model
    
############################################   shformer original attention/shformer original attention+FE
    if model_name=="shformer_add_atto":
        from ..shformer.model_compare.shfromer_add_atto import shformer_add_atto
        model = shformer_add_atto(num_classes=num_classes)
        return model

    if model_name=="shformer_add_atto_sci":
        from ..shformer.model_compare.shfromer_add_atto_sci import \
            shformer_add_atto_sci
        model = shformer_add_atto_sci(num_classes=num_classes)
        return model
    
############################################   shformer Spatial_Channel_Interaction position compare
    if model_name=="shformer_add_f":
        from ..shformer.model_compare.shfromer_add_f import shformer_add_f
        model = shformer_add_f(num_classes=num_classes)
        return model

    if model_name=="shformer_add_l":
        from ..shformer.model_compare.shfromer_add_l import shformer_add_l
        model = shformer_add_l(num_classes=num_classes)
        return model

    if model_name=="shformer_add_mid":
        from ..shformer.model_compare.shfromer_add_mid import shformer_add_mid
        model = shformer_add_mid(num_classes=num_classes)
        return model

    if model_name=="shformer_add_mlpf":
        from ..shformer.model_compare.shfromer_add_mlpf import shformer_add_mlpf
        model = shformer_add_mlpf(num_classes=num_classes)
        return model

    if model_name=="shformer_add_attf":
        from ..shformer.model_compare.shfromer_add_attf import shformer_add_attf
        model = shformer_add_attf(num_classes=num_classes)
        return model

    if model_name=="shformer_add_attb":
        from ..shformer.model_compare.shfromer_add_attb import shformer_add_attb
        model = shformer_add_attb(num_classes=num_classes)
        return model

############################################   block ablation
    if model_name == "shformer_add_SE":
        from ..shformer.model_compare.shfromer_add_SE import shformer_add_SE
        model = shformer_add_SE(num_classes=num_classes)
        return model

    if model_name == "shformer_add_SE_S":
        from ..shformer.model_compare.shfromer_add_SE_S import shformer_add_SE_S
        model = shformer_add_SE_S(num_classes=num_classes)
        return model
    
    if model_name == "shformer_add_CBAM":
        from ..shformer.model_compare.shfromer_add_CBAM import shformer_add_CBAM
        model = shformer_add_CBAM(num_classes=num_classes)
        return model

############################################   Attention Mechanism Ablation
    # FLASH
    if model_name == 'shformer_flash':
        from ..flash.shformer_flash import shformer_flash
        model = shformer_flash(num_classes)
        return model

    if model_name == 'shformer_flash_d2':
        from ..flash.shformer_flash_d2 import shformer_flash_d2
        model = shformer_flash_d2(num_classes)
        return model
    
    # PVT
    if model_name == 'shformer_pvt':
        from ..pvt.shformer_pvt import shformer_pvt
        model = shformer_pvt(num_classes)
        return model
    
############################################   UNet-family
    if model_name == "unet":
        from ..unet.UNet import UNet
        model = UNet(in_channels=3, n_classes=num_classes, is_fe=False, feature_scale=2)
        # model = UNet(in_channels=3, n_classes=num_classes, is_fe=True, feature_scale=2)
        return model

    if model_name == "unet_sci":
        from ..unet.UNet import UNet
        model = UNet(in_channels=3, n_classes=num_classes, is_fe=True, feature_scale=2)
        # model = UNet(in_channels=3, n_classes=num_classes, is_fe=True, feature_scale=2)
        return model
    
    if model_name == "unet++":
        from ..unet.UNet_nested import UNet_Nested
        model = UNet_Nested(in_channels=3, n_classes=num_classes, is_ds=True)
        return model