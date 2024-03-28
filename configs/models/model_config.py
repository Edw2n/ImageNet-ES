from .efficientNet_config import config as EFFICIENTNET_CONFIG
from .swin_config import config as SWIN_CONFIG
from .resnet18_config import config as RESNET18_CONFING
from .resnet50_config import config as RESNET50_CONFING
from .vit_config import config as VIT_CONFING

timm_config = {
    'swin': SWIN_CONFIG,          # swin_base_patch4_window7_224
    'en': EFFICIENTNET_CONFIG,    # efficientnet_b0.ra_in1k
    'resnet18': RESNET18_CONFING, # resnet18.a1_in1k
    'resnet50': RESNET50_CONFING, # resnet50.a1_in1k
    'vit': VIT_CONFING ,          # deit_base_patch16_224.fb_in1k.pth
}
