from .efficientNet_config import config as EFFICIENTNET_CONFIG
from .swin_config import config as SWIN_CONFIG
from .resnet18_config import config as RESNET18_CONFING

timm_config = {
    'swin': SWIN_CONFIG,         # swin_base_patch4_window7_224
    'en': EFFICIENTNET_CONFIG,   # efficientnet_b0.ra_in1k
    'resnet18': RESNET18_CONFING # resnet18.a1_in1k
}
