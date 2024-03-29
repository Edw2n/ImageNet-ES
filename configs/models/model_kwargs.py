from ..user_configs import SWIN_PT, RESNET18_PT, EN_PT, VIT_PT

RN18_kwargs = {
    'model_type': 'timm',
    'arch': 'resnet18.a1_in1k',
    'resume_path':  RESNET18_PT,
    'fv_name': 'fc',
    'fv_out': 200,
    'fv_in': 512,
    'fv_out_name': 'global_pool',
}

SWIN_kwargs = {
    'model_type': 'timm',
    'arch': 'swin_base_patch4_window7_224',
    'resume_path': SWIN_PT,
    'fv_name': 'head.fc',
    'fv_out': 200,
    'fv_in': 1024,
    'fv_out_name': 'head.global_pool',
}

EN_kwargs = {
    'model_type': 'timm',
    'arch': 'efficientnet_b0.ra_in1k',
    'resume_path': EN_PT,
    'fv_name': 'classifier',
    'fv_out': 200,
    'fv_in': 1280,
    'fv_out_name': 'global_pool',
}

VIT_kwargs = {
    'model_type': 'timm',
    'arch': 'deit_base_patch16_224.fb_in1k',
    'resume_path': VIT_PT,
    'fv_name': 'head',
    'fv_out': 200,
    'fv_in': 768,
    'fv_out_name': 'head_drop',
}

KWARGS_MAP = {
    'resnet18': RN18_kwargs,
    'swin': SWIN_kwargs,
    'en': EN_kwargs,
    'vit': VIT_kwargs,
}