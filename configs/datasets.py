from .user_configs import IMAGENET_ES_ROOT_DIR

DATA_ROOT_DIR = IMAGENET_ES_ROOT_DIR

SAMPLE_DIR_MAP = {
    'es-train': 'tin_no_resize_sample_removed', #S3
    'es-val': 'sampled_tin_no_resize', #S1
    'es-test': 'sampled_tin_no_resize2', #S2
}

OPS_NUM = {
    'es-val': 64,
    'es-test': 27,
}

ENVS = {
    'es-val': ['l1', 'l5'],
    'es-test': ['l1', 'l5'],
}