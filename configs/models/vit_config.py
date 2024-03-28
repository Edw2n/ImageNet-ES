cfg = {
    "learning_rate": 5e-3,
    "epochs": 20,
    "batch_size": 1024,
    "patience": 2,
    "factor": 0.5,
    "threshold": 1e-3
}

config = {
    'MODEL_NAME': 'deit_base_patch16_224.fb_in1k',
    'TARGETS':  ['head', 'norm'],
    'TRANSFER_TARGET': 'head',
    'CFG': cfg,
}
