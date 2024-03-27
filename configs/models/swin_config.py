cfg = {
    "learning_rate": 5e-3,
    "epochs": 20,
    "batch_size": 128,
    "patience": 2,
    "factor": 0.5,
    "threshold": 1e-3
}

config = {
    'MODEL_NAME': 'swin_base_patch4_window7_224',
    'TARGETS':  ['head', 'norm'],
    'TRANSFER_TARGET': 'head.fc',
    'CFG': cfg,
}
