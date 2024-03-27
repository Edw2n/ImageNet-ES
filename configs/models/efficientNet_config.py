cfg = {
    "learning_rate": 5e-3,
    "epochs": 20,
    "batch_size": 128,
    "patience": 2,
    "factor": 0.5,
    "threshold": 1e-2
}

config = {
    'MODEL_NAME': 'efficientnet_b0.ra_in1k',
    'TARGETS':  ['bn2', 'classifier'],
    'TRANSFER_TARGET': 'classifier',
    'CFG': cfg,
}
