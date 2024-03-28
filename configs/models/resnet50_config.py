cfg = {
    "learning_rate": 5e-3,
    "epochs": 15,
    "batch_size": 128,
    "patience": 2,
    "factor": 0.5,
    "threshold": 1e-2
}

config = {
    'MODEL_NAME': 'resnet50.a1_in1k',
    'TARGETS':  ['fc'],
    'TRANSFER_TARGET': 'fc',
    'CFG': cfg,
}
