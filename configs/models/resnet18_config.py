cfg = {
    "learning_rate": 5e-2,
    "epochs": 15,
    "batch_size": 128,
    "patience": 1,
    "factor": 0.5,
    "threshold": 1e-2
}

config = {
    'MODEL_NAME': 'resnet18.a1_in1k',
    'TARGETS':  ['fc'],
    'TRANSFER_TARGET': 'fc',
    'CFG': cfg,
}
