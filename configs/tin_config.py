from torchvision import transforms

data_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'draw': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

cfg = {
    "learning_rate": 5e-3,
    "epochs": 20,
    "batch_size": 256,
    "patience": 2,
    "factor": 0.5,
    "threshold": 1e-2
}

LOADER_CONFIG = {
    'DATA_TRANSFORMS': data_transforms,
    'num_workers': {'train' : 8, 'val'   : 8,'test'  : 8},
    'class_num': 200,
    'CFG': cfg,
}
