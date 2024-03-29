METHOD_DICT = {
    'vim': 'ViM',
    'react': 'ReAct',
    'msp': 'MSP',
    'odin': 'ODIN',
    'knn': 'KNN',
    'ash': 'ASH',
}

LABEL_DICT = {
    'worst': 'Worst',
    'best': 'Best',
    'auto': 'Auto',
}

COLOR = {
    # OOD detection methods
    'msp': 'blue',
    'odin': 'red',
    'react': 'orange',
    'vim': 'darkcyan',
    'knn': 'black',
    'ash': 'magenta',
    
    # OOD detection framework settings
    'MS-OOD': 'green',
    'ENHANCEMENT': 'orange',
    
    # Feauture Activation analysis
    'worst': 'magenta',
    'best': 'orange',
    'auto': 'blue',
}

HIST_COLOR = {
    'ID': 'blue',
    'S-OOD': 'red', # 이거 ID를 red로 통일 시키는게 날지....
    'ID+': 'red',
    'ID-': 'red',
    'ImageNet-ES': 'darkcyan',
    'ImageNet-ES+': 'darkcyan',
    'ImageNet-ES-': 'darkcyan',
}

CLASS_MAP = {
    'n03026506': 'christmas stocking',
    'n03404251': 'fur coat',
    'n02917067': 'bullet train',
    'n02236044': 'mantis',
    'n02950826': 'cannon',
}

# SETTING_COLOR = {
#     'MS-OOD':
# }

S_OOD_CATEGORY = {
    'ssb_hard': 'N1',
    'ninco': 'N2',
    'inaturalist': 'F1',
    'textures': 'F2',
    'openimage_o': 'F3'
}

NF_MARKER = {
    'msp': '*',
    'odin': '*',
    'react': '*',
    'vim': 'x',
    'knn': 'x',
    'ash': 'x',
    
    'MS-OOD': 'o',
    'ENHANCEMENT': 'o',
}