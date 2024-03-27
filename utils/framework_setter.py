SAMPLE_OPS_NUM =2000
from configs.datasets import ENVS

class FrameworkSetter():
    def __init__(self, DATA_ROOT_DIR, DATA_INFO, args, model_name, envs=ENVS['es-test'], train_dir='es-train'):
        self.DATA_ROOT_DIR = DATA_ROOT_DIR
        self.DATA_INFO = DATA_INFO
        self.__dict__.update(args)
        self.envs = envs
        self.model_name = model_name
        self.train_dir = train_dir

    def generate_validation_groups(self):
        '''
        add dataset groups ('val_id', 'val_id-', 'val_id+', 'val_es-', 'val_es+') to near ood
        
        - 'standard_id': ground-truth, id
        - 'sample': original image
        - 'es': ImageNet-ES version
        
        - 'val_id':  S1 (val/standard_sample_id)
        - 'val_id-': S1- (val/sample_ood)
        - 'val_id+': S1+ (val/sample_id)
        - 'val_es-': ES-VAL - (val/es_ood)
        - 'val-es+': ES-VAL - (val/es_id)
        '''
        
        # S1 (val/standard_sample_id)
        self.DATA_INFO['ood']['near']['val_id']= {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path':f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/sample_standard_id.txt'
        }
        self.DATA_INFO['ood']['near']['datasets'].append('val_id')

        # S1+/- (val/sample_id, sample_ood)
        self.DATA_INFO['ood']['near']['val_id-']= {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path':f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/sample_ood.txt'
        }
        self.DATA_INFO['ood']['near']['datasets'].append('val_id-')
        self.DATA_INFO['ood']['near']['val_id+']= {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path':f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/sample_id.txt'
        }
        self.DATA_INFO['ood']['near']['datasets'].append('val_id+')

        # es+/es- (val/es_id, es_ood)
        self.DATA_INFO['ood']['near']['val_es-']= {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path':f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/es_ood.txt'
        }
        self.DATA_INFO['ood']['near']['datasets'].append('val_es-')
        self.DATA_INFO['ood']['near']['val_es+']= {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path':f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/es_id.txt'
        }
        self.DATA_INFO['ood']['near']['datasets'].append('val_es+')
        
    def generate_es_testset(self):
        '''
        add dataset groups in testset of ImageNet-ES to near ood
        - format : f'test-{setting_group}-id/ood'
        - setting_group: f'{env}-param_{i}'
        '''
        data_dir = f'{self.DATA_ROOT_DIR}/{self.test_dir}'
        testset_info = {}
        ad_testsets = sum([[f'{env}-param_{i}' for i in range(1,self.test_op_num+1)] for env in self.envs], [])
        datasets = []
        for dataset in ad_testsets:
            name = 'test-' + dataset +'-ood'
            datasets.append(name)
            testset_info[name] = {
                'data_dir': data_dir,
                'imglist_path': f'{data_dir}/{self.model_name}/imglist/param_control/{dataset}_ood.txt'
            }
            name = 'test-' + dataset +'-id'
            datasets.append(name)
            testset_info[name] = {
                'data_dir': data_dir,
                'imglist_path': f'{data_dir}/{self.model_name}/imglist/param_control/{dataset}_id.txt'
            }
        self.DATA_INFO['ood']['near']['datasets'] += datasets
        self.DATA_INFO['ood']['near'].update(testset_info)

    def set_semantics_centric(self):
        ID_SETTING = {
            'train': { #TODO 요부분 Train 이어야함. val 아니고 s3
                'data_dir': f'{self.DATA_ROOT_DIR}/{self.train_dir}',
                'imglist_path': f'{self.DATA_ROOT_DIR}/{self.train_dir}/{self.model_name}/imglist/sample_standard_id.txt'
            },
            'val': {
                'data_dir': f'{self.DATA_ROOT_DIR}/{self.train_dir}',
                'imglist_path': f'{self.DATA_ROOT_DIR}/{self.train_dir}/{self.model_name}/imglist/sample_standard_id.txt'
            },
            'test': { # S2 standard id // but not experimeted in the papaer (not valid framework)
                'data_dir': f'{self.DATA_ROOT_DIR}/{self.test_dir}',
                'imglist_path': f'{self.DATA_ROOT_DIR}/{self.test_dir}/{self.model_name}/imglist/sample_standard_id.txt'
            }
            }

        self.DATA_INFO['id'] = ID_SETTING
    
    def set_model_centric(self):
        ID_SETTING = {
            'train': { #S3+
                'data_dir': f'{self.DATA_ROOT_DIR}/{self.train_dir}',
                'imglist_path': f'{self.DATA_ROOT_DIR}/{self.train_dir}/{self.model_name}/imglist/sample_id.txt'
                },
            'val': { #S3+
                'data_dir': f'{self.DATA_ROOT_DIR}/{self.train_dir}',
                'imglist_path': f'{self.DATA_ROOT_DIR}/{self.train_dir}/{self.model_name}/imglist/sample_id.txt'
                },
            'test':{ #S2+
                    'data_dir': f'{self.DATA_ROOT_DIR}/{self.test_dir}',
                    'imglist_path': f'{self.DATA_ROOT_DIR}/{self.test_dir}/{self.model_name}/imglist/sample_id.txt' #S2+ in paper
                }        
        }
        self.DATA_INFO['id'] = ID_SETTING

        OOD_VAL_SETTING = { #S3-
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.train_dir}',
            'imglist_path': f'{self.DATA_ROOT_DIR}/{self.train_dir}/{self.model_name}/imglist/sample_ood.txt',
        }
        self.DATA_INFO['ood']['val'] = OOD_VAL_SETTING
        
        self.DATA_INFO['ood']['near']['test_id-']= {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.test_dir}',
            'imglist_path':f'{self.DATA_ROOT_DIR}/{self.test_dir}/{self.model_name}/imglist/sample_ood.txt'
        }
        self.DATA_INFO['ood']['near']['datasets'].append('test_id-') #S2- in paper
    
    def set_enhancement(self):
        
        self.DATA_INFO['id']['train'] = {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path': f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/es_id.txt',
        }

        self.DATA_INFO['id']['val'] = {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path': f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/es_id.txt',
        }
        
        self.DATA_INFO['ood']['val'] = {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.val_dir}',
            'imglist_path': f'{self.DATA_ROOT_DIR}/{self.val_dir}/{self.model_name}/imglist/es_ood.txt',
        }

        self.DATA_INFO['id']['test'] = {
                    'data_dir': f'{self.DATA_ROOT_DIR}/{self.test_dir}',
                    'imglist_path': f'{self.DATA_ROOT_DIR}/{self.test_dir}/{self.model_name}/imglist/sample_id.txt'
        }

        self.DATA_INFO['ood']['near']['test_id-']= {
            'data_dir': f'{self.DATA_ROOT_DIR}/{self.test_dir}',
            'imglist_path':f'{self.DATA_ROOT_DIR}/{self.test_dir}/{self.model_name}/imglist/sample_ood.txt'
        }
        self.DATA_INFO['ood']['near']['datasets'].append('test_id-')


    def set_data_info(self, id_name):
        '''
        set data_info to support experiment of id_name
        [input]
        'id_name': experiment name
        '''    
        self.generate_es_testset()
        self.generate_validation_groups()

        if 'MC' in id_name:
            self.set_model_centric()
        elif 'SC' in id_name:
            self.set_semantics_centric()
        elif 'ES' in id_name:
            self.set_enhancement()
        else:
            print('error: unknown id_name, {id_name}')