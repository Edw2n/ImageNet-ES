import os, sys
import pickle 

def get_file_dict(root_dir):
    files_dict = {}
    for directory in os.listdir(root_dir):
        cls_name = directory    
        files = sorted(os.listdir(os.path.join(root_dir, directory)))
        files_dict[cls_name] = files

    return files_dict

def merge_dict(dict1, dict2):

    merged_dict = dict1.copy()
    for k, v in dict1.items():
        if k in dict2:
            merged_dict[k] += dict2[k]

    for k, v in dict2.items():
        if k not in dict1:
            merged_dict[k] = v
    return merged_dict




our_samples_val = '/datasets/ImageNet-ES/val/param_control/l1/param_1'
our_samples_test = '/datasets/ImageNet-ES/test/param_control/l1/param_1'

val_dict = get_file_dict(our_samples_val)
test_dict = get_file_dict(our_samples_test)


with open('fnames_tin_val.pkl', 'wb') as f:
    pickle.dump(val_dict, f)

with open('fnames_tin_test.pkl', 'wb') as f:
    pickle.dump(test_dict, f)





