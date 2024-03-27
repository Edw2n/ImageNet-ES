import argparse

from configs.datasets import DATA_ROOT_DIR
from configs.user_configs import TARGET_OOD_POSTPROCESSORS
from utils.eval import Evaluator_ES
from utils.experiment_setup import get_evalood_args, oodexp_setup

parser = argparse.ArgumentParser()
args = get_evalood_args(parser)

if __name__ == '__main__':
    device_num, EXPERIMENT_ID, OUTPUT_ROOT_DIR = oodexp_setup(args)
    
    evaluator = Evaluator_ES(EXPERIMENT_ID,
                             args, 
                             data_root_dir=DATA_ROOT_DIR,
                             output_dir=OUTPUT_ROOT_DIR,
                             device_num=device_num)
    if not evaluator:
        exit()

    results, scores = evaluator.eval_ood(fsood_in=False,
                                         score_return=True,
                                         target_postprocessor=TARGET_OOD_POSTPROCESSORS)