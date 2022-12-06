from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('../')
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.utils import aggregate_results
import tensorflow.compat.v1 as tf
import gin.tf
from liftoff import parse_opts
import argparse
from glob import glob
import pdb
    
def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }
def run(params):
    
    out_dir = params.train.out_dir
    run_id = params.train.run_id
    folder_num = params.train.folder_num
    post_seed = params.postprocess.postprocess.random_seed
    # write evaluation.gin
    for folder in sorted(glob(out_dir+"/*/", recursive = True)):
        if folder_num in folder:
            folder_repre_seed = os.path.join(folder, str(run_id), 'representation', '@mean_representation', str(post_seed))
        continue

    file = open(folder_repre_seed + "/evaluation.gin", "w")
    file.write('# coding=utf-8')
    file.write(os.linesep)
    cfg_dict = namespace_to_dict(params.evaluation)
    print(cfg_dict)
    for name in cfg_dict.keys():
        if name not in ['out_dir','run_id','title','cfg_id','full_title','train']:
            for subname in cfg_dict[name].keys():
                file.write(name+ "." + subname+ " = " + str(cfg_dict[name][subname]))
                file.write(os.linesep)
    file.close()
    # gin_bindings = [
    #     "evaluation.evaluation_fn = @mig",
    #     "dataset.name='auto'",
    #     "evaluation.random_seed = 0",
    #     "mig.num_train=1000",
    #     "discretizer.discretizer_fn = @histogram_discretizer",
    #     "discretizer.num_bins = 20"
    # ]
    overwrite= True
    evaluation_gin = [folder_repre_seed +"/evaluation.gin"]
    eval_seed = params.evaluation.evaluation.random_seed
    metric = params.evaluation.evaluation.evaluation_fn
    gin_bindings = ["evaluation.evaluation_fn = "+metric]
    result_path = os.path.join(folder_repre_seed, "metrics", metric, str(eval_seed))
    evaluate.evaluate_with_gin(
        folder_repre_seed, result_path, overwrite, evaluation_gin, gin_bindings=gin_bindings)



if __name__ == "__main__":
    run(parse_opts())
