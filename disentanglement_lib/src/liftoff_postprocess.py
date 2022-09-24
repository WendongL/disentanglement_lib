from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('../')
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.postprocessing import postprocess
import tensorflow.compat.v1 as tf
import gin.tf
from liftoff import parse_opts
import argparse
from glob import glob
    
def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def run(params):
    out_dir = params.train.out_dir
    run_id = params.train.run_id
    folder_num = params.train.folder_num

    # write postprocess.gin
    for folder in sorted(glob(out_dir+"/*/", recursive = True)):
        if folder_num in folder:
            folder_runid = os.path.join(folder, str(run_id))
    file = open(folder_runid + "/postprocess.gin", "w")
    file.write('# coding=utf-8')
    file.write(os.linesep)
    cfg_dict = namespace_to_dict(params.postprocess)
    print(cfg_dict)
    for name in cfg_dict.keys():
        if name not in ['out_dir','run_id','title','cfg_id','full_title']:
            for subname in cfg_dict[name].keys():
                file.write(name+ "." + subname+ " = " + str(cfg_dict[name][subname]))
                file.write(os.linesep)
    file.close()
    postprocess_gin = [folder_runid +"/postprocess.gin"] # This contains the settings.
    
    # postprocess.postprocess_with_gin defines the standard extraction protocol.
    overwrite= True
    post_seed = params.postprocess.postprocess.random_seed
    gin_bindings = ['postprocess.random_seed = '+ str(post_seed)]
    postprocess_fn = params.postprocess.postprocess.postprocess_fn
    
    
    for folder in sorted(glob(out_dir+"/*/", recursive = True)):
        if folder_num in folder:
            model_path = os.path.join(folder, str(run_id), "model")
            representation_path = os.path.join(folder, str(run_id), "representation", postprocess_fn, str(post_seed))
    postprocess.postprocess_with_gin(model_dir=model_path, output_dir=representation_path, overwrite=overwrite,
                                    gin_config_files=postprocess_gin, gin_bindings=gin_bindings)

    
if __name__ == "__main__":
    run(parse_opts())
