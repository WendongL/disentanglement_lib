from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('../')
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
import tensorflow.compat.v1 as tf
import gin.tf
from liftoff import parse_opts
import argparse


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }
def run(params):
    out_dir = params.out_dir
    # write train.gin
    file = open(out_dir + "/train.gin", "w")
    file.write('# coding=utf-8')
    file.write(os.linesep)
    cfg_dict = namespace_to_dict(params.train)
    print(cfg_dict)
    for name in cfg_dict.keys():
        if name not in ['out_dir','run_id','title','cfg_id','full_title']:
            for subname in cfg_dict[name].keys():
                file.write(name+ "." + subname+ " = " + str(cfg_dict[name][subname]))
                file.write(os.linesep)
    file.write('model.random_seed = '+str(params.run_id))
    file.close()
    

    # os.system("dlib_train --gin_config="+ out_dir +"/model.gin --model_dir=" + out_dir+"/output")
    #########
    # out_dir = '/is/ei/wliang/disentanglement_lib/src/results/2022Sep15-183457_default/0000_default/0'
    overwrite= True
    gin_bindings = []
    model_path = os.path.join(out_dir, "model")
    train.train_with_gin(model_path, overwrite, [out_dir +"/train.gin"], gin_bindings)
    

    
if __name__ == "__main__":
    run(parse_opts())
