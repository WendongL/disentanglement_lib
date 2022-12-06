from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('../')
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.visualize.visualize_model import visualize
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
    for folder in sorted(glob(out_dir+"/*/", recursive = True)):
        print(folder)
        if folder_num in folder:
            folder_runid = os.path.join(folder, str(run_id))
    visualize(os.path.join(folder_runid,"model"),
                os.path.join(folder_runid,"visual2"),
                overwrite=True,
                num_animations=5,
                num_frames=20,
                fps=10,
                num_points_irs=100)
if __name__ == "__main__":
    run(parse_opts())
