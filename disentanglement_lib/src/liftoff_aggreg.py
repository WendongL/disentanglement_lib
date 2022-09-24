from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('../')
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.utils import aggregate_results
import tensorflow.compat.v1 as tf
import gin.tf
from liftoff import parse_opts
import argparse
from glob import glob

def run(params):
    # Aggregate the results.
    # ------------------------------------------------------------------------------
    # In the previous steps, we saved the scores to several output directories. We
    # can aggregate all the results using the following command.
    pattern = os.path.join(out_dir,
                        "*/*/metrics/*/results/aggregate/evaluation.json")
    results_path = os.path.join(out_dir, "results.json")
    aggregate_results.aggregate_results_to_json(
        pattern, results_path)

    # Print out the final Pandas data frame with the results.
    # ------------------------------------------------------------------------------
    # The aggregated results contains for each computed metric all the configuration
    # options and all the results captured in the steps along the pipeline. This
    # should make it easy to analyze the experimental results in an interactive
    # Python shell. At this point, note that the scores we computed in this example
    # are not realistic as we only trained the models for a few steps and our custom
    # metric always returns 1.
    model_results = aggregate_results.load_aggregated_json_results(results_path)
    print(model_results)

if __name__ == "__main__":
    run(parse_opts())
