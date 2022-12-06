# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import os
import sys
sys.path.append('../')
from disentanglement_lib.utils import aggregate_results
import pdb
from liftoff import parse_opts
# from glob import glob

def run(params):
    out_dir = params.train.out_dir
    if params.evaluation.evaluation.evaluation_fn == "@hsic":
        metric_name="hsic"
    elif params.evaluation.evaluation.evaluation_fn == "@dci":
        metric_name="dci"
    elif params.evaluation.evaluation.evaluation_fn == "@beta_vae_sklearn":
        metric_name="beta"
    elif params.evaluation.evaluation.evaluation_fn == "@factor_vae_score":
        metric_name="factor"
    # Aggregate the results.
    # ------------------------------------------------------------------------------
    # In the previous steps, we saved the scores to several output directories. We
    # can aggregate all the results using the following command.
    # pattern = os.path.join(out_dir,
    #                     "*/*/representation/@mean_representation/0/metrics/"+params.evaluation.evaluation.evaluation_fn+"/*/results/aggregate/evaluation.json")
    pattern = os.path.join(out_dir,
                        "*/*/representation/@mean_representation/0/metrics/"+params.evaluation.evaluation.evaluation_fn+"/*/results/aggregate/evaluation.json")
    results_path = os.path.join(out_dir, "results_"+metric_name+".json")
    print(pattern)
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


""" only grouping by alpha and beta. There are 5 seeds for training models, and 3 seeds for random classfiers in evaluation step.
  "train_config.HsicBetaVAE.alpha": "0.0",
  "train_config.HsicBetaVAE.beta": "1.0",
  "train_results.elbo": -54.515193939208984,
  "train_results.hsic_loss": 0.003946063108742237,
  "train_results.kl_loss": 27.080379486083984,
  "train_results.loss": 54.515193939208984,
  "train_results.reconstruction_loss": 27.434730529785156,
  "train_results.regularizer": 27.080379486083984,
  "evaluation_results.train_accuracy": 0.8664,
  "evaluation_results.eval_accuracy": 0.858,
"""