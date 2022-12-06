
# from unittest import result
import pdb
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
sys.path.append('../')
from disentanglement_lib.utils import aggregate_results
import numpy as np
from liftoff import parse_opts
from matplotlib.pyplot import cm
import matplotlib as mpl
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

def run(params):
    sns.set()
    mpl.rcParams['figure.dpi'] = 120
    mpl.rcParams['savefig.dpi'] = 200
    out_dir = params.train.out_dir
    if params.evaluation.evaluation.evaluation_fn == "@hsic":
        metric_name="hsic"
    elif params.evaluation.evaluation.evaluation_fn == "@dci":
        metric_name="dci"
    elif params.evaluation.evaluation.evaluation_fn == "@beta_vae_sklearn":
        metric_name="beta"
    elif params.evaluation.evaluation.evaluation_fn == "@factor_vae_score":
        metric_name="factor"
    # To tune ##############################################
    # alpha_list = [0., 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    # beta_list = [0, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]
    #####################################################
    results_path = os.path.join(out_dir, "results_"+metric_name+".json")
    model_results = aggregate_results.load_aggregated_json_results(results_path)
    if params.evaluation.evaluation.evaluation_fn == "@hsic":
        metric_name="hsic"
        # pdb.set_trace()
        model_results_small = model_results[["train_config.dip_vae.alpha",
                                            "train_config.dip_vae.lambda_od",
                                            "evaluation_results.hsic100"
                                            ]] #"train_results.hsic_loss"

        model_results_small=model_results_small.rename(columns = {"train_config.dip_vae.alpha":"alpha", "train_config.dip_vae.lambda_od":"lambda_od"})
    elif params.evaluation.evaluation.evaluation_fn == "@dci":
        metric_name="dci"
        model_results_small = model_results[["train_config.dip_vae.alpha",
                            "train_config.dip_vae.lambda_od",
                            "train_results.kl_loss",
                            "train_results.elbo",
                            "train_results.loss",
                            "train_results.reconstruction_loss",
                            "train_results.regularizer",
                            "evaluation_results.elapsed_time",
                            "evaluation_results.informativeness_train",
                            "evaluation_results.informativeness_test",
                            "evaluation_results.disentanglement",
                            "evaluation_results.completeness"
                            ]]

        model_results_small=model_results_small.rename(columns = {"train_config.dip_vae.alpha":"alpha", "train_config.dip_vae.lambda_od":"lambda_od"})
    
    elif params.evaluation.evaluation.evaluation_fn == "@beta_vae_sklearn":
        metric_name="beta"
        model_results_small = model_results[["train_config.dip_vae.alpha",
                                            "train_config.dip_vae.lambda_od",
                                            "evaluation_results.train_accuracy",
                                            "evaluation_results.eval_accuracy"]]
        model_results_small=model_results_small.rename(columns = {"train_config.dip_vae.alpha":"alpha",
                                                                "train_config.dip_vae.lambda_od":"lambda_od",
                                                                "evaluation_results.train_accuracy":"beta_train_accuracy",
                                                                "evaluation_results.eval_accuracy":"beta_eval_accuracy"})
    elif params.evaluation.evaluation.evaluation_fn == "@factor_vae_score":
        metric_name="factor"
        model_results_small = model_results[["train_config.dip_vae.alpha",
                                            "train_config.dip_vae.lambda_od",
                                            "evaluation_results.train_accuracy", 
                                            "evaluation_results.eval_accuracy"]]
        model_results_small=model_results_small.rename(columns = {"train_config.dip_vae.alpha":"alpha",
                                                                  "train_config.dip_vae.lambda_od":"lambda_od",
                                                                  "evaluation_results.train_accuracy":"factor_train_accuracy",
                                                                  "evaluation_results.eval_accuracy":"factor_eval_accuracy"})
    df_mean = model_results_small.groupby(['alpha', 'lambda_od'], group_keys=True).mean()
    labels = list(df_mean.columns)

    sns.set_theme()
    for label in labels:
        plt.figure()
        heatmap_pt=pd.pivot_table(df_mean, values =label, index=['alpha'], columns='lambda_od')
        if params.evaluation.evaluation.evaluation_fn == "@hsic":
            sns.heatmap(heatmap_pt, annot=True, fmt=".0e", norm=LogNorm(), annot_kws={"size": 8}) ##### change in case fmt=".0e", norm=LogNorm() / fmt=".2f"
        else:
            sns.heatmap(heatmap_pt, annot=True, fmt=".2f", annot_kws={"size": 8}) ##### change in case fmt=".0e", norm=LogNorm() / fmt=".2f"
        # ax.set(xlabel='beta', ylabel='alpha')
        plt.title(label)
        plt.savefig(os.path.join(out_dir, label+'_3_3seeds.png'), bbox_inches='tight')
    if params.evaluation.evaluation.evaluation_fn == "@dci":
        labels2 = ["train_results.elbo",
                                "train_results.loss",
                                "train_results.kl_loss",
                                "train_results.reconstruction_loss",
                                "train_results.regularizer",
                                "evaluation_results.elapsed_time"]
        for label in labels2:
            plt.figure()
            heatmap_pt=pd.pivot_table(df_mean, values =label, index=['alpha'], columns='lambda_od')
            sns.heatmap(heatmap_pt, annot=True, fmt=".1f", annot_kws={"size": 8}) ##### change in case fmt=".0e", norm=LogNorm()
            # ax.set(xlabel='beta', ylabel='alpha')
            plt.title(label)
            plt.savefig(os.path.join(out_dir, label+'_3_3seeds.png'), bbox_inches='tight')
if __name__ == "__main__":
    run(parse_opts())
