from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import time
import warnings

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import fairness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import hsic  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import strong_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unified_scores  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

import gin.tf

def _has_kwarg_or_kwargs(f, kwarg):
  """Checks if the function has the provided kwarg or **kwargs."""
  # For gin wrapped functions, we need to consider the wrapped function.
  if hasattr(f, "__wrapped__"):
    f = f.__wrapped__
  (args, _, kwargs, _, _, _, _) = inspect.getfullargspec(f)
  if kwarg in args or kwargs is not None:
    return True
  return False

model_dir = "/home/wliang/miniconda3/envs/tf/lib/python3.7/site-packages/disentanglement_lib/src/results/2022Sep17-235755_train_hsicbetavae/0000_train.HsicBetaVAE.beta_0.5__train.HsicBetaVAE.alpha_0/0/representation/@mean_representation/0"
evaluation_fn = hsic.hsic_batch
random_seed = 0
dataset = named_data.get_named_ground_truth_data("dsprites_full")
name=""


# Path to TFHub module of previously trained representation.
module_path_train = os.path.join(model_dir,"../../..", "model", "tfhub")
### the model_dir that evaluate takes is acutually representation folder, in which the Hub is different from trained model, e.g. the signature is different!!!
if _has_kwarg_or_kwargs(evaluation_fn, "decoder"):
  with hub.eval_function_for_module(module_path_train) as f2:
    ### Added decoder
    def decoder_function(x):
        # f2 = hub.eval_function_for_module(module_path_train)
        # output = f2(images=x, signature="decoder", as_dict=True)
        # output =f2(dict(images=x), signature="gaussian_encoder", as_dict=True)
        output =f2(dict(images=x),  signature="decoder", as_dict=True)
        print("decoder:", output.keys())
        return np.array(output["images"])
    
    def _representation_function(x):
      """Computes representation vector for input images."""
      output = f(dict(images=x), signature="representation", as_dict=True)
      return np.array(output["default"])  # what is "default"?

    ## ERR: 'decoder' is not recognized
    # def decoder_function(x):
    #   return gaussian_encoder_model.decode(latent_vector, observation_shape, is_training)

score = hsic.hsic_batch(ground_truth_data, representation_function,decoder, random_state,
                batch_size=16,
                artifact_dir=None,
                num_points=gin.REQUIRED,
                s_x=gin.REQUIRED,
                s_y=gin.REQUIRED,
                num_sample_reparam = gin.REQUIRED)
results_dict = evaluation_fn(
            dataset,
            _representation_function,
            random_state=np.random.RandomState(random_seed),
            artifact_dir=artifact_dir,
            decoder = decoder_function)