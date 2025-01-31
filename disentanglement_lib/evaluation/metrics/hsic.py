from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import scipy
from six.moves import range
import gin.tf
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import pdb

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm =tf.reshape(tf.math.reduce_sum(x**2,axis=-1),[-1,1])
    return -2*tf.matmul(x,tf.transpose(x)) + instances_norm + tf.transpose(instances_norm)

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    # print(pairwise_distances_)
    return tf.math.exp(-pairwise_distances_ /sigma)

def HSIC(x, y, s_x=1, s_y=1):
    m,_ = x.shape #batch size
    m=int(m)
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = tf.eye(m) - 1.0/m * tf.ones([m,m])
    result = tf.linalg.trace(tf.matmul(L,tf.matmul(H,tf.matmul(K,H))))/((m-1)**2)
    return result


### numpy version
def pairwise_distances_np(x):
    #x should be two dimensional
    instances_norm =np.reshape(np.sum(x**2,axis=-1),[-1,1])
    return -2*np.matmul(x,np.transpose(x)) + instances_norm + np.transpose(instances_norm)

def GaussianKernelMatrix_np(x, sigma=1):
    pairwise_distances_ = pairwise_distances_np(x)
    # print(pairwise_distances_)
    return np.exp(-pairwise_distances_ /sigma)

def HSIC_np(x, y, s_x=1, s_y=1):
    m,_ = x.shape #batch size
    m=int(m)
    K = GaussianKernelMatrix_np(x,s_x)
    L = GaussianKernelMatrix_np(y,s_y)
    H = np.eye(m) - 1.0/m * np.ones([m,m])
    result = np.trace(np.matmul(L,np.matmul(H,np.matmul(K,H))))/((m-1)**2)
    return result


# The tf version with reparamatrization
def pairwise_distances_(x):
    #x should be 3 dimensional
    batch_size,_,_ = x.shape
    instances_norm = tf.math.reduce_sum(x**2, axis=-1)
    instances_norm = tf.reshape(instances_norm, [batch_size,-1,1])
    print("############")
    print('instances_norm', instances_norm.shape)
    print('x', x.shape)
    return -2*tf.einsum('aij,akj->aik', x, x) + instances_norm + tf.einsum('kij->kji', instances_norm)
    # return -2*torch.bmm(x,torch.permute(x, (0,2,1))) + instances_norm + torch.permute(instances_norm, (0,2,1))

def GaussianKernelMatrix_(x, sigma=1):
    pairwise_distances = pairwise_distances_(x)
    return tf.linalg.expm(-pairwise_distances /sigma)
def HSIC_(x, y, s_x=100, s_y=100):
    # calculate HSIC over a dataset like: [batch_size, num_reparam, dim_x]
    # calculate over dim1,2
    # output: HSIC [batch_size]
    # pdb.set_trace()
    batch_size,m,_ = x.get_shape().as_list()
    K = GaussianKernelMatrix_(x,s_x) # [batch_size, num_reparam, num_reparam]
    L = GaussianKernelMatrix_(y,s_y)
    H = tf.eye(m) - 1.0/m * tf.ones([m,m])
    H = tf.expand_dims(H, axis=0)
    BH = tf.tile(H, [batch_size,1,1]) # repeat batch_size times H
    A = tf.einsum('aij,ajk->aik', L, tf.einsum('aij,ajk->aik', BH, tf.einsum('aij,ajk->aik', K, BH))) 
    result = tf.linalg.trace(A)/((m-1)**2) #[batch_size]
    return tf.reduce_mean(result) # mean over samples in a batch

@gin.configurable(
    "hsic",blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
### eval HSIC
def hsic_batch(ground_truth_data, representation_function, decoder, random_state,
                batch_size=16,
                artifact_dir=None,
                num_points=gin.REQUIRED,
                sigma=gin.REQUIRED,
                num_sample_reparam = gin.REQUIRED,
                ):
            # num_sample_reparam: num of samples of reparamatrizing z using mu and sigma
    del artifact_dir
    score={}
    logging.info("Generating training set.")
    for s in sigma:
        hsic_score = 0
        for i in tqdm(range(num_sample_reparam)):
            feats, observations, reconstructions = utils.generate_batch_code_output(ground_truth_data, representation_function, decoder,
                                num_points, random_state, batch_size)
            print("############hsic_batch start")
            # print('feats', feats.shape)
            # print('observations', observations.shape)
            # print('reconstructions', reconstructions.shape)
            # print('reconstructions', type(reconstructions))
            # print(reconstructions)
            observations = np.reshape(observations, [observations.shape[0],-1])
            reconstructions = np.reshape(reconstructions, [reconstructions.shape[0],-1])
            # print("after")
            # print('feats', feats.shape)
            # print('observations', observations.shape)
            # print('reconstructions', reconstructions.shape)
            hsic_score += HSIC_np(feats, observations - reconstructions, s, s)
        hsic_score /= num_sample_reparam
        score['hsic'+str(s)]= hsic_score
    return score