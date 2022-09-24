from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import tensorflow.compat.v1 as tf
from disentanglement_lib.evaluation.metrics.hsic import HSIC
# sess = tf.InteractiveSession()
tf.enable_eager_execution()

x = tf.random.normal([200,5])
y = tf.random.normal([200,50])




print('dimension unbased scaling')
print('xy HSIC = ', HSIC(x,y,s_x = 1000, s_y=1000).numpy())
print('xy HSIC = ', HSIC(x,y,s_x = 100, s_y=100).numpy())
print('xy HSIC = ', HSIC(x,y,s_x = 10, s_y=10).numpy())
# xy HSIC =  tensor(2.9368e-05, device='cuda:0')

print('dimension based scaling')
print('xy HSIC = ', HSIC(x,y,s_x = 100, s_y=1000).numpy())
print('xy HSIC = ', HSIC(x,y,s_x = 10, s_y=100).numpy())
print('xy HSIC = ', HSIC(x,y,s_x = 1, s_y=10).numpy())

z = x 
#check dep different dimension scaling sigma
print('dep')
print('xx HSIC = ', HSIC(x,z,s_x = 1000, s_y=1000).numpy())
print('xx HSIC = ', HSIC(x,z,s_x = 100, s_y=100).numpy())
print('xx HSIC = ', HSIC(x,z,s_x = 10, s_y=10).numpy())
# xx HSIC =  tensor(0.0018, device='cuda:0')
# xx HSIC =  tensor(0.0063, device='cuda:0')
# xx HSIC =  tensor(0.0012, device='cuda:0')
# xx HSIC =  tensor(0.0005, device='cuda:0')


# # Test batchwise HSIC
# x = tf.random.uniform(100,200,5)
# y = tf.random.uniform(100,200,5)
# print('dimension unbased scaling')
# print('xy HSIC = ', HSIC(x,y,s_x = 1000, s_y=1000))
# xy HSIC =  tensor([4.5136e-06, 4.8490e-06, 4.4989e-06, 4.6712e-06, 5.6281e-06, 5.1141e-06,
#         4.7008e-06, 4.6331e-06, 5.7584e-06, 5.3174e-06, 4.8064e-06, 3.7694e-06,
#         4.7421e-06, 4.3642e-06, 4.9041e-06, 5.0329e-06, 5.6243e-06, 5.2292e-06,
#         4.3169e-06, 4.2757e-06, 4.9875e-06, 5.0479e-06, 4.9199e-06, 5.4121e-06,
#         4.5346e-06, 5.1154e-06, 4.5003e-06, 4.6267e-06, 4.4921e-06, 4.6903e-06,
#         5.9266e-06, 4.2881e-06, 4.5944e-06, 4.8834e-06, 4.0744e-06, 4.8113e-06,
#         4.4940e-06, 4.9660e-06, 4.5582e-06, 5.0674e-06, 5.0693e-06, 4.8605e-06,
#         5.0480e-06, 4.4115e-06, 3.9799e-06, 5.3122e-06, 5.2128e-06, 4.3391e-06,
#         3.7135e-06, 5.0158e-06, 4.8219e-06, 4.8755e-06, 4.8974e-06, 5.8190e-06,
#         4.3698e-06, 5.3434e-06, 4.7514e-06, 5.0550e-06, 4.6217e-06, 4.6923e-06,
#         5.2878e-06, 4.1746e-06, 5.1770e-06, 4.1578e-06, 5.2196e-06, 4.4700e-06,
#         4.8828e-06, 5.4164e-06, 4.0633e-06, 5.5128e-06, 4.1357e-06, 5.5183e-06,
#         4.5766e-06, 4.5919e-06, 4.7875e-06, 5.3946e-06, 4.9806e-06, 4.7055e-06,
#         4.5704e-06, 4.3785e-06, 4.6322e-06, 3.8181e-06, 4.8668e-06, 4.0748e-06,
#         4.6744e-06, 4.3339e-06, 5.1452e-06, 3.9519e-06, 4.4084e-06, 4.7304e-06,
#         4.3480e-06, 4.8946e-06, 5.0657e-06, 5.4453e-06, 4.7079e-06, 5.2662e-06,
#         4.0309e-06, 4.7707e-06, 4.6514e-06, 4.3616e-06], device='cuda:0')

# print('xy HSIC = ', HSIC_(x,y,s_x = 100, s_y=100))
# xy HSIC =  tensor([0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0002, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
#         0.0003], device='cuda:0')

# print('xy HSIC = ', HSIC_(x,y,s_x = 10, s_y=10))
# xy HSIC =  tensor([0.0028, 0.0029, 0.0029, 0.0028, 0.0029, 0.0029, 0.0028, 0.0029, 0.0029,
#         0.0028, 0.0029, 0.0026, 0.0029, 0.0028, 0.0029, 0.0027, 0.0029, 0.0029,
#         0.0029, 0.0028, 0.0029, 0.0029, 0.0029, 0.0027, 0.0029, 0.0029, 0.0029,
#         0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0029,
#         0.0027, 0.0029, 0.0028, 0.0028, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028,
#         0.0029, 0.0028, 0.0029, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0029,
#         0.0028, 0.0029, 0.0030, 0.0028, 0.0028, 0.0030, 0.0030, 0.0030, 0.0030,
#         0.0028, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0029, 0.0028, 0.0029,
#         0.0028, 0.0028, 0.0029, 0.0029, 0.0029, 0.0028, 0.0029, 0.0028, 0.0029,
#         0.0029, 0.0029, 0.0027, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028, 0.0028,
#         0.0028, 0.0028, 0.0029, 0.0029, 0.0029, 0.0030, 0.0028, 0.0028, 0.0028,
#         0.0029], device='cuda:0')

# print('xx HSIC = ', HSIC_(x,x,s_x = 1000, s_y=1000))
# print('xx HSIC = ', HSIC_(x,x,s_x = 100, s_y=100))
# print('xx HSIC = ', HSIC_(x,x,s_x = 10, s_y=10))

# xy HSIC =  tensor(4.7645e-06, device='cuda:0')
# xy HSIC =  tensor(0.0003, device='cuda:0')
# xy HSIC =  tensor(0.0028, device='cuda:0')
# xx HSIC =  tensor(1.9914e-05, device='cuda:0')
# xx HSIC =  tensor(0.0016, device='cuda:0')
# xx HSIC =  tensor(0.0260, device='cuda:0')