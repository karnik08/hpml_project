'''
params.py

Managers of all hyper-parameters

'''

import torch

# choose the value according the the hyper parameters you want to use. 
hyper_set=8 


#constants
epochs = 500
soft_label = False
adv_weight = 0
d_thresh = 0.8
z_dim = 200
beta = (0.5, 0.999)
cube_len = 32
leak_value = 0.2
bias = False
model_save_step = 1


if hyper_set==1:
    batch_size = 100
    g_lr = 0.0025
    d_lr = 0.00001
    z_dis = "norm"
elif hyper_set==2:
    batch_size = 32
    g_lr = 0.0020
    d_lr = 0.000005
    z_dis = "norm"
elif hyper_set==3 or hyper_set==4:
    batch_size = 100
    g_lr = 0.0025
    d_lr = 0.00001
    z_dis = "norm"
elif hyper_set==5:
    batch_size = 32
    g_lr = 0.0025
    d_lr = 0.00001
    z_dis = "uni"
elif hyper_set==6 or hyper_set==7 or hyper_set==8:
    batch_size = 256
#     batch_size = 32
    g_lr = 0.0025
    d_lr = 0.00001
    z_dis = "norm"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = '../volumetric_data/'
model_dir = 'chair/'  # change it to train on other data models
# model_dir = 'desk/'
# output_dir = '../outputs_uni'
# output_dir = '../outputs_LReLU'
# output_dir = '../outputs_3dgan'
# output_dir = '../outputs_lr'
# output_dir = '../outputs_tanh'
# output_dir = '../outputs_loss'
# output_dir = '../outputs'
output_dir = '../outputs_256bs'
# output_dir = '../outputs_real'
# output_dir = '../outputs_desk'
# images_dir = '../test_outputs'


def print_params():
    l = 16
    print(l * '*' + 'hyper-parameters' + l * '*')

    print('epochs =', epochs)
    print('batch_size =', batch_size)
    print('soft_labels =', soft_label)
    print('adv_weight =', adv_weight)
    print('d_thresh =', d_thresh)
    print('z_dim =', z_dim)
    print('z_dis =', z_dis)
    print('model_images_save_step =', model_save_step)
    print('data =', model_dir)
    print('device =', device)
    print('g_lr =', g_lr)
    print('d_lr =', d_lr)
    print('cube_len =', cube_len)
    print('leak_value =', leak_value)
    print('bias =', bias)

    print(l * '*' + 'hyper-parameters' + l * '*')
