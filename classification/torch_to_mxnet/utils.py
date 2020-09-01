import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# import torch.onnx as torch_onnx
# import torch
# from classification import *
#
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from classification.dataset.dataloader import *
# from classification.models.model import *
# from utils import *
# from classification.dataset.augmentations import get_test_transform
# import cv2
# import warnings
# from classification import *
from classification.models import load_inference_model_classification
# import onnx
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
# import numpy as np


class torch_config:
    class singlecore:
        root = '/home/imdl/workspace/pytorch-image-classification/checkpoints/best_model-'
        model_path = root + "single-core/resnet18/200/model_best-201904_07.pth.tar"
        model = load_inference_model_classification(model_path)
        input_size = (3, 200, 200)
    class multicore:
        root = '/home/imdl/workspace/pytorch-image-classification/checkpoints/best_model-'
        model_path = root + "multicore/resnet18/500/model_best-201904_07.pth.tar"
        model = load_inference_model_classification(model_path)
        input_size = (3, 500, 500)



def load_net_params_mxnet(net, params_path):
    # Import the ONNX model into MXNet's symbolic interface
    sym, arg_params, aux_params = onnx_mxnet.import_model(params_path)
    print("Loaded torch_model.onnx!")
    net_params = net.collect_params()
    ctx = mx.gpu(0)
    for param in aux_params:
        temp = param
        param = 'resnetv10_' + param.replace('.', '_').replace('bias', 'beta').replace('bn', 'batchnorm')  # .replace()
        if 'batchnorm' in param:
            param = param.replace('batchnorm' + param.split('batchnorm')[1][0],
                                  'batchnorm' + str(int(param.split('batchnorm')[1][0]) - 1))
        if 'layer' in param:
            param = param.replace('layer', 'stage')
            i, j = map(int, param.split('stage')[1].split('_')[:2])
            if i < 2 and j == 1: t = 2
            if i >= 2 and j == 1: t = 3
            if j == 1:
                if 'batchnorm' in param:
                    param = param.replace('batchnorm' + param.split('batchnorm')[1][0],
                                          'batchnorm' + str(int(param.split('batchnorm')[1][0]) + t))
            if 'downsample_1' in param:
                param = param.replace('downsample_1', 'batchnorm2')
            param = '_'.join(param.split('_%d_' % j))
        net_params[param]._load_init(aux_params[temp], ctx=ctx)
    for param in arg_params:
        temp = param
        param = 'resnetv10_' + param.replace('.', '_').replace('bias', 'beta').replace('bn', 'batchnorm')#.replace()
        if 'conv' in param:
            param = param.replace('conv' + param.split('conv')[1][0], 'conv' + str(int(param.split('conv')[1][0])-1))
        if 'batchnorm' in param:
            param = param.replace('batchnorm' + param.split('batchnorm')[1][0], 'batchnorm' + str(int(param.split('batchnorm')[1][0]) - 1))
            param = param.replace('weight', 'gamma')
        if 'layer' in param:
            param = param.replace('layer', 'stage')
            i, j = map(int, param.split('stage')[1].split('_')[:2])
            if i < 2 and j == 1: t = 2
            if i >= 2 and j == 1: t = 3
            if j == 1:
                if 'conv' in param:
                    param = param.replace('conv' + param.split('conv')[1][0], 'conv' + str(int(param.split('conv')[1][0]) + t))
                if 'batchnorm' in param:
                    param = param.replace('batchnorm' + param.split('batchnorm')[1][0],
                                          'batchnorm' + str(int(param.split('batchnorm')[1][0]) + t))
                    param = param.replace('weight', 'gamma')
            if 'downsample' in param:
                if 'downsample_0' in param: param = param.replace('downsample_0', 'conv2')
                if 'downsample_1' in param:
                    if 'weight' in param: param = param.replace('downsample_1_weight', 'batchnorm2_gamma')
                    if 'beta' in param: param = param.replace('downsample_1_beta', 'batchnorm2_beta')
            param = '_'.join(param.split('_%d_' % j))
        if 'fc' in param:
            param = param.replace('fc_weight', 'dense0_weight').replace('fc_beta', 'dense0_bias')
        net_params[param]._load_init(arg_params[temp], ctx=ctx)
    net.hybridize()
    return net





