import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx
import torch
from classification import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from classification.dataset.dataloader import *
from classification.models.model import *
from utils import *
from classification.dataset.augmentations import get_test_transform
import cv2
import warnings
from classification import *
import onnx
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
from classification.torch_to_mxnet.utils import load_net_params_mxnet, torch_config
from gluoncv.model_zoo import resnet18_v1
from mxnet import nd
import glob
from classification.utils.confusion_matrix import plot_confusion_matrix


if __name__ == '__main__':
    #-------------------------torch model to onnx---------------------------
    input_shape = torch_config.multicore.input_size
    model_onnx_path = "/home/imdl/workspace/pytorch-image-classification/classification/mxmodel/torch_model_multicore.onnx"
    model = torch_config.multicore.model
    model.train(False)
    dummy_input = Variable(torch.randn(1, *input_shape)).cuda()
    output = torch_onnx.export(model,
                              dummy_input,
                              model_onnx_path,
                              verbose=False)
    print("Export of torch_model.onnx complete!")

    #---------------------------onnx to mx model-----------------------
    mx_model_path = '/home/imdl/workspace/pytorch-image-classification/classification/mxmodel/mx_model_multicore.params'
    net = resnet18_v1(classes=2)
    net.initialize(ctx=mx.gpu(0))
    # save params
    net = load_net_params_mxnet(net, model_onnx_path)
    net.save_parameters(mx_model_path)
    # net.load_parameters(mx_model_path)
    print('finish')

    # ------------------------------test mxnet model acc-----------------------
    # ims = glob.glob('/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/test/*/*.png')
    ims = glob.glob('/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/multicore/test/*/*.png')
    # ims = glob.glob('/home/imdl/DataSets/vol2/TCT/PATCH/853-512/inf/TP/classification/single-core/test/*/*.png')


    y_t, y_p = [[], []]
    ctx = mx.gpu(0)
    for im in tqdm(ims):
        img_bgr = cv2.imread(im)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = get_test_transform_mxnet(input_shape[1:])(image=img_rgb)["image"]
        img = nd.array([img], ctx=mx.gpu()).astype('float32')
        img = nd.transpose(img, (0, 3, 1, 2))
        test_data = mx.gluon.data.DataLoader(img, batch_size=1)
        for data1 in test_data:
            data1 = data1.as_in_context(ctx)
            output = nd.softmax(net(data1))
            output = output.argmax(axis=1)
            label = int(output.asnumpy()[0])
            y_t.append(int(im.split('/')[-2]))
            y_p.append(label)
    print('test imgs num: %d, POS num: %d,    acc= %.3f' %
          (len(y_t), sum(np.array(y_t)), sum((1 - np.array(y_p)) ^ np.array(y_t)) / len(y_t)))
    plot_confusion_matrix(y_t, y_p, 2, figsize=(5, 5))



