import shutil
import torch
import os
from classification.utils.config import config
from classification import *
from torch.utils.data import DataLoader
from classification.models.model import get_net
from classification.dataset.augmentations import get_test_transform
from torch.autograd import Variable
from torch import nn
import numpy as np


def save_checkpoint(state, is_best,fold):
    filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = config.best_models + config.model_name+ os.sep +str(fold)  + os.sep + 'model_best.pth.tar'
        print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"],message))
        with open("./logs/%s.txt"%config.model_name,"a") as f:
            print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"],message),file=f)
        shutil.copyfile(filename, message)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = config.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def schedule(current_epoch, current_lrs, **logs):
        lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
        epochs = [0, 1, 6, 8, 12]
        for lr, epoch in zip(lrs, epochs):
            if current_epoch >= epoch:
                current_lrs[5] = lr
                if current_epoch >= 2:
                    current_lrs[4] = lr * 1
                    current_lrs[3] = lr * 1
                    current_lrs[2] = lr * 1
                    current_lrs[1] = lr * 1
                    current_lrs[0] = lr * 0.1
        return current_lrs

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError


def load_inference_model_classification(path, model_name=config.model_name):
    model = get_net(model_name)
    model.cuda()
    best_model = torch.load(path)
    model.load_state_dict(best_model["state_dict"])
    model.eval()
    return model


def inference(model, imgs_rgb):
    imgs = [get_test_transform((config.img_height, config.img_weight))(image=img_rgb)["image"] for img_rgb in imgs_rgb]
    test_loader = DataLoader(imgs, batch_size=len(imgs), shuffle=False, pin_memory=False)
    for i, input in enumerate(test_loader):
        image_var = Variable(input).cuda()
        y_pred = model(image_var)
        smax = nn.Softmax(1)
        smax_out = smax(y_pred).cpu().data.numpy()
        label = np.argmax(smax_out, axis=1)
        prob = np.max(smax_out, axis=1)
    return label, prob

