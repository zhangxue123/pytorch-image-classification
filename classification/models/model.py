import torchvision
import torch.nn.functional as F 
from torch import nn, load
from classification.utils.config import config
from classification import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from classification.dataset import get_test_transform
import numpy as np


def generate_model():
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.classifier = nn.Linear(pretrained_model.classifier.in_features, config.num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

            self.features = pretrained_model.features
            self.layer1 = pretrained_model.features._modules['denseblock1']
            self.layer2 = pretrained_model.features._modules['denseblock2']
            self.layer3 = pretrained_model.features._modules['denseblock3']
            self.layer4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            print(out.size())
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.sigmoid(self.classifier(out))
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))


from simpledet.zxjob.Utils.GeM import GeneralizedMeanPooling, GeneralizedMeanPoolingP
def get_net(model_name, pooling='avg'):
    if pooling == 'max':
        adpool = nn.AdaptiveMaxPool2d(output_size=1)
    elif pooling == 'avg':
        adpool = nn.AdaptiveAvgPool2d(output_size=1)
    elif pooling == 'gem':
        adpool = GeneralizedMeanPoolingP(norm=3)
    else:
        raise ValueError(pooling)
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.avgpool = adpool
        model.fc = nn.Linear(512, config.num_classes)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.avgpool = adpool
        model.fc = nn.Linear(2048, config.num_classes)
    elif model_name == 'resnet101':
        # return MyModel(torchvision.models.resnet101(pretrained = True))
        model = torchvision.models.resnet101(pretrained = True)
        model.avgpool = adpool
        model.fc = nn.Linear(2048, config.num_classes)
    elif model_name == 'resnet152':
        model = torchvision.models.resnet152(pretrained = True)
        #for param in model.parameters():
        #    param.requires_grad = False
        model.avgpool = adpool
        model.fc = nn.Linear(2048,config.num_classes)
        # model.fc = nn.Linear(512,config.num_classes)
    elif model_name == 'deeplab':
        from classification.models.deeplab.deeplabv3 import DeepLabV3
        model = DeepLabV3(num_classes=config.num_classes)
    elif model_name == 'res2net50':
        from classification.models.res2net_v1b import res2net50_v1b
        model = res2net50_v1b(pretrained=True)
        model.avgpool = adpool
        model.fc = nn.Linear(2048, config.num_classes)
    return model


def load_inference_model_classification(path, model_name='resnet18'):
    model = get_net(model_name)
    model.cuda()
    best_model = load(path)
    model.load_state_dict(best_model["state_dict"])
    model.eval()
    return model


def inference_single_img_classification(model, img, input_size=(config.img_height, config.img_weight)):
    img = get_test_transform(input_size)(image=img)["image"]
    test_loader = DataLoader([img], batch_size=1, shuffle=True, pin_memory=False)
    for i, input in enumerate(test_loader):
        image_var = Variable(input).cuda()
        y_pred = model(image_var)
        smax = nn.Softmax(1)
        smax_out = smax(y_pred)
        label = np.argmax(smax_out.cpu().data.numpy())
    return label