import torch.nn as nn
import torchvision
from torch.autograd import Function
import torch
import torch.nn.functional as F
from torchvision import models

"""
    D:fc(1024)->relu->dr->fc->re->dr->fc(2) 0.001-0.001-0.3
    F:fc(128)->relu->dr->fc->re->dr->fc(31) Sequential
"""


class ReverseLayerF(Function):
    r"""Gradient Reverse Layer(Unsupervised Domain Adaptation by Backpropagation)
    Definition: During the forward propagation, GRL acts as an identity transform. During the back propagation though,
    GRL takes the gradient from the subsequent level, multiplies it by -alpha  and pass it to the preceding layer.

    Args:
        x (Tensor): the input tensor
        alpha (float): \alpha =  \frac{2}{1+\exp^{-\gamma \cdot p}}-1 (\gamma =10)
        out (Tensor): the same output tensor as x
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SilenceLayer(torch.autograd.Function):
    def __init__(self):
        pass

    def forward(self, input):
        return input * 1.0

    def backward(self, gradOutput):
        return 0 * gradOutput

class LeNetBase(nn.Module):
    """Long usps2mnist
    """
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(50*4*4, 500),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.__in_features = 500

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)

        return x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        # 暂定学习率都一样
        return [{'params': self.conv_params.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                {'params': self.fc_params.parameters(), 'lr_mult': 1, 'decay_mult': 2}]


class LeNetBase2(nn.Module):
    """Maximize usps2mnist"""
    def __init__(self):
        super(LeNetBase2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.__in_features = 48*4*4

    def forward(self, x):
        x = torch.mean(x, 1).view(x.size()[0], 1, x.size()[2], x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = x.view(x.size(0), 48*4*4)
        return x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        # 暂定学习率都一样
        return [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]


class DTN(nn.Module):
    """Long svhn2mnist
    """
    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.__in_features = 512

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)

        return x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        # 暂定学习率都一样
        parameter_list = [{'params': self.conv_params.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                          {'params': self.fc_params.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return parameter_list


class AlexBase(nn.Module):
    """Long alex base"""
    def __init__(self, use_bottleneck=True, bottleneck_dim=256):
        super(AlexBase, self).__init__()
        model_alex = torchvision.models.alexnet(pretrained=True)
        self.features = model_alex.features
        self.classifier = nn.Sequential()
        for i in range(6):
            # dropout -> fc -> relu -> dropout -> fc -> relu
            self.classifier.add_module('classifier' + str(i), model_alex.classifier[i])
        self.use_bottleneck = use_bottleneck

        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
            self.bottleneck.weight.data.normal_(0, 0.005)
            self.bottleneck.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.__in_features = 4096

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        return x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        parameter_list = [{'params': self.features.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                          {'params': self.classifier.parameters(), 'lr_mult': 10, 'decay_mult': 2},
                          {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
        return parameter_list


class Domain(nn.Module):
    r"""domain classifier
    connect to tne feature extractor via a gradient reverse layer that multiplies by
    a certain negative constant during the backpropagation-based training

    Distinguish the features as a source or target (minimize domain loss)

    Args:
        in_features: size of input layer unit, default: 256
        hidden_size: size of hidden layer unit, default: 1024
    """

    def __init__(self, in_features=256, hidden_size=1024):
        super(Domain, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.__in_features = 2
        self.init_weight()

    def forward(self, x, alpha):
        r"""flip all the samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :param alpha: ratio
        :return: the domain label prediction(2 dimension use CrossEntropyLoss)
        """
        x = ReverseLayerF.apply(x, alpha)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

    def output_num(self):
        return self.__in_features


class DomainLong(nn.Module):
    r"""Long domain classifier
        connect to tne feature extractor via a gradient reverse layer that multiplies by
        a certain negative constant during the backpropagation-based training

        Distinguish the features as a source or target (minimize domain loss)

        Args:
            in_features: size of input layer unit, default: 256
            hidden_size: size of hidden layer unit, default: 1024
        """

    def __init__(self, in_features=256, hidden_size=1024):
        super(DomainLong, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.__in_features = 1
        self.init_weight()

    def forward(self, x, alpha):
        r"""flip all the samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :param alpha: ratio
        :return: the domain label prediction(1 dimension and use BCEloss)
        """
        x = ReverseLayerF.apply(x, alpha)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

    def output_num(self):
        return self.__in_features


class Classifier(nn.Module):
    r"""Task-specific Classifier & label predictor
    utilize the task-specific classifier as discriminators that try to detect target samples
    that are far from the support of the source.

    1. predict class labels (source, minimize prediction loss)
    2. align distributions (maximize discrepancy loss)

    Args:
        in_features: size of input layer unit, default: 256
        hidden_size: size of hidden layer unit, default: 128
        class_num: number of categories, default: 31(office-31)
    """

    def __init__(self, in_features=256, hidden_size=128, class_num=31):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, class_num)
        )
        # self.fc1 = nn.Linear(in_features, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, class_num)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        # self.init_weight()

    def forward(self, x, alpha):
        r"""flip the target samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :param alpha: ratio
        :return: the class label prediction
        """
        xs = x[:x.size(0) // 2, :]
        xt = x[x.size(0) // 2:, :]
        xt = ReverseLayerF.apply(xt, alpha)
        x = torch.cat((xs, xt), 0)
        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        x = self.classifier(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]


class ResBase(nn.Module):
    r"""Constructs a feature extractor based on ResNet-50 model.
    remove the last layer, replace with bottleneck layer(out_dim=256)

    1. extract the discriminative feature (minimize the label prediction loss)
    2. extract the domain-invariance feature to confuse the domain classifier (maximize domain loss)
    3. learn to generate target features near the support to fool the classifiers (minimize discrepancy loss)
    """

    def __init__(self, option="resnet50", use_bottleneck=False):
        super(ResBase, self).__init__()
        self.use_bottleneck = use_bottleneck
        if option == "resnet50":
            model_res = torchvision.models.resnet50(pretrained=True)
        elif option == "resnet101":
            model_res = torchvision.models.resnet101(pretrained=True)
        self.conv1 = model_res.conv1
        self.bn1 = model_res.bn1
        self.relu = model_res.relu
        self.maxpool = model_res.maxpool
        self.layer1 = model_res.layer1
        self.layer2 = model_res.layer2
        self.layer3 = model_res.layer3
        self.layer4 = model_res.layer4
        self.avgpool = model_res.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.__in_features = 2048

        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_res.fc.in_features, 256)
            # init bottleneck parameters
            nn.init.normal_(self.bottleneck.weight.data, 0, 0.005)
            nn.init.constant_(self.bottleneck.bias.data, 0.1)
            self.__in_features = 256

    def forward(self, x):
        """
        :param x: the input Tensor as [bs, 3, 224, 224]
        :return: 256-dim feature
        """
        feature = self.feature_layers(x)
        feature = feature.view(feature.size(0), -1)
        if self.use_bottleneck:
            feature = self.bottleneck(feature)
        return feature

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        if self.use_bottleneck:
            parameter_list = [{'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                              {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
        else:
            parameter_list = [{'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2}]

        return parameter_list

    def output_num(self):
        return self.__in_features

resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        # 特征层 feature
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)   # batchsize * channel * width * height -> batchsize * columns
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

class ClassifierMCD(nn.Module):
    r"""Task-specific Classifier & label predictor
    utilize the task-specific classifier as discriminators that try to detect target samples
    that are far from the support of the source.

    1. predict class labels (source, minimize prediction loss)
    2. align distributions (maximize discrepancy loss)

    Args:
        in_features: size of input layer unit, default: 256
        hidden_size: size of hidden layer unit, default: 128
        class_num: number of categories, default: 31(office-31)
    """

    def __init__(self, in_features=2048, hidden_size=1000, class_num=31):
        super(ClassifierMCD, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, affine=True),
            nn.ReLU(True),
            nn.Linear(hidden_size, class_num)
        )
        # self.init_weight()

    def forward(self, x, reverse=False, alpha=1.0):
        r"""flip the target samples' sign of gradients when back-propagation
        :param reverse: reverse gradient or not
        :param x: the input Tensor as [bs, features_dim]
        :param alpha: ratio
        :return: the class label prediction
        """
        if reverse:
            x = ReverseLayerF.apply(x, alpha)
        x = self.classifier(x)
        return x

    def init_weight(self):
        class_name = self.__class__.__name__
        if class_name.find('Conv') != -1:
            self.weight.data.normal_(0.0, 0.01)
            self.bias.data.normal_(0.0, 0.01)
        elif class_name.find('BatchNorm') != -1:
            self.weight.data.normal_(1.0, 0.01)
            self.bias.data.fill_(0)
        elif class_name.find('Linear') != -1:
            self.weight.data.normal_(0.0, 0.01)
            self.bias.data.normal_(0.0, 0.01)


    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]


class DomainIWAN(nn.Module):
    r"""IWAN domain classifier
        connect to tne feature extractor via a gradient reverse layer that multiplies by
        a certain negative constant during the backpropagation-based training

        Distinguish the features as a source or target (minimize domain loss)

        Args:
            in_features: size of input layer unit, default: 256
            hidden_size: size of hidden layer unit, default: 1024
        """

    def __init__(self, in_features=256, hidden_size=1024):
        super(DomainIWAN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.__in_features = 1
        self.init_weight()

    def forward(self, x):
        r"""flip all the samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :return: the domain label prediction(1 dimension and use BCEloss)
        """
        x = ReverseLayerF.apply(x, 1.0)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

    def output_num(self):
        return self.__in_features

class DomainWasserstein(nn.Module):

    def __init__(self, in_features=256, hidden_size=1024):
        super(DomainIWAN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.__in_features = 1
        self.init_weight()

    def forward(self, x):
        r"""flip all the samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :return: the domain label prediction(1 dimension and use BCEloss)
        """
        x = ReverseLayerF.apply(x, 1.0)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

    def output_num(self):
        return self.__in_features

class Svhn2MnistFeature(nn.Module):
    def __init__(self):
        super(Svhn2MnistFeature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        parameter_list = [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return parameter_list

class Svhn2MnistPredictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Svhn2MnistPredictor, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

class usps_feature(nn.Module):
    def __init__(self):
        super(usps_feature, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x,1).view(x.size()[0],1,x.size()[2],x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        #print(x.size())
        x = x.view(x.size(0), 48*4*4)
        return x
    def get_parameters(self):
        r"""get the different parameters of each layer"""
        parameter_list = [{'params': self.parameters(), 'lr_mult': 1, 'decay_mult': 2}]
        return parameter_list

class usps_predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(usps_predictor, self).__init__()
        self.fc1 = nn.Linear(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x, reverse=False):
        if reverse:
            x = ReverseLayerF(x, self.lambd)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]