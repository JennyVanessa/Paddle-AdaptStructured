import paddle.nn as nn
import math
import paddle
import numpy as np
from init_weights import init_weights


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, stride=stride) # change
        self.bn1 = nn.BatchNorm2D(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding,dilation = dilation)
        self.bn2 = nn.BatchNorm2D(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Layer):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.LayerList()
        print("conv2d_list:  ",self.conv2d_list)
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2D(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation))

        # for m in self.conv2d_list:
            # print("m:  ",m)
            # m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out



class ResNet(nn.Layer):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2D(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.01)
        #     elif isinstance(m, nn.BatchNorm):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

          # for m in self.modules():
        for m in self.sublayers():
            print(m)
            if isinstance(m, nn.Conv2D):
                # 该接口实现Kaiming正态分布方式的权重初始化
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print(50 * '*')
                m.weight_attr = init_weights(init_type='kaiming')
                m.bias_attr = init_weights(init_type='kaiming')
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                # nn.initializer.Constant(m.weight, 1)
                # m.weight = nn.initializer.Constant(1)
                # nn.initializer.Constant(m.bias, 0)
                # m.bias = nn.initializer.Constant(0)
                m.param_attr = init_weights(init_type='constant')
                m.bias_attr = init_weights(init_type='constant')


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2D(planes * block.expansion))
        for i in downsample['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

    
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i
            


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}] 


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model


if __name__ == "__main__":
   r=Res_Deeplab()
   print(r)
