import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
import torch.nn.functional as F
import resnet
from resnet import conv3x3,BasicBlock,Bottleneck,ResNet

def myconv(in_channels,outchannels,INnorm=True,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    conv2d = nn.Conv2d(in_channels, outchannels, kernel_size=3, padding=d_rate, dilation=d_rate)
    if INnorm:
        layers += [conv2d, nn.InstanceNorm2d(outchannels), nn.ReLU(inplace=True)]
    elif batch_norm:
        layers += [conv2d, nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def mydeconv(in_channels,outchannels,INnorm=True,batch_norm=False):
    layers = []
    conv2d = nn.ConvTranspose2d(in_channels, outchannels, kernel_size=3, stride=2, padding=1, output_padding=1)
    if INnorm:
        layers += [conv2d, nn.InstanceNorm2d(outchannels), nn.ReLU(inplace=True)]
    elif batch_norm:
        layers += [conv2d, nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def my_make_layers(cfg, in_channels = 64,INnorm=True, batch_norm=True,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'T':
            transconv=nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
            if INnorm:
                layers += [transconv, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [transconv, nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        # self.FME=Xception()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512,512,512,256,128,64]
        # self.dilatedconv1 = myconv(512,256,INnorm=False,batch_norm=True,dilation=True)
        # self.deconv1=mydeconv(256,256,INnorm=False,batch_norm=True)
        # self.dilatedconv2 = myconv(256, 128, INnorm=False, batch_norm=True, dilation=True)
        # self.deconv2 = mydeconv(128, 128, INnorm=False, batch_norm=True)
        # self.dilatedconv3 = myconv(128, 64, INnorm=False, batch_norm=True, dilation=True)
        # self.deconv3 = mydeconv(64, 64, INnorm=False, batch_norm=True)
        # self.dilatedconv4 = myconv(64, 64, INnorm=False, batch_norm=True, dilation=True)
        # self.mydme_feat = [64, 'T', 64]
        self.frontend = make_layers(self.frontend_feat)
        # self.mydme = my_make_layers(self.mydme_feat,in_channels = 64)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in xrange(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
    def forward(self,x):
        # x = self.FME(x)
        x = self.frontend(x)
        x = self.backend(x)
        # x = self.mydme(x)
        # x = self.dilatedconv1(x)
        # x = self.deconv1(x)
        # x = self.dilatedconv2(x)
        # x = self.deconv2(x)
        # x = self.dilatedconv3(x)
        # x = self.deconv3(x)
        # x = self.dilatedconv4(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MCNN(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=False):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(3, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16, 8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d(3, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d(3, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fuse(x)

        return x

class model(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, relu=True, bn=False, IN= False):
        super(model, self).__init__()
        # padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 5, stride, padding=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 7, stride, padding=3)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.IN = nn.InstanceNorm2d(out_channels,affine=True) if IN else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.bn is not None:
            x1 = self.bn(x1)
            x2 = self.bn(x2)
            x3 = self.bn(x3)
            x4 = self.bn(x4)
        if self.IN is not None:
            x1 = self.IN(x1)
            x2 = self.IN(x2)
            x3 = self.IN(x3)
            x4 = self.IN(x4)
        if self.relu is not None:
            x1 = self.relu(x1)
            x2 = self.relu(x2)
            x3 = self.relu(x3)
            x4 = self.relu(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        return y

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,3,1,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,3,1,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,2,2,start_with_relu=True,grow_first=True)

        self.block12=Block(728,512,3,1,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(512,512,3,1,1)
        self.bn3 = nn.BatchNorm2d(512)

        #do relu here
        self.conv4 = SeparableConv2d(512,512,3,1,1)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        # x = self.block8(x)
        # x = self.block9(x)
        # x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        # x = self.logits(x)
        return x

class SANet(nn.Module):
    def __init__(self):
        super(SANet, self).__init__()
        self.FME=Xception()
        # self.FME = nn.Sequential(model(3, 16, bn=False, IN=True),
        #                        nn.MaxPool2d(2),
        #                        model(64, 32, bn=False, IN=True),
        #                        nn.MaxPool2d(2),
        #                        model(128, 32, bn=False, IN=True),
        #                        nn.MaxPool2d(2),
        #                        model(128, 16, bn=False, IN=True))
        # self.FME = nn.Sequential(model(3, 16, bn=True, IN=False),
        #                          nn.MaxPool2d(2),
        #                          model(64, 32, bn=True, IN=False),
        #                          nn.MaxPool2d(2),
        #                          model(128, 32, bn=True, IN=False),
        #                          nn.MaxPool2d(2),
        #                          model(128, 16, bn=True, IN=False))
        self.DME = nn.Sequential(nn.Conv2d(64, 64, 9, 1, padding=4),
                                 nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(inplace=True),
                               nn.Conv2d(64, 32, 7, 1, padding=3),
                                 nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(inplace=True),
                               nn.Conv2d(32, 16, 5, 1, padding=2),
                                 nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(inplace=True),
                               nn.Conv2d(16, 16, 3, 1, padding=1),
                                 nn.ReLU(inplace=True),
                               nn.Conv2d(16, 16, 5, 1, padding=2),
                                 nn.ReLU(inplace=True),
                               # nn.Conv2d(16, 16, 3, 1, padding=2, dilation=2),
                               nn.Conv2d(16, 1, 1, 1, padding=0))
        # self.DME = nn.Sequential(nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 32, 7, 1, padding=3),
        #                          nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(32, 16, 5, 1, padding=2),
        #                          nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(16, 16, 3, 1, padding=1),
        #                          nn.Conv2d(16, 16, 5, 1, padding=2),
        #                          nn.Conv2d(16, 1, 1, 1, padding=0))
        # self.DME = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 64, 7, 1, padding=3),
        #                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 64, 5, 1, padding=2),
        #                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 64, 3, 1, padding=1),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.Conv2d(64, 16, 3, 1, padding=2, dilation=2),
        #                          nn.Conv2d(16, 16, 5, 1, padding=2),
        #                          nn.Conv2d(16, 1, 1, 1, padding=0))
        # self.DME = nn.Sequential(nn.Conv2d(64, 64, 9, 1, padding=4),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 64, 7, 1, padding=3),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 64, 5, 1, padding=2),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 64, 3, 1, padding=1),
        #                          nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
        #                          nn.Conv2d(64, 16, 5, 1, padding=2),
        #                          nn.Conv2d(16, 1, 1, 1, padding=0))
        # self.DME = nn.Sequential(nn.Conv2d(64, 64, 9, 1, padding=8, dilation=2),
        #                          nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(64, 32, 7, 1, padding=6, dilation=2),
        #                          nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(32, 16, 5, 1, padding=4, dilation=2),
        #                          nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                          nn.Conv2d(16, 16, 3, 1, padding=2, dilation=2),
        #                          nn.Conv2d(16, 16, 5, 1, padding=4, dilation=2),
        #                          nn.Conv2d(16, 1, 1, 1, padding=0, dilation=2))
    def forward(self, x):
        # x = self.layer1(x)
        # x = self.layer2(x)
        x = self.FME(x)
        x = self.DME(x)
        return x