import torch
import torch.nn as nn
import torchvision.models as models

from utils import device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyResNet(nn.Module):

    def __init__(self, args):
        super(MyResNet, self).__init__()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            self.resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            self.resnet = models.resnet50(pretrained=True)

    def forward(self, context_images):

        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        #comment by guofei这里思考是否可以选择restnet多层的特征，从而而已从不同感受野来提取特征
        #用resnet初步提取特征，context_features 40，3，112，112--->40,512,4,4 ;  target_images 200,3,112,112-> 200, 512, 4,4
        #patch一共就是4*4=16个
        ###########start############################3
        x = self.resnet.conv1(context_images)
        x = self.resnet.bn1(x)
        x_= self.resnet.relu(x)
        x = self.resnet.maxpool(x_)
        p1 = self.resnet.layer1(x)
        p2 = self.resnet.layer2(p1)
        p3 = self.resnet.layer3(p2)
        p4 = self.resnet.layer4(p3)
        out = [x_, p1, p2, p3, p4]
        return out
