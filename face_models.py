'''
This file load pretrained models which are trained on MS1 datastet with Arcface loss function.
These pretrained models are downloaded from insightface github repository
'''

from backbones import get_model
import torch
from torchsummary import summary
import torch.nn as nn




class my_resnet(nn.Module):
    def __init__(self, resnet_reduced):
        super(my_resnet, self).__init__()
        # super().__init__()
        self.resnet_reduced = resnet_reduced
        self.my_layers = nn.Sequential(nn.Dropout(0.2),
                                       nn.Flatten(),
                                       nn.Linear(25088, 10572),
                                       )
    def forward(self, x):
        x = self.resnet_reduced(x)
        x = self.my_layers(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


resnet50 = get_model("r50", dropout=0.0, fp16=True, num_features=512).cuda()
resnet50.load_state_dict(torch.load('./pretrained_models/resnet50.pth'))
resnet50_reduced = torch.nn.Sequential(*(list(resnet50.children())[:-3]))
resnet50_customized = my_resnet(resnet_reduced=resnet50_reduced)
resnet50_customized.resnet_reduced.requires_grad_(False)
resnet50_customized = resnet50_customized.to(device)


resnet34 = get_model("r34", dropout=0.0, fp16=True, num_features=512).cuda()
resnet34.load_state_dict(torch.load('./pretrained_models/resnet34.pth'))
resnet34_reduced = torch.nn.Sequential(*(list(resnet34.children())[:-3]))
resnet34_customized = my_resnet(resnet_reduced=resnet34_reduced)
resnet34_customized.resnet_reduced.requires_grad_(False)
resnet34_customized = resnet34_customized.to(device)