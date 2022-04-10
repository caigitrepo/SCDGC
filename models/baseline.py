import torch
import torch.nn as nn
from torchvision import models

class Baseline(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        model = models.resnet101(pretrained = True)
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.AdaptiveMaxPool2d((1,1))
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[7].parameters():
            param.requires_grad = True
        
        self.classifier = nn.Linear(2048, 80)
    
    def train_mode(self):
        self.backbone.eval()
        self.backbone[7].train()
    

    def forward(self, x):
        batch_size = x.size()[0]
        img_feature_map = self.backbone(x)
        feature = img_feature_map.squeeze(3).squeeze(2)
        output = self.classifier(feature)
        output = torch.sigmoid(output)
        return output, output
