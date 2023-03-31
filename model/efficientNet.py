import torch.nn as nn
import torchvision.models as models
import torch

class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_b1(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, input):
        x = self.model(input)
        return x
    