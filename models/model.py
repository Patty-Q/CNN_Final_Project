import torchvision
import torch.nn.functional as F 
from torch import nn
from config import config

def get_net():
    model = torchvision.models.resnet50(pretrained = True) # Based on ResNet50 pre-trained model.
    model.avgpool = nn.AdaptiveAvgPool2d(1) # The average pooling layer and fully connected layer of the model are modified.。
    model.fc = nn.Linear(2048,config.num_classes) # Suitable for classification tasks that require efficient feature extraction.。
    return model

