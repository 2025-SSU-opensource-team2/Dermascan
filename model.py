import torchvision.models as models
import torch.nn as nn

def get_resnet34(num_classes):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
