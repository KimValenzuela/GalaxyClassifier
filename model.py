import torch
import torch.nn as nn
from torchvision import models

class ModelGalaxyClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Adaptar primera capa a 1 canal
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Inicializar pesos promediando RGB
        with torch.no_grad():
            model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                model.conv1.bias[:] = old_conv.bias
        
        # Congelar model ANTES de modificar fc
        for name, param in model.named_parameters():
            if not name.startswith('fc'):  # No congelar fc (será reemplazado)
                param.requires_grad = False
        
        # Clasificador final
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
        # Descongelar conv1 para ajuste fino
        for param in model.conv1.parameters():
            param.requires_grad = True
        
        self.model = model
    
    def forward(self, x):
        return self.model(x)



def get_resnet50(num_classes=5, in_channels=1):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if in_channels == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet34(num_classes=5, in_channels=1):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    if in_channels == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet152(num_classes=5, in_channels=1):
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

    if in_channels == 1:
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
