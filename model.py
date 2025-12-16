import torch
import torch.nn as nn
from torchvision import models

def get_resnet50_improved(num_classes=5, in_channels=1, dropout=0.3):
    """
    ResNet50 mejorado con:
    - Adaptación correcta de pesos RGB → Grayscale
    - Congelamiento progresivo de capas
    - Clasificador mejorado con dropout
    """
    model = models.resnet50(weights="IMAGENET1K_V2")
    
    # Adapta la primera capa promediando pesos RGB → Grayscale
    if in_channels == 1:
        pretrained_weights = model.conv1.weight.data  # (64, 3, 7, 7)
        avg_weights = pretrained_weights.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.conv1.weight.data = avg_weights
    
    # Congela las primeras capas (features básicas de bordes, texturas)
    for name, param in model.named_parameters():
        if "layer1" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False
    
    # Clasificador mejorado con regularización
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )
    
    return model


def get_efficientnet_b0(num_classes=5, in_channels=1, dropout=0.3):
    """
    EfficientNet-B0: Más eficiente para imágenes pequeñas
    Mejor relación precisión/parámetros que ResNet
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Adapta primera capa
    if in_channels == 1:
        pretrained = model.features[0][0].weight.data  # (32, 3, 3, 3)
        avg_weights = pretrained.mean(dim=1, keepdim=True)  # (32, 1, 3, 3)
        
        model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.features[0][0].weight.data = avg_weights
    
    # Clasificador mejorado
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
    
    return model


def get_resnet34_improved(num_classes=5, in_channels=1, dropout=0.3):
    """
    ResNet34: Más ligero que ResNet50, menos propenso a overfitting
    """
    model = models.resnet34(weights="IMAGENET1K_V1")
    
    if in_channels == 1:
        pretrained_weights = model.conv1.weight.data
        avg_weights = pretrained_weights.mean(dim=1, keepdim=True)
        
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.conv1.weight.data = avg_weights
    
    # Congela layer1
    for name, param in model.named_parameters():
        if "layer1" in name or "conv1" in name or "bn1" in name:
            param.requires_grad = False
    
    # Clasificador
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
    
    return model


def unfreeze_model(model, unfreeze_from_layer=None):
    """
    Descongela capas del modelo para fine-tuning progresivo
    
    Args:
        model: modelo PyTorch
        unfreeze_from_layer: nombre de la capa desde donde descongelar
                            None = descongela todo
    """
    if unfreeze_from_layer is None:
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze = False
        for name, param in model.named_parameters():
            if unfreeze_from_layer in name:
                unfreeze = True
            if unfreeze:
                param.requires_grad = True
    
    return model


# Funciones auxiliares para crear modelos
def create_model(model_name, num_classes=5, in_channels=1, dropout=0.3):
    """
    Factory function para crear modelos
    
    Args:
        model_name: 'resnet50', 'resnet34', 'efficientnet_b0'
        num_classes: número de clases
        in_channels: canales de entrada (1 para grayscale)
        dropout: tasa de dropout
    """
    models_dict = {
        'resnet50': get_resnet50_improved,
        'resnet34': get_resnet34_improved,
        'efficientnet_b0': get_efficientnet_b0,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Modelo {model_name} no disponible. Usa: {list(models_dict.keys())}")
    
    return models_dict[model_name](num_classes, in_channels, dropout)