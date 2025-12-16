import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar desbalance extremo de clases.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    donde:
    - alpha_t: peso por clase (mayor para clases minoritarias)
    - gamma: factor de enfoque (típicamente 2.0)
    - p_t: probabilidad predicha de la clase correcta
    
    Ventajas:
    - Penaliza más los errores en clases minoritarias
    - Reduce el peso de ejemplos fáciles
    - Funciona bien con soft labels
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Tensor de pesos por clase (C,) o None
            gamma: Factor de enfoque (mayor = más peso a ejemplos difíciles)
            reduction: 'mean', 'sum', o 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) salida sin softmax del modelo
            targets: (B, C) soft labels (distribuciones de probabilidad)
        
        Returns:
            loss: escalar o tensor (B,) según reduction
        """
        # Probabilidades predichas
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Calcula p_t: probabilidad de la clase correcta
        # Para soft labels: p_t = sum(target_i * prob_i)
        p_t = (targets * probs).sum(dim=1, keepdim=True)  # (B, 1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma  # (B, 1)
        
        # Cross entropy: -sum(target_i * log(prob_i))
        ce = -(targets * log_probs)  # (B, C)
        
        # Aplica focal weight
        focal_loss = focal_weight * ce  # (B, C)
        
        # Aplica pesos por clase (alpha) si están definidos
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Calcula alpha efectivo por sample según la distribución target
            # alpha_t = sum(target_i * alpha_i)
            alpha_t = (targets * self.alpha.unsqueeze(0)).sum(dim=1, keepdim=True)  # (B, 1)
            focal_loss = alpha_t * focal_loss  # (B, C)
        
        # Reduce por clase
        focal_loss = focal_loss.sum(dim=1)  # (B,)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SoftCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss que acepta soft labels (distribuciones de probabilidad).
    
    Útil cuando las etiquetas no son one-hot sino probabilidades.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) salida sin softmax
            targets: (B, C) distribuciones de probabilidad
        
        Returns:
            loss: escalar
        """
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy con Label Smoothing.
    
    Suaviza las etiquetas one-hot para evitar overconfidence:
    [0, 1, 0] -> [ε/K, 1-ε+ε/K, ε/K]
    
    Útil para mejorar generalización.
    """
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C)
            targets: (B, C) soft labels
        """
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Aplica smoothing
        smoothed_targets = targets * (1 - self.epsilon) + self.epsilon / n_classes
        
        loss = -(smoothed_targets * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combinación de múltiples losses con pesos.
    
    Útil para combinar Focal Loss + Label Smoothing, etc.
    """
    def __init__(self, losses, weights=None):
        """
        Args:
            losses: lista de nn.Module losses
            weights: lista de pesos para cada loss (suman 1.0)
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        self.weights = weights
    
    def forward(self, logits, targets):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(logits, targets)
        return total_loss


def mixup_data(x, y, alpha=0.4, device='cuda'):
    """
    Mixup augmentation: mezcla pares de ejemplos y sus labels.
    
    x_mixed = lambda * x_i + (1 - lambda) * x_j
    y_mixed = lambda * y_i + (1 - lambda) * y_j
    
    Args:
        x: (B, C, H, W) batch de imágenes
        y: (B, num_classes) soft labels
        alpha: parámetro de la distribución Beta
        device: dispositivo
    
    Returns:
        x_mixed, y_mixed, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y, lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix augmentation: recorta y pega regiones entre imágenes.
    
    Útil para que el modelo aprenda características locales.
    
    Args:
        x: (B, C, H, W)
        y: (B, num_classes)
        alpha: parámetro Beta
        device: dispositivo
    
    Returns:
        x_mixed, y_mixed, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    # Coordenadas del recorte
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Mezcla
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Ajusta lambda según área real
    lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    mixed_y = lam_adjusted * y + (1 - lam_adjusted) * y[index]
    
    return mixed_x, mixed_y, lam_adjusted


# Factory function para crear losses fácilmente
def create_loss(loss_name, class_weights=None, **kwargs):
    """
    Crea una función de pérdida por nombre.
    
    Args:
        loss_name: 'focal', 'cross_entropy', 'label_smoothing', 'combined'
        class_weights: tensor de pesos por clase
        **kwargs: parámetros adicionales
    
    Returns:
        loss_fn: nn.Module
    """
    if loss_name == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_name == 'cross_entropy':
        return SoftCrossEntropyLoss()
    
    elif loss_name == 'label_smoothing':
        epsilon = kwargs.get('epsilon', 0.1)
        return LabelSmoothingCrossEntropy(epsilon=epsilon)
    
    elif loss_name == 'combined':
        # Focal + Label Smoothing
        focal = FocalLoss(alpha=class_weights, gamma=2.0)
        smooth = LabelSmoothingCrossEntropy(epsilon=0.1)
        return CombinedLoss([focal, smooth], weights=[0.7, 0.3])
    
    else:
        raise ValueError(f"Loss {loss_name} no reconocida. Usa: focal, cross_entropy, label_smoothing, combined")
