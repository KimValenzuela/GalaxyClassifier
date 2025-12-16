import torch
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if current_score < (self.best_score + self.min_delta):
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best_score = current_score
            self.counter = 0
            return False


class TrainerGalaxyClassifier:
    def __init__(
        self, 
        model,
        model_name,
        optimizer,
        fn_loss,
        train_loader,
        val_loader,
        num_classes,
        device,
        scheduler=None,
        use_soft_labels=False,
        class_names=None,
        use_mixup=False,
        mixup_alpha=0.4,
        use_cutmix=False,
        cutmix_alpha=1.0,
        grad_clip=None
    ):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.fn_loss = fn_loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.scheduler = scheduler
        self.use_soft_labels = use_soft_labels
        self.class_names = class_names
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.grad_clip = grad_clip

        # Métricas
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.train_f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.val_f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        
        # F1 por clase
        self.val_f1score_per_class = F1Score(
            task="multiclass", 
            num_classes=num_classes, 
            average=None
        ).to(device)
        
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true").to(device)
        self.best_confusion_matrix = None
        self.best_f1_per_class = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_f1score": [],
            "val_f1score": [],
            "learning_rate": []
        }

    def mixup_data(self, x, y):
        """Aplica mixup augmentation"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y

    def cutmix_data(self, x, y):
        """Aplica cutmix augmentation"""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
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
        
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        mixed_y = lam_adjusted * y + (1 - lam_adjusted) * y[index]
        
        return mixed_x, mixed_y

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0

        for x, y in tqdm(self.train_loader, desc="Training"):
            x = x.to(self.device)
            y = y.to(self.device)

            # Aplica mixup o cutmix aleatoriamente
            if self.use_mixup and np.random.rand() < 0.5:
                x, y = self.mixup_data(x, y)
            elif self.use_cutmix and np.random.rand() < 0.5:
                x, y = self.cutmix_data(x, y)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.fn_loss(logits, y)

            loss.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            train_loss += loss.item()

            preds = logits
            targets = y.argmax(dim=1)

            self.train_accuracy.update(preds, targets)
            self.train_f1score.update(preds, targets)

        return train_loss / len(self.train_loader)


    def validate(self):
        self.model.eval()
        val_loss = 0.0

        self.confusion_matrix.reset()
        self.val_f1score_per_class.reset()

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation"):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.fn_loss(logits, y)

                val_loss += loss.item()

                targets = y.argmax(dim=1)
                preds = logits

                self.val_accuracy.update(preds, targets)
                self.val_f1score.update(preds, targets)
                self.val_f1score_per_class.update(preds, targets)
                self.confusion_matrix.update(preds, targets)

        return val_loss / len(self.val_loader)


    def fit(self, epochs, early_stopping=None):
        best_val_f1 = 0.0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            train_accuracy = self.train_accuracy.compute().item()
            train_f1score = self.train_f1score.compute().item()
            val_accuracy = self.val_accuracy.compute().item()
            val_f1score = self.val_f1score.compute().item()
            f1_per_class = self.val_f1score_per_class.compute()

            # Scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Guarda learning rate actual
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["learning_rate"].append(current_lr)

            # Early stopping
            if early_stopping:
                stop = early_stopping.step(val_f1score)
                if stop:
                    print(f"\n🛑 Early Stopping at epoch {epoch+1}")
                    break

            # Guarda historial
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["train_f1score"].append(train_f1score)
            self.history["val_f1score"].append(val_f1score)

            # Imprime métricas
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.2e}")
            print(f"{'='*70}")
            print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | F1: {train_f1score:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f} | F1: {val_f1score:.4f}")
            print(f"\n  F1 por clase:")
            for i, (name, f1) in enumerate(zip(self.class_names, f1_per_class)):
                print(f"    {name:20s}: {f1:.4f}")

            # Guarda mejor modelo
            if val_f1score > best_val_f1:
                best_val_f1 = val_f1score
                torch.save(self.model.state_dict(), f"{self.model_name}_best_weights.pth")
                self.best_confusion_matrix = self.confusion_matrix.compute().detach().cpu()
                self.best_f1_per_class = f1_per_class.detach().cpu()
                print(f"\n  ✅ Mejor modelo guardado (F1: {best_val_f1:.4f})")

            # Reset métricas
            self.train_accuracy.reset()
            self.train_f1score.reset()
            self.val_accuracy.reset()
            self.val_f1score.reset()
            self.val_f1score_per_class.reset()


    def plot_confusion_matrix(self, save_path=None):
        """Grafica matriz de confusión del mejor modelo"""
        if self.best_confusion_matrix is None:
            raise RuntimeError("Matriz de confusión no disponible")

        cm = self.best_confusion_matrix.numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proporción'}
        )
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.title("Confusion Matrix (Best Model - Validation)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def plot_metrics_history(self, save_prefix=None):
        """Grafica evolución de métricas durante entrenamiento"""
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Loss
        ax = axes[0, 0]
        best_val_loss = min(self.history["val_loss"])
        best_epoch_loss = self.history["val_loss"].index(best_val_loss) + 1
        
        ax.plot(epochs, self.history["train_loss"], label="Train", linewidth=2)
        ax.plot(epochs, self.history["val_loss"], label="Validation", linewidth=2)
        ax.axhline(best_val_loss, linestyle="--", color="red", alpha=0.7, 
                   label=f"Best Val Loss = {best_val_loss:.4f}")
        ax.axvline(best_epoch_loss, linestyle=":", color="gray", alpha=0.5)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Training & Validation Loss", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        best_val_acc = max(self.history["val_accuracy"])
        best_epoch_acc = self.history["val_accuracy"].index(best_val_acc) + 1
        
        ax.plot(epochs, self.history["train_accuracy"], label="Train", linewidth=2)
        ax.plot(epochs, self.history["val_accuracy"], label="Validation", linewidth=2)
        ax.axhline(best_val_acc, linestyle="--", color="red", alpha=0.7,
                   label=f"Best Val Acc = {best_val_acc:.4f}")
        ax.axvline(best_epoch_acc, linestyle=":", color="gray", alpha=0.5)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title("Training & Validation Accuracy", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Score
        ax = axes[1, 0]
        best_val_f1 = max(self.history["val_f1score"])
        best_epoch_f1 = self.history["val_f1score"].index(best_val_f1) + 1
        
        ax.plot(epochs, self.history["train_f1score"], label="Train", linewidth=2)
        ax.plot(epochs, self.history["val_f1score"], label="Validation", linewidth=2)
        ax.axhline(best_val_f1, linestyle="--", color="red", alpha=0.7,
                   label=f"Best Val F1 = {best_val_f1:.4f}")
        ax.axvline(best_epoch_f1, linestyle=":", color="gray", alpha=0.5)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("F1-Score", fontsize=11)
        ax.set_title("Training & Validation F1-Score", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs, self.history["learning_rate"], linewidth=2, color='green')
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Learning Rate", fontsize=11)
        ax.set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_prefix:
            plt.savefig(f"{save_prefix}_metrics_history.png", dpi=300, bbox_inches='tight')
        plt.show()


    def plot_f1_per_class(self, save_path=None):
        """Grafica F1-Score por clase del mejor modelo"""
        if self.best_f1_per_class is None:
            raise RuntimeError("F1 por clase no disponible")

        f1_scores = self.best_f1_per_class.numpy()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.class_names, f1_scores, color='skyblue', edgecolor='navy')
        
        # Colores según performance
        for i, bar in enumerate(bars):
            if f1_scores[i] >= 0.7:
                bar.set_color('green')
            elif f1_scores[i] >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.axhline(f1_scores.mean(), linestyle='--', color='black', 
                    label=f'Mean F1: {f1_scores.mean():.3f}')
        
        # Valores sobre las barras
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("F1-Score", fontsize=12)
        plt.title("F1-Score per Class (Best Model)", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    def print_summary(self):
        """Imprime resumen del mejor modelo"""
        if self.best_f1_per_class is None:
            raise RuntimeError("Entrenar el modelo primero")
        
        best_val_f1 = max(self.history["val_f1score"])
        best_epoch = self.history["val_f1score"].index(best_val_f1) + 1
        
        print("\n" + "="*70)
        print("📊 RESUMEN DEL MEJOR MODELO")
        print("="*70)
        print(f"Mejor Epoch: {best_epoch}")
        print(f"Val F1 (Macro): {best_val_f1:.4f}")
        print(f"\nF1-Score por clase:")
        for name, f1 in zip(self.class_names, self.best_f1_per_class):
            print(f"  {name:20s}: {f1:.4f}")
        print("="*70)
