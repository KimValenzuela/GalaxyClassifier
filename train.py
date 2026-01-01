import torch
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score
from torchmetrics.regression import MeanSquaredError
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        """
        Early stopping for metrics that must be MINIMIZED (e.g. RMSE)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


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
        class_names=None
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

        #Metricas
        self.train_rmse = MeanSquaredError(squared=False).to(device)
        self.val_rmse = MeanSquaredError(squared=False).to(device)

        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true").to(device)
        self.best_confusion_matrix = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": []
        }

    def train_one_epoch(self):
        self.model.train()
        self.train_rmse.reset()
        train_loss = 0.0

        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.fn_loss(logits, y)

            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            self.train_rmse.update(probs, y)

        return train_loss / len(self.train_loader), self.train_rmse.compute().item()


    def validate(self):
        self.model.eval()
        val_loss = 0.0

        self.val_rmse.reset()

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.fn_loss(logits, y)

                val_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                self.val_rmse.update(probs, y)

        return val_loss / len(self.val_loader), self.val_rmse.compute().item()


    def fit(self, epochs, early_stopping=None):
        best_val_rmse = float("inf")

        for epoch in range(epochs):
   
            train_loss, train_rmse = self.train_one_epoch()
            val_loss, val_rmse = self.validate()

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_rmse)
                else:
                    self.scheduler.step()

            if early_stopping:
                if early_stopping.step(val_rmse):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history.setdefault("train_rmse", []).append(train_rmse)
            self.history.setdefault("val_rmse", []).append(val_rmse)

            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f" Train → Loss: {train_loss:.4f} | RMSE: {train_rmse:.4f}")
            print(f" Val   → Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f}")
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(self.model.state_dict(), f"{self.model_name}_weights.pth")
                print("Modelo guardado")



    def plot_rmse_history(self):
        epochs = range(1, len(self.history["train_rmse"]) + 1)

        best_val_rmse = min(self.history["val_rmse"])
        best_epoch = self.history["val_rmse"].index(best_val_rmse) + 1

        plt.figure(figsize=(7,5))
        plt.plot(epochs, self.history["train_rmse"], label="Train RMSE")
        plt.plot(epochs, self.history["val_rmse"], label="Validation RMSE")
        plt.axhline(
            best_val_rmse,
            linestyle="--",
            color="tab:red",
            alpha=0.8,
            label=f"Best Val RMSE = {best_val_rmse:.4f}"
        )
        plt.scatter(best_epoch, best_val_rmse, color="tab:red", zorder=3)

        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("RMSE vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.model_name}_rmse_curve.png", dpi=300)
        plt.show()

