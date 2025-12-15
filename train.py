import torch
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        # macro -> calcula estadísticas por cada etiqueta y las promedia
        self.train_f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.val_f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        #self.val_f1score_topk = F1Score(task="multiclass", num_classes=num_classes, average=None, top_k=2).to(device)
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="true").to(device)
        self.best_confusion_matrix = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_f1score": [],
            "val_f1score": []
        }

    def train_one_epoch(self):
        self.model.train()
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

            preds = logits
            targets = y.argmax(dim=1)

            self.train_accuracy.update(preds, targets)
            self.train_f1score.update(preds, targets)

        return train_loss / len(self.train_loader)


    def validate(self):
        self.model.eval()
        val_loss = 0.0

        self.confusion_matrix.reset()

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.fn_loss(logits, y)

                val_loss += loss.item()

                targets = y.argmax(dim=1)
                
                preds = logits
                targets = y.argmax(dim=1)

                self.val_accuracy.update(preds, targets)
                self.val_f1score.update(preds, targets)
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

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if early_stopping:
                stop = early_stopping.step(val_f1score)
                if stop:
                    print(f"Early Stopping at epoch {epoch+1}")
                    break

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["train_f1score"].append(train_f1score)
            self.history["val_f1score"].append(val_f1score)

            print(f"Epoch [{epoch + 1}/{epochs}]")
            print(f"\tTrain → Loss: {train_loss: .4f} | Acc: {train_accuracy: .4f} | F1: {train_f1score: .4f}")
            print(f"\tVal   → Loss: {val_loss: .4f} | Acc: {val_accuracy: .4f} | F1: {val_f1score: .4f}")

            if val_f1score > best_val_f1:
                best_val_f1 = val_f1score
                torch.save(self.model.state_dict(), f"{self.model_name}_weights.pth")
                self.best_confusion_matrix = self.confusion_matrix.compute().detach().cpu()
                print("Modelo guardado")

            self.train_accuracy.reset()
            self.train_f1score.reset()
            self.val_accuracy.reset()
            self.val_f1score.reset()


    def plot_confusion_matrix(self):
        if self.best_confusion_matrix is None:
            raise RuntimeError("Matriz de confusión no disponible")

        cm = self.best_confusion_matrix.numpy()

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Validation)")
        plt.tight_layout()
        plt.show()


    def plot_metrics_history(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(15,4))

        # Loss
        plt.subplot(1,3,1)
        plt.plot(epochs, self.history["train_loss"], label="Train")
        plt.plot(epochs, self.history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.model_name}_loss_curve.png", dpi=300)

        # Accuracy
        plt.subplot(1,3,2)
        plt.plot(epochs, self.history["train_accuracy"], label="Train")
        plt.plot(epochs, self.history["val_accuracy"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.model_name}_accuracy_curve.png", dpi=300)

        # F1 Score
        plt.subplot(1,3,3)
        plt.plot(epochs, self.history["train_f1score"], label="Train")
        plt.plot(epochs, self.history["val_f1score"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.title("F1-Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.model_name}_f1score_curve.png", dpi=300)

        plt.tight_layout()
        plt.show()
