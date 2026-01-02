import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torchmetrics.classification import Accuracy, ConfusionMatrix, F1Score

class PostTrainingEvaluator:
    def __init__(self, model, device, num_classes, class_names):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names

    def _init_metrics(self):
        acc = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device)
        cm = ConfusionMatrix(task="multiclass", num_classes=self.num_classes, normalize="true").to(self.device)
        return acc, f1, cm


    def evaluate_map(self, dataloader):
        self.model.eval()
        acc, f1, cm = self._init_metrics()

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = F.softmax(logits, dim=1)

                preds = probs.argmax(dim=1)
                targets = y.argmax(dim=1)

                acc.update(preds, targets)
                f1.update(preds, targets)
                cm.update(preds, targets)

        return {
            "accuracy": acc.compute().item(),
            "f1_macro": f1.compute().item(),
            "confusion_matrix": cm.compute().cpu()
        }

    def evaluate_threshold(self, dataloader, threshold=0.75):
        self.model.eval()
        acc, f1, cm = self._init_metrics()

        total = 0
        kept = 0

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = F.softmax(logits, dim=1)

                max_probs, preds = probs.max(dim=1)
                targets = y.argmax(dim=1)

                mask = max_probs >= threshold
                if mask.sum() == 0:
                    continue

                acc.update(preds[mask], targets[mask])
                f1.update(preds[mask], targets[mask])
                cm.update(preds[mask], targets[mask])

                kept += mask.sum().item()
                total += x.size(0)

        metrics = {
            "accuracy": acc.compute().item() if kept > 0 else None,
            "f1_macro": f1.compute().item() if kept > 0 else None,
            "fraction_kept": kept / total
        }

        cm_dict = {
            "confusion_matrix": cm.compute().cpu() if kept > 0 else None,
            "fraction_kept": kept / total
        }

        return metrics, cm_dict


    def sweep_thresholds(self, dataloader, thresholds):
        metrics, cms = [], []

        for t in thresholds:
            res, cm = self.evaluate_threshold(dataloader=dataloader, threshold=t)
            res["threshold"] = t
            cm["threshold"] = t

            metrics.append(res)
            cms.append(cm)

        return pd.DataFrame(metrics), cms


    def plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm.numpy(),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    
    def collect_examples_by_threshold(
        self,
        dataloader,
        threshold,
        max_examples=8
    ):
        self.model.eval()
        kept, discarded = [], []

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                probs = F.softmax(logits, dim=1)

                max_probs, preds = probs.max(dim=1)
                targets = y.argmax(dim=1)

                for i in range(x.size(0)):
                    sample = {
                        "image": x[i].cpu(),
                        "pred": preds[i].item(),
                        "target": targets[i].item(),
                        "confidence": max_probs[i].item(),
                        "probs": probs[i].cpu()
                    }

                    if max_probs[i] >= threshold and len(kept) < max_examples:
                        kept.append(sample)

                    if max_probs[i] < threshold and len(discarded) < max_examples:
                        discarded.append(sample)

                    if len(kept) >= max_examples and len(discarded) >= max_examples:
                        return kept, discarded

        return kept, discarded


    def plot_examples_by_threshold(self, examples, title):
        n = len(examples)
        plt.figure(figsize=(3 * n, 3))

        for i, ex in enumerate(examples):
            plt.subplot(1, n, i + 1)
            img = ex["image"].squeeze()
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.title(
                f"Pred: {self.class_names[ex['pred']]}\n"
                f"Conf: {ex['confidence']:.2f}"
            )

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


    def class_retention_by_threshold(
        self,
        dataloader,
        thresholds
    ):
        self.model.eval()
        results = []

        with torch.no_grad():
            for t in thresholds:
                total_per_class = np.zeros(self.num_classes)
                kept_per_class = np.zeros(self.num_classes)

                for x, y in dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(x)
                    probs = F.softmax(logits, dim=1)

                    max_probs, preds = probs.max(dim=1)
                    targets = y.argmax(dim=1)

                    for c in range(self.num_classes):
                        mask = targets == c
                        total_per_class[c] += mask.sum().item()
                        kept_per_class[c] += (
                            mask & (max_probs >= t)
                        ).sum().item()

                fraction = np.divide(
                    kept_per_class,
                    total_per_class,
                    out=np.zeros_like(kept_per_class),
                    where=total_per_class > 0
                )

                for c in range(self.num_classes):
                    results.append({
                        "threshold": t,
                        "class": c,
                        "fraction_kept": fraction[c]
                    })

        return pd.DataFrame(results)


    def plot_class_retention_by_threshold(self, df):
        plt.figure(figsize=(8, 6))

        for i, cname in enumerate(self.class_names):
            subset = df[df["class"] == i]
            plt.plot(
                subset["threshold"],
                subset["fraction_kept"],
                marker="o",
                label=cname
            )

        plt.xlabel("Probability threshold")
        plt.ylabel("Fraction kept per class")
        plt.title("Class-dependent retention vs threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()