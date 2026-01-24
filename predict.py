import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class GalaxyPredictor:
    def __init__(
        self,
        root_path,
        model,
        device,
        class_names=None
    ):
        self.root_path = root_path
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names

        self.model.eval()


    def predict_batch(self, dataloader, threshold=None):
        results = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # batch puede ser (x, y) o solo x
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device)

                logits = self.model(x)
                probs = F.softmax(logits, dim=1)

                max_probs, preds = probs.max(dim=1)
                probs_np = probs.cpu().numpy()

                for i in range(len(preds)):
                    row = {
                        "pred_class": preds[i].item(),
                        "pred_label": self.class_names[preds[i].item()],
                        "confidence": max_probs[i].item(),
                    }

                    # Soft labels con nombres reales
                    for class_name, p in zip(self.class_names, probs_np[i]):
                        row[class_name] = float(p)

                    if threshold is not None:
                        row["accepted"] = max_probs[i].item() >= threshold

                    results.append(row)

        return pd.DataFrame(results)


    def save_csv(self, df):
        df.to_csv(os.path.join(self.root_path, "clash_predictions.csv"), index=False)

    
    def visualize_sample(
        self,
        dataset,
        idx,
        threshold=None
    ):
        img = dataset[idx][0]  # imagen ya preprocesada

        x = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

        max_prob, pred = probs.max(dim=1)

        label = self.class_names[pred.item()]

        plt.figure(figsize=(4, 4))
        plt.imshow(img.squeeze(), cmap="gray", origin="lower")
        plt.axis("off")

        title = f"Pred: {label}\nConf: {max_prob.item():.3f}"

        if threshold is not None:
            title += f"\nAccepted: {max_prob.item() >= threshold}"

        plt.title(title)
        plt.show()