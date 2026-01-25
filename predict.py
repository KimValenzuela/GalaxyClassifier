import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

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


    def predict_batch(self, dataloader, dataset, threshold=None):
        results = []

        idx_global = 0

        with torch.no_grad():
            for x in tqdm(dataloader):
                x = x.to(self.device)

                probs = F.softmax(self.model(x), dim=1)
                max_probs, preds = probs.max(dim=1)
                probs_np = probs.cpu().numpy()

                for i in range(len(preds)):
                    img_id = dataset.df.iloc[idx_global]["ID"]

                    row = {
                        "id": img_id,
                        "pred_class": preds[i].item(),
                        "pred_label": self.class_names[preds[i].item()],
                        "confidence": max_probs[i].item(),
                    }

                    for name, p in zip(self.class_names, probs_np[i]):
                        row[name] = float(p)

                    if threshold is not None:
                        row["accepted"] = max_probs[i].item() >= threshold

                    results.append(row)
                    idx_global += 1

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


    def visualize_by_label(
        self,
        dataset,
        predictions_df,
        label,
        n=9
    ):

        df_label = predictions_df[
            predictions_df["pred_label"] == label
        ].head(n)

        n = len(df_label)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(3 * cols, 3 * rows))

        for i, (_, row) in enumerate(df_label.iterrows()):
            idx = dataset.df.index[
                dataset.df["ID"] == row["id"]
            ][0]

            img = dataset[idx]
            if isinstance(img, (tuple, list)):
                img = img[0]

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img.squeeze(), cmap="gray", origin="lower")
            plt.axis("off")
            plt.title(f"{row['pred_label']}\n{row['confidence']:.2f}")

        plt.tight_layout()
        plt.show()