import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class GalaxyPredictor:
    def __init__(
        self,
        model,
        device,
        class_names=None
    ):
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

                for i in range(len(preds)):
                    row = {
                        "pred_class": preds[i].item(),
                        "confidence": max_probs[i].item(),
                    }

                    # Soft labels
                    for c, p in enumerate(probs[i].cpu().numpy()):
                        row[f"prob_class_{c}"] = p

                    if threshold is not None:
                        row["accepted"] = max_probs[i].item() >= threshold

                    if self.class_names:
                        row["pred_label"] = self.class_names[preds[i].item()]

                    results.append(row)

        return pd.DataFrame(results)


    def save_csv(self, df, path):
        df.to_csv(path, index=False)

    
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

        label = (
            self.class_names[pred.item()]
            if self.class_names else pred.item()
        )

        plt.figure(figsize=(4, 4))
        plt.imshow(img.squeeze(), cmap="gray", origin="lower")
        plt.axis("off")

        title = f"Pred: {label}\nConf: {max_prob.item():.3f}"

        if threshold is not None:
            title += f"\nAccepted: {max_prob.item() >= threshold}"

        plt.title(title)
        plt.show()