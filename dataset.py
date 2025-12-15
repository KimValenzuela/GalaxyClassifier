import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

CLASS_COLS = [
    "p_smooth",
    "p_features",
    "p_irregular",
    "p_point_source",
    "p_unclassifiable",
]

class PreprocessImage(Dataset):
    def __init__(self, df, img_dir, transform=None, log_scale=True, eps=1e-8):
        """
        df: DataFrame con columnas [ID] + CLASS_COLS
        img_dir: carpeta donde están los .fits
        transform: transforms de torchvision (que acepten tensores CHW)
        log_scale: si True, aplica log(1 + x) a la imagen
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.log_scale = log_scale
        self.eps = eps

    def __len__(self):
        return len(self.df)

    def _load_fits(self, img_id):
        path = os.path.join(self.img_dir, f"{img_id}.fits")
        with fits.open(path) as hdul:
            img = hdul[0].data.astype("float32")
        return img

    def _preprocess_image(self, img):
        img = np.nan_to_num(img, nan=0.0)

        # Quita outliers
        vmin, vmax = np.percentile(img, [1, 99])
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin + self.eps)  # [0,1]

        if self.log_scale:
            img = np.log1p(img)  # resalta estructura débil

        # (H, W) -> (1, H, W)
        img = img.astype("float32")
        img = torch.from_numpy(img).unsqueeze(0)  # 1 canal

        # Normalización
        img = (img - img.mean()) / (img.std() + self.eps)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["ID"]

        # Imagen
        img = self._load_fits(img_id)
        img = self._preprocess_image(img)

        if self.transform:
            img = self.transform(img)

        # Soft label
        probs = row[CLASS_COLS].astype(float).to_numpy(dtype="float32")
        y = torch.from_numpy(probs)

        return img, y


class GalaxyDataset:
    def __init__(
        self, 
        labels_path: str,
        img_dir: str,
        class_cols: list,
        img_size: int = 224,
        batch_size: int = 32,
        test_size: float = 0.2,
        random_state: int = 42,
        num_workers: int = 4,
        use_weighted_sampler: bool = True,
        eps: float = 1e-6,
        train_transform = None, 
        val_transform = None
    ):
        self.labels_path = labels_path
        self.img_dir = img_dir
        self.class_cols = class_cols
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler
        self.eps = eps

        self.num_classes = len(class_cols)

        # Transforms
        self.train_transform = train_transform or self._default_train_transform()
        self.val_transform = val_transform or self._default_val_transform()

        # Inicialización en setup()
        self.train_df = None
        self.val_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None

    def _default_train_transform(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180)
        ])
        return train_transform

    def _default_val_transform(self):
        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size))
        ])
        return val_transform

    def setup(self):
        df = pd.read_csv(self.labels_path)

        for col in self.class_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Hard labels para estratificación
        probs = df[self.class_cols].values
        df["hard_label"] = probs.argmax(axis=1)

        # Split estratificado
        self.train_df, self.val_df = train_test_split(
            df,
            test_size=self.test_size,
            shuffle=True,
            stratify=df["hard_label"],
            random_state=self.random_state
        )

        # Datasets
        self.train_dataset = PreprocessImage(
            self.train_df,
            self.img_dir,
            transform=self.train_transform
        )

        self.val_dataset = PreprocessImage(
            self.val_df,
            self.img_dir,
            transform=self.val_transform
        )

        # Sampler opcional
        if self.use_weighted_sampler:
            self.train_sampler = self._create_weighted_sampler()
        else:
            self.train_sampler = None

    def _create_weighted_sampler(self):
        hard_labels = self.train_df["hard_label"].values
        class_counts = np.bincount(hard_labels, minlength=self.num_classes)

        class_weights = 1.0 / (class_counts + self.eps)
        sample_weights = class_weights[hard_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

    # Dataloaders
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=False if self.train_sampler else True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return val_dataloader

