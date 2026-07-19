from importlib.resources import path
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class ConvBlock(nn.Module):
    """[Conv3x3 -> ReLU -> Conv3x3 -> BatchNorm -> ReLU -> MaxPool2]"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BoneAgeCNN(nn.Module):
    """CNN feature extractor + gender late fusion + regression head.

    Parameters
    ----------
    n_blocks      : number of conv blocks (each halves spatial size).
    base_filters  : filters in the first block; doubles every block.
    in_channels   : input image channels (1 for grayscale X-ray).
    fc_hidden     : width of the hidden FC layer in the head.
    dropout       : dropout probability before the final regression layer.
    img_size      : input side length (used to infer the flatten dimension).
    """

    def __init__(self, n_blocks: int = 5, base_filters: int = 8,
                 in_channels: int = 1, fc_hidden: int = 512, dropout: float = 0.3,
                 img_size: int = 256):
        super().__init__()

        blocks = []
        ch = in_channels
        for i in range(n_blocks):
            out = base_filters * (2 ** i)
            blocks.append(ConvBlock(ch, out))
            ch = out
        self.features = nn.Sequential(*blocks)

        # Each block halves the spatial size and the last block has
        # base_filters * 2**(n_blocks-1) channels.
        side = img_size // (2 ** n_blocks)
        self.flatten_dim = base_filters * (2 ** (n_blocks - 1)) * side * side

        # Regression head operating on [visual features ++ gender].
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim + 1, fc_hidden), # 1 = gender dimension
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, image: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        """image: [B, C, H, W], gender: [B, gender_dim] -> [B, 1]."""
        feats = self.features(image).flatten(1)        # [B, flatten_dim]
        fused = torch.cat([feats, gender], dim=1)      # late fusion
        return self.head(fused)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(self, loaders, cfg, device, name):
        """Train the model. tqdm over epochs, save the model every 5 epochs."""
        self.to(device)

        if cfg["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=cfg["lr"],
                momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
            )
        elif cfg["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            )
        else:
            raise ValueError("Unknown optimizer: {}".format(cfg["optimizer"]))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["lr_decay"])
        criterion = nn.L1Loss()

        ckpt_dir = Path("../data/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        val_loss = None
        train_loss_history = []
        val_loss_history = []

        for epoch in tqdm(range(cfg["epochs"]), desc=name):
            # train
            self.train()
            train_total, train_n = 0.0, 0
            for image, gender, target in loaders["train"]:
                image, gender, target = image.to(device), gender.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(self(image, gender), target)
                loss.backward()
                optimizer.step()
                train_total += loss.item() * image.size(0)
                train_n += image.size(0)
            scheduler.step()
            train_loss_history.append(train_total / train_n)

            # validation
            self.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for image, gender, target in loaders["valid"]:
                    image, gender, target = image.to(device), gender.to(device), target.to(device)
                    loss = criterion(self(image, gender), target)
                    total += loss.item() * image.size(0)
                    n += image.size(0)
            val_loss = total / n
            val_loss_history.append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == cfg["epochs"] - 1:
                torch.save({
                    "model_state": self.state_dict(),
                    "config": cfg,
                    "val_loss": val_loss,
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                }, ckpt_dir / f"{name}.pt")
        return val_loss

    @torch.no_grad()
    def predict_group(self,loader, scaler):
        """Run inference; return (y_true, y_pred, gender) as months/numpy arrays."""
        self.eval()
        device = next(self.parameters()).device
        ys, ps, gs = [], [], []
        for image, gender, target in tqdm(loader, desc="predict", leave=False):
            image = image.to(device)
            gender = gender.to(device)
            pred = self(image, gender).cpu().numpy().ravel()
            ps.append(scaler.inverse(pred))
            ys.append(scaler.inverse(target.numpy().ravel()))
            gs.append(gender.cpu().numpy().ravel())
        return np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)


def load_checkpoint(path):
    """Rebuild the model from a checkpoint and load weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BoneAgeCNN(
        n_blocks=cfg["n_blocks"],
        base_filters=cfg["base_filters"],
        in_channels=cfg["in_channels"],
        fc_hidden=cfg["fc_hidden"],
        dropout=cfg["dropout"],
        img_size=cfg["img_size"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to("cpu")
    return model, ckpt
