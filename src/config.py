import torch

def get_device() -> torch.device:
    """Return the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


COMMON = dict(in_channels=1, fc_hidden=512,  img_size=256,
              batch_size=32, epochs=30, weight_decay=5e-5, momentum=0.9,  lr_decay=0.97
)

MODEL_CONFIGS: dict[str, dict] = {
    # 1) Faithful Cole-style baseline: 5 blocks, light dropout, SGD lr 0.01.
    "baseline": {**COMMON, "n_blocks": 5, "base_filters": 8,
                 "dropout": 0.3, "optimizer": "sgd", "lr": 0.01},

    # 2) Stronger regularization to fight overfitting (heavier dropout).
    "high_dropout": {**COMMON, "n_blocks": 5, "base_filters": 8,
                     "dropout": 0.5, "optimizer": "sgd", "lr": 0.01},

    # 3) Shallower network: capacity vs. speed trade-off (4 blocks).
    "shallow": {**COMMON, "n_blocks": 4, "base_filters": 8,
                "dropout": 0.3, "optimizer": "sgd", "lr": 0.01},

    # 4) Optimizer comparison: Adam instead of SGD.
    "adam": {**COMMON, "n_blocks": 5, "base_filters": 8,
             "dropout": 0.3, "optimizer": "adam", "lr": 1e-3},
}