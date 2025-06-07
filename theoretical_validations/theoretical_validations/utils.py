import torch.nn as nn


class PositionReconstructionHead(nn.Module):
    """Simple interpolation head making predictions on the original tubelet
    values from the learned MASK tokens.
    """
    def __init__(self, d_model: int, d_output: int, layer_norm_eps: float,
                 head_drop_rate: float, cls: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls = cls
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x):
        if self.cls:
            x = x[:, 1:, :]
        return self.head(x).transpose(1, 2)


class RelativeReconstructionHead(nn.Module):
    """Simple interpolation head making predictions on the original tubelet
    values from the learned MASK tokens.
    """
    def __init__(self, d_model: int, d_output: int, layer_norm_eps: float,
                 head_drop_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(head_drop_rate),
            nn.RMSNorm(d_model, layer_norm_eps),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x):
        cls = x[:, 1, :]
        logit = self.head(cls)
        return logit
