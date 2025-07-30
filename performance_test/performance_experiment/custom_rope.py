import torch
import torch.nn as nn

from romae.positional_embeddings import BasePosEmbedding

class NDPRope(nn.Module,  BasePosEmbedding):
    """
    N-dimensional continuous p-RoPE. The initial p-RoPE code was converted from
    the JAX implementation here:
    "Round and Round We Go! What Makes Rotary Positional Encodings Useful?"
    https://openreview.net/forum?id=GtvuNrk58a
    """
    def __init__(self, head_dim: int, positions, B, base=10000, p=1,
                 n_dims: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if head_dim % n_dims != 0:
            raise AttributeError(f"The head dimension ({head_dim}) is not "
                                 f"divisible by the number of positional axis ({n_dims})!")
        if 0 > p or p > 1:
            raise AttributeError(f"Provided p value ({p}) is not between 0 and 1!")

        self.axis_dim = head_dim // n_dims
        self.n_dims = n_dims

        rope_angles = int(p * self.axis_dim // 2)
        nope_angles = self.axis_dim // 2 - rope_angles

        fraction = 2. * torch.arange(0, rope_angles) / self.axis_dim
        self.register_buffer("timescale", nn.Parameter(nn.functional.pad(
            base ** fraction,
            (0, nope_angles),
            mode='constant',
            value=torch.inf
        )))
        self.cache = []
        for i in range(self.n_dims):
            self.cache.append(self.get_sin_cos(positions[:, i].reshape(B, -1)))


    def reset_cache(self):
        ...

    def get_sin_cos(self, positions):
        sinusoid_inp = (
                positions[..., torch.newaxis] / self.timescale[torch.newaxis,
                                                torch.newaxis, :]
        )
        sinusoid_inp = sinusoid_inp[..., torch.newaxis, :]
        sin = torch.sin(sinusoid_inp).to("cuda")
        cos = torch.cos(sinusoid_inp).to("cuda")

        return sin, cos

    def apply_ndprope(self, x, angles):
        sin, cos = angles
        sin, cos = sin[:x.shape[0], :x.shape[1]], cos[:x.shape[0], :x.shape[1]]
        first_half, second_half = torch.tensor_split(x, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        out = torch.concatenate([first_part, second_part], dim=-1)
        return out.to(x.dtype)

    def forward(self, x, positions):
        """
        Parameters
        ----------
        x
        positions : Tensor, shape ``[batch_size, ndim, seq_len]``
            For 3D position, this would be ```[batch_size, 3, seq_len]```.
        """
        B, seq_len, nhead, head_dim = x.shape

        # Collapse embeddings into the sequence dimension
        views = []
        for i in range(self.n_dims):
            views.append(self.apply_ndprope(
                x[..., self.axis_dim*i:self.axis_dim*(i+1)],
                self.cache[i],
            ))
        x = torch.cat(views, dim=-1)
        return x
