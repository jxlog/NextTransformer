"""Canon layers A-D on the full-attention Dense shell with its standard residual stream."""
from torch import nn

from nexttransformer.methods.attnres_canon_abcd import Canon, ffn, init, mixer  # noqa: F401
from nexttransformer.models.backbone import norm


class CanonBlock(nn.Module):
    """Add Canon-A/C to the branch inputs before their RMSNorms."""

    def __init__(self, dense_block, width):
        super().__init__()
        self.attn = dense_block.attn
        self.mlp = dense_block.mlp
        self.canon_a = Canon(width)
        self.canon_c = Canon(width)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(self.canon_a(x)), ve, cos_sin, window_size, kv_cache)
        return x + self.mlp(norm(self.canon_c(x)))


def block(dense_block, config, layer):
    return CanonBlock(dense_block, config.width)
