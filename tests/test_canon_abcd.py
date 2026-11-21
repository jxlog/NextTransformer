"""Check that standalone Canon preserves the Dense shell and inserts causal convolutions."""
import torch

from nexttransformer.methods.attnres_canon_abcd import Canon
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model


def test_zero_initialized_canon_matches_dense_full():
    dense = build_model(debug_config("dense_full"), "cpu", seed=42).eval()
    canon = build_model(debug_config("canon_abcd"), "cpu", seed=42).eval()
    convolutions = {
        f"{name}.weight": module.weight for name, module in canon.named_modules() if isinstance(module, Canon)
    }
    assert len(convolutions) == 6 * canon.config.n_layer
    assert all(weight.shape[-1] == 4 and torch.count_nonzero(weight) == 0 for weight in convolutions.values())

    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        # Exercise the sublayers: the shared shell initializes both output projections to zero.
        for layer in dense.transformer.h:
            layer.attn.c_proj.weight.uniform_(-0.1, 0.1, generator=rng)
            layer.mlp.c_proj.weight.uniform_(-0.1, 0.1, generator=rng)
        missing, unexpected = canon.load_state_dict(dense.state_dict(), strict=False)
        assert set(missing) == set(convolutions)
        assert not unexpected
        tokens = torch.randint(0, canon.config.vocab_size, (2, canon.config.sequence_len), generator=rng)
        torch.testing.assert_close(canon(tokens), dense(tokens), rtol=0, atol=0)


def test_active_canon_a_and_c_leave_the_residual_stream_intact():
    model = build_model(debug_config("canon_abcd"), "cpu", seed=42).eval()
    layer = model.transformer.h[0]
    x = torch.ones(1, model.config.sequence_len, model.config.width)
    cos_sin = (model.cos[:, :x.size(1)], model.sin[:, :x.size(1)])

    with torch.no_grad():
        layer.canon_a.weight.fill_(0.25)
        layer.canon_c.weight.fill_(0.25)
        # The two branch projections are zero, so active Canon on either branch cannot change x.
        torch.testing.assert_close(layer(x, None, cos_sin, (x.size(1), 0), None), x, rtol=0, atol=0)
