import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

from run_spice_mnist_local_feature_torch_train import LocalFeatureReadout  # noqa: E402


def pm1_targets(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    target = -torch.ones((len(labels), n_classes), dtype=torch.float32)
    target[torch.arange(len(labels)), labels] = 1.0
    return target


def test_mse_tanh_sgd_maps_to_spice_delta_update_up_to_lr_scale():
    torch.manual_seed(3)
    batch = 3
    blocks = 2
    channels = 3
    block_len = 4
    n_classes = 10
    lr_torch = 4.0
    lr_spice = 2.0 * lr_torch / n_classes

    model = LocalFeatureReadout(blocks, channels, block_len, seed=7)
    x_blocks = torch.randn(batch, blocks, block_len)
    labels = torch.tensor([1, 4, 7], dtype=torch.long)
    target = pm1_targets(labels, n_classes)

    w0 = model.local_weights.detach().clone()
    hb0 = model.local_bias.detach().clone()
    v0 = model.readout.detach().clone()
    ob0 = model.output_bias.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_torch)
    optimizer.zero_grad()
    logits = model(x_blocks)
    loss = F.mse_loss(torch.tanh(logits), target)
    loss.backward()
    optimizer.step()

    h = torch.tanh(torch.einsum("nbp,bcp->nbc", x_blocks, w0) + hb0)
    score = torch.einsum("nbc,kbc->nk", h, v0) + ob0
    y = torch.tanh(score)
    d = (target - y) * (1.0 - y * y)
    dh = torch.einsum("nk,kbc->nbc", d, v0) * (1.0 - h * h)

    expected_w = w0 + lr_spice * torch.einsum("nbc,nbp->bcp", dh, x_blocks) / batch
    expected_hb = hb0 + lr_spice * dh.mean(dim=0)
    expected_v = v0 + lr_spice * torch.einsum("nk,nbc->kbc", d, h) / batch
    expected_ob = ob0 + lr_spice * d.mean(dim=0)

    torch.testing.assert_close(model.local_weights.detach(), expected_w, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.local_bias.detach(), expected_hb, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.readout.detach(), expected_v, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.output_bias.detach(), expected_ob, rtol=1e-5, atol=1e-6)
