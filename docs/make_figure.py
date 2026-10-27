"""Train the same small network twice on a noisy sine wave on the Apple GPU: once built only from this repo's custom
Metal layers (CustomLinear, CustomLeakyReLU), once from standard PyTorch layers. Writes docs/training.png."""
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from custom import CustomLeakyReLU, CustomLinear  # noqa: E402

torch.manual_seed(0)
x = torch.rand(1000, 1) * 2 - 1
y = torch.sin(3 * x) + 0.1 * torch.randn_like(x)
xm, ym = x.to("mps"), y.to("mps")


def net(linear, act):
    return nn.Sequential(linear(1, 128), act(), linear(128, 64), act(), linear(64, 1)).to("mps")


def train(model, epochs=400):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    t0 = time.time()
    for _ in range(epochs):
        loss = nn.functional.mse_loss(model(xm), ym)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, time.time() - t0


custom = net(CustomLinear, CustomLeakyReLU)
standard = net(nn.Linear, nn.LeakyReLU)
lc, tc = train(custom)
ls, ts = train(standard)
grid = torch.linspace(-1, 1, 200).reshape(-1, 1)
with torch.no_grad():
    pc = custom(grid.to("mps")).cpu()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(x, y, s=3, color="#9aa7ad", label="training data")
ax[0].plot(grid, pc, color="#c2410c", lw=2.5, label="network of custom Metal layers")
ax[0].set_title("Fit: CustomLinear + CustomLeakyReLU on the GPU")
ax[0].legend(frameon=False)
ax[1].plot(lc, color="#c2410c", lw=2, label=f"custom Metal layers ({tc:.1f} s)")
ax[1].plot(ls, color="#1c6b8a", lw=2, ls="--", label=f"standard PyTorch layers ({ts:.1f} s)")
ax[1].set_yscale("log")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("mean squared error")
ax[1].set_title("Training loss, same architecture and optimizer")
ax[1].legend(frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("docs/training.png", dpi=150)
print(f"final loss custom {lc[-1]:.4f} standard {ls[-1]:.4f}; time custom {tc:.1f} s standard {ts:.1f} s")
