import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data.pac_nerf import PACNeRFDataset
from models.navier_cauchy import NavierCauchy
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------------
# 1. Dataset & DataLoader
# -------------------------------------
 
dataset = PACNeRFDataset(
    dataroot="dataset/pac-nerf",
    instance="cat",
    verbose=True,
)
num_samples = 40000

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)

DEVICE = torch.device('cuda', 0)

# -------------------------------------
# 2. PINN Model
# -------------------------------------

model = NavierCauchy(
    hid_dim=128,
    depth=6,
    density=9.0e3,
    youngs=1.0e6,
    poissons=0.3,
    ground_pos=dataset.ground_pos,
    up_index=dataset.up_index,
    optimize_properties=True,
    activation=nn.Tanh,
).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr=1.0E-5,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    verbose=True,
)

# -------------------------------------
# 3. Training Loop
# -------------------------------------

loss_history = []
best_loss = torch.inf
n_epochs = 200

for epoch in range(n_epochs):
    epoch_loss = 0.0

    for sample in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}"):

        geometry = sample["geometry"][0]  # [N_points, 3]
        displacement = sample["displacement"][0]  # [N_points, 3]
        # geometry = sample[0][0]
        if len(geometry) > num_samples:
            indices = torch.randperm(len(geometry))[:num_samples]
            geometry = geometry[indices]
            displacement = displacement[indices]
        geometry = geometry.to(DEVICE)
        displacement = displacement.to(DEVICE)
        time_value = sample["time"].to(DEVICE)[0] 
        # time_value = sample[1].to(DEVICE)
        
        num_points = geometry.shape[0]

        # build xyzt
        t_values = torch.full(
            (num_points, 1), time_value[0], 
            dtype=geometry.dtype, device=geometry.device
        )
        xyzt = torch.cat([geometry, t_values], dim=1)  # [N_points, 4]

        # forward
        losses: dict[str, torch.Tensor] = model.compute_loss(xyzt, displacement=displacement, pde=True, ic=True)

        # total loss
        total_loss = sum([
            losses['pde_loss'] / 1.0E+8,
            losses['ic_loss'],
            losses['bc_loss'],
            losses['gt_loss'],
        ])
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += total_loss.clone().detach().cpu().item()

    scheduler.step(epoch_loss)
    loss_history.append(epoch_loss)
    ckpt = {
        'epoch': epoch + 1,
        'model': {
            key: val.clone().detach().cpu()
            for key, val in model.state_dict().items()
        },
        'model_config': model.config,
        'optimizer': optimizer.state_dict(),
        'loss_list': loss_history,
    }
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(ckpt, 'best.pt')
    torch.save(ckpt, 'last.pt')
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.6f}")

# -------------------------------------
# 4. Plot Loss
# -------------------------------------

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("PINN Loss Curve")
plt.savefig('loss.png')
