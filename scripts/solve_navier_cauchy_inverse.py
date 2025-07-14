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
    instance="torus",
    verbose=True,
)
num_samples = 20000

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
    density=9.0e4,  # GT
    youngs=1.0e5,   # GT
    poissons=0.1,   # GT
    # density=3.0e2,
    # youngs=1.0e9,
    # poissons=0.8,
    ground_pos=dataset.ground_pos,
    up_index=dataset.up_index,
    optimize_properties=True,
    activation=nn.Tanh,
).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
)
optimizers = [
    optim.Adam(
        model.network_parameters(),
        lr=1e-3,
    ), 
    optim.Adam(
        model.property_parameters(),
        lr=1e-1,
    ), 
]

schedulers = [
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
    ) for optimizer in optimizers
] 

# -------------------------------------
# 3. Training Loop
# -------------------------------------

loss_history = []
loss_history_detailed = {}
prop_history = {
    'density': [model.density.data.item()],
    'youngs': [model.youngs.data.item()],
    'poissons': [model.poissons.data.item()],
}
best_loss = torch.inf
n_epochs = 200

for epoch in range(n_epochs):
    epoch_loss = 0.0
    epoch_loss_detailed = {}

    for sample in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}"):

        geometry = sample["geometry"][0]  # [N_points, 3]
        sampling_indices = torch.randperm(len(geometry) - 1)[:num_samples]
        
        geometry = geometry[sampling_indices].to(DEVICE)
        displacement = sample['displacement'][0, sampling_indices].to(DEVICE)
        time_value = sample["time"].to(DEVICE)[0]
        
        num_points = geometry.shape[0]

        # build xyzt
        t_values = torch.full(
            (num_points, 1), time_value[0], 
            dtype=geometry.dtype, device=geometry.device
        )
        xyzt = torch.cat([geometry, t_values], dim=1)  # [N_points, 4]

        # forward
        losses: dict[str, torch.Tensor] = model.compute_loss(xyzt, displacement=displacement)

        # total loss
        losses['pde_loss'] = losses['pde_loss'] / 117_214_320.0
        losses['bc_loss'] = losses['bc_loss'] * 100000.0
        losses['gt_loss'] = losses['gt_loss'] * 100.0
        total_loss = sum([
            losses['pde_loss'],
            losses['bc_loss'],
            losses['gt_loss'],
        ])
        # print({key: f'{val.item():.3E}' for key, val in losses.items()})
        
        total_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += total_loss.clone().detach().cpu().item()
        for key, val in losses.items():
            if key in epoch_loss_detailed:
                epoch_loss_detailed[key] += val.clone().detach().cpu().item()
            else:
                epoch_loss_detailed[key] = val.clone().detach().cpu().item()

    for scheduler in schedulers:
        scheduler.step(epoch_loss)
        
    loss_history.append(epoch_loss)
    for key, val in epoch_loss_detailed.items():
        if key in loss_history_detailed:
            loss_history_detailed[key].append(val)
        else:
            loss_history_detailed[key] = [val]
    prop_history['youngs'].append(model.youngs.data.item())
    prop_history['poissons'].append(model.poissons.data.item())
    prop_history['density'].append(model.density.data.item())
        
    ckpt = {
        'epoch': epoch + 1,
        'model': {
            key: val.clone().detach().cpu()
            for key, val in model.state_dict().items()
        },
        'optimizers': [
            optimizer.state_dict() 
            for optimizer in optimizers
        ],
        'loss_list': loss_history,
        'loss_detailed': loss_history_detailed,
        'prop_traj': prop_history,
    }
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(ckpt, 'best.pt')
    torch.save(ckpt, 'last.pt')
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.6f} ρ: {model.density.item():.2E} ν: {model.poissons.item():.2E} E: {model.youngs.item():.2E}")

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
