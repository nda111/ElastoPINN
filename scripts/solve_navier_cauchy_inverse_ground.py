import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data.pac_nerf import PACNeRFDataset
from models.navier_cauchy_ground import NavierCauchyGround

from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse
import os


parser = argparse.ArgumentParser(description="PINN for Navier-Cauchy with Ground Contact")
parser.add_argument('--pde_weight', type=float, default=1e-6, help='Weight for the PDE loss')
parser.add_argument('--gt_weight', type=float, default=1e5, help='Weight for the ground truth loss')
parser.add_argument('--contact_weight', type=float, default=1e4, help='Weight for the contact loss')
parser.add_argument('--exp_name', type=str, default='default_exp', help='Experiment name for saving results')
args = parser.parse_args()

# 실험 결과 저장을 위한 디렉토리 생성
output_dir = f"experiments/{args.exp_name}"
os.makedirs(output_dir, exist_ok=True)
print(f"--- Running Experiment: {args.exp_name} ---")
print(f"Parameters: pde_weight={args.pde_weight}, gt_weight={args.gt_weight}, contact_weight={args.contact_weight}")

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

model = NavierCauchyGround(
    hid_dim=128,
    depth=6,
    
    # ─────────────── 재료 실제값(또는 초기 추정치) ───────────────
    density = 9.0e4,   # kg m⁻³
    youngs  = 1.0e5,   # Pa
    poissons= 0.1,
    
    # ─────────────── 환경·지오메트리 ───────────────
    ground_pos = 0, # dataset.ground_pos,   # 지면 y-좌표
    up_index   = dataset.up_index,     # y축이 1 → 그대로
    gravity    = 9.8,                  # ↓ 방향 중력가속도 (기본값이라 생략 가능)
    
    # ─────────────── 학습 옵션 ───────────────
    optimize_properties = True,        # ρ,E,ν 도 함께 추정
    activation = nn.Tanh               # MLP 활성화 함수
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
        pde_weight = args.pde_weight # 1e-6
        gt_weight = args.gt_weight# 1e5
        contact_weight = args.contact_weight# 1e4

        losses['pde_loss'] = losses['pde_loss'] * pde_weight
        losses['gt_loss'] = losses['gt_loss'] * gt_weight
        losses['contact_loss'] = losses['contact_loss'] * contact_weight
        # losses['ic_loss'] = losses['ic_loss']
        # losses['bc_loss'] = losses['bc_loss'] * 100_000.0


        total_loss = sum([
            losses['pde_loss'],
            losses['gt_loss'],
            losses['contact_loss']
        ])
        # print({key: f'{val.item():.3E}' for key, val in losses.items()})
        
        total_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += total_loss.clone().detach().cpu().item()
        for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()

                # 누적
                if key in epoch_loss_detailed:
                    epoch_loss_detailed[key] += val
                else:
                    epoch_loss_detailed[key] = val

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
        'model_config': model.config,
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
        torch.save(ckpt, os.path.join(output_dir, 'best.pt'))
    
    torch.save(ckpt, os.path.join(output_dir, 'last.pt'))
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
plt.savefig(os.path.join(output_dir, 'loss.png'))
plt.close() # 메모리 누수 방지를 위해 그림 닫기

print(f"--- Experiment {args.exp_name} Finished ---")