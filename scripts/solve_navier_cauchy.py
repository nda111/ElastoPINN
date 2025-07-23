from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs.config import CFG as cfg
from data.pac_nerf import PACNeRFDataset
from models.navier_cauchy import NavierCauchy


# -------------------------------------
# Load configuration
# -------------------------------------
object_name = 'bird'
cfg_fname = f'configs/{object_name}.yaml'
cfg.merge_from_file(cfg_fname)

# -------------------------------------
# Dataset & DataLoader
# -------------------------------------

dataset = PACNeRFDataset(
    dataroot="dataset/pac-nerf",
    instance=object_name,
    verbose=True,
    max_frames=14
)
num_samples = 20000

loader = DataLoader(
    dataset,
    batch_size=14,
    shuffle=True,
)

DEVICE = torch.device('cuda', 0)

# -------------------------------------
# PINN Model
# -------------------------------------

model = NavierCauchy(
    hid_dim=128,
    depth=8,

    # Actual physical parameters (or initial values)
    density = cfg.ELASTOMER.DENSITY,    # kg m⁻³
    youngs  = cfg.ELASTOMER.YOUNGS,     # Pa
    poissons= cfg.ELASTOMER.POISSONS,
    
    # Environment
    ground_pos = 0,                    # The y-coord of the ground
    up_index   = dataset.up_index,     # The y axis will be the gravity direction
    gravity    = 9.80665,              # The gravitational acceleration
    
    # Training options
    optimize_properties = False,       # Indicates whether optimize ρ, E, ν
    activation = nn.Tanh               # The activation function type of the MLP
).to(DEVICE)

# -------------------------------------
# Optimizers & Schedulers
# -------------------------------------
n_epochs = 100_000

optimizers = [
    optim.Adam(
        model.network_parameters(),
        lr=1e-4 ,
    ), 
    optim.Adam(
        model.property_parameters(),
        lr=1e-1,
    ), 
]

schedulers = [
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=0,
    ) for optimizer in optimizers
]

# -------------------------------------
# Training Loop
# -------------------------------------

# The loss weights
loss_weight_pde: float = cfg.ELASTOMER.LOSS.PDE
loss_weight_gt: float = cfg.ELASTOMER.LOSS.GT
loss_weight_bc: float = cfg.ELASTOMER.LOSS.BC
loss_weight_ic: float = cfg.ELASTOMER.LOSS.IC

# Logger
loss_history = []
loss_history_detailed = {}
prop_history = {
    'density': [model.density.data.item()],
    'youngs': [model.youngs.data.item()],
    'poissons': [model.poissons.data.item()],
}
best_loss = torch.inf

# The training loop
for epoch in range(n_epochs):
    epoch_loss = 0.0
    epoch_loss_detailed = {}

    for sample in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        
        # input preparation
        num_timesteps, num_points, _ = sample['geometry'].shape

        geometry = sample["geometry"].flatten(0, 1)  # [n_samples, 3]
        sampling_indices = torch.randperm(len(geometry) - 1)[:num_samples]
        
        geometry = geometry[sampling_indices].to(DEVICE)
        displacement = sample['displacement'].flatten(0, 1)[sampling_indices].to(DEVICE)
        time_value = sample['time'][:, 0:1].expand(num_timesteps, num_points).flatten(0, 1)

        t_values = time_value[sampling_indices, None].to(DEVICE)
        xyzt = torch.cat([geometry, t_values], dim=1)  # [N_points, 4]

        # forward
        losses = model.compute_loss(xyzt, displacement=displacement)
        losses: dict[str, torch.Tensor] = {
            'pde_loss': losses['pde_loss'] * loss_weight_pde,
            'gt_loss': losses['gt_loss'] * loss_weight_gt,
            'bc_loss': losses['bc_loss'] * loss_weight_bc,
            'ic_loss': losses['ic_loss'] * loss_weight_ic,
        }

        # loss backward and gradient steps
        total_loss = sum(losses.values())
        total_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        # logging the losses
        epoch_loss += total_loss.clone().detach().cpu().item()
        for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()
                if key in epoch_loss_detailed:
                    epoch_loss_detailed[key] += val
                else:
                    epoch_loss_detailed[key] = val

    for scheduler in schedulers:
        scheduler.step(epoch_loss)

    # logging the losses 
    loss_history.append(epoch_loss)
    for key, val in epoch_loss_detailed.items():
        if key in loss_history_detailed:
            loss_history_detailed[key].append(val)
        else:
            loss_history_detailed[key] = [val]

    # logging the physical parameters
    prop_history['youngs'].append(model.youngs.data.item())
    prop_history['poissons'].append(model.poissons.data.item())
    prop_history['density'].append(model.density.data.item())
    
    # save the checkpoints to the file(s)
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
        torch.save(ckpt,  'best.pt')
    torch.save(ckpt, 'last.pt')

    print(
        f"Epoch {epoch+1}", 
        f"Loss: {epoch_loss:.6f}", 
        f"ρ: {model.density.item():.2E}", 
        f"ν: {model.poissons.item():.2E}", 
        f"E: {model.youngs.item():.2E}", 
        sep=' ',
    )
