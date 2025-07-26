from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs.config import CFG as cfg
from data.pac_nerf import PACNeRFDataset
from models.navier_cauchy import NavierCauchy
from utils.logging import (
    Averager,
    CheckpointWriter,
)


# -------------------------------------
# Load configuration
# -------------------------------------
parser = ArgumentParser('Navier-Cauchy')
add = parser.add_argument
add('--device', '-d', type=int, default=0)
add('--object', '-o', type=str, default='bird')
add('--num-frames', '-nf', type=int, default=14)
add('--batch-size', '-bs', type=int, default=20000)
add('--learning-rate', '-lr', type=float, default=1.0E-4)
add('--epochs', '-e', type=int, default=100_000)
add('--tag', type=str, default=None)
args = parser.parse_args()

# -------------------------------------
# Load configuration
# -------------------------------------
object_name = args.object
cfg_fname = f'configs/{object_name}.yaml'
cfg.merge_from_file(cfg_fname)

if torch.cuda.is_available():
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda', args.device)

# -------------------------------------
# Dataset & DataLoader
# -------------------------------------

num_frames = args.num_frames
num_samples = args.batch_size // num_frames

dataset = PACNeRFDataset(
    dataroot="dataset/pac-nerf",
    instance=object_name,
    verbose=True,
    max_frames=num_frames
)
loader = DataLoader(
    dataset,
    batch_size=num_frames,
    shuffle=True,
)

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
n_epochs = args.epochs

optimizers = [
    optim.Adam(
        model.network_parameters(),
        lr=args.learning_rate,
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
loss_history = Averager()
loss_history_detailed = Averager()
prop_history = Averager()
prop_history.push({
    'density': [model.density.data.item()],
    'youngs': [model.youngs.data.item()],
    'poissons': [model.poissons.data.item()],
}, flush=True)
ckpt_writer = CheckpointWriter(
    dir_name=f'./output/{object_name}' if args.tag else f'./output/{object_name}_{args.tag}',
    save_first=False,
    save_every=0,
    save_best=True,
    save_last=True,
    larger_better=False,
)

# The training loop
for epoch in range(n_epochs):
    
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
        loss_history_detailed.push(losses)

    # logging the losses 
    epoch_loss_detailed = loss_history_detailed.flush()
    epoch_loss = sum(epoch_loss_detailed.values())

    # scheduler step w.r.t. to the total loss
    for scheduler in schedulers:
        scheduler.step(epoch_loss)

    # logging the physical parameters and total loss
    loss_history.push(epoch_loss_detailed, flush=True)
    prop_history.push({
        'youngs': model.youngs.data,
        'poissons': model.poissons.data,
        'density': model.density.data,
    }, flush=True)
    
    # save the checkpoints to the file(s)
    ckpt_writer.write({
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
    })

    print(
        f"Epoch {epoch+1}", 
        f"Loss: {epoch_loss:.6f}", 
        f"ρ: {model.density.item():.2E}", 
        f"ν: {model.poissons.item():.2E}", 
        f"E: {model.youngs.item():.2E}", 
        sep=' ',
    )
