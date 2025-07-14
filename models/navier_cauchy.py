import torch
from torch import nn, autograd
from .mlp import MLP


class NavierCauchy(MLP):
    def __init__(
        self,
        hid_dim: int=200, depth: int=6,
        density: float=1.0E+3, 
        youngs: float=1.0E+6, 
        poissons: float=0.3,
        ground_pos: float=0.0,
        optimize_properties: bool=True,
        activation=nn.Tanh,
        up_index: int=1,
    ):
        MLP.__init__(
            self,
            in_dim=4,
            hid_dim=hid_dim,
            out_dim=3,
            depth=depth,
            activation=activation,
        )
        self.config = dict(
            hid_dim=hid_dim,
            depth=depth,
            density=density,
            youngs=youngs,
            poissons=poissons,
            ground_pos=ground_pos,
            optimize_properties=optimize_properties,
            up_index=up_index,
        )
        
        self.log_density = nn.Parameter(torch.tensor(density).log(), requires_grad=optimize_properties)
        self.log_youngs = nn.Parameter(torch.tensor(youngs).log(), requires_grad=optimize_properties)
        self.log_poissons = nn.Parameter(torch.tensor(poissons).log(), requires_grad=optimize_properties)
        self.ground_pos = ground_pos
        self.up_index = up_index
        
    @property
    def density(self):
        return torch.exp(self.log_density)

    @property
    def youngs(self):
        return torch.exp(self.log_youngs)

    @property
    def poissons(self):
        return torch.exp(self.log_poissons)
    
    def network_parameters(self):
        yield from MLP.parameters(self)

    def property_parameters(self):
        yield self.log_density
        yield self.log_youngs
        yield self.log_poissons
    
    def compute_loss(
        self,
        xyzt: torch.Tensor,
        f: torch.Tensor=None,
        displacement: torch.Tensor=None,
        pde: bool=True,
        ic: bool=True,
        bc: bool=True,
        po: bool=True,
    ) -> dict[str, torch.Tensor]:
        xyzt = xyzt.flatten(0, -2)
        device = xyzt.device
        num_points = xyzt.size(0)
        tensor_kwargs = dict(device=device, dtype=xyzt.dtype)
            
        if f is None:
            f = torch.zeros(num_points, 3, **tensor_kwargs)
            f[:, self.up_index].fill_(-9.8)
        
        if pde:
            xyzt_ = xyzt.clone().detach().requires_grad_(True)
            
            u = self.forward(xyzt_)
            ux, uy, uz = torch.chunk(u, 3, dim=-1)

            # Gradient wrt spatial coordinates
            grads = []
            for u_comp in [ux, uy, uz]:
                grad_u = autograd.grad(
                    u_comp, xyzt_,
                    grad_outputs=torch.ones_like(u_comp),
                    retain_graph=True,
                    create_graph=True,
                )[0]
                grads.append(grad_u[:, :3])

            # grad_u = [du/dx, du/dy, du/dz, du/dt]
            # Stack grads for tensor operations
            # grads = [∇ux, ∇uy, ∇uz]
            # Each ∇u ∈ [N, 4]

            # Compute strain tensor ε_ij = 0.5*(∂ui/∂xj + ∂uj/∂xi)
            eps = torch.zeros(num_points, 3, 3, device=device)
            for i in range(3):
                for j in range(3):
                    eps[:, i, j] = 0.5 * (grads[i][:, j] + grads[j][:, i])

            # Compute divergence of stress tensor
            mu = self.youngs / (2.0 * (1.0 + self.poissons))
            lam = self.youngs * self.poissons / ((1.0 + self.poissons)*(1.0 - 2.0 * self.poissons))
            div_sigma = torch.zeros(num_points, 3, device=device)
            for i in range(3):
                # σ_ij = λ*tr(ε) δ_ij + 2μ*ε_ij
                eps_trace = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]
                sigma_ij = lam * eps_trace[:, None] * (i == torch.arange(3, device=device)).float() + 2 * mu * eps[:, i, :]
                # Compute divergence ∂σ_ij/∂xj
                div = torch.zeros(num_points, device=device)
                for j in range(3):
                    d_sigma = autograd.grad(
                        sigma_ij[:,j], xyzt_,
                        grad_outputs=torch.ones_like(sigma_ij[:,j]),
                        retain_graph=True,
                        create_graph=True
                    )[0][:, j]   # derive w.r.t. x_j
                    div += d_sigma
                div_sigma[:, i] = div

            # Time acceleration
            dudt = grads[0][:, 3:4]  # dux/dt
            dvdt = grads[1][:, 3:4]  # duy/dt
            dwdt = grads[2][:, 3:4]  # duz/dt

            d2ux_dt2 = autograd.grad(
                dudt, xyzt_,
                grad_outputs=torch.ones_like(dudt),
                retain_graph=True,
                create_graph=True
            )[0][:, 3]

            d2uy_dt2 = autograd.grad(
                dvdt, xyzt_,
                grad_outputs=torch.ones_like(dvdt),
                retain_graph=True,
                create_graph=True
            )[0][:, 3]

            d2uz_dt2 = autograd.grad(
                dwdt, xyzt_,
                grad_outputs=torch.ones_like(dwdt),
                retain_graph=True,
                create_graph=True
            )[0][:, 3]

            acceleration = torch.stack([d2ux_dt2, d2uy_dt2, d2uz_dt2], dim=1)

            residual = self.density * acceleration - div_sigma - f
            pde_loss = torch.mean(torch.square(residual))
        else:
            pde_loss = torch.tensor(0.0, **tensor_kwargs)
    
        # Initial condition
        ic_mask = xyzt[:, 3] == 0.0
        if ic and (ic_mask.sum() > 0):
            ic_xyzt = xyzt[ic_mask]
            ic_u = self.forward(ic_xyzt)
            u0 = torch.zeros_like(ic_u)
            ic_loss = torch.nn.functional.mse_loss(ic_u, u0)
        else:
            ic_loss = torch.tensor(0.0, **tensor_kwargs)

        # Boundary condition
        if bc:
            bc_xyzt = xyzt.clone()
            bc_u = self.forward(bc_xyzt)
            y_deformed = bc_xyzt[:, self.up_index] + bc_u[:, self.up_index]
            ground_margin = y_deformed - self.ground_pos
            bc_loss = torch.mean(torch.square(torch.relu(-ground_margin))) \
                    + torch.mean(torch.square(torch.relu(ground_margin)))
        else:
            bc_loss = torch.tensor(0.0, **tensor_kwargs)
        
        if po:
            # Gravity-based pseudo-observation
            u = self.forward(xyzt)
            u_mean = torch.mean(u, dim=0)

            t = xyzt[:, 3:4]
            u_rigid = -0.5 * f * torch.square(t)
            po_loss = torch.nn.functional.mse_loss(
                u_mean, u_rigid,
            )
        else:
            po_loss = torch.tensor(0.0, **tensor_kwargs)
        
        if displacement is not None:
            u = self.forward(xyzt)
            gt_loss = torch.nn.functional.mse_loss(u, displacement)
        else:
            gt_loss = torch.tensor(0.0, **tensor_kwargs)

        return {
            'pde_loss': pde_loss,
            'ic_loss': ic_loss,
            'bc_loss': bc_loss,
            'po_loss': po_loss,
            'gt_loss': gt_loss,
        }


if __name__ == '__main__':
    model = NavierCauchy()
    xyzt = torch.cat([
        torch.randn(10, 512, 3),
        torch.linspace(0, 1, 10).reshape(10, 1, 1).repeat(1, 512, 1)
    ], dim=2)
    loss_dict = model.compute_loss(xyzt)
    print(loss_dict)
