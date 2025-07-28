import torch
from torch import nn, autograd
# from .mlp import MLP 
from .mlp import GlobalMLP as MLP 
# from .time_attn_mlp import TimeAttnMLP as MLP 
from typing import Any, Dict  


class NavierCauchy(MLP):
    def __init__(
        self,
        hid_dim: int = 200, depth: int = 6,
        density: float = 1.0e3,
        youngs: float = 1.0e6,
        poissons: float = 0.30,
        ground_pos: float = 0.0,      # the ground will be at `y = ground_pos`
        gravity: float = 9.8,         # positive sign for downward
        optimize_properties: bool = False,
        activation = nn.Tanh,
        up_index: int = 1             # 0:x, 1:y, 2:z -> suppose `y` is the vertical axis`
    ):
        super().__init__(             # MLP intiialization
            in_dim = 4,               # (x,y,z,t)
            hid_dim = hid_dim,
            out_dim = 3,              # (u,v,w)
            depth = depth,
            activation = activation,
        )

        self.config: Dict[str, Any] = dict(
            hid_dim=hid_dim, 
            depth=depth,
            density=density, youngs=youngs, poissons=poissons,
            ground_pos=ground_pos, gravity=gravity,
            optimize_properties=optimize_properties,
             up_index=up_index,
        )

        # Log parameterization for positive values
        self.log_density  = nn.Parameter(torch.log(torch.tensor(density)),
                                         requires_grad = optimize_properties)
        self.log_youngs   = nn.Parameter(torch.log(torch.tensor(youngs)),
                                         requires_grad = optimize_properties)
        self.log_poissons = nn.Parameter(torch.log(torch.tensor(poissons)),
                                         requires_grad = optimize_properties)

        self.ground_pos  = ground_pos
        self.gravity     = gravity
        self.up_index    = up_index

    # 재료 상수 property
    @property
    def density(self):  return torch.exp(self.log_density)
    @property
    def youngs(self):   return torch.exp(self.log_youngs)
    @property
    def poissons(self): return torch.exp(self.log_poissons)

    def network_parameters(self):
        """ Returns only the MLP parameters. """
        yield from MLP.parameters(self)

    def property_parameters(self):
        """ Returns the material properties parameters only. """
        yield self.log_density
        yield self.log_youngs
        yield self.log_poissons
    
    def compute_stress_tensor(self, xyzt: torch.Tensor):
        """
        Caluclates the Cauchy stress tensor for the given input points.
        It contains 6 independent components: `(sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx)` because the stress tensor is symmetric.
        """        
        # 1. MLP forward
        uvw = self.forward(xyzt)
        u, v, w = uvw[:, 0:1], uvw[:, 1:2], uvw[:, 2:3]

        # 2. First-order spatial derivatives
        grad_u = autograd.grad(outputs=u, inputs=xyzt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx, du_dy, du_dz, du_dt = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2], grad_u[:, 3]

        # Differentiation of v with respect to xyzt.
        grad_v = autograd.grad(outputs=v, inputs=xyzt, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dx, dv_dy, dv_dz, dv_dt = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2], grad_v[:, 3]

        # Differentiation of w with respect to xyzt.
        grad_w = autograd.grad(outputs=w, inputs=xyzt, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        dw_dx, dw_dy, dw_dz, dw_dt = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2], grad_w[:, 3]
        
        # 3. Calculate the strain tensor.
        # epsilon_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        epsilon_xx = du_dx
        epsilon_yy = dv_dy
        epsilon_zz = dw_dz
        
        epsilon_xy = 0.5 * (du_dy + dv_dx)
        epsilon_yz = 0.5 * (dv_dz + dw_dy)
        epsilon_zx = 0.5 * (dw_dx + du_dz)
        
        # Volumetric strain
        div_u = epsilon_xx + epsilon_yy + epsilon_zz

        # 4. Lamé parameters
        lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
        mu = self.youngs / (2 * (1 + self.poissons))
        
        # 5. Calculate the stress tensor.
        # σ = λ * tr(ε) * I + 2μ * ε
        sigma_xx = lmbda * div_u + 2 * mu * epsilon_xx
        sigma_yy = lmbda * div_u + 2 * mu * epsilon_yy
        sigma_zz = lmbda * div_u + 2 * mu * epsilon_zz
        
        sigma_xy = 2 * mu * epsilon_xy
        sigma_yz = 2 * mu * epsilon_yz
        sigma_zx = 2 * mu * epsilon_zx
        
        # Returns the six components as a dictionary.
        stress = {
            'sigma_xx': sigma_xx, 'sigma_yy': sigma_yy, 'sigma_zz': sigma_zz,
            'sigma_xy': sigma_xy, 'sigma_yz': sigma_yz, 'sigma_zx': sigma_zx
        }
        
        return stress

    def compute_loss(
        self,
        xyzt: torch.Tensor,
        time_dim: int, point_dim: int,
        displacement: torch.Tensor = None,
        time: torch.Tensor = None,
        f_ext: torch.Tensor = None,
        use_pde: bool = True,
        use_ic: bool = True,
        use_bc: bool = True,
        use_vel: bool = False,
    ) -> dict[str, torch.Tensor]:
        """ Calculates the loss for the Navier-Cauchy equation. """
        
        kwargs = {'device': xyzt.device, 'dtype': xyzt.dtype}
        xyzt.requires_grad_(True)
        
        # 1. MLP forward
        uvw = self.forward(xyzt)
        u, v, w = torch.unbind(uvw, dim=-1)


        # 2. autograds (u, v, w) with respect to xyzt.
        grad_u = autograd.grad(outputs=u, inputs=xyzt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_u = grad_u.reshape(-1, 4)
        du_dx, du_dy, du_dz, du_dt = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2], grad_u[:, 3]

        grad_v = autograd.grad(outputs=v, inputs=xyzt, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        grad_v = grad_v.reshape(-1, 4)
        dv_dx, dv_dy, dv_dz, dv_dt = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2], grad_v[:, 3]

        grad_w = autograd.grad(outputs=w, inputs=xyzt, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        grad_w = grad_w.reshape(-1, 4)
        dw_dx, dw_dy, dw_dz, dw_dt = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2], grad_w[:, 3]
        
        du_dt2 = autograd.grad(outputs=du_dt, inputs=xyzt, grad_outputs=torch.ones_like(du_dt), create_graph=True)[0][:, 3]
        dv_dt2 = autograd.grad(outputs=dv_dt, inputs=xyzt, grad_outputs=torch.ones_like(dv_dt), create_graph=True)[0][:, 3]
        dw_dt2 = autograd.grad(outputs=dw_dt, inputs=xyzt, grad_outputs=torch.ones_like(dw_dt), create_graph=True)[0][:, 3]


        # 3. PDE Loss (residual of the Navier-Cauchy equation)
        pde_loss = torch.tensor(0.0, **kwargs)
        if use_pde:
            lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
            mu = self.youngs / (2 * (1 + self.poissons))

            # The stress divergence term in Navier-Cauchy equation requires the second-order spatial derivatives.
            div_u = du_dx + dv_dy + dw_dz
            
            grad_div_u = autograd.grad(div_u, xyzt, torch.ones_like(div_u), create_graph=True)[0]
            d_div_u_dx = grad_div_u[:, 0]
            d_div_u_dy = grad_div_u[:, 1]
            d_div_u_dz = grad_div_u[:, 2]

            lap_u = autograd.grad(du_dx, xyzt, torch.ones_like(du_dx), create_graph=True)[0][:, 0] + \
                    autograd.grad(du_dy, xyzt, torch.ones_like(du_dy), create_graph=True)[0][:, 1] + \
                    autograd.grad(du_dz, xyzt, torch.ones_like(du_dz), create_graph=True)[0][:, 2]

            lap_v = autograd.grad(dv_dx, xyzt, torch.ones_like(dv_dx), create_graph=True)[0][:, 0] + \
                    autograd.grad(dv_dy, xyzt, torch.ones_like(dv_dy), create_graph=True)[0][:, 1] + \
                    autograd.grad(dv_dz, xyzt, torch.ones_like(dv_dz), create_graph=True)[0][:, 2]

            lap_w = autograd.grad(dw_dx, xyzt, torch.ones_like(dw_dx), create_graph=True)[0][:, 0] + \
                    autograd.grad(dw_dy, xyzt, torch.ones_like(dw_dy), create_graph=True)[0][:, 1] + \
                    autograd.grad(dw_dz, xyzt, torch.ones_like(dw_dz), create_graph=True)[0][:, 2]

            # The Navier-Cauchy equation residuals
            pde_x = self.density * du_dt2 - ( (lmbda + mu) * d_div_u_dx + mu * lap_u )
            pde_y = self.density * dv_dt2 - ( (lmbda + mu) * d_div_u_dy + mu * lap_v )
            pde_z = self.density * dw_dt2 - ( (lmbda + mu) * d_div_u_dz + mu * lap_w )

            # The gravity term.
            pde_y -= self.density * self.gravity
            
            if f_ext is not None:
                pde_x -= f_ext[:, 0]
                pde_y -= f_ext[:, 1]
                pde_z -= f_ext[:, 2]

            pde_loss = torch.mean(pde_x**2 + pde_y**2 + pde_z**2)

        # 4. Initial Condition (IC) Loss

        ic_mask = xyzt[..., 3] == 0.0
        if use_ic and (ic_mask.sum() > 0):
            if xyzt.ndim == 2:
                ic_u = self.forward(xyzt[ic_mask]).reshape(-1, 3)
            elif xyzt.ndim == 3:
                ic_u = self.forward(xyzt).reshape(time_dim, point_dim, -1)[ic_mask]
            u0 = torch.zeros_like(ic_u)
            ic_loss = torch.nn.functional.mse_loss(ic_u, u0)
        else:
            ic_loss = torch.tensor(0.0, **kwargs)

        # 5. Boundary Condition (BC) Loss
        
        bc_loss = torch.tensor(0.0, **kwargs)
        if use_bc:
            # 5.1. y-coordinate of the current position
            # Initial y + y-direction displacement
            y_initial = xyzt[..., self.up_index].reshape(-1, 1)
            y_current = y_initial + v

            # 5.2. Penetration Loss
            penetration = torch.relu(self.ground_pos - y_current)
            loss_penetration = torch.mean(penetration**2) 

            # 5.3. Calculate the orthogonal stress in the y-direction.
            lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
            mu = self.youngs / (2 * (1 + self.poissons))
            div_u = du_dx + dv_dy + dw_dz  # strain: How much the surrounding space has moved due to the displacement.
            epsilon_yy = dv_dy
            sigma_y = lmbda * div_u + 2 * mu * epsilon_yy

            # 5.4. Complementarity and stress conditions:
            # C1: When `y_current ≈ ground_pos`, the stress must be compression.
            contact_mask = (y_current <= self.ground_pos + 1e-6).float() 
            loss_tensile_stress = torch.mean((contact_mask * torch.relu(sigma_y))**2)

            # C2: When `y_current > ground_pos`, the stress must be zero.
            no_contact_mask = (y_current > self.ground_pos + 1e-6).float()
            loss_no_contact_stress = torch.mean((no_contact_mask * sigma_y)**2)
            
            # The final BC loss is the sum of the terms.
            bc_loss = loss_penetration + loss_tensile_stress + loss_no_contact_stress
            
        # 6. Velocity-Driven GT Loss
        if use_vel and (displacement is not None) and (time is not None):
            uvw_unflat = u.reshape(time_dim, point_dim, -1)
            disp_unflat = displacement.reshape(time_dim, point_dim, -1)
            time_unflat = time.reshape(time_dim, point_dim, -1)
            delta_time = time_unflat[+1:] - time_unflat[:-1]
            vel_unflat = (uvw_unflat[+1:] - uvw_unflat[:-1]) / delta_time
            vel_gt_unflat = (disp_unflat[+1:] - disp_unflat[:-1]) / delta_time
            l1_loss = torch.mean((vel_unflat - vel_gt_unflat).abs())
            l2_loss = torch.mean((vel_unflat - vel_gt_unflat).square())
            vel_loss = l2_loss + l1_loss
        else:
            vel_loss = torch.tensor(0.0, **kwargs)

        # 7. Ground Truth (GT) Loss 
        gt_loss = torch.tensor(0.0, **kwargs)
        if displacement is not None:
            if xyzt.ndim == 2:
                uvw_unflat = u
            elif xyzt.ndim == 3:
                uvw_unflat = u.reshape(time_dim, point_dim, -1)
            l1_loss = torch.mean((uvw_unflat - displacement).abs())
            l2_loss = torch.mean((uvw_unflat - displacement).square())
            gt_loss = l2_loss + l1_loss


        return {
            "pde_loss":      pde_loss,
            "ic_loss":       ic_loss,
            "vel_loss":      vel_loss,
            "gt_loss":       gt_loss,
            "bc_loss" :      bc_loss,
        }



if __name__ == "__main__":
    model = NavierCauchy()
    xyz  = torch.randn(10, 512, 3)
    t    = torch.linspace(0, 1, 10).view(10, 1, 1).repeat(1, 512, 1)
    xyzt = torch.cat([xyz, t], dim = 2)

    loss = model.compute_loss(xyzt)
    print({k:f"{v.item():.3e}" for k,v in loss.items()})
