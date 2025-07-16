import torch
from torch import nn, autograd
from .mlp import MLP           # mlp.py 그대로 재사용
from typing import Optional, Any, Dict  

class NavierCauchyGround(MLP):
    def __init__(
        self,
        hid_dim: int = 200, depth: int = 6,
        density: float = 1.0e3,
        youngs: float = 1.0e6,
        poissons: float = 0.30,
        ground_pos: float = 0.0,      # y = ground_pos 가 지면
        gravity: float = 9.8,         # [+] = 아래로
        optimize_properties: bool = True,
        activation = nn.Tanh,
        up_index: int = 1             # 0:x, 1:y, 2:z  ‑-> y축을 “위”로 가정
    ):
        super().__init__(               # MLP 초기화
            in_dim = 4,                # (x,y,z,t)
            hid_dim = hid_dim,
            out_dim = 3,               # (u,v,w)
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

        # 로그-파라미터(양수 보장) ― 옵션에 따라 학습 대상
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
        """MLP(가중치·편향)만 반환"""
        yield from MLP.parameters(self)          # 부모 모듈의 parameters()

    def property_parameters(self):
        """ρ, E, ν 로그 파라미터만 반환"""
        yield self.log_density
        yield self.log_youngs
        yield self.log_poissons

    # ------------------------------------------------------------------ #
    # 손실 계산
    # ------------------------------------------------------------------ #
    def compute_loss(
        self,
        xyzt: torch.Tensor,
        displacement: torch.Tensor = None,
        f_ext: torch.Tensor = None,
        use_pde: bool = True,
        use_ic: bool = True,
        use_bc: bool = True,
        use_contact: bool = True,      # Signorini
    ) -> dict[str, torch.Tensor]:
        """
        주어진 입력에 대한 손실 함수들을 계산합니다.
        """
        
        kwargs = {'device': xyzt.device, 'dtype': xyzt.dtype}
        xyzt.requires_grad_(True)
        
        # 1. 신경망 순전파
        uvw = self.forward(xyzt)
        u, v, w = uvw[:, 0:1], uvw[:, 1:2], uvw[:, 2:3] # 슬라이싱으로 차원 유지
        
        # 2. 자동 미분을 이용한 각 변수에 대한 미분 계산 (수정된 부분)
        # 각 출력(u, v, w)에 대해 개별적으로 그래디언트를 계산합니다.
        
        # u에 대한 1차 미분, autograd.grad 는 outputs 이 미분 대상 Y, input이 미분의 변수 X dY/dX 
        grad_u = torch.autograd.grad(outputs=u, inputs=xyzt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx, du_dy, du_dz, du_dt = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2], grad_u[:, 3]

        # v에 대한 1차 미분
        grad_v = torch.autograd.grad(outputs=v, inputs=xyzt, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dx, dv_dy, dv_dz, dv_dt = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2], grad_v[:, 3]

        # w에 대한 1차 미분
        grad_w = torch.autograd.grad(outputs=w, inputs=xyzt, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        dw_dx, dw_dy, dw_dz, dw_dt = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2], grad_w[:, 3]
        
        # 2차 시간 미분 계산
        du_dt2 = torch.autograd.grad(outputs=du_dt, inputs=xyzt, grad_outputs=torch.ones_like(du_dt), create_graph=True)[0][:, 3]
        dv_dt2 = torch.autograd.grad(outputs=dv_dt, inputs=xyzt, grad_outputs=torch.ones_like(dv_dt), create_graph=True)[0][:, 3]
        dw_dt2 = torch.autograd.grad(outputs=dw_dt, inputs=xyzt, grad_outputs=torch.ones_like(dw_dt), create_graph=True)[0][:, 3]

        # 3. PDE Loss (나비에-코시 방정식 잔차)
        pde_loss = torch.tensor(0.0, **kwargs)
        if use_pde:
            lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
            mu = self.youngs / (2 * (1 + self.poissons))

            # 나비에-코시 방정식의 응력 발산 항을 계산하기 위해 2차 공간 미분이 필요합니다.
            # div(sigma) 계산
            div_u = du_dx + dv_dy + dw_dz
            
            grad_div_u = torch.autograd.grad(div_u, xyzt, torch.ones_like(div_u), create_graph=True)[0]
            d_div_u_dx = grad_div_u[:, 0]
            d_div_u_dy = grad_div_u[:, 1]
            d_div_u_dz = grad_div_u[:, 2]

            lap_u = torch.autograd.grad(du_dx, xyzt, torch.ones_like(du_dx), create_graph=True)[0][:, 0] + \
                    torch.autograd.grad(du_dy, xyzt, torch.ones_like(du_dy), create_graph=True)[0][:, 1] + \
                    torch.autograd.grad(du_dz, xyzt, torch.ones_like(du_dz), create_graph=True)[0][:, 2]

            lap_v = torch.autograd.grad(dv_dx, xyzt, torch.ones_like(dv_dx), create_graph=True)[0][:, 0] + \
                    torch.autograd.grad(dv_dy, xyzt, torch.ones_like(dv_dy), create_graph=True)[0][:, 1] + \
                    torch.autograd.grad(dv_dz, xyzt, torch.ones_like(dv_dz), create_graph=True)[0][:, 2]

            lap_w = torch.autograd.grad(dw_dx, xyzt, torch.ones_like(dw_dx), create_graph=True)[0][:, 0] + \
                    torch.autograd.grad(dw_dy, xyzt, torch.ones_like(dw_dy), create_graph=True)[0][:, 1] + \
                    torch.autograd.grad(dw_dz, xyzt, torch.ones_like(dw_dz), create_graph=True)[0][:, 2]

            # 나비에-코시 방정식 잔차
            pde_x = self.density * du_dt2 - ( (lmbda + mu) * d_div_u_dx + mu * lap_u )
            pde_y = self.density * dv_dt2 - ( (lmbda + mu) * d_div_u_dy + mu * lap_v )
            pde_z = self.density * dw_dt2 - ( (lmbda + mu) * d_div_u_dz + mu * lap_w )

            # 중력 항 추가 
            pde_y -= self.density * self.gravity
            
            if f_ext is not None:
                pde_x -= f_ext[:, 0]
                pde_y -= f_ext[:, 1]
                pde_z -= f_ext[:, 2]

            pde_loss = torch.mean(pde_x**2 + pde_y**2 + pde_z**2)

        # 4. Initial Condition (IC) Loss
        if use_ic:
            ic_mask = (xyzt[:, 3] == 0)
            ic_displacement_loss = torch.mean(uvw[ic_mask]**2) # 4.1. 시작할 때 변위 = 0 
            ic_velocity_loss = torch.mean(du_dt[ic_mask]**2 + dv_dt[ic_mask]**2 + dw_dt[ic_mask]**2) # 4.2. 시작할 때 속도 = 0
            ic_loss = ic_displacement_loss + ic_velocity_loss
        else : 
            ic_loss = torch.tensor(0.0, **kwargs)

        # 5. Boundary Condition (BC) Loss
        
        if use_bc:
            bc_loss = torch.tensor(0.0, **kwargs)
        else : 
            bc_loss = torch.tensor(0.0, **kwargs)

        # 6. Contact Loss
        contact_loss = torch.tensor(0.0, **kwargs)
        if use_contact:
            # 라메 상수를 여기서 다시 계산하거나 위에서 계산한 값을 사용
            lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
            mu = self.youngs / (2 * (1 + self.poissons))
            
            y_pos = xyzt[:, self.up_index] + v.squeeze()
            
            penetration = torch.relu(self.ground_pos - y_pos)
            loss_penetration = torch.mean(penetration**2)
            
            epsilon_yy = dv_dy
            div_u = du_dx + dv_dy + dw_dz
            sigma_y = lmbda * div_u + 2 * mu * epsilon_yy
            
            # 접촉 시(y_pos가 ground_pos에 가까울 때) 응력이 양수(인장)가 되면 페널티
            loss_stress = torch.mean(torch.relu(sigma_y * (y_pos < self.ground_pos + 1e-6))**2)
            
            # 비접촉 시(y_pos가 ground_pos보다 클 때) 응력이 0이 아니면 페널티
            loss_no_contact_stress = torch.mean((sigma_y * (y_pos > self.ground_pos + 1e-6))**2)
            
            contact_loss = loss_penetration + loss_stress + loss_no_contact_stress

        # 7. Ground Truth (GT) Loss
        if displacement is not None:
            gt_loss = torch.mean((uvw - displacement)**2)
        else: 
            gt_loss = torch.tensor(0.0, **kwargs)

        return {
            "pde_loss":      pde_loss,
            "ic_loss":       ic_loss,
            "bc_loss":       bc_loss,
            "contact_loss":  contact_loss,
            "gt_loss":       gt_loss,
        }



if __name__ == "__main__":
    model = NavierCauchyGround()
    # 10 단계(t) × 512 점(x,y,z)
    xyz  = torch.randn(10, 512, 3)
    t    = torch.linspace(0, 1, 10).view(10, 1, 1).repeat(1, 512, 1)
    xyzt = torch.cat([xyz, t], dim = 2)

    loss = model.compute_loss(xyzt)
    print({k:f"{v.item():.3e}" for k,v in loss.items()})
