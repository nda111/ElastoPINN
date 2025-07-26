import torch
from torch import nn, autograd
import torch.nn.functional as F
# from .mlp import MLP           # mlp.py 그대로 재사용
from .mlp import GlobalMLP as MLP           # mlp.py 그대로 재사용
from typing import Optional, Any, Dict  


class NavierCauchy(MLP):
    def __init__(
        self,
        hid_dim: int = 200, depth: int = 6,
        density: float = 1.0e3,
        youngs: float = 1.0e6,
        poissons: float = 0.30,
        ground_pos: float = 0.0,      # y = ground_pos 가 지면
        gravity: float = 9.8,         # [+] = 아래로
        optimize_properties: bool = False,
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
    
    def compute_stress_tensor(self, xyzt: torch.Tensor):
        """
        주어진 입력 포인트(xyzt)에 대한 전체 응력 텐서(Cauchy stress tensor)를 계산합니다.
        응력 텐서는 대칭 행렬이므로 6개의 독립적인 성분을 반환합니다.
        (sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx)
        """
        # xyzt.requires_grad_(False)
        
        # 1. 신경망 순전파
        uvw = self.forward(xyzt)
        u, v, w = uvw[:, 0:1], uvw[:, 1:2], uvw[:, 2:3]

        # 2. 1차 공간 미분 계산
        grad_u = torch.autograd.grad(outputs=u, inputs=xyzt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx, du_dy, du_dz, du_dt = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2], grad_u[:, 3]

        # v에 대한 1차 미분
        grad_v = torch.autograd.grad(outputs=v, inputs=xyzt, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dx, dv_dy, dv_dz, dv_dt = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2], grad_v[:, 3]

        # w에 대한 1차 미분
        grad_w = torch.autograd.grad(outputs=w, inputs=xyzt, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        dw_dx, dw_dy, dw_dz, dw_dt = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2], grad_w[:, 3]
        
        # 3. 변형률(strain) 텐서의 성분 계산
        # epsilon_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        epsilon_xx = du_dx
        epsilon_yy = dv_dy
        epsilon_zz = dw_dz # 2D 문제인 경우 0으로 가정하거나 3D 미분 필요
        
        epsilon_xy = 0.5 * (du_dy + dv_dx)
        epsilon_yz = 0.5 * (dv_dz + dw_dy)
        epsilon_zx = 0.5 * (dw_dx + du_dz)
        
        # 체적 변형률 (Volumetric strain)
        div_u = epsilon_xx + epsilon_yy + epsilon_zz

        # 4. 라메 상수(Lamé parameters) 계산
        lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
        mu = self.youngs / (2 * (1 + self.poissons))
        
        # 5. 응력(stress) 텐서 계산 (σ = λ * tr(ε) * I + 2μ * ε)
        sigma_xx = lmbda * div_u + 2 * mu * epsilon_xx
        sigma_yy = lmbda * div_u + 2 * mu * epsilon_yy
        sigma_zz = lmbda * div_u + 2 * mu * epsilon_zz
        
        sigma_xy = 2 * mu * epsilon_xy
        sigma_yz = 2 * mu * epsilon_yz
        sigma_zx = 2 * mu * epsilon_zx
        
        # 6개의 응력 성분을 딕셔너리로 반환
        stress = {
            'sigma_xx': sigma_xx, 'sigma_yy': sigma_yy, 'sigma_zz': sigma_zz,
            'sigma_xy': sigma_xy, 'sigma_yz': sigma_yz, 'sigma_zx': sigma_zx
        }
        
        return stress

    # ------------------------------------------------------------------ #
    # 손실 계산
    # ------------------------------------------------------------------ #
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
        # if use_ic:
        #     ic_mask = (xyzt[:, 3] == 0)
        #     ic_displacement_loss = torch.mean(uvw[ic_mask]**2) # 4.1. 시작할 때 변위 = 0 
        #     ic_velocity_loss = torch.mean(du_dt[ic_mask]**2 + dv_dt[ic_mask]**2 + dw_dt[ic_mask]**2) # 4.2. 시작할 때 속도 = 0
        #     ic_loss = ic_displacement_loss + ic_velocity_loss
        # else : 
        #     ic_loss = torch.tensor(0.0, **kwargs)
        # Initial condition

        ic_mask = xyzt[:, 3] == 0.0
        if (ic_mask.sum() > 0):
            ic_xyzt = xyzt[ic_mask]
            ic_u = self.forward(ic_xyzt)
            u0 = torch.zeros_like(ic_u)
            ic_loss = torch.nn.functional.mse_loss(ic_u, u0)
        else:
            ic_loss = torch.tensor(0.0, **kwargs)

        # 5. Boundary Condition (BC) Loss
        
        bc_loss = torch.tensor(0.0, **kwargs)
        if use_bc:
            # 5.1. 현재 y 좌표 계산
            # 초기 y좌표 + y방향 변위
            y_initial = xyzt[:, self.up_index:self.up_index+1]
            y_current = y_initial + v

            # 5.2. 비침투 손실 (Penetration Loss)
            # 물체가 지면(ground_pos) 아래로 내려가면 페널티를 부과
            penetration = torch.relu(self.ground_pos - y_current)
            loss_penetration = torch.mean(penetration**2) 

            # 5.3. y방향 수직 응력(sigma_y) 계산
            lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
            mu = self.youngs / (2 * (1 + self.poissons))
            div_u = du_dx + dv_dy + dw_dz # strain : 주변 공간이 이동한 것에 대해서 얼마나 변위가 발생했는가 
            epsilon_yy = dv_dy
            sigma_y = lmbda * div_u + 2 * mu * epsilon_yy

            # 5.4. 보완성(Complementarity) 및 응력 조건 손실
            # 조건 1: 접촉 시(y_current ≈ ground_pos), 응력은 압축(sigma_y <= 0)이어야 한다.
            #   -> 인장 응력(양수)이 발생하면 페널티
            contact_mask = (y_current <= self.ground_pos + 1e-6).float() # 부동소수점 오차 고려
            loss_tensile_stress = torch.mean((contact_mask * torch.relu(sigma_y))**2)

            # 조건 2: 비접촉 시(y_current > ground_pos), 응력은 0이어야 한다.
            #   -> 0이 아닌 응력이 발생하면 페널티
            no_contact_mask = (y_current > self.ground_pos + 1e-6).float()
            loss_no_contact_stress = torch.mean((no_contact_mask * sigma_y)**2)
            
            
            # 최종 BC Loss는 각 조건들의 합
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
            l1_loss = torch.mean((uvw_unflat - displacement).abs())
            l2_loss = torch.mean((uvw_unflat - displacement).square())
            cosine_loss = torch.mean(1 - F.cosine_similarity(uvw, displacement, dim=-1))
            lambda_val = 1.0 
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
    # 10 단계(t) × 512 점(x,y,z)
    xyz  = torch.randn(10, 512, 3)
    t    = torch.linspace(0, 1, 10).view(10, 1, 1).repeat(1, 512, 1)
    xyzt = torch.cat([xyz, t], dim = 2)

    loss = model.compute_loss(xyzt)
    print({k:f"{v.item():.3e}" for k,v in loss.items()})
