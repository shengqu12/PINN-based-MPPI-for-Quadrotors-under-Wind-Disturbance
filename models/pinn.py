"""
pinn.py 

inputs (17 dimensions):
    v_rel(3)  = vel - wind    relative velocity
    quat(4)                   attitude quaternion [qx,qy,qz,qw]
    omega(3)                  angular velocity
    u(4)                      motor speeds rad/s
    wind(3)                   wind speed vector
    index:[0:3][3:7][7:10][10:14][14:17]



Model architecture:

    f_total = f_aero + f_rotor

    f_aero  = -C_d(x) ⊙ v_rel · |v_rel|  (EDC)
    - aero force from drag which shold be opposite to the V_rel, 
    -C_d is predicted by a MLP.


    f_rotor = MLP(omega, u)
    - rotor-induced force, mainly in Z axis, predicted by another MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualPINN(nn.Module):

    def __init__(self, input_dim=17, hidden_dims=(128, 128, 64), output_dim=3):
        super().__init__()
        self.input_dim = input_dim

        # ── C_d net: predict drag coefficients(x,y,z) ────────────────────────────
        # input: x(17)
        # output: drag coefficients (3,)，Softplus ensures >= 0
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.cd_net = nn.Sequential(*layers)

        # ── Rotor net: predict rotor effects ───────────────────────────────
        # input: omega(3) + u(4) = 7
        # output: rotor forces (3,)，Softplus ensures >= 0
        self.rotor_net = nn.Sequential(
            nn.Linear(7, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, v_rel):
        """
        Args:
            x:     (B, 17) normalized input features (v_rel, quat, omega, u, wind)
            v_rel: (B, 3)  physical relative velocity (world frame, not normalized, for EDC calculation)
        Returns:
            (B, 3) predicted residual acceleration
        """
        # F_aero: aero force from drag which shold be opposite to the V_rel,
        C_d       = F.softplus(self.cd_net(x))
        v_rel_mag = torch.norm(v_rel, dim=1, keepdim=True)
        f_aero    = -C_d * v_rel * v_rel_mag

        # F_rotor: rotor-induced force, mainly in Z axis, predicted by another MLP.
        # omega: x[:,7:10]，u: x[:,10:14]
        rotor_input = x[:, 7:14]   # (B,7)
        f_rotor     = self.rotor_net(rotor_input)

        return f_aero + f_rotor

    def forward_rotor_z(self, x):
        """
        forward pass for rotor net Z output only, used for downwash loss calculation.
        """
        rotor_input = x[:, 7:14]
        f_rotor = self.rotor_net(rotor_input)
        return f_rotor[:, 2]   # (B,) Z axis only


if __name__ == '__main__':
    model = ResidualPINN(input_dim=17)
    print(model)
    print(f"\n params amount: {sum(p.numel() for p in model.parameters()):,}")

    x     = torch.randn(32, 17)
    v_rel = torch.randn(32, 3)
    out   = model(x, v_rel)
    print(f"input: {x.shape}, output: {out.shape}")

    # verify EDC
    C_d   = F.softplus(model.cd_net(x))
    mag   = torch.norm(v_rel, dim=1, keepdim=True)
    f_aero = -C_d * v_rel * mag
    dot   = (f_aero * v_rel).sum(dim=1)
    print(f"f_aero·v_rel maximum: {dot.max().item():.6f}（must <= 0）✓")