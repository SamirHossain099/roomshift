"""Small encoders + multi-task heads (3-D pose regression, 27-way action classification).

RadarNet: PointNet (shared per-point MLP, masked max-pool) over the W-frame point window.
CSINet:   2-D CNN over (6, 114, 10) amplitude+phase.
Both BatchNorm-based so the norm-statistics TTA methods apply.
"""
import torch
import torch.nn as nn

N_J, N_A = 17, 27


class RadarNet(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.point = nn.Sequential(nn.Conv1d(6, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                   nn.Conv1d(128, d, 1), nn.BatchNorm1d(d), nn.ReLU())

    def forward(self, b):
        x = self.point(b["radar"].transpose(1, 2))                       # B x d x N
        x = x.masked_fill(b["rmask"][:, None, :] == 0, -1e4)
        return x.max(2).values.clamp_min(0)


class CSINet(nn.Module):
    def __init__(self, d=256):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(6, 32), nn.MaxPool2d((2, 1)), blk(32, 64), nn.MaxPool2d((2, 2)),
                                 blk(64, 128), nn.MaxPool2d((2, 1)), blk(128, d), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, b):
        return self.net(b["csi"])


class SenseNet(nn.Module):
    def __init__(self, modality="radar", d=256):
        super().__init__()
        self.modality = modality
        self.enc = RadarNet(d) if modality == "radar" else CSINet(d)
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.pose_head = nn.Linear(256, N_J * 3)
        self.act_head = nn.Linear(256, N_A)

    def features(self, b):
        return self.trunk(self.enc(b))

    def forward(self, b):
        z = self.features(b)
        return self.pose_head(z).view(-1, N_J, 3), self.act_head(z)
