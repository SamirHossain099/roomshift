"""Online adaptation methods for MMFi pose + action (protocol as in project 01).

step(b) -> (pose_pred B x 17 x 3, action_logits B x 27), made before any feedback about b.
feedback(b, fb): fb["label"] marks frames whose ground-truth pose (and action) is revealed: a
"calibration frame" budget of one every K frames (a person stands in a known pose / a camera is briefly
available); 0 = label-free.

Label-free:  source, norm (EMA normalisation stats, all layers), tent (entropy of the action head, norm
             affine params), csi-align (Wi-Fi: per-antenna amplitude re-centred to the source's
             statistics with running target statistics; no gradients).
Label-using: rigid (gradient-free Procrustes rotation + translation of predicted poses onto the labelled
             frames in a sliding window, applied to every prediction), supft (supervised fine-tuning of all
             weights on the labelled frames, replay buffer), rigid+norm, rigid+csi-align.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- EMA normalisation (as in project 01)
class EMANorm(nn.Module):
    ref_batch = 16

    def __init__(self, bn, momentum):
        super().__init__()
        self.weight, self.bias, self.eps, self.m = bn.weight, bn.bias, bn.eps, momentum
        self.register_buffer("mean", bn.running_mean.clone())
        self.register_buffer("var", bn.running_var.clone())

    def forward(self, x):
        dims = [0] + list(range(2, x.dim()))
        shape = [1, -1] + [1] * (x.dim() - 2)
        if self.m > 0:
            m = 1 - (1 - self.m) ** (x.shape[0] / self.ref_batch)
            with torch.no_grad():
                self.mean.mul_(1 - m).add_(m * x.mean(dims))
                self.var.mul_(1 - m).add_(m * x.var(dims, unbiased=False))
        return (x - self.mean.view(shape)) / torch.sqrt(self.var.view(shape) + self.eps) * self.weight.view(shape) \
            + self.bias.view(shape)


def swap_norms(model, momentum):
    for name, mod in model.named_children():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            setattr(model, name, EMANorm(mod, momentum))
        else:
            swap_norms(mod, momentum)
    return model


def similarity_fit(pred, gt):
    """Rotation, uniform scale and translation (Umeyama) with s * R @ pred + t ~ gt."""
    P, G = pred.reshape(-1, 3), gt.reshape(-1, 3)
    mp, mg = P.mean(0), G.mean(0)
    Pc, Gc = P - mp, G - mg
    u, sv, vt = np.linalg.svd(Pc.T @ Gc)
    d = np.sign(np.linalg.det(u @ vt))
    D = np.diag([1, 1, d])
    R = (u @ D @ vt).T
    scale = (sv * np.diag(D)).sum() / max((Pc ** 2).sum(), 1e-12)
    return scale * R, mg - scale * R @ mp


def affine_fit(pred, gt, ridge=1e-3):
    """Full 3-D affine map A @ pred + t ~ gt (12 parameters), lightly ridge-regularised towards identity."""
    P, G = pred.reshape(-1, 3), gt.reshape(-1, 3)
    X = np.c_[P, np.ones(len(P))]
    target = G - P                                   # regress the correction so the prior is identity
    W = np.linalg.solve(X.T @ X + ridge * len(P) * np.eye(4), X.T @ target)
    return np.eye(3) + W[:3].T, W[3]


def rigid_fit(pred, gt):
    """Least-squares rotation R and translation t with R @ pred + t ~ gt over all joints of all frames.
    pred, gt: N x 17 x 3 numpy. Returns (R, t)."""
    P, G = pred.reshape(-1, 3), gt.reshape(-1, 3)
    mp, mg = P.mean(0), G.mean(0)
    u, _, vt = np.linalg.svd((P - mp).T @ (G - mg))
    d = np.sign(np.linalg.det(u @ vt))
    R = (u @ np.diag([1, 1, d]) @ vt).T
    return R, mg - R @ mp


# ---------------------------------------------------------------- methods
class Method:
    name = "source"

    def __init__(self, model, momentum=0.0, **kw):
        self.model = swap_norms(copy.deepcopy(model), momentum).eval()

    @torch.no_grad()
    def step(self, b):
        return self.model(b)

    def feedback(self, b, fb):
        pass


class Norm(Method):
    name = "norm"

    def __init__(self, model, momentum=0.05, **kw):
        super().__init__(model, momentum)


class Tent(Method):
    name = "tent"

    def __init__(self, model, momentum=0.05, lr=1e-3, **kw):
        super().__init__(model, momentum)
        ps = [p for m in self.model.modules() if isinstance(m, EMANorm) for p in (m.weight, m.bias)]
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in ps:
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(ps, lr=lr)

    def step(self, b):
        pose, act = self.model(b)
        ent = -(act.softmax(1) * act.log_softmax(1)).sum(1).mean()
        self.opt.zero_grad()
        ent.backward()
        self.opt.step()
        return pose.detach(), act.detach()


class CSIAlign(Method):
    """Re-centre each antenna's CSI amplitude to the source mean using running target means (label-free)."""
    name = "csi-align"
    needs_source = True

    def __init__(self, model, src_stats=None, momentum=0.0, rate=0.02, **kw):
        super().__init__(model, momentum)
        self.src = src_stats            # (3,) per-antenna mean of standardised amplitude on the source
        self.tgt, self.rate = None, rate

    def _align(self, b):
        if "csi" not in b or self.src is None:
            return b
        amp = b["csi"][:, :3]
        m = amp.mean((0, 2, 3))
        self.tgt = m if self.tgt is None else (1 - self.rate) * self.tgt + self.rate * m
        b = dict(b)
        b["csi"] = torch.cat([amp - (self.tgt - self.src).view(1, 3, 1, 1), b["csi"][:, 3:]], 1)
        return b

    @torch.no_grad()
    def step(self, b):
        return self.model(self._align(b))


class Rigid(Method):
    """Gradient-free: rigid transform of predicted poses fitted on the labelled frames of a sliding window."""
    name = "rigid"
    fit = staticmethod(rigid_fit)

    def __init__(self, model, momentum=0.0, window=64, min_n=2, freeze_after=0, **kw):
        super().__init__(model, momentum)
        self.window, self.min_n = window, min_n
        self.freeze_after, self.n_seen = freeze_after, 0   # freeze_after > 0: fit once on that many labels, then hold
        self.P, self.G = [], []
        self.R, self.t = np.eye(3), np.zeros(3)
        self.raw = None

    def _base(self, b):
        return self.model(b)

    @torch.no_grad()
    def step(self, b):
        pose, act = self._base(b)
        self.raw = pose.detach().cpu().numpy()
        out = self.raw @ self.R.T + self.t
        return torch.as_tensor(out, dtype=pose.dtype, device=pose.device), act

    def feedback(self, b, fb):
        lab = fb["label"].cpu().numpy()
        if not lab.any() or (self.freeze_after and self.n_seen >= self.freeze_after):
            return
        self.n_seen += int(lab.sum())
        self.P = (self.P + list(self.raw[lab]))[-self.window:]
        self.G = (self.G + list(b["pose"][fb["label"]].cpu().numpy()))[-self.window:]
        if len(self.P) >= self.min_n:
            self.R, self.t = type(self).fit(np.array(self.P), np.array(self.G))


def shift_fit(pred, gt):
    """Translation only (3 parameters): t = mean(gt - pred)."""
    return np.eye(3), (gt.reshape(-1, 3) - pred.reshape(-1, 3)).mean(0)


class Shift(Rigid):
    name = "shift"
    fit = staticmethod(shift_fit)


class Similarity(Rigid):
    name = "similarity"
    fit = staticmethod(similarity_fit)


class Affine(Rigid):
    name = "affine"
    fit = staticmethod(affine_fit)


class Smooth(Method):
    """Label-free: causal exponential smoothing of predicted poses within a sequence (resets at a new one)."""
    name = "smooth"

    def __init__(self, model, momentum=0.0, alpha=0.3, **kw):
        super().__init__(model, momentum)
        self.alpha, self.state, self.seq = alpha, None, None

    def _smooth(self, pose, seq):
        out = torch.empty_like(pose)
        for i in range(len(pose)):
            s = int(seq[i])
            if self.state is None or s != self.seq:
                self.state, self.seq = pose[i].clone(), s
            else:
                self.state = self.alpha * pose[i] + (1 - self.alpha) * self.state
            out[i] = self.state
        return out

    @torch.no_grad()
    def step(self, b):
        pose, act = self.model(b)
        return self._smooth(pose, b["seq"]), act


class RigidSmooth(Rigid):
    """Rigid calibration (labels) applied to causally smoothed predictions."""
    name = "rigid+smooth"

    def __init__(self, model, alpha=0.3, **kw):
        super().__init__(model, **kw)
        self.sm = Smooth(model, alpha=alpha)
        self.sm.model = self.model

    def _base(self, b):
        pose, act = self.model(b)
        return self.sm._smooth(pose, b["seq"]), act


class AffineSmooth(RigidSmooth):
    """Affine calibration (labels) of causally smoothed predictions (ours, best combination)."""
    name = "affine+smooth"
    fit = staticmethod(affine_fit)


class RigidNorm(Rigid):
    name = "rigid+norm"

    def __init__(self, model, momentum=0.05, **kw):
        super().__init__(model, momentum, **kw)


class RigidCSIAlign(Rigid):
    name = "rigid+csi-align"
    needs_source = True

    def __init__(self, model, src_stats=None, **kw):
        super().__init__(model, 0.0, **kw)
        self.align = CSIAlign(model, src_stats)
        self.align.model = self.model

    def _base(self, b):
        return self.model(self.align._align(b))


class SupFT(Method):
    """Supervised online fine-tuning on the labelled frames (replay buffer of the last `replay`).
    `steps` gradient steps per labelled batch; `part` = "all" weights or the "head" (pose/action heads only)."""
    name = "supft"

    def __init__(self, model, momentum=0.0, lr=1e-4, replay=64, steps=1, part="all", freeze_after=0, **kw):
        super().__init__(model, momentum)
        ps = list(self.model.parameters()) if part == "all" else \
            list(self.model.pose_head.parameters()) + list(self.model.act_head.parameters())
        self.opt = torch.optim.Adam(ps, lr=lr)
        self.replay, self.buf, self.steps = replay, [], steps
        self.freeze_after, self.n_seen = freeze_after, 0

    def feedback(self, b, fb):
        lab = fb["label"]
        if not lab.any() or (self.freeze_after and self.n_seen >= self.freeze_after):
            return
        self.n_seen += int(lab.sum())
        keys = [k for k in ("radar", "rmask", "csi", "pose", "action") if k in b]
        self.buf.append({k: b[k][lab] for k in keys})
        bb = {k: torch.cat([x[k] for x in self.buf])[-self.replay:] for k in keys}
        self.buf = [bb]
        if len(bb["pose"]) < 2:
            return
        self.model.train()
        for m in self.model.modules():            # keep normalisation statistics frozen while fine-tuning
            if isinstance(m, EMANorm):
                m.m = 0.0
        with torch.enable_grad():
            for _ in range(self.steps):
                pose, act = self.model(bb)
                loss = F.l1_loss(pose, bb["pose"]) + 0.1 * F.cross_entropy(act, bb["action"])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self.model.eval()


METHODS = {c.name: c for c in (Method, Norm, Tent, CSIAlign, Rigid, Shift, Similarity, Affine, Smooth, RigidSmooth,
                               AffineSmooth, RigidNorm, RigidCSIAlign, SupFT)}
