"""Pose and action metrics (millimetres; MM-Fi keypoints are in metres)."""
import numpy as np
import torch


def procrustes_align(pred, gt):
    """Similarity-transform pred onto gt per sample (for PA-MPJPE). pred, gt: B x J x 3 numpy."""
    mp, mg = pred.mean(1, keepdims=True), gt.mean(1, keepdims=True)
    p, g = pred - mp, gt - mg
    out = np.empty_like(pred)
    for i in range(len(p)):
        u, s, vt = np.linalg.svd(p[i].T @ g[i])
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            s[-1] *= -1
            r = u @ vt
        scale = s.sum() / max((p[i] ** 2).sum(), 1e-12)
        out[i] = scale * p[i] @ r + mg[i]
    return out


class PoseMeter:
    def __init__(self):
        self.n = 0
        self.s = {"mpjpe_mm": 0.0, "pa_mpjpe_mm": 0.0, "rel_mpjpe_mm": 0.0, "root_err_mm": 0.0, "act_acc": 0.0}

    def update(self, pose_pred, act_logits, pose_gt, act_gt):
        p = pose_pred.detach().float().cpu().numpy()
        g = pose_gt.detach().float().cpu().numpy()
        err = np.linalg.norm(p - g, axis=2)                       # B x J
        self.s["mpjpe_mm"] += 1000 * err.mean(1).sum()
        self.s["pa_mpjpe_mm"] += 1000 * np.linalg.norm(procrustes_align(p, g) - g, axis=2).mean(1).sum()
        pr, gr = p - p.mean(1, keepdims=True), g - g.mean(1, keepdims=True)   # root = keypoint centroid
        self.s["rel_mpjpe_mm"] += 1000 * np.linalg.norm(pr - gr, axis=2).mean(1).sum()
        self.s["root_err_mm"] += 1000 * np.linalg.norm(p.mean(1) - g.mean(1), axis=1).sum()
        if act_logits is not None:
            self.s["act_acc"] += (act_logits.detach().argmax(1).cpu() == act_gt.cpu()).float().sum().item()
        self.n += len(p)

    def result(self):
        return {k: float(v / max(self.n, 1)) for k, v in self.s.items()} | {"n": int(self.n)}
