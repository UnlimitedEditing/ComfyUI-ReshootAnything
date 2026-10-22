"""Pure-torch anchor renderer for Reshoot-Anything (no PyTorch3D / no compiled deps).

For each output frame j: unproject frame j's MoGe point map (camera space, OpenCV axes) and
re-render it from the authored camera pose j (relative to the source camera) with a z-buffered
2x2 forward splat. Mirrors the official pipeline: render.mp4 (holes black), render_pink.mp4
(holes pink), and a hole mask.

Trajectory string (same as the Reshoot Path Editor's "Copy prompt string"):
    reshoot1 n=77 fps=16 fov=60 k=t:x:y:z:yaw:pitch:roll[:zoom],...
t is a 0..1 fraction of the clip; x/y/z in scene units (1 = median scene depth);
x right, y up, z forward; yaw right, pitch up, roll clockwise (degrees); zoom = focal multiplier.
"""
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

CHANNELS = ["x", "y", "z", "yaw", "pitch", "roll", "zoom"]
DEFAULTS = {"zoom": 1.0}
PINK = (255 / 255.0, 65 / 255.0, 174 / 255.0)


def parse_trajectory(text):
    m = re.search(r"k=(\S+)", text or "")
    if not m:
        raise ValueError("no 'k=' keyframe field found in the trajectory string")
    keys = []
    for item in m.group(1).split(","):
        a = [float(v) for v in item.split(":")]
        vals = {c: (a[i + 1] if i + 1 < len(a) else DEFAULTS.get(c, 0.0)) for i, c in enumerate(CHANNELS)}
        keys.append((a[0], vals))
    keys.sort(key=lambda k: k[0])
    return keys


def eval_channel(keys, ch, u):
    """Monotone cubic Hermite (Fritsch-Carlson) - identical to the editor's 'Smooth' curve."""
    n = len(keys)
    if n == 1 or u <= keys[0][0]:
        return keys[0][1][ch]
    if u >= keys[-1][0]:
        return keys[-1][1][ch]
    i = 0
    while i < n - 2 and u >= keys[i + 1][0]:
        i += 1
    h = keys[i + 1][0] - keys[i][0]
    if h <= 1e-9:
        return keys[i + 1][1][ch]
    s = (u - keys[i][0]) / h
    y0, y1 = keys[i][1][ch], keys[i + 1][1][ch]

    def slope(j):
        return (keys[j + 1][1][ch] - keys[j][1][ch]) / max(keys[j + 1][0] - keys[j][0], 1e-9)

    def tang(j):
        if j == 0:
            return slope(0)
        if j == n - 1:
            return slope(n - 2)
        d0, d1 = slope(j - 1), slope(j)
        if d0 * d1 <= 0:
            return 0.0
        h0, h1 = keys[j][0] - keys[j - 1][0], keys[j + 1][0] - keys[j][0]
        return 3 * (h0 + h1) / ((2 * h1 + h0) / d0 + (h1 + 2 * h0) / d1)

    m0, m1 = tang(i), tang(i + 1)
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * m1


def pose_at(keys, u):
    return {c: eval_channel(keys, c, u) for c in CHANNELS}


def rot_cv(yaw, pitch, roll):
    """R = Ry(yaw) Rx(pitch) Rz(roll), OpenCV axes (x right, y down, z forward)."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def _smooth_log(x, sigma):
    if sigma <= 0 or len(x) < 3:
        return x
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1).float() / sigma) ** 2)
    k = k / k.sum()
    xp = F.pad(x[None, None], (r, r), mode="replicate")
    return F.conv1d(xp, k[None, None])[0, 0]


def _fill_pinholes(img, valid, passes=2):
    """img (1,3,H,W), valid (1,1,H,W) float. Fill 1-2 px holes from >=5 valid neighbours."""
    for _ in range(passes):
        cnt = F.avg_pool2d(valid, 3, 1, 1) * 9 - valid
        num = F.avg_pool2d(img * valid, 3, 1, 1) * 9
        fill = (valid == 0) & (cnt >= 5)
        img = torch.where(fill, num / cnt.clamp_min(1), img)
        valid = torch.where(fill, torch.ones_like(valid), valid)
    return img, valid


@torch.no_grad()
def render_anchor(frames, points, intrinsics, mask, keys, motion_scale=1.0,
                  depth_smoothing=2.0, device=None, progress=None):
    """frames (N,H,W,3) 0..1; points (N,h,w,3) camera space; intrinsics (N,3,3) normalized
    (u = (fx*x/z + cx) * W); mask (N,h,w) bool or None.
    Returns (anchor_black (N,H,W,3), anchor_pink (N,H,W,3), hole (N,H,W) float 1=hole)."""
    N, H, W, _ = frames.shape
    dev = torch.device(device) if device is not None else frames.device
    pts = points
    msk = mask
    if pts.shape[1] != H or pts.shape[2] != W:
        pts = F.interpolate(pts.permute(0, 3, 1, 2).float(), size=(H, W), mode="nearest").permute(0, 2, 3, 1)
        if msk is not None:
            msk = F.interpolate(msk[:, None].float(), size=(H, W), mode="nearest")[:, 0] > 0.5

    med = []
    for j in range(N):
        z = pts[j, ..., 2].float()
        ok = torch.isfinite(z) & (z > 0)
        if msk is not None:
            ok &= msk[j]
        med.append(z[ok].median().cpu() if bool(ok.any()) else torch.tensor(1.0))
    med = torch.stack(med).float()
    med_s = _smooth_log(med.log(), depth_smoothing).exp()
    scene = float(med_s.median()) * float(motion_scale)
    far = 25.0 * float(med_s.median())

    ys, xs = torch.meshgrid(torch.arange(H, device=dev).float(), torch.arange(W, device=dev).float(), indexing="ij")
    out_black, out_pink, out_hole = [], [], []
    for j in range(N):
        u_frac = j / max(N - 1, 1)
        p = pose_at(keys, u_frac)
        R = torch.tensor(rot_cv(p["yaw"], p["pitch"], p["roll"]), dtype=torch.float32, device=dev)
        c = torch.tensor([p["x"], -p["y"], p["z"]], dtype=torch.float32, device=dev) * scene
        K = intrinsics[j]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        P = pts[j].to(dev).float() * float(med_s[j] / med[j])
        ok = torch.isfinite(P).all(-1) & (P[..., 2] > 1e-4)
        if msk is not None:
            ok &= msk[j].to(dev)
        ray = torch.stack([((xs + 0.5) / W - cx) / fx, ((ys + 0.5) / H - cy) / fy, torch.ones_like(xs)], -1)
        P = torch.where(ok[..., None], P, ray * far).reshape(-1, 3)
        rgb = frames[j].to(dev).float().reshape(-1, 3)

        Q = (P - c) @ R                       # R^T (P - c)
        z = Q[:, 2]
        front = z > 1e-3
        zc = z.clamp_min(1e-3)
        zoom = max(p["zoom"], 1e-3)
        u = (fx * zoom * Q[:, 0] / zc + cx) * W - 0.5
        v = (fy * zoom * Q[:, 1] / zc + cy) * H - 0.5
        u0, v0 = torch.floor(u).long(), torch.floor(v).long()
        idxs, zs, srcs = [], [], []
        src = torch.arange(P.shape[0], device=dev)
        for dv in (0, 1):
            for du in (0, 1):
                ui, vi = u0 + du, v0 + dv
                inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                idxs.append((vi * W + ui)[inb]); zs.append(z[inb]); srcs.append(src[inb])
        idx, zz, sr = torch.cat(idxs), torch.cat(zs), torch.cat(srcs)
        zbuf = torch.full((H * W,), float("inf"), device=dev).scatter_reduce(0, idx, zz, "amin", include_self=True)
        win = zz <= zbuf[idx]
        img = torch.zeros(H * W, 3, device=dev)
        img[idx[win]] = rgb[sr[win]]
        valid = torch.isfinite(zbuf).float().view(1, 1, H, W)
        img = img.view(H, W, 3).permute(2, 0, 1)[None]
        img, valid = _fill_pinholes(img, valid)
        hole = (valid == 0)
        black = torch.where(hole, torch.zeros_like(img), img)
        pink = torch.where(hole, torch.tensor(PINK, device=dev).view(1, 3, 1, 1).expand_as(img), img)
        out_black.append(black[0].permute(1, 2, 0).cpu())
        out_pink.append(pink[0].permute(1, 2, 0).cpu())
        out_hole.append(hole[0, 0].float().cpu())
        if progress is not None:
            progress(j + 1)
    return torch.stack(out_black), torch.stack(out_pink), torch.stack(out_hole)


# ---- presets (index for the Graydient numeric slot) ----
def _orbit(deg, dolly=0.0, steps=6):
    out = []
    for i in range(steps + 1):
        f = i / steps
        a = math.radians(deg * f)
        out.append((f, dict(x=math.sin(a), z=(1 - math.cos(a)) + dolly * f, yaw=-deg * f)))
    return out


PRESETS = {
    1: ("Dolly in", lambda: [(0, {}), (1, dict(z=0.6))]),
    2: ("Dolly out", lambda: [(0, {}), (1, dict(z=-0.5))]),
    3: ("Orbit right 40", lambda: _orbit(40)),
    4: ("Orbit left 40", lambda: _orbit(-40)),
    5: ("Truck right", lambda: [(0, {}), (1, dict(x=0.5))]),
    6: ("Truck left", lambda: [(0, {}), (1, dict(x=-0.5))]),
    7: ("Pedestal up", lambda: [(0, {}), (1, dict(y=0.4))]),
    8: ("Zoom in", lambda: [(0, {}), (1, dict(zoom=2.2))]),
    9: ("Vertigo", lambda: [(i / 5, dict(z=0.55 * i / 5, zoom=1 - 0.55 * i / 5)) for i in range(6)]),
    10: ("Spiral in right", lambda: _orbit(35, 0.35)),
    11: ("Crane down", lambda: [(0, dict(y=0.5, pitch=-14)), (1, {})]),
}


def preset_string(idx):
    name, fn = PRESETS[idx]
    ks = []
    for t, v in fn():
        vals = [v.get(c, DEFAULTS.get(c, 0.0)) for c in CHANNELS]
        ks.append(":".join(("%.3f" % (round(x, 3) + 0.0)).rstrip("0").rstrip(".") or "0" for x in [t] + vals))
    return "reshoot1 k=" + ",".join(ks)
