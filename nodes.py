"""ComfyUI nodes for Reshoot-Anything (morphic/reshoot-anything, Wan2.2-I2V-A14B LoRAs).

Pipeline these nodes serve:
  ReshootLoadVideo -> frames (+ source audio)  -> MoGe (core) -> ReshootAnchorRender
  -> ReshootCondition (x2: pink anchor for the high-noise expert, plain for the low-noise expert)
  -> native Wan2.2 I2V models + ReshootLoraLoader (+ lightx2v) -> KSamplerAdvanced x2 -> decode.
"""
import os
import re
import subprocess
import tempfile

import numpy as np
import torch

from . import anchor as A

SAMPLE_RATE = 44100


def _probe(path):
    import json
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                          "-of", "json", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    st = (json.loads(out.stdout or b"{}").get("streams") or [{}])[0]
    w, h = int(st.get("width", 0)), int(st.get("height", 0))
    rot = (st.get("tags") or {}).get("rotate")
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    if rot is not None and abs(int(float(rot))) % 180 == 90:
        w, h = h, w
    return w, h


class ReshootLoadVideo:
    """Frames at a fixed fps / 4k+1 length / 16-multiple size, plus the source audio over the same
    span (silent if the source has none), so the result can be muxed back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "url": ("STRING", {"default": ""}),
            "long_side": ("INT", {"default": 512, "min": 128, "max": 1280, "step": 16}),
            "length": ("INT", {"default": 77, "min": 5, "max": 161}),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 60.0}),
        }, "optional": {"filename": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("frames", "audio", "width", "height", "length", "fps", "has_audio")
    FUNCTION = "load"
    CATEGORY = "Reshoot"

    def load(self, url, long_side, length, fps, filename=""):
        import folder_paths
        src = (url or "").strip() or (filename or "").strip()
        if not src:
            raise RuntimeError("ReshootLoadVideo: no source video (url or filename)")
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "in.mp4")
        if src.startswith(("http://", "https://")):
            import requests
            r = requests.get(src, timeout=120, stream=True)
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(65536):
                    fh.write(chunk)
        else:
            path = folder_paths.get_annotated_filepath(src)
            if not os.path.isfile(path):
                raise RuntimeError(f"ReshootLoadVideo: local video not found: {src}")

        fps = float(fps) if fps and fps > 0 else 16.0
        n = max(5, ((int(length) - 1) // 4) * 4 + 1)
        sw, sh = _probe(path)
        if sw <= 0 or sh <= 0:
            sw, sh = 832, 480
        s = long_side / max(sw, sh)
        W = max(64, int(round(sw * s / 16.0)) * 16)
        H = max(64, int(round(sh * s / 16.0)) * 16)

        p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf",
                            f"fps={fps},scale={W}:{H}:flags=lanczos", "-frames:v", str(n),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buf = np.frombuffer(p.stdout, dtype=np.uint8)
        got = buf.size // (W * H * 3)
        if got == 0:
            raise RuntimeError(f"ReshootLoadVideo: ffmpeg decoded no frames: {p.stderr[-300:]!r}")
        arr = buf[: got * W * H * 3].reshape(got, H, W, 3)
        frames = torch.from_numpy(arr.copy()).float() / 255.0
        if got < n:
            print(f"[ReshootLoadVideo] clip has only {got} frames at {fps} fps; repeating the last to reach {n}")
            frames = torch.cat([frames, frames[-1:].expand(n - got, -1, -1, -1)], 0)

        dur = n / fps
        a = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-t", f"{dur:.4f}", "-ac", "2",
                            "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        want = int(round(dur * SAMPLE_RATE))
        raw = np.frombuffer(a.stdout, dtype=np.float32)
        has_audio = a.returncode == 0 and raw.size >= 2
        if has_audio:
            wav = torch.from_numpy(raw[: raw.size // 2 * 2].reshape(-1, 2).T.copy())
        else:
            wav = torch.zeros(2, 0)
        if wav.shape[1] < want:
            wav = torch.cat([wav, torch.zeros(2, want - wav.shape[1])], 1)
        wav = wav[:, :want]
        return (frames, {"waveform": wav[None].contiguous(), "sample_rate": SAMPLE_RATE}, W, H, n, fps, has_audio)


_TRAJ_RE = re.compile(r"reshoot1\b[^\n]*?\bk=\S+")


class ReshootParsePrompt:
    """Splits the Graydient prompt into (caption, trajectory string). Trajectory = the pasted
    'reshoot1 ... k=...' string if present, else numbered preset, else a gentle dolly-in."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "preset": ("INT", {"default": 0, "min": 0, "max": 99}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "trajectory")
    FUNCTION = "parse"
    CATEGORY = "Reshoot"

    def parse(self, prompt, preset):
        m = _TRAJ_RE.search(prompt or "")
        if m:
            traj = m.group(0)
            caption = (prompt[: m.start()] + " " + prompt[m.end():]).strip()
        elif int(preset) in A.PRESETS:
            traj, caption = A.preset_string(int(preset)), (prompt or "").strip()
        else:
            print("[ReshootParsePrompt] no trajectory given: using preset 1 (dolly in)")
            traj, caption = A.preset_string(1), (prompt or "").strip()
        return (caption or "a cinematic shot", traj)


class ReshootAnchorRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "moge_geometry": ("MOGE_GEOMETRY",),
            "trajectory": ("STRING", {"default": "", "multiline": True}),
            "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            "depth_smoothing": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("anchor", "anchor_pink", "hole_mask")
    FUNCTION = "render"
    CATEGORY = "Reshoot"

    def render(self, images, moge_geometry, trajectory, motion_scale, depth_smoothing):
        import comfy.model_management as mm
        import comfy.utils
        keys = A.parse_trajectory(trajectory)
        pts = moge_geometry.get("points")
        if pts is None:
            raise RuntimeError("ReshootAnchorRender: MoGe geometry has no 'points'")
        if pts.shape[0] != images.shape[0]:
            raise RuntimeError(f"frame count mismatch: {images.shape[0]} images vs {pts.shape[0]} geometry")
        K = moge_geometry["intrinsics"]
        mask = moge_geometry.get("mask")
        pbar = comfy.utils.ProgressBar(images.shape[0])
        black, pink, hole = A.render_anchor(images[..., :3], pts, K, mask, keys, motion_scale,
                                            depth_smoothing, device=mm.get_torch_device(),
                                            progress=lambda i: pbar.update_absolute(i))
        return (black, pink, hole)


class ReshootCondition:
    """Feeds the anchor to Wan2.2 I2V through its normal concat channels. Default mask = zeros
    (matches the official inference code, whose hole mask collapses to zeros)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "vae": ("VAE",),
            "anchor": ("IMAGE",),
            "use_hole_mask": ("BOOLEAN", {"default": False}),
        }, "optional": {"hole_mask": ("MASK",)}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "run"
    CATEGORY = "Reshoot"

    def run(self, positive, negative, vae, anchor, use_hole_mask, hole_mask=None):
        import comfy.model_management
        import node_helpers
        N, H, W, _ = anchor.shape
        T = (N - 1) // 4 + 1
        latent = torch.zeros([1, 16, T, H // 8, W // 8], device=comfy.model_management.intermediate_device())
        concat = vae.encode(anchor[:, :, :, :3])
        values = {"concat_latent_image": concat}
        if use_hole_mask and hole_mask is not None:
            valid = (1.0 - hole_mask[:N]).unsqueeze(0)                      # (1,N,H,W): 1 = anchor pixel present
            m = torch.nn.functional.interpolate(valid, size=(H // 8, W // 8), mode="bilinear") > 0
            m = m.float()
            m = torch.cat([m[:, 0:1].repeat_interleave(4, dim=1), m[:, 1:]], dim=1)   # (1,N+3,h,w)
            m = m.view(1, m.shape[1] // 4, 4, H // 8, W // 8).transpose(1, 2)           # (1,4,T,h,w)
            values["concat_mask"] = 1.0 - m                                              # core inverts it again
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)
        return (positive, negative, {"samples": latent})


class ReshootLoraLoader:
    """Loads a morphic rank-512 LoRA (bare 'blocks.N.*.lora_A/B.weight' keys) and also applies the
    fine-tuned patch_embedding it ships, which a plain LoraLoaderModelOnly would drop."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "apply_patch_embedding": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Reshoot"

    def load(self, model, lora_name, strength, apply_patch_embedding):
        import comfy.sd
        import comfy.utils
        import folder_paths
        if strength == 0:
            return (model,)
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        pe = {k: sd.pop(k) for k in list(sd) if k.startswith("patch_embedding.")}
        # core resolves bare 'blocks.N...' keys only with the 'diffusion_model.' prefix; also halve host RAM
        # (9.8 GB fp32 -> ~4.9 GB bf16 per file)
        sd = {"diffusion_model." + k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in sd.items()}
        new_model, _ = comfy.sd.load_lora_for_models(model, None, sd, strength, 0)
        if apply_patch_embedding and pe:
            patches = {f"diffusion_model.{k}": ("set", (v.to(torch.bfloat16),)) for k, v in pe.items()}
            new_model.add_patches(patches, 1.0, 1.0)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "ReshootLoadVideo": ReshootLoadVideo,
    "ReshootParsePrompt": ReshootParsePrompt,
    "ReshootAnchorRender": ReshootAnchorRender,
    "ReshootCondition": ReshootCondition,
    "ReshootLoraLoader": ReshootLoraLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReshootLoadVideo": "Reshoot Load Video (+audio)",
    "ReshootParsePrompt": "Reshoot Parse Prompt",
    "ReshootAnchorRender": "Reshoot Anchor Render",
    "ReshootCondition": "Reshoot Condition",
    "ReshootLoraLoader": "Reshoot LoRA Loader",
}
