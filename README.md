# ComfyUI-ReshootAnything

Nodes for running [morphic/reshoot-anything](https://huggingface.co/morphic/reshoot-anything)
(Wan2.2-I2V-A14B LoRAs, Apache 2.0) inside ComfyUI with **no compiled dependencies** and no PyTorch3D:
the point-cloud anchor renderer is pure torch.

| Node | What it does |
|---|---|
| `Reshoot Load Video (+audio)` | Source video -> frames at a fixed fps / 4k+1 length / 16-multiple size, plus the source audio over the same span (silent if the source has none), ready to mux back. |
| `Reshoot Parse Prompt` | Splits the prompt into caption + camera path. Path = pasted `reshoot1 ... k=...` string, or a numbered preset (1 dolly in, 2 dolly out, 3 orbit right, 4 orbit left, 5 truck right, 6 truck left, 7 pedestal up, 8 zoom in, 9 vertigo, 10 spiral in, 11 crane down). |
| `Reshoot Anchor Render` | Needs core `MoGeInference` output. Warps each frame's point map along the authored per-frame camera (truck/pedestal/dolly/pan/tilt/roll/zoom) with a z-buffered 2x2 splat. Outputs plain anchor, pink-hole anchor, hole mask. |
| `Reshoot Condition` | VAE-encodes the anchor into Wan I2V's concat channels. Mask defaults to zeros, matching the official inference code (its hole mask collapses to zeros). |
| `Reshoot LoRA Loader` | Loads the rank-512 LoRAs. Adds the `diffusion_model.` key prefix (without it core patches 0 of 400 modules) and applies the fine-tuned `patch_embedding` they ship. |

Camera paths are authored with the Reshoot Path Editor (`camera-path-editor.html`, in the
`reshoot-studio` folder of the Graydient project): keyframe poses over a video timeline, then "Copy prompt string".
Format: `reshoot1 n=77 fps=16 fov=60 k=t:x:y:z:yaw:pitch:roll:zoom,...` (t = 0..1 of the clip; x/y/z in
fractions of the scene's median depth; angles in degrees; zoom = focal multiplier).

## Pipeline
HIGH expert (morphic `jan06_scaling_80k_ckpt1400`) runs on the pink-hole anchor, LOW expert
(`dec23_v2v_lownoise_black_lora_512_ckpt1000`) on the plain anchor, as in the official `video2video.py`.

## Not reproduced
The official code also passes the source video as `reference_latent`; the public `model.py` does not accept
that argument, so this port is anchor-only.
