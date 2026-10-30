"""Generate / edit images with FLUX 3 Image Klein on Apple Silicon (MPS), CUDA or CPU.

Usage:
    uv run python -m flux_3_klein.generate "A neon shop sign that reads KLEIN"
    uv run python -m flux_3_klein.generate "Edit: sunset" --input photo.png
    uv run flux-3-klein "Prompt" -m ~/models/flux-3-image-klein --pup-model ~/models/flux-3-image-klein-pup
    uv run flux-3-klein "Prompt" -m ~/models/flux-3-image-klein-base --guidance 4  # 50 steps, with CFG
    FLUX_3_KLEIN_PATH=~/models/flux-3-image-klein uv run flux-3-klein "Prompt"
    uv run flux-3-klein "a red balloon over a field" --boxes boxes.json
    uv run flux-3-klein "A caption the upsampler already wrote" --no-upsample
    uv run flux-3-klein "Prompt" --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import ExifTags, Image, ImageOps


def patch_mps_empty_cache() -> None:
    # torch 2.14: mps.empty_cache() waits on the GPU while holding the GIL, but finishing the
    # non_blocking weight copies device_map makes needs the GIL -> deadlock. synchronize() releases it.
    # ponytail: global monkeypatch, drop once torch's empty_cache releases the GIL itself
    empty_cache = torch.mps.empty_cache

    def safe_empty_cache() -> None:
        torch.mps.synchronize()
        empty_cache()

    torch.mps.empty_cache = safe_empty_cache


DEVICES = ("auto", "mps", "cuda", "cpu")


def pick_device(requested: str) -> str:
    """The torch device to run on: the one asked for, or with "auto" the first of MPS, CUDA, CPU present."""
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        return next(d for d, ok in available.items() if ok)
    if not available[requested]:
        raise ValueError(f"{requested.upper()} is not available in this torch build")
    return requested


def empty_cache(device: str) -> None:
    """Hand freed memory back to the device, so the next model has room."""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def noise_seed(seed: int, images: list[Image.Image] | None) -> int:
    """The seed for the starting noise: `seed` itself, or for an edit, `seed` mixed with the inputs' pixels.

    Editing an image with the seed and size it was generated with would start from the exact noise that made it,
    and the model then over-sharpens it into a crunchy, HDR-like texture instead of editing. Mixing in the inputs
    gives an edit its own noise, while the same seed and inputs still reproduce the same take.
    """
    if not images:
        return seed
    digest = hashlib.sha256(str(seed).encode())
    for img in images:
        digest.update(f"{img.mode}{img.size}".encode())
        digest.update(img.tobytes())
    return int.from_bytes(digest.digest()[:8], "big") >> 1


def is_distilled(model: str) -> bool:
    """Whether `model` is the distilled checkpoint: its transformer is a flow map. Works for a Hub ID too."""
    from diffusers import Flux3Transformer2DModel  # type: ignore[import-not-found]

    config = Flux3Transformer2DModel.load_config(model, subfolder="transformer")
    return bool(config.get("flow_map"))


def load_pipe(
    model: str,
    distilled: bool,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
):
    from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel  # type: ignore[import-not-found]
    from diffusers.modular_pipelines import (  # type: ignore[import-not-found]
        Flux3AutoBlocks,
        Flux3DistilledAutoBlocks,
    )
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    blocks = Flux3DistilledAutoBlocks() if distilled else Flux3AutoBlocks()
    # the upsampler runs on its own first, and is gone before the image model loads
    blocks.sub_blocks.pop("prompt_upsampler")
    # tqdm's bar has no newline, so it would glue itself to the @step lines the studio matches from the line's start.
    # pipe.set_progress_bar_config only reaches the top-level blocks, and the denoise loops sit below an auto block
    for route in blocks.sub_blocks["denoise"].sub_blocks.values():
        route.sub_blocks["denoise"].set_progress_bar_config(disable=True)
    pipe = blocks.init_pipeline()
    # device_map loads each weight straight to the GPU instead of building a full CPU copy first
    pipe.update_components(
        transformer=Flux3Transformer2DModel.from_pretrained(
            model, subfolder="transformer", dtype=dtype, device_map=device
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", dtype=dtype, device_map=device
        ).eval(),
        tokenizer=AutoProcessor.from_pretrained(model, subfolder="text_encoder"),
        vae=AutoencoderKLFlux2.from_pretrained(
            model, subfolder="vae", dtype=torch.float32, device_map=device
        ),
    )
    return pipe


def stage(name: str) -> None:
    # machine-readable marker for the studio UI: upsample, load, denoise, decode, save
    print(f"@stage {name}", flush=True)


def track_steps(pipe, total: int, device: str) -> None:
    # the modular pipeline has no step callback, but it calls scheduler.step exactly once per denoising step
    # ponytail: wraps a scheduler method, swap for a real callback if the pipeline gets one
    scheduler_step = pipe.scheduler.step
    done = 0

    def counted_step(*args, **kwargs):
        nonlocal done
        result = scheduler_step(*args, **kwargs)
        if device == "mps":
            # MPS only queues the work: without the wait all 16 steps print at once and the real time lands in decode
            torch.mps.synchronize()
        done += 1
        print(f"@step {done}/{total}", flush=True)
        if done == total:  # the pipeline decodes the latents right after the last step
            stage("decode")
        return result

    pipe.scheduler.step = counted_step


def exif(model: str) -> Image.Exif:
    # the same disclosure BFL's own CLI writes
    tags = Image.Exif()
    tags[ExifTags.Base.Software] = f"AI generated;{model}"
    tags[ExifTags.Base.Make] = "Black Forest Labs"
    return tags


# what Pillow writes EXIF to: any other format saves fine and silently drops the disclosure
EXIF_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")


def valid_box(v) -> bool:
    """Whether `v` is [y_min, x_min, y_max, x_max] in 0-1 with an area: diffusers clamps anything else unannounced."""
    return (
        isinstance(v, list)
        and len(v) == 4
        and all(isinstance(n, (int, float)) and 0 <= n <= 1 for n in v)
        and v[0] < v[2]
        and v[1] < v[3]
    )


ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (1184, 880),
    "3:4": (880, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1360, 768),
    "9:16": (768, 1360),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="FLUX 3 Image Klein generation")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--input", help="Input image(s) for editing, comma-separated")
    parser.add_argument(
        "--ratio",
        choices=ASPECT_RATIOS,
        help="Output aspect ratio (default: 1:1, or the first input image's size with --input)",
    )
    parser.add_argument("--width", type=int, help="Override width (multiple of 16)")
    parser.add_argument("--height", type=int, help="Override height (multiple of 16)")
    parser.add_argument(
        "--steps",
        type=int,
        help="Denoising steps (default: 16 for the distilled model, 50 for Base)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        help="CFG scale, Base only (default: 3.5); the distilled model samples without CFG",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--boxes",
        help="JSON file with a list of layout boxes, each "
        '{"id": "obj_1", "desc": "a red balloon", "bbox": [0.1, 0.1, 0.5, 0.4]} '
        "with bbox [y_min, x_min, y_max, x_max] in 0-1, top-left origin",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=os.path.expanduser,
        default=os.environ.get("FLUX_3_KLEIN_PATH")
        or "black-forest-labs/flux-3-image-klein",
        help="Local path or HF repo ID (default: %(default)s, set via FLUX_3_KLEIN_PATH)",
    )
    parser.add_argument(
        "--no-upsample",
        action="store_true",
        help="Send the prompt as typed: the model is not meant to run without the upsampler's "
        "caption, so only for a prompt that already is one",
    )
    parser.add_argument(
        "--pup-model",
        type=os.path.expanduser,
        default=os.environ.get("FLUX_3_KLEIN_PUP_PATH")
        or "black-forest-labs/flux-3-image-klein-pup",
        help="Prompt upsampler path or HF repo ID (default: %(default)s, set via FLUX_3_KLEIN_PUP_PATH)",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=os.environ.get("FLUX_3_KLEIN_DEVICE") or "auto",
        help="Where to run (default: %(default)s = MPS, else CUDA, else CPU; set via FLUX_3_KLEIN_DEVICE)",
    )
    parser.add_argument("--output", default="output/output.png")
    args = parser.parse_args()

    for side in (args.width, args.height):
        if side is not None and (side < 16 or side % 16):
            parser.error("--width and --height must be multiples of 16")

    boxes: list[dict] | None = None
    if args.boxes:
        if args.no_upsample:
            parser.error("--boxes needs the upsampler, which cites them in the caption")
        try:
            boxes = json.loads(Path(args.boxes).read_text())
        except (OSError, ValueError) as e:
            parser.error(f"could not read --boxes: {e}")
        if not isinstance(boxes, list):
            parser.error("--boxes must hold a JSON list")
        # here, before the 19 GB upsampler loads. A null src_bbox or tgt_bbox is an edit row's way of saying none
        for row in boxes:
            keys = ("bbox", "src_bbox", "tgt_bbox")
            found = [row[k] for k in keys if row.get(k)] if isinstance(row, dict) else []
            if not found or not all(map(valid_box, found)):
                parser.error(
                    f"--boxes: {row} needs a bbox, or a src_bbox or tgt_bbox, "
                    "as [y_min, x_min, y_max, x_max] in 0-1"
                )

    # ponytail: a relative "dir/name" path looks like a Hub ID, so only absolute, dotted or multi-segment paths are checked
    for model in [args.model] if args.no_upsample else [args.model, args.pup_model]:
        if not os.path.isdir(model) and (
            model.startswith((".", "/")) or model.count("/") != 1
        ):
            parser.error(f"model directory not found: {model}")

    distilled = is_distilled(args.model)
    if distilled and args.guidance is not None:
        parser.error("--guidance is for Base: the distilled model samples without CFG")
    steps = args.steps or (16 if distilled else 50)

    try:
        device = pick_device(args.device)
    except ValueError as e:
        parser.error(str(e))
    print(f"Device: {device}", flush=True)

    images: list[Image.Image] | None = None
    if args.input:
        paths = args.input.split(",")
        if len(paths) > 10:
            parser.error("FLUX 3 Image Klein supports at most 10 input images")
        # upright, as BFL prepares references: the upsampler rotates its copy, the pipeline's processor does not
        images = [ImageOps.exif_transpose(Image.open(p.strip())) for p in paths]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if device == "mps":
        patch_mps_empty_cache()

    prompt = args.prompt
    if not args.no_upsample:
        from flux_3_klein.upsample import upsample

        stage("upsample")
        prompt = upsample(prompt, images, boxes, args.pup_model, device)
        print(f"Upsampled prompt:\n{prompt}\n")
        print(f"@upsampled {json.dumps({'prompt': prompt})}", flush=True)

    ratio = args.ratio or (None if args.input else "1:1")
    width, height = ASPECT_RATIOS[ratio] if ratio else (None, None)
    width, height = args.width or width, args.height or height

    stage("load")
    pipe = load_pipe(args.model, distilled, device)
    if not distilled:
        pipe.guider = pipe.guider.new(
            guidance_scale=3.5 if args.guidance is None else args.guidance
        )

    stage("denoise")
    track_steps(pipe, steps, device)

    out = pipe(
        prompt=prompt,
        **({"image": images} if images else {}),
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(
            noise_seed(args.seed, images)
        ),
        output="images",
    )[0]

    stage("save")
    # the checkpoint's kind, as BFL writes it: args.model can be a local path, which must not ship in the image
    tags = exif("distilled" if distilled else "base")
    try:
        if out_path.suffix.lower() not in EXIF_SUFFIXES:
            raise ValueError("it would lose the AI disclosure")
        out.save(out_path, exif=tags)
    except (OSError, ValueError) as e:  # e.g. unknown extension: keep the render as PNG
        out_path = out_path.with_suffix(".png")
        print(f"Could not save {args.output} ({e}); saving PNG instead")
        out.save(out_path, exif=tags)
    print(f"Saved {out_path.resolve()} ({out.size[0]}x{out.size[1]}, mode={out.mode})")


if __name__ == "__main__":
    main()
