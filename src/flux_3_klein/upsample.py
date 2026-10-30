"""Expand a prompt into the full caption FLUX 3 Image Klein is trained on, with its prompt upsampler."""

from __future__ import annotations

import gc

import torch
from PIL import Image

from flux_3_klein.generate import empty_cache


def upsample(
    prompt: str,
    images: list[Image.Image] | None,
    boxes: list[dict] | None,
    model_path: str,
    device: str = "mps",
) -> str:
    """Return the caption. Loads the upsampler on `device` and frees it before returning."""
    from diffusers.modular_pipelines import Flux3DistilledAutoBlocks  # type: ignore[import-not-found]
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # the same block for the distilled model and Base
    up = Flux3DistilledAutoBlocks().sub_blocks.pop("prompt_upsampler").init_pipeline()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    ).eval()
    up.update_components(
        prompt_upsampler=model,
        prompt_upsampler_processor=AutoProcessor.from_pretrained(model_path),
    )

    inputs = {}
    if images:
        inputs["image"] = images
    if boxes:
        inputs["boxes"] = boxes
    caption = up(prompt=prompt, **inputs, output="prompt")

    # the 19 GB upsampler and the image model don't fit in memory together
    del up, model
    gc.collect()
    empty_cache(device)
    return caption
