#!/usr/bin/env python3
"""Checks what the page's form turns into. Run with .venv/bin/python web/test_server.py"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image
from server import clean_settings, image_size, resolve_mentions

BOX = '[{"id": "obj_1", "desc": "a red balloon", "bbox": [0.1, 0.1, 0.5, 0.4]}]'


def refused(**req: Any) -> bool:
    try:
        clean_settings({"prompt": "a cat", **req})
    except ValueError:
        return True
    return False


def test_resolve_mentions() -> None:
    names = [f"r{i}.png" for i in range(1, 11)]
    assert resolve_mentions("@r1.png beside @r2.png", names) == "the first image beside the second image"
    assert resolve_mentions("into @r10.png.", names) == "into the tenth image."
    assert resolve_mentions("mail me@example.com or @nobody", names) == "mail me@example.com or @nobody"


def test_defaults() -> None:
    distilled = clean_settings({"prompt": " a cat "})
    assert distilled["prompt"] == "a cat"
    assert (distilled["steps"], distilled["seed"], distilled["upsample"]) == (16, 42, True)
    assert "guidance" not in distilled and "boxes" not in distilled
    base = clean_settings({"prompt": "a cat", "model": "base"})
    assert (base["steps"], base["guidance"]) == (50, 3.5)
    assert "guidance" not in clean_settings({"prompt": "a cat", "guidance": 7})
    assert clean_settings({"prompt": "a cat", "upsample": False})["upsample"] is False


def test_ranges() -> None:
    assert clean_settings({"prompt": "a cat", "width": 1040, "height": "880"})["height"] == 880
    assert refused(width=1032) and refused(height=240) and refused(width=4112)
    assert refused(steps=0) and refused(steps=201)
    assert refused(model="base", guidance=0.5) and refused(model="base", guidance="nan")
    assert refused(prompt=" ") and refused(ratio="5:4") and refused(model="turbo") and refused(device="tpu")


def test_boxes() -> None:
    assert clean_settings({"prompt": "a cat", "boxes": f" {BOX}\n"})["boxes"] == BOX
    assert "boxes" not in clean_settings({"prompt": "a cat", "boxes": "  ", "upsample": False})
    assert refused(boxes=BOX, upsample=False)
    for bad in ("[", "[]", "{}", '"x"', "[1]", '[{"desc": "no id"}]', '[{"id": 1}]', [{"id": "not text"}], 0):
        assert refused(boxes=bad), bad


def test_image_size() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "photo.jpg"
        for orientation, size in ((1, (40, 30)), (6, (30, 40))):
            exif = Image.Exif()
            exif[ExifTags.Base.Orientation] = orientation
            Image.new("RGB", (40, 30)).save(path, exif=exif)
            assert image_size(path) == size


if __name__ == "__main__":
    test_resolve_mentions()
    test_defaults()
    test_ranges()
    test_boxes()
    test_image_size()
    print("ok")
