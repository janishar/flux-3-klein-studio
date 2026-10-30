"""Checks the seed mixing, the preset sizes and the box check. Run: .venv/bin/python tests/test_generate.py"""

from PIL import Image

from flux_3_klein.generate import ASPECT_RATIOS, noise_seed, valid_box

black, red = Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16), "red")

# text-to-image uses the seed as given
assert noise_seed(42, None) == 42
assert noise_seed(42, []) == 42

# an edit gets its own noise, and the same seed and inputs reproduce it
assert noise_seed(42, [black]) == noise_seed(42, [black.copy()])
assert noise_seed(42, [black]) not in (
    42,
    noise_seed(43, [black]),
    noise_seed(42, [red]),
)
assert noise_seed(42, [black, red]) != noise_seed(42, [red, black])
assert 0 <= noise_seed(42, [black]) < 2**63  # fits manual_seed

assert all(side % 16 == 0 for size in ASPECT_RATIOS.values() for side in size)

assert valid_box([0.1, 0.1, 0.5, 0.4]) and valid_box([0, 0, 1, 1])
for bad in (
    [100, 100, 500, 400],  # the upsampler's own 0-1000 rows
    [0.5, 0.5, 0.1, 0.1],
    [0.1, 0.2],
    [float("nan"), 0, 1, 1],
    "0.1",
    None,
):
    assert not valid_box(bad), bad

print("ok")
