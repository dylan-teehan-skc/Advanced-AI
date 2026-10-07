"""Single source of image preprocessing. Import from here; do not re-implement elsewhere.

Every image goes 64x64 PIL -> bicubic resize to 224x224 -> tensor in [0, 1] -> normalise (mean 0.5, std 0.5) -> [-1, 1].
Augmentation is one of the 8 right-angle orientations (4 rotations x optional flip), applied after resizing,
so a train_transform sample is always exactly one of eight_views(img). That is what makes the cached 8-view
features for the frozen head and the live LoRA augmentation the same augmentation (Plan, decision 2).

Self-check: python -m src.transforms
"""
import torch
from torchvision.transforms import v2 as T

# ponytail: 0.5/0.5 matches google/vit-base-patch16-224-in21k's preprocessor_config.json; bicubic is our choice
_base = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),  # on PIL, so output stays in [0, 255]
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


def orient(x, k, flip):
    """Rotate a CxHxW tensor by k * 90 degrees, then optionally flip horizontally. Lossless pixel moves."""
    x = torch.rot90(x, k, dims=(1, 2))
    return x.flip(2) if flip else x


def eval_transform(img):
    """Validation, test and single-view features: resize and normalise only."""
    return _base(img)


def train_transform(img):
    """Training: one of the 8 orientations, drawn from torch's RNG so torch.manual_seed controls it."""
    k = int(torch.randint(4, ()))
    flip = bool(torch.randint(2, ()))
    return orient(_base(img), k, flip)


def eight_views(img):
    """All 8 orientations as an 8x3x224x224 tensor. View 0 is identical to eval_transform(img)."""
    x = _base(img)
    return torch.stack([orient(x, k, flip) for flip in (False, True) for k in range(4)])


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8))

    x = eval_transform(img)
    assert x.shape == (3, 224, 224) and x.min() >= -1 and x.max() <= 1, (x.shape, x.min(), x.max())

    views = eight_views(img)
    assert views.shape == (8, 3, 224, 224)
    assert torch.equal(views[0], x), "view 0 must equal the eval transform"
    assert all(not torch.equal(views[i], views[j]) for i in range(8) for j in range(i + 1, 8)), "views not distinct"

    torch.manual_seed(0)
    seen = set()
    for _ in range(200):
        t = train_transform(img)
        matches = [i for i in range(8) if torch.equal(t, views[i])]
        assert len(matches) == 1, "train sample is not one of the 8 views"
        seen.add(matches[0])
    assert seen == set(range(8)), f"train transform only produced views {seen}"

    print("transforms OK")
