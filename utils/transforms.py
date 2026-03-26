import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(split: str, input_resolution: int = 200) -> A.Compose:
    """
    Return albumentations transform pipeline for a given split.

    Train:     RandomResizedCrop → HorizontalFlip → colour jitter → light blur
               → JPEG compression simulation → Normalize
    Val/Test:  Resize (256/224 ratio) → CenterCrop → Normalize
    Inference: Resize to input_resolution → Normalize  (no cropping — preserves
               full user-uploaded image content for the API)
    """
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(
                size=(input_resolution, input_resolution),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
            ),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            # Simulate social-media JPEG re-compression (ArtiFact paper §3.2).
            # quality_lower=65 matches the paper's minimum; applied stochastically
            # so the model sees both compressed and uncompressed examples.
            A.ImageCompression(quality=(65, 100), p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    if split == "inference":
        # Direct resize — no cropping — so uploaded images are not spatially
        # clipped before the model sees them.
        return A.Compose([
            A.Resize(height=input_resolution, width=input_resolution),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    # val / test — deterministic
    resize_to = int(input_resolution * 256 / 224)  # standard 256→224 ratio
    return A.Compose([
        A.Resize(height=resize_to, width=resize_to),
        A.CenterCrop(height=input_resolution, width=input_resolution),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
