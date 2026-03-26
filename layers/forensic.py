from __future__ import annotations # allow class methods to reference the class itself in type hints without string literals

from pathlib import Path # A convenient class for handling filesystem paths, used for loading config files

import numpy as np
import timm # A popular library of pretrained vision models, used here to load ConvNeXt-Large with ImageNet weights
import torch # Gives access to everything PyTorch offers - creating tensors, moving data to GPU, saving and loading models, math operations.
import torch.nn as nn # Stands for neural networks, it's a submodule of PyTorch that contains specifically the building blocks for constructing neural network architectures.
import yaml # A library for parsing YAML files, esentially a file format for writing configuration and settings in a very human-readable way
from PIL import Image

from utils.dataset import CLASS_NAMES
from utils.transforms import get_transforms


class ForensicDetector(nn.Module):
    """
    ConvNeXt-Large based fake-image detector for the ArtiFact benchmark.

    Architecture
    ------------
    - Backbone : ConvNeXt-Large, pretrained on ImageNet-1k via timm.
    - FSR      : When enabled, the stem Conv2d stride is halved (4 → 2) while
                 retaining all pretrained weights.  This preserves high-frequency
                 compression artefacts that a stride-4 stem would discard.
    - Head     : The 1000-class ImageNet head is replaced with a fresh
                 Linear(1536, num_classes) layer.

    Usage
    -----
    # Build from config
    model = ForensicDetector.from_config("configs/layer_a.yaml")

    # Training
    logits = model(batch_tensor)          # (B, 7)

    # Inference (single image)
    result = model.predict(pil_image)
    # → {"synthetic_score": 0.94, "predicted_class": "StyleGan2",
    #    "class_probabilities": {"Real": 0.06, "StyleGan2": 0.73, ...}}
    """

    _REAL_IDX: int = 0  # CLASS_NAMES[0] == "Real"

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        input_resolution: int,
        fsr: bool,
    ) -> None:
        super().__init__()

        self.input_resolution = input_resolution
        self._inference_transform = get_transforms("inference", input_resolution)

        # ------------------------------------------------------------------ #
        # Backbone — load with pretrained ImageNet weights                    #
        # ------------------------------------------------------------------ #
        self.model = timm.create_model(backbone, pretrained=True)

        # ------------------------------------------------------------------ #
        # FSR: Filter Stride Reduction                                        #
        # Halve the stem stride (4 → 2) to retain spatial detail lost during  #
        # JPEG compression and resize.  Pretrained weights are preserved;     #
        # only the stride parameter changes.                                  #
        # ------------------------------------------------------------------ #
        if fsr:
            self._apply_fsr()

        # ------------------------------------------------------------------ #
        # Head — replace the 1000-class classifier with num_classes outputs  #
        # ------------------------------------------------------------------ #
        # reset_classifier is timm's canonical API; it replaces head.fc and
        # keeps the pooling/norm layers intact.
        self.model.reset_classifier(num_classes)
        assert num_classes == len(CLASS_NAMES), (
            f"num_classes={num_classes} but CLASS_NAMES has {len(CLASS_NAMES)} entries"
        )

    # ---------------------------------------------------------------------- #
    # Constructor helpers                                                     #
    # ---------------------------------------------------------------------- #

    @classmethod
    def from_config(cls, config_path: str | Path) -> "ForensicDetector":
        """Instantiate from a layer YAML config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open() as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        return cls(
            backbone=m["backbone"],
            num_classes=m["num_classes"],
            input_resolution=m["input_resolution"],
            fsr=m.get("fsr", False),
        )

    def _apply_fsr(self) -> None:
        """
        Replace stem[0] Conv2d(stride=4) with Conv2d(stride=2).

        ConvNeXt-Large stem:
            stem[0]  Conv2d(3, 192, kernel_size=4, stride=4, padding=0)
            stem[1]  LayerNorm2d(192)

        Only the stride attribute changes; all pretrained filter weights are
        copied exactly so the learned low-level features are preserved.
        """
        old = self.model.stem[0]  # Conv2d(3, 192, k=4, stride=4, padding=0)

        new = nn.Conv2d(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=(2, 2),              # FSR: 4 → 2
            padding=old.padding,
            bias=old.bias is not None,
        )

        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)

        self.model.stem[0] = new

        # Sanity check — runs once at init, never again
        assert new.out_channels == old.out_channels, (
            f"FSR sanity check failed: stem out_channels changed from "
            f"{old.out_channels} to {new.out_channels}"
        )
        print(f"[FSR] stem[0] after modification: {self.model.stem[0]}")

    # ---------------------------------------------------------------------- #
    # Forward — training path                                                 #
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of preprocessed images, shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (B, num_classes).  Apply softmax / cross-entropy
            externally (do not softmax inside forward — loss functions expect
            raw logits).
        """
        return self.model(x)

    # ---------------------------------------------------------------------- #
    # Predict — inference path                                                #
    # ---------------------------------------------------------------------- #

    def predict(self, image: Image.Image | np.ndarray) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        image : PIL.Image.Image | np.ndarray
            RGB image in any common format.  Numpy arrays may be HWC uint8 or
            HWC float32.  Grayscale images are broadcast to 3 channels.

        Returns
        -------
        dict with keys:
            synthetic_score      float [0, 1] — probability of being AI-generated
                                 (1 − P(Real))
            class_probabilities  dict mapping each class name to its probability
            predicted_class      str — the most probable class name
        """
        image_np = self._to_numpy_rgb(image)

        tensor: torch.Tensor = self._inference_transform(image=image_np)["image"]
        tensor = tensor.unsqueeze(0).to(self._device)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self(tensor)                            # (1, num_classes)
            probs  = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
        if was_training:
            self.train()

        probs_list = probs.cpu().tolist()

        return {
            "synthetic_score":     round(1.0 - probs_list[self._REAL_IDX], 6),
            "class_probabilities": {
                name: round(probs_list[i], 6)
                for i, name in enumerate(CLASS_NAMES)
            },
            "predicted_class": CLASS_NAMES[int(probs.argmax())],
        }

    # ---------------------------------------------------------------------- #
    # Internal utilities                                                      #
    # ---------------------------------------------------------------------- #

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _to_numpy_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
        """Normalise any input image to HWC uint8 numpy RGB."""
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        # numpy path
        arr = np.array(image)
        if arr.dtype != np.uint8:
            # float [0, 1] → uint8
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)  # grayscale → RGB
        return arr
