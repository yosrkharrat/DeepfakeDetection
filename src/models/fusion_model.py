"""Dual-stream fusion classifier for deepfake detection."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """Fuse RGB and FFT feature embeddings for binary classification."""

    RGB_FEATURE_DIM = 512
    FFT_FEATURE_DIM = 256
    FUSED_FEATURE_DIM = RGB_FEATURE_DIM + FFT_FEATURE_DIM

    def __init__(
        self,
        rgb_model: nn.Module,
        fft_model: nn.Module,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()
        self.rgb_model = rgb_model
        self.fft_model = fft_model

        self._rgb_headless_layers = self._build_headless_layers(rgb_model)
        self._fft_headless_layers = self._build_headless_layers(fft_model)

        self.classifier = nn.Sequential(
            nn.Linear(self.FUSED_FEATURE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        self.set_backbone_trainable(trainable=not freeze_backbones)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the RGB and FFT backbones in-place."""
        for backbone in (self.rgb_model, self.fft_model):
            for parameter in backbone.parameters():
                parameter.requires_grad = trainable

    def forward(self, rgb_input: torch.Tensor, fft_input: torch.Tensor) -> torch.Tensor:
        rgb_features = self._extract_features(
            model=self.rgb_model,
            inputs=rgb_input,
            expected_dim=self.RGB_FEATURE_DIM,
            headless_layers=self._rgb_headless_layers,
            stream_name="RGB",
        )
        fft_features = self._extract_features(
            model=self.fft_model,
            inputs=fft_input,
            expected_dim=self.FFT_FEATURE_DIM,
            headless_layers=self._fft_headless_layers,
            stream_name="FFT",
        )

        fused_features = torch.cat([rgb_features, fft_features], dim=1)
        return self.classifier(fused_features)

    @staticmethod
    def _build_headless_layers(model: nn.Module) -> Optional[Sequence[nn.Module]]:
        children = list(model.children())
        if len(children) <= 1:
            return None
        return tuple(children[:-1])

    @staticmethod
    def _run_headless_layers(
        layers: Sequence[nn.Module],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs
        for layer in layers:
            x = layer(x)
        return x

    @staticmethod
    def _as_feature_matrix(output: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(output):
            raise TypeError(f"Expected a tensor output, received {type(output)!r}.")

        if output.dim() == 1:
            output = output.unsqueeze(0)
        if output.dim() > 2:
            output = torch.flatten(output, start_dim=1)
        if output.dim() != 2:
            raise ValueError(f"Expected a 2D feature matrix, received shape {tuple(output.shape)}.")
        return output

    @classmethod
    def _extract_features(
        cls,
        model: nn.Module,
        inputs: torch.Tensor,
        expected_dim: int,
        headless_layers: Optional[Sequence[nn.Module]],
        stream_name: str,
    ) -> torch.Tensor:
        strategies: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = []

        forward_features = getattr(model, "forward_features", None)
        if callable(forward_features):
            strategies.append(("forward_features", forward_features))

        extract_features = getattr(model, "extract_features", None)
        if callable(extract_features):
            strategies.append(("extract_features", extract_features))

        backbone = getattr(model, "backbone", None)
        if callable(backbone):
            strategies.append(("backbone", backbone))

        strategies.append(("forward", model))

        if headless_layers is not None:
            strategies.append(
                (
                    "trimmed_forward",
                    lambda tensor, layers=headless_layers: cls._run_headless_layers(layers, tensor),
                )
            )

        failures: list[str] = []
        for name, strategy in strategies:
            try:
                features = cls._as_feature_matrix(strategy(inputs))
            except Exception as exc:
                failures.append(f"{name} failed: {exc}")
                continue

            if features.shape[1] == expected_dim:
                return features

            failures.append(
                f"{name} returned shape {tuple(features.shape)} instead of (*, {expected_dim})"
            )

        failure_text = "; ".join(failures) if failures else "no extraction strategies available"
        raise RuntimeError(f"{stream_name} backbone did not expose {expected_dim}-d features: {failure_text}")


if __name__ == "__main__":
    from src.models.fft_stream import FFTStreamCNN
    from src.models.rgb_stream import RGBStreamResNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rgb_backbone = RGBStreamResNet(pretrained=False).to(device)
    fft_backbone = FFTStreamCNN().to(device)
    fusion_model = FusionModel(rgb_backbone, fft_backbone, freeze_backbones=True).to(device)

    rgb_batch = torch.randn(4, 3, 224, 224, device=device)
    fft_batch = torch.randn(4, 1, 224, 224, device=device)
    logits = fusion_model(rgb_batch, fft_batch)

    print(f"Fusion logits shape: {logits.shape}")
